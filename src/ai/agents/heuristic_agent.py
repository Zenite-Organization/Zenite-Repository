from typing import Any, Dict
import json

from ai.core.effort_calibration import (
    bucket_rank_to_default_range_index,
    bucket_to_rank,
    clamp_bucket_rank,
    clamp_range_index,
    range_index_to_bucket_rank,
    range_index_to_label,
    range_index_to_payload,
    rank_to_bucket,
)
from ai.core.llm_client import LLMClient
from ai.core.json_utils import parse_llm_json_response
from ai.core.prompt_utils import build_system_prompt
from ai.core.token_usage import coerce_token_usage


MODE_GUIDANCE = {
    "scope": {
        "role": "You are a senior analyst focused on functional scope and delivery breadth.",
        "focus": [
            "number of concrete deliverables hidden in the request",
            "whether the issue reads like a local fix or a broad change",
            "how many modules or business rules appear to be touched",
            "whether the work sounds like one outcome or multiple outcomes grouped together",
        ],
    },
    "complexity": {
        "role": "You are a senior engineer focused on technical complexity and coupling.",
        "focus": [
            "integration points and contracts that may be affected",
            "database, api, auth, queue, cache, or observability changes",
            "likelihood of regression-sensitive testing",
            "hidden technical coordination implied by the text",
        ],
    },
    "uncertainty": {
        "role": "You are a senior analyst focused on ambiguity and discovery risk.",
        "focus": [
            "missing details or vague acceptance criteria",
            "need for investigation before implementation",
            "dependency on clarification or alignment",
            "signals that confidence should go down without automatically pushing size up",
        ],
    },
    "agile_fit": {
        "role": "You are an agile lead focused on healthy issue sizing and split readiness.",
        "focus": [
            "whether the issue fits as a single backlog item",
            "signals of epic-like aggregation",
            "whether the request should be split into smaller items",
            "whether the wording indicates too much work for one issue",
        ],
    },
}


def _build_prompt(issue_context: Dict[str, Any], mode: str) -> str:
    guidance = MODE_GUIDANCE[mode]
    available_ranges = [
        "1=1-3h",
        "2=3-6h",
        "3=6-9h",
        "4=9-12h",
        "5=12-15h",
        "6=15-18h",
        "7=18-21h",
        "8=21-24h",
        "9=24-27h",
        "10=27-30h",
        "11=30-33h",
        "12=33-36h",
        "13=36-40h",
    ]
    instruction = f"""
You received the full context of a software issue.

Your task is to estimate the most plausible effort range, not an exact number of hours.

Rules:
- Choose one ordinal range only from this catalog:
{chr(10).join(f"- {item}" for item in available_ranges)}
- Prefer lower ranges for local fixes, small validations, text/config changes, single-screen defects, and isolated bug fixes.
- Use 9-15h when the issue implies investigation plus implementation, multi-step debugging, or more than one technical surface.
- Use 15-24h when the issue implies investigation plus correction plus validation across more than one subsystem, integration path, environment, or operational surface.
- Use 21h or above when the wording strongly suggests aggregate scope, likely split, multiple deliverables, or broad technical coordination.
- Do not compress everything into 6-12h by default.
- Do not inflate the range only because the issue looks risky, vague, or mentions an exception.
- Use uncertainty to reduce confidence first. Only move the range up if the text clearly indicates broader work.
- If the issue sounds large, prefer expressing that through the range rather than a generic warning.

Primary focus for this mode:
{guidance["role"]}

What to analyze:
{chr(10).join(f"- {item}" for item in guidance["focus"])}

Return JSON only:
{{
  "mode": "{mode}",
  "range_index": 1,
  "range_label": "1-3h|3-6h|6-9h|9-12h|12-15h|15-18h|18-21h|21-24h|24-27h|27-30h|30-33h|33-36h|36-40h",
  "confidence": 0.0,
  "justification": "short objective text",
  "evidence": ["short list"],
  "warnings": ["short list"],
  "assumptions": ["short list"]
}}

Issue:
{json.dumps(issue_context, ensure_ascii=False, indent=2)}
"""
    return build_system_prompt(guidance["role"], instruction)


def _normalize_bucket_payload(mode: str, parsed: Dict[str, Any]) -> Dict[str, Any]:
    raw_range_index = parsed.get("range_index")
    raw_range_label = str(parsed.get("range_label") or "").strip().lower()
    if raw_range_index in (None, "") and raw_range_label:
        for idx in range(1, 14):
            if raw_range_label == range_index_to_label(idx).lower():
                raw_range_index = idx
                break
    if raw_range_index in (None, ""):
        raw_rank = parsed.get("bucket_rank")
        if raw_rank in (None, ""):
            bucket_rank = bucket_to_rank(parsed.get("size_bucket"))
        else:
            bucket_rank = clamp_bucket_rank(raw_rank)
        range_index = bucket_rank_to_default_range_index(bucket_rank)
    else:
        range_index = clamp_range_index(raw_range_index)
    range_payload = range_index_to_payload(range_index)
    bucket_rank = range_index_to_bucket_rank(range_index)
    size_bucket = rank_to_bucket(bucket_rank)

    return {
        "mode": mode,
        "size_bucket": size_bucket,
        "bucket_rank": bucket_rank,
        "estimated_hours": float(range_payload["display_hours"]),
        "estimated_hours_raw": float(range_payload["display_hours"]),
        "min_hours": float(range_payload["range_min_hours"]),
        "max_hours": float(range_payload["range_max_hours"]),
        "range_index": range_payload["range_index"],
        "range_label": range_payload["range_label"],
        "range_min_hours": range_payload["range_min_hours"],
        "range_max_hours": range_payload["range_max_hours"],
        "display_hours": range_payload["display_hours"],
        "confidence": max(0.05, min(0.95, float(parsed.get("confidence", 0.5) or 0.5))),
        "justification": str(parsed.get("justification", "") or ""),
        "evidence": list(parsed.get("evidence", []) or []),
        "warnings": list(parsed.get("warnings", []) or []),
        "assumptions": list(parsed.get("assumptions", []) or []),
    }


def _fallback(mode: str) -> Dict[str, Any]:
    range_payload = range_index_to_payload(4)
    return {
        "mode": mode,
        "size_bucket": rank_to_bucket(range_index_to_bucket_rank(range_payload["range_index"])),
        "bucket_rank": range_index_to_bucket_rank(range_payload["range_index"]),
        "estimated_hours": float(range_payload["display_hours"]),
        "estimated_hours_raw": float(range_payload["display_hours"]),
        "min_hours": float(range_payload["range_min_hours"]),
        "max_hours": float(range_payload["range_max_hours"]),
        "range_index": range_payload["range_index"],
        "range_label": range_payload["range_label"],
        "range_min_hours": range_payload["range_min_hours"],
        "range_max_hours": range_payload["range_max_hours"],
        "display_hours": range_payload["display_hours"],
        "confidence": 0.25,
        "justification": f"Resposta de contingência aplicada ao modo heurístico {mode}.",
        "evidence": [],
        "warnings": ["Não foi possível interpretar a resposta do modelo."],
        "assumptions": [],
    }


def run_heuristic(
    issue_context: Dict[str, Any],
    llm: LLMClient,
    temperature: float = 0.0,
    mode: str = "scope",
) -> Dict[str, Any]:
    if mode not in MODE_GUIDANCE:
        raise ValueError(f"Unsupported heuristic mode: {mode}")

    try:
        prompt = _build_prompt(issue_context, mode)
        raw = llm.send_prompt(prompt, temperature=temperature)
        parsed = parse_llm_json_response(raw)
        normalized = _normalize_bucket_payload(mode, parsed)
        normalized["source"] = mode
        normalized["token_usage"] = coerce_token_usage(llm.get_last_token_usage())
        return normalized
    except Exception as exc:
        data = _fallback(mode)
        data["warnings"].append(str(exc))
        data["source"] = mode
        data["token_usage"] = coerce_token_usage(llm.get_last_token_usage())
        return data
