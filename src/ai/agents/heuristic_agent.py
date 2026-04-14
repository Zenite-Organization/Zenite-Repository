from typing import Any, Dict
import json

from ai.core.effort_calibration import bucket_to_rank, clamp_bucket_rank, rank_to_bucket
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
    instruction = f"""
You received the full context of a software issue.

Your task is to estimate only the relative size bucket, not direct hours.

Rules:
- Work with ordinal effort buckets only: XS, S, M, L, XL, XXL.
- Treat these buckets as relative project size classes, not as a fixed hour table.
- Prefer XS/S for local fixes, small validations, text/config changes, single-screen defects, and isolated bug fixes.
- Use M when the issue implies investigation plus implementation, multi-step debugging, SDK/runtime/version compatibility work, CI/CD or certificate remediation, or more than one technical surface.
- Use L or above only when the text strongly suggests broader multi-component change, cross-cutting refactor, or aggregate scope.
- Do not inflate the bucket only because the issue looks risky, vague, or mentions an exception.
- Use uncertainty to reduce confidence first. Only move the bucket up if the text clearly indicates broader work.
- Do not default to S when the issue clearly requires both root-cause analysis and a concrete fix.

Primary focus for this mode:
{guidance["role"]}

What to analyze:
{chr(10).join(f"- {item}" for item in guidance["focus"])}

Return JSON only:
{{
  "mode": "{mode}",
  "size_bucket": "XS|S|M|L|XL|XXL",
  "bucket_rank": 1,
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
    raw_rank = parsed.get("bucket_rank")
    if raw_rank in (None, ""):
        bucket_rank = bucket_to_rank(parsed.get("size_bucket"))
    else:
        bucket_rank = clamp_bucket_rank(raw_rank)
    size_bucket = str(parsed.get("size_bucket") or rank_to_bucket(bucket_rank)).strip().upper()
    if size_bucket not in {"XS", "S", "M", "L", "XL", "XXL"}:
        size_bucket = rank_to_bucket(bucket_rank)

    return {
        "mode": mode,
        "size_bucket": size_bucket,
        "bucket_rank": bucket_rank,
        "confidence": max(0.05, min(0.95, float(parsed.get("confidence", 0.5) or 0.5))),
        "justification": str(parsed.get("justification", "") or ""),
        "evidence": list(parsed.get("evidence", []) or []),
        "warnings": list(parsed.get("warnings", []) or []),
        "assumptions": list(parsed.get("assumptions", []) or []),
    }


def _fallback(mode: str) -> Dict[str, Any]:
    return {
        "mode": mode,
        "size_bucket": "M",
        "bucket_rank": 3,
        "confidence": 0.25,
        "justification": f"Fallback applied for heuristic mode {mode}.",
        "evidence": [],
        "warnings": ["Could not parse the model response."],
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
