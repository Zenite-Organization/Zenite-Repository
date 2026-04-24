from typing import Any, Dict
import json

from ai.core.llm_client import LLMClient
from ai.core.json_utils import parse_llm_json_response
from ai.core.prompt_utils import build_system_prompt
from ai.core.token_usage import coerce_token_usage


def _normalize_delta(value: Any) -> int:
    try:
        delta = int(round(float(value)))
    except Exception:
        delta = 0
    return max(-1, min(1, delta))


def run_agile_guard(issue_context: Dict[str, Any], llm: LLMClient) -> Dict[str, Any]:
    role = "You are an agile lead focused on healthy backlog item size."
    instruction = f"""
Review the issue with focus on agile fit.

Your job is not to generate a full hour estimate.
Your job is to say whether the issue looks healthy, borderline, or oversized as a single item.

Rules:
- bucket_delta can be -1, 0, or 1.
- Use -1 only when the issue clearly looks simpler and more local than it first appears.
- Use 0 in the common case.
- Use 1 only when the wording strongly suggests aggregate scope, multiple deliverables, or work that likely should be split.
- Prefer should_split and fit_status over pushing the estimate upward.
- Write every human-readable field in Brazilian Portuguese.

Return JSON only:
{{
  "mode": "agile_guard",
  "fit_status": "healthy|borderline|oversized",
  "bucket_delta": 0,
  "confidence": 0.0,
  "justification": "short text",
  "evidence": ["short list"],
  "warnings": ["short list"],
  "assumptions": ["short list"],
  "should_split": false,
  "split_reason": null
}}

Issue:
{json.dumps(issue_context, ensure_ascii=False, indent=2)}
"""
    try:
        raw = llm.send_prompt(build_system_prompt(role, instruction))
        parsed = parse_llm_json_response(raw)
        fit_status = str(parsed.get("fit_status", "healthy") or "healthy").strip().lower()
        if fit_status not in {"healthy", "borderline", "oversized"}:
            fit_status = "healthy"
        confidence = max(0.05, min(0.95, float(parsed.get("confidence", 0.5) or 0.5)))
        bucket_delta = _normalize_delta(parsed.get("bucket_delta"))
        if bucket_delta > 0 and fit_status == "healthy" and confidence < 0.8:
            bucket_delta = 0

        out = {
            "mode": "agile_guard",
            "fit_status": fit_status,
            "bucket_delta": bucket_delta,
            "confidence": confidence,
            "justification": str(parsed.get("justification", "") or ""),
            "evidence": list(parsed.get("evidence", []) or []),
            "warnings": list(parsed.get("warnings", []) or []),
            "assumptions": list(parsed.get("assumptions", []) or []),
            "should_split": bool(parsed.get("should_split", False)) or fit_status == "oversized",
            "split_reason": parsed.get("split_reason"),
            "source": "agile_guard",
            "token_usage": coerce_token_usage(llm.get_last_token_usage()),
        }
        return out
    except Exception as exc:
        return {
            "mode": "agile_guard",
            "fit_status": "healthy",
            "bucket_delta": 0,
            "confidence": 0.25,
            "justification": "Resposta de contingência da análise ágil.",
            "evidence": [],
            "warnings": [str(exc)],
            "assumptions": [],
            "should_split": False,
            "split_reason": None,
            "source": "agile_guard",
            "token_usage": coerce_token_usage(llm.get_last_token_usage()),
        }
