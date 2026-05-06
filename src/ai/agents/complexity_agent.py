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
    return max(0, min(2, delta))


def run_complexity_review(issue_context: Dict[str, Any], llm: LLMClient) -> Dict[str, Any]:
    role = "You are a senior software architect focused on hidden technical complexity."
    instruction = f"""
Review the issue below without using historical analogies.

Your job is not to re-estimate the whole issue in hours.
Your job is to indicate whether hidden technical complexity should increase the calibrated base estimate.

Rules:
- Return a bounded bucket delta only: 0, 1, or 2.
- Default to 0.
- Use 0 when the work still looks local or isolated.
- Use 1 only when the issue clearly touches more than one technical surface, such as API plus database, runtime plus build pipeline, auth plus deployment, or investigation plus non-trivial fix.
- Use 2 only when the text strongly suggests broad technical change and likely split/refinement.
- Prefer confidence and warnings over inflating the delta.

Return JSON only:
{{
  "mode": "complexity_review",
  "bucket_delta": 0,
  "confidence": 0.0,
  "risk_hidden_complexity": 0.0,
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
        confidence = max(0.05, min(0.95, float(parsed.get("confidence", 0.5) or 0.5)))
        risk_hidden_complexity = max(0.0, min(1.0, float(parsed.get("risk_hidden_complexity", 0.0) or 0.0)))
        bucket_delta = _normalize_delta(parsed.get("bucket_delta"))
        if bucket_delta == 2 and not bool(parsed.get("should_split", False)):
            bucket_delta = 1
        if bucket_delta > 0 and (risk_hidden_complexity < 0.78 or confidence < 0.82):
            bucket_delta = 0

        out = {
            "mode": "complexity_review",
            "bucket_delta": bucket_delta,
            "confidence": confidence,
            "risk_hidden_complexity": risk_hidden_complexity,
            "justification": str(parsed.get("justification", "") or ""),
            "evidence": list(parsed.get("evidence", []) or []),
            "warnings": list(parsed.get("warnings", []) or []),
            "assumptions": list(parsed.get("assumptions", []) or []),
            "should_split": bool(parsed.get("should_split", False)),
            "split_reason": parsed.get("split_reason"),
            "source": "complexity_review",
            "token_usage": coerce_token_usage(llm.get_last_token_usage()),
        }
        return out
    except Exception as exc:
        return {
            "mode": "complexity_review",
            "bucket_delta": 0,
            "confidence": 0.25,
            "risk_hidden_complexity": 0.0,
            "justification": "Resposta de contingência da análise de complexidade.",
            "evidence": [],
            "warnings": [str(exc)],
            "assumptions": [],
            "should_split": False,
            "split_reason": None,
            "source": "complexity_review",
            "token_usage": coerce_token_usage(llm.get_last_token_usage()),
        }
