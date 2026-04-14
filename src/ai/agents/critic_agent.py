from typing import Any, Dict, List
import json

from ai.core.llm_client import LLMClient
from ai.core.json_utils import parse_llm_json_response
from ai.core.prompt_utils import build_system_prompt
from ai.core.token_usage import coerce_token_usage


def run_estimation_critic(
    issue_context: Dict[str, Any],
    analogical: Dict[str, Any] | None,
    heuristic_candidates: List[Dict[str, Any]],
    complexity_review: Dict[str, Any] | None,
    agile_guard_review: Dict[str, Any] | None,
    llm: LLMClient,
) -> Dict[str, Any]:
    role = "You are a critical reviewer of software effort estimates."
    payload = {
        "issue": issue_context,
        "analogical": analogical,
        "heuristics": heuristic_candidates,
        "complexity_review": complexity_review,
        "agile_guard_review": agile_guard_review,
    }
    instruction = f"""
Review the signals already produced for the same issue.

Your job:
- identify contradictions between the reviewers
- estimate risk of underestimation
- estimate risk of overestimation
- state the strongest signal for final consolidation
- avoid generating a brand new point estimate

Return JSON only:
{{
  "risk_of_underestimation": 0.0,
  "risk_of_overestimation": 0.0,
  "contradictions": ["short list"],
  "hidden_complexities": ["short list"],
  "strongest_signal": "short text",
  "recommendation": "short text"
}}

Data:
{json.dumps(payload, ensure_ascii=False, indent=2)}
"""
    try:
        raw = llm.send_prompt(build_system_prompt(role, instruction))
        parsed = parse_llm_json_response(raw)
        parsed["token_usage"] = coerce_token_usage(llm.get_last_token_usage())
        return parsed
    except Exception as exc:
        return {
            "risk_of_underestimation": 0.5,
            "risk_of_overestimation": 0.5,
            "contradictions": [f"Critic fallback: {exc}"],
            "hidden_complexities": [],
            "strongest_signal": "No reliable critic output.",
            "recommendation": "Keep the calibrated estimate conservative and confidence moderate.",
            "token_usage": coerce_token_usage(llm.get_last_token_usage()),
        }
