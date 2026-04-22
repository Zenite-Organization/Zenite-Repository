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

    # [B1.3] Extract retrieval signal so the critic can weigh analogical properly.
    # Sem isso, o crítico trata a discordância analogical vs heurístico como
    # sinal forte de viés, mesmo quando analogical tem top1_score ~0.99.
    retrieval_stats = (analogical or {}).get("retrieval_stats") or {}
    retrieval_signal = {
        "route": (analogical or {}).get("retrieval_route"),
        "top1_score": retrieval_stats.get("top1_score"),
        "top3_avg_score": retrieval_stats.get("top3_avg_score"),
        "useful_count": retrieval_stats.get("useful_count"),
        "has_strong_anchor": retrieval_stats.get("has_strong_anchor"),
        "anchor_overlap": retrieval_stats.get("anchor_overlap"),
    }

    payload = {
        "issue": issue_context,
        "retrieval_signal": retrieval_signal,
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

Rules for weighing the signals:
- When retrieval_signal.has_strong_anchor is true AND top1_score >= 0.90, the analogical
  estimate is grounded in a very similar historical issue. Treat it as highly reliable.
  In this case, keep risk_of_underestimation and risk_of_overestimation both <= 0.3 unless
  you have concrete textual evidence the analogical is misleading (e.g., the issue explicitly
  mentions multiple deliverables that the historical match did not have).
- When retrieval_signal.route is "analogical_primary" or "analogical_support", the analogical
  evidence is stronger than the heuristics. Prefer it.
- When retrieval_signal.route is "analogical_weak" or top1_score < 0.65, the analogical is
  not trustworthy. In this case, heuristic disagreement is meaningful — lean on heuristics.
- Heuristic disagreement alone (when analogical is strong) is NOT evidence of underestimation.
  The heuristics cannot see similar historical issues; their votes naturally differ from a
  strong analogical, and that difference should not inflate risk.

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
