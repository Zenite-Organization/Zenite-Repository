from typing import Dict, Any, List
import json
import re

from ai.core.effort_calibration import weighted_neighbor_estimate
from ai.core.llm_client import LLMClient
from ai.core.json_utils import parse_llm_json_response
from ai.core.prompt_utils import build_system_prompt
from ai.core.token_usage import coerce_token_usage


STRONG_ANCHOR_SCORE = 0.90
STRONG_ANCHOR_OVERLAP = 0.25
USEFUL_SCORE_THRESHOLD = 0.70
PRIMARY_TOP1_MIN = 0.86
PRIMARY_TOP3_MIN = 0.76
PRIMARY_USEFUL_MIN = 2
PRIMARY_SPREAD_MAX = 2.4
SUPPORT_TOP1_MIN = 0.72
SUPPORT_TOP3_MIN = 0.68
SUPPORT_USEFUL_MIN = 1
SOFT_SIGNAL_TOP1_MIN = 0.64
SOFT_SIGNAL_TOP3_MIN = 0.60
SOFT_SIGNAL_ANCHOR_OVERLAP_MIN = 0.18
SUPPORT_ANCHOR_OVERLAP_MIN = 0.22
TOP_K = 10


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _normalize_text(text: str) -> List[str]:
    cleaned = re.sub(r"[^a-z0-9\s]+", " ", str(text or "").lower())
    return [token for token in cleaned.split() if token]


def _jaccard_similarity(left: str, right: str) -> float:
    left_tokens = set(_normalize_text(left))
    right_tokens = set(_normalize_text(right))
    if not left_tokens or not right_tokens:
        return 0.0
    union = left_tokens | right_tokens
    if not union:
        return 0.0
    return len(left_tokens & right_tokens) / len(union)


def _prepare_similar_issues(similar_issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ordered = sorted(
        similar_issues or [],
        key=lambda item: _safe_float(item.get("score"), 0.0),
        reverse=True,
    )[:TOP_K]

    cleaned: List[Dict[str, Any]] = []
    for item in ordered:
        cleaned.append(
            {
                "issue_id": item.get("issue_id") or item.get("id"),
                "title": item.get("title"),
                "description": (item.get("description") or "")[:800],
                "issue_type": item.get("issue_type"),
                "labels": item.get("labels") or [],
                "project_key": item.get("project_key"),
                "total_effort_hours": _safe_float(item.get("total_effort_hours"), 0.0),
                "score": _safe_float(item.get("score"), 0.0),
            }
        )
    return cleaned


def _compute_retrieval_stats(
    issue_context: Dict[str, Any],
    similar_issues: List[Dict[str, Any]],
) -> Dict[str, Any]:
    prepared = _prepare_similar_issues(similar_issues)
    if not prepared:
        return {
            "top_k_count": 0,
            "top1_score": 0.0,
            "top3_avg_score": 0.0,
            "useful_count": 0,
            "hours_spread": 0.0,
            "has_strong_anchor": False,
            "anchor_score": 0.0,
            "anchor_overlap": 0.0,
            "route": "analogical_weak",
        }

    scores = [_safe_float(item.get("score"), 0.0) for item in prepared]
    hours = [
        _safe_float(item.get("total_effort_hours"), 0.0)
        for item in prepared
        if _safe_float(item.get("total_effort_hours"), 0.0) > 0
    ]
    useful_count = sum(1 for score in scores if score >= USEFUL_SCORE_THRESHOLD)
    top1_score = scores[0]
    top3_avg_score = sum(scores[:3]) / max(1, min(3, len(scores)))

    issue_title = str(issue_context.get("title") or "")
    anchor_candidate = prepared[0]
    anchor_score = _safe_float(anchor_candidate.get("score"), 0.0)
    anchor_overlap = _jaccard_similarity(issue_title, str(anchor_candidate.get("title") or ""))
    has_strong_anchor = (
        anchor_score >= STRONG_ANCHOR_SCORE
        and anchor_overlap >= STRONG_ANCHOR_OVERLAP
    )

    if len(hours) >= 2:
        ordered_hours = sorted(hours)
        center = ordered_hours[len(ordered_hours) // 2]
        spread_denom = max(abs(center), 1.0)
        hours_spread = (max(ordered_hours) - min(ordered_hours)) / spread_denom
    else:
        hours_spread = 0.0

    if has_strong_anchor or (
        top1_score >= PRIMARY_TOP1_MIN
        and top3_avg_score >= PRIMARY_TOP3_MIN
        and useful_count >= PRIMARY_USEFUL_MIN
        and hours_spread <= PRIMARY_SPREAD_MAX
    ):
        route = "analogical_primary"
    elif (
        (top1_score >= SUPPORT_TOP1_MIN and useful_count >= SUPPORT_USEFUL_MIN)
        or (top1_score >= 0.74 and top3_avg_score >= SUPPORT_TOP3_MIN)
        or (top1_score >= 0.70 and useful_count >= 2)
        or (top1_score >= 0.76 and anchor_overlap >= SUPPORT_ANCHOR_OVERLAP_MIN)
    ):
        route = "analogical_support"
    elif (
        top1_score >= SOFT_SIGNAL_TOP1_MIN
        or (top1_score >= 0.60 and top3_avg_score >= SOFT_SIGNAL_TOP3_MIN)
        or (top1_score >= 0.62 and useful_count >= 1)
        or (anchor_overlap >= SOFT_SIGNAL_ANCHOR_OVERLAP_MIN and top1_score >= 0.58)
    ):
        route = "analogical_soft_signal"
    else:
        route = "analogical_weak"

    return {
        "top_k_count": len(prepared),
        "top1_score": round(top1_score, 4),
        "top3_avg_score": round(top3_avg_score, 4),
        "useful_count": useful_count,
        "hours_spread": round(hours_spread, 4),
        "has_strong_anchor": bool(has_strong_anchor),
        "anchor_score": round(anchor_score, 4),
        "anchor_overlap": round(anchor_overlap, 4),
        "route": route,
    }


def _llm_explanation(
    issue_context: Dict[str, Any],
    prepared_similar: List[Dict[str, Any]],
    deterministic_result: Dict[str, Any],
    retrieval_stats: Dict[str, Any],
    repository_technologies: Dict[str, float],
    llm: LLMClient,
) -> Dict[str, Any]:
    payload = {
        "issue": {
            "issue_number": issue_context.get("issue_number"),
            "repository": issue_context.get("repository"),
            "issue_type": issue_context.get("issue_type"),
            "title": issue_context.get("title"),
            "description": issue_context.get("description"),
            "labels": issue_context.get("labels", []),
        },
        "deterministic_estimate": deterministic_result,
        "retrieval_stats": retrieval_stats,
        "similar_issues": prepared_similar[:5],
        "repository_technologies": repository_technologies,
    }
    role = "You are a senior engineer explaining an analogical effort estimate."
    instruction = f"""
You already received a deterministic analogical estimate computed from historical neighbors.

Do not invent a new point estimate.
Explain whether the deterministic analogical estimate looks reasonable.

Return JSON only:
{{
  "justification": "short text",
  "evidence": ["short list"],
  "warnings": ["short list"],
  "assumptions": ["short list"],
  "should_split": false,
  "split_reason": null
}}

Data:
{json.dumps(payload, ensure_ascii=False, indent=2)}
"""
    raw = llm.send_prompt(build_system_prompt(role, instruction))
    parsed = parse_llm_json_response(raw)
    return {
        "justification": str(parsed.get("justification", "") or ""),
        "evidence": list(parsed.get("evidence", []) or []),
        "warnings": list(parsed.get("warnings", []) or []),
        "assumptions": list(parsed.get("assumptions", []) or []),
        "should_split": bool(parsed.get("should_split", False)),
        "split_reason": parsed.get("split_reason"),
    }


def run_analogical(
    issue_context: Dict[str, Any],
    similar_issues: List[Dict[str, Any]],
    repository_technologies: Dict[str, float],
    llm: LLMClient,
) -> Dict[str, Any]:
    prepared_similar = _prepare_similar_issues(similar_issues)
    retrieval_stats = _compute_retrieval_stats(issue_context, prepared_similar)
    deterministic_result = weighted_neighbor_estimate(
        issue_context=issue_context,
        similar_issues=prepared_similar,
        useful_score_threshold=USEFUL_SCORE_THRESHOLD,
        top_k=5,
    )

    out = {
        "mode": "analogical",
        "estimated_hours": deterministic_result.get("estimated_hours"),
        "min_hours": deterministic_result.get("min_hours"),
        "max_hours": deterministic_result.get("max_hours"),
        "confidence": deterministic_result.get("confidence"),
        "size_bucket": deterministic_result.get("size_bucket"),
        "bucket_rank": deterministic_result.get("bucket_rank"),
        "base_hours": deterministic_result.get("estimated_hours"),
        "calibration_source": deterministic_result.get("calibration_source"),
        "neighbor_count": deterministic_result.get("neighbor_count"),
        "supporting_hours": deterministic_result.get("supporting_hours"),
        "weighted_hours": deterministic_result.get("weighted_hours"),
        "source": "analogical",
        "retrieval_stats": retrieval_stats,
        "retrieval_route": retrieval_stats.get("route"),
        "should_split": False,
        "split_reason": None,
        "justification": "Deterministic analogical estimate from historical neighbors.",
        "evidence": [],
        "warnings": [],
        "assumptions": [],
    }

    try:
        explanation = _llm_explanation(
            issue_context=issue_context,
            prepared_similar=prepared_similar,
            deterministic_result=deterministic_result,
            retrieval_stats=retrieval_stats,
            repository_technologies=repository_technologies,
            llm=llm,
        )
        out.update(explanation)
        out["token_usage"] = coerce_token_usage(llm.get_last_token_usage())
        return out
    except Exception as exc:
        out["warnings"] = [str(exc)]
        out["token_usage"] = coerce_token_usage(llm.get_last_token_usage())
        return out


__all__ = ["run_analogical", "_compute_retrieval_stats"]
