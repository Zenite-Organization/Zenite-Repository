from typing import Any, Dict, List, Optional


AGILE_HOURS_LIMIT = 40.0


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _collect_text_list(*groups: Any) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for group in groups:
        for item in group or []:
            text = str(item or "").strip()
            if text and text not in seen:
                seen.add(text)
                out.append(text)
    return out


def _review_confidence(
    calibrated_estimation: Dict[str, Any],
    critic_review: Optional[Dict[str, Any]],
    heuristic_candidates: List[Dict[str, Any]],
) -> float:
    base = _safe_float(calibrated_estimation.get("base_confidence"), 0.55)
    route = str(calibrated_estimation.get("retrieval_route") or "")
    finalization_mode = str(calibrated_estimation.get("finalization_mode") or "")

    if route == "analogical_primary":
        base += 0.10
    elif route == "analogical_support":
        base += 0.03
    elif finalization_mode == "heuristic_bucket_calibrated":
        base -= 0.04

    ranks = []
    for candidate in heuristic_candidates or []:
        rank = candidate.get("bucket_rank")
        if rank not in (None, ""):
            ranks.append(int(rank))
    spread = max(ranks) - min(ranks) if len(ranks) >= 2 else 0
    base -= 0.05 * spread

    if critic_review:
        base -= 0.12 * _safe_float(critic_review.get("risk_of_underestimation"), 0.0)
        base -= 0.08 * _safe_float(critic_review.get("risk_of_overestimation"), 0.0)

    return max(0.08, min(0.95, round(base, 4)))


def _build_justification(
    calibrated_estimation: Dict[str, Any],
    analogical: Optional[Dict[str, Any]],
    complexity_review: Optional[Dict[str, Any]],
    agile_guard_review: Optional[Dict[str, Any]],
) -> str:
    finalization_mode = str(calibrated_estimation.get("finalization_mode") or "calibrated")
    route = str(calibrated_estimation.get("retrieval_route") or "unknown")
    size_bucket = str(calibrated_estimation.get("size_bucket") or "M")
    calibration_source = str(calibrated_estimation.get("calibration_source") or "unknown")
    adjustment_delta = _safe_float(calibrated_estimation.get("adjustment_delta"), 0.0)

    parts = [
        f"Finalized via {finalization_mode}",
        f"size bucket {size_bucket}",
        f"calibrated from {calibration_source}",
        f"retrieval route {route}",
    ]
    if analogical and analogical.get("estimated_hours") is not None:
        parts.append(f"analogical base {_safe_float(analogical.get('estimated_hours')):.1f}h")
    if adjustment_delta > 0.2:
        parts.append("bounded upward adjustment from hidden technical complexity")
    elif adjustment_delta < -0.2:
        parts.append("bounded downward adjustment after review")
    if (agile_guard_review or {}).get("should_split"):
        parts.append("agile review recommends split")
    elif (complexity_review or {}).get("should_split"):
        parts.append("technical review recommends split")
    return "; ".join(parts) + "."


def combine_multi_agent_estimations(
    issue_context: Dict[str, Any],
    strategy: str,
    analogical: Optional[Dict[str, Any]],
    heuristic_candidates: List[Dict[str, Any]],
    complexity_review: Optional[Dict[str, Any]],
    agile_guard_review: Optional[Dict[str, Any]],
    critic_review: Optional[Dict[str, Any]],
    calibrated_estimation: Optional[Dict[str, Any]] = None,
    llm: Any = None,
) -> Dict[str, Any]:
    calibrated = dict(calibrated_estimation or {})
    estimated_hours = _safe_float(
        calibrated.get("adjusted_hours", calibrated.get("base_hours")),
        8.0,
    )
    min_hours = _safe_float(calibrated.get("min_hours"), max(1.0, estimated_hours * 0.8))
    max_hours = _safe_float(calibrated.get("max_hours"), max(estimated_hours, estimated_hours * 1.2))
    should_split = (
        bool(calibrated.get("should_split"))
        or bool((agile_guard_review or {}).get("should_split"))
        or bool((complexity_review or {}).get("should_split"))
        or estimated_hours > AGILE_HOURS_LIMIT
    )
    split_reason = (
        calibrated.get("split_reason")
        or (agile_guard_review or {}).get("split_reason")
        or (complexity_review or {}).get("split_reason")
    )
    if should_split and not split_reason:
        split_reason = f"Consolidated estimate exceeds the healthy agile limit of {AGILE_HOURS_LIMIT:.0f}h."

    out = {
        "estimated_hours": round(max(1.0, estimated_hours), 1),
        "min_hours": round(max(1.0, min(min_hours, estimated_hours)), 1),
        "max_hours": round(max(max_hours, estimated_hours), 1),
        "confidence": _review_confidence(calibrated, critic_review, heuristic_candidates),
        "justification": _build_justification(
            calibrated_estimation=calibrated,
            analogical=analogical,
            complexity_review=complexity_review,
            agile_guard_review=agile_guard_review,
        ),
        "evidence": _collect_text_list(
            (analogical or {}).get("evidence"),
            calibrated.get("evidence"),
            (complexity_review or {}).get("evidence"),
            (agile_guard_review or {}).get("evidence"),
        ),
        "warnings": _collect_text_list(
            (analogical or {}).get("warnings"),
            (complexity_review or {}).get("warnings"),
            (agile_guard_review or {}).get("warnings"),
            (critic_review or {}).get("contradictions"),
        ),
        "assumptions": _collect_text_list(
            (analogical or {}).get("assumptions"),
            (complexity_review or {}).get("assumptions"),
            (agile_guard_review or {}).get("assumptions"),
        ),
        "should_split": should_split,
        "split_reason": split_reason,
        "dominant_strategy": str(calibrated.get("dominant_strategy") or strategy or "multiagent_heuristic_consensus"),
        "selected_model": str(calibrated.get("selected_model") or calibrated.get("finalization_mode") or strategy or ""),
        "finalization_mode": calibrated.get("finalization_mode"),
        "calibration_source": calibrated.get("calibration_source"),
        "size_bucket": calibrated.get("size_bucket"),
        "bucket_rank": calibrated.get("bucket_rank"),
        "base_hours": calibrated.get("base_hours"),
        "adjusted_hours": calibrated.get("adjusted_hours", calibrated.get("base_hours")),
        "adjustment_delta": calibrated.get("adjustment_delta"),
        "retrieval_route": calibrated.get("retrieval_route") or (analogical or {}).get("retrieval_route"),
        "retrieval_stats": calibrated.get("retrieval_stats") or (analogical or {}).get("retrieval_stats") or {},
        "agent_trace": {
            "analogical": analogical,
            "heuristic_candidates": heuristic_candidates,
            "complexity_review": complexity_review,
            "agile_guard_review": agile_guard_review,
            "critic_review": critic_review,
            "calibrated_estimation": calibrated,
        },
    }
    return out
