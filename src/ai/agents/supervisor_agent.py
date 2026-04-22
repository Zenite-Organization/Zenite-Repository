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
    range_label = str(calibrated_estimation.get("range_label") or "")
    calibration_source = str(calibrated_estimation.get("calibration_source") or "unknown")
    adjustment_delta = _safe_float(calibrated_estimation.get("adjustment_delta"), 0.0)
    meta_applied = bool(calibrated_estimation.get("meta_applied"))
    meta_source = str(calibrated_estimation.get("meta_source") or "").strip()

    parts = [
        f"Finalized via {finalization_mode}",
        f"size bucket {size_bucket}",
        f"final range {range_label}" if range_label else "final range unknown",
        f"calibrated from {calibration_source}",
        f"retrieval route {route}",
    ]
    if meta_applied and meta_source:
        parts.append(f"meta-calibrated with {meta_source}")
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


def _confidence_label(confidence: float) -> str:
    if confidence >= 0.78:
        return "alta"
    if confidence >= 0.58:
        return "moderada"
    return "baixa"


def _build_user_justification(
    calibrated_estimation: Dict[str, Any],
    analogical: Optional[Dict[str, Any]],
    complexity_review: Optional[Dict[str, Any]],
    agile_guard_review: Optional[Dict[str, Any]],
) -> str:
    range_label = str(calibrated_estimation.get("range_label") or "")
    size_bucket = str(calibrated_estimation.get("size_bucket") or "M")
    route = str(calibrated_estimation.get("retrieval_route") or "")
    confidence = _safe_float(calibrated_estimation.get("base_confidence"), 0.55)
    adjustment_delta = _safe_float(calibrated_estimation.get("adjustment_delta"), 0.0)
    should_split = bool(calibrated_estimation.get("should_split"))
    fit_status = str((agile_guard_review or {}).get("fit_status") or "").strip().lower()
    complexity_risk = _safe_float((complexity_review or {}).get("risk_hidden_complexity"), 0.0)

    opening = (
        f"A estimativa ficou na faixa de {range_label}"
        if range_label
        else "A estimativa ficou em uma faixa intermediária"
    )

    if size_bucket in {"XS", "S"}:
        scope_text = "porque o trabalho parece mais localizado e com menor impacto no restante do sistema"
    elif size_bucket in {"XL", "XXL"}:
        scope_text = "porque há sinais de impacto amplo, com mais etapas de trabalho e maior coordenação técnica"
    elif size_bucket == "L":
        scope_text = "porque a issue sugere um esforço acima da média, com mais de uma etapa relevante de implementação e validação"
    else:
        scope_text = "porque a issue sugere um esforço intermediário, combinando implementação e validação sem indicar uma mudança muito ampla"

    if route == "analogical_primary":
        history_text = "Também encontramos itens bem parecidos no histórico, o que reforçou essa decisão."
    elif route in {"analogical_support", "analogical_soft_signal"}:
        history_text = "O histórico trouxe alguns itens parecidos, que ajudaram a calibrar a faixa final."
    else:
        history_text = "Como o histórico parecido foi fraco, demos mais peso ao escopo descrito na própria issue."

    adjustments: List[str] = []
    if adjustment_delta > 0.2:
        adjustments.append("A faixa foi puxada um pouco para cima por sinais de complexidade adicional.")
    elif adjustment_delta < -0.2:
        adjustments.append("A faixa foi mantida mais enxuta porque o trabalho parece mais isolado do que um caso amplo.")

    if fit_status == "oversized" or should_split:
        adjustments.append("Há sinais de que a issue pode estar agrupando trabalho demais e talvez precise ser quebrada.")
    elif fit_status == "borderline":
        adjustments.append("A issue parece estar perto do limite de um item saudável de backlog, então vale atenção no refinamento.")

    if complexity_risk >= 0.8:
        adjustments.append("Também apareceram sinais de complexidade oculta que podem exigir investigação e validação adicionais.")

    confidence_text = f"A confiança nessa leitura é {_confidence_label(confidence)}."
    return " ".join([opening, scope_text + ".", history_text, *adjustments, confidence_text]).strip()


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
    estimated_hours_raw = _safe_float(
        calibrated.get("adjusted_hours", calibrated.get("base_hours")),
        8.0,
    )
    estimated_hours = _safe_float(
        calibrated.get("display_hours"),
        estimated_hours_raw,
    )
    min_hours = _safe_float(calibrated.get("range_min_hours"), _safe_float(calibrated.get("min_hours"), max(1.0, estimated_hours_raw * 0.8)))
    max_hours = _safe_float(calibrated.get("range_max_hours"), _safe_float(calibrated.get("max_hours"), max(estimated_hours_raw, estimated_hours_raw * 1.2)))
    should_split = (
        bool(calibrated.get("should_split"))
        or bool((agile_guard_review or {}).get("should_split"))
        or bool((complexity_review or {}).get("should_split"))
        or estimated_hours_raw > AGILE_HOURS_LIMIT
    )
    split_reason = (
        calibrated.get("split_reason")
        or (agile_guard_review or {}).get("split_reason")
        or (complexity_review or {}).get("split_reason")
    )
    if should_split and not split_reason:
        split_reason = f"Consolidated estimate exceeds the healthy agile limit of {AGILE_HOURS_LIMIT:.0f}h."

    analysis_justification = _build_justification(
        calibrated_estimation=calibrated,
        analogical=analogical,
        complexity_review=complexity_review,
        agile_guard_review=agile_guard_review,
    )
    user_justification = _build_user_justification(
        calibrated_estimation=calibrated,
        analogical=analogical,
        complexity_review=complexity_review,
        agile_guard_review=agile_guard_review,
    )

    out = {
        "estimated_hours": int(max(1, round(estimated_hours))),
        "estimated_hours_raw": round(max(1.0, estimated_hours_raw), 1),
        "min_hours": int(max(1, round(min_hours))),
        "max_hours": int(max(round(max_hours), round(min_hours))),
        "confidence": _review_confidence(calibrated, critic_review, heuristic_candidates),
        "justification": user_justification,
        "user_justification": user_justification,
        "analysis_justification": analysis_justification,
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
        "range_index": calibrated.get("range_index"),
        "range_label": calibrated.get("range_label"),
        "range_min_hours": calibrated.get("range_min_hours"),
        "range_max_hours": calibrated.get("range_max_hours"),
        "display_hours": calibrated.get("display_hours"),
        "base_hours": calibrated.get("base_hours"),
        "adjusted_hours": calibrated.get("adjusted_hours", calibrated.get("base_hours")),
        "adjustment_delta": calibrated.get("adjustment_delta"),
        "meta_applied": calibrated.get("meta_applied"),
        "meta_hours": calibrated.get("meta_hours"),
        "meta_min_hours": calibrated.get("meta_min_hours"),
        "meta_max_hours": calibrated.get("meta_max_hours"),
        "meta_confidence": calibrated.get("meta_confidence"),
        "meta_source": calibrated.get("meta_source"),
        "meta_prior_source": calibrated.get("meta_prior_source"),
        "meta_prior_count": calibrated.get("meta_prior_count"),
        "meta_blend_weight": calibrated.get("meta_blend_weight"),
        "meta_model_version": calibrated.get("meta_model_version"),
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
