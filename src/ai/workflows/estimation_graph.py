import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TypedDict, Dict, Any, List, Optional

from langgraph.graph import StateGraph, START, END

from ai.core.effort_calibration import (
    aggregate_bucket_consensus,
    bounded_adjustment_from_reviews,
    build_calibration_profile,
    calibrate_bucket_rank_to_hours,
    clamp_bucket_rank,
    infer_bucket_rank_from_hours,
    rank_to_bucket,
)
from ai.core.meta_calibrator import (
    build_meta_feature_payload,
    predict_meta_calibration,
)
from ai.core.retriever import Retriever
from ai.core.llm_client import LLMClient
from ai.core.pinecone_vector_store import PineconeVectorStoreClient
from ai.core.token_usage import TokenUsage, coerce_token_usage
from config.settings import settings
from ai.dtos.issues_estimation_dto import IssueEstimationDTO

from ai.agents.analogical_agent import run_analogical
from ai.agents.heuristic_agent import run_heuristic
from ai.agents.complexity_agent import run_complexity_review
from ai.agents.agile_guard_agent import run_agile_guard
from ai.agents.critic_agent import run_estimation_critic
from ai.agents.supervisor_agent import combine_multi_agent_estimations


class Estimation(TypedDict, total=False):
    estimated_hours: Optional[float]
    min_hours: Optional[float]
    max_hours: Optional[float]
    confidence: float
    justification: str
    mode: str
    source: str
    error: str
    size_bucket: str
    bucket_rank: int
    bucket_delta: int
    fit_status: str
    risk_hidden_complexity: float
    token_usage: TokenUsage
    evidence: List[str]
    warnings: List[str]
    assumptions: List[str]
    should_split: bool
    split_reason: str
    latency_ms: int
    retrieval_route: str
    retrieval_stats: Dict[str, Any]
    calibration_source: str
    base_hours: float
    adjusted_hours: float
    adjustment_delta: float
    neighbor_count: int
    supporting_hours: List[float]
    weighted_hours: List[float]


class CriticReview(TypedDict, total=False):
    risk_of_underestimation: float
    risk_of_overestimation: float
    contradictions: List[str]
    hidden_complexities: List[str]
    strongest_signal: str
    recommendation: str
    token_usage: TokenUsage
    latency_ms: int


class CalibratedEstimation(TypedDict, total=False):
    size_bucket: str
    bucket_rank: int
    heuristic_size_bucket: str
    heuristic_bucket_rank: int
    analogical_size_bucket: str
    analogical_bucket_rank: int
    base_hours: float
    adjusted_hours: float
    adjustment_delta: float
    min_hours: float
    max_hours: float
    calibration_source: str
    finalization_mode: str
    dominant_strategy: str
    selected_model: str
    base_confidence: float
    retrieval_route: str
    retrieval_stats: Dict[str, Any]
    meta_applied: bool
    meta_hours: float
    meta_min_hours: float
    meta_max_hours: float
    meta_confidence: float
    meta_source: str
    meta_prior_source: str
    meta_prior_count: int
    meta_blend_weight: float
    meta_model_version: str
    should_split: bool
    split_reason: str
    evidence: List[str]


class EstimationState(TypedDict, total=False):
    issue: Dict[str, Any]
    similar_issues: List[Dict[str, Any]]
    repository_technologies: Dict[str, float]
    rag_context_sufficient: bool
    strategy: str
    rag_stats: Dict[str, Any]
    token_usage_summary: Dict[str, int]
    execution_metrics: Dict[str, Any]

    analogical: Estimation
    heuristic_candidates: List[Estimation]
    heuristic_ensemble_metrics: Dict[str, Any]
    complexity_review: Estimation
    agile_guard_review: Estimation
    critic_review: CriticReview
    calibrated_estimation: CalibratedEstimation

    final_estimation: Dict[str, Any]


logger = logging.getLogger(__name__)
vector_store = PineconeVectorStoreClient()


def _optional_float(value: Any) -> Optional[float]:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except Exception:
        return None


def normalize_estimation(res: Dict[str, Any], *, fallback_mode: str) -> Estimation:
    out: Estimation = {
        "estimated_hours": _optional_float(res.get("estimated_hours")),
        "min_hours": _optional_float(res.get("min_hours", res.get("estimated_hours"))),
        "max_hours": _optional_float(res.get("max_hours", res.get("estimated_hours"))),
        "confidence": max(0.0, min(1.0, float(res.get("confidence", 0.5) or 0.5))),
        "justification": str(res.get("justification", "") or ""),
        "mode": str(res.get("mode", fallback_mode) or fallback_mode),
        "source": str(res.get("source", fallback_mode) or fallback_mode),
        "evidence": list(res.get("evidence", []) or []),
        "warnings": list(res.get("warnings", []) or []),
        "assumptions": list(res.get("assumptions", []) or []),
        "should_split": bool(res.get("should_split", False)),
        "split_reason": str(res.get("split_reason", "") or ""),
    }
    passthrough_keys = (
        "latency_ms",
        "retrieval_route",
        "retrieval_stats",
        "error",
        "size_bucket",
        "bucket_rank",
        "bucket_delta",
        "fit_status",
        "risk_hidden_complexity",
        "calibration_source",
        "base_hours",
        "adjusted_hours",
        "adjustment_delta",
        "neighbor_count",
        "supporting_hours",
        "weighted_hours",
    )
    for key in passthrough_keys:
        if key in res:
            out[key] = res.get(key)
    if "token_usage" in res:
        out["token_usage"] = coerce_token_usage(res.get("token_usage"))
    return out


def normalize_critic(res: Dict[str, Any]) -> CriticReview:
    out: CriticReview = {
        "risk_of_underestimation": max(0.0, min(1.0, float(res.get("risk_of_underestimation", 0.5) or 0.5))),
        "risk_of_overestimation": max(0.0, min(1.0, float(res.get("risk_of_overestimation", 0.5) or 0.5))),
        "contradictions": list(res.get("contradictions", []) or []),
        "hidden_complexities": list(res.get("hidden_complexities", []) or []),
        "strongest_signal": str(res.get("strongest_signal", "") or ""),
        "recommendation": str(res.get("recommendation", "") or ""),
    }
    if "latency_ms" in res:
        out["latency_ms"] = int(res.get("latency_ms") or 0)
    if "token_usage" in res:
        out["token_usage"] = coerce_token_usage(res.get("token_usage"))
    return out


def model_from_finalization(
    finalization_mode: Optional[str],
    dominant_strategy: Optional[str],
) -> str:
    if dominant_strategy == "analogical_consensus":
        return "analogical+multiagent_consensus"
    if finalization_mode in {"hybrid_calibrated", "soft_hybrid_calibrated", "analogical_calibrated"}:
        return "analogical+multiagent_consensus"
    return "multiagent_heuristic_consensus"


def retriever_node(state: EstimationState) -> EstimationState:
    retriever = Retriever(vector_store)
    issue = state["issue"]
    similar = retriever.get_similar_issues(issue)

    min_score = float(settings.RAG_MIN_SCORE_MAIN)
    min_hits = int(settings.RAG_MIN_HITS_MAIN)
    qualified_hits = 0
    for item in similar:
        try:
            score = float(item.get("score") or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        if score >= min_score:
            qualified_hits += 1

    rag_context_sufficient = qualified_hits >= min_hits
    strategy = "analogical_consensus" if rag_context_sufficient else "multiagent_heuristic_consensus"

    techs = (
        vector_store.get_repository_technologies()
        if hasattr(vector_store, "get_repository_technologies")
        else {}
    )

    rag_usage = getattr(retriever, "last_rag_usage", None)
    return {
        "similar_issues": similar,
        "repository_technologies": techs,
        "rag_context_sufficient": rag_context_sufficient,
        "strategy": strategy,
        "rag_stats": {
            "qualified_hits": qualified_hits,
            "min_hits": min_hits,
            "min_score": min_score,
            "token_usage": rag_usage or {"embedding_tokens": 0, "embedding_calls": 0},
        },
    }


def _run_heuristic_ensemble(issue: Dict[str, Any]) -> tuple[List[Estimation], Dict[str, Any]]:
    modes = ["scope", "complexity", "uncertainty", "agile_fit"]
    temperature = float(getattr(settings, "HEURISTIC_ENSEMBLE_TEMPERATURE", 0.0) or 0.0)
    max_concurrency = int(
        getattr(settings, "HEURISTIC_ENSEMBLE_MAX_CONCURRENCY", len(modes)) or len(modes)
    )
    max_concurrency = max(1, min(max_concurrency, len(modes)))

    def _run_once(mode_name: str) -> Estimation:
        llm = LLMClient(temperature=temperature)
        started_at = time.perf_counter()
        res = run_heuristic(
            issue_context=issue,
            llm=llm,
            temperature=temperature,
            mode=mode_name,
        )
        normalized = normalize_estimation(res, fallback_mode=mode_name)
        normalized["latency_ms"] = int((time.perf_counter() - started_at) * 1000)
        return normalized

    started_at = time.perf_counter()
    results: List[Optional[Estimation]] = [None] * len(modes)
    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        future_to_idx = {executor.submit(_run_once, mode): idx for idx, mode in enumerate(modes)}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            mode_name = modes[idx]
            try:
                results[idx] = future.result()
            except Exception as exc:
                results[idx] = normalize_estimation(
                    {
                        "mode": mode_name,
                        "size_bucket": "M",
                        "bucket_rank": 3,
                        "confidence": 0.25,
                        "justification": f"Heuristic mode {mode_name} failed; fallback applied.",
                        "warnings": [str(exc)],
                        "latency_ms": 0,
                    },
                    fallback_mode=mode_name,
                )

    candidates = [item for item in results if item is not None]
    metrics = {
        "latency_ms": int((time.perf_counter() - started_at) * 1000),
        "candidate_count": len(candidates),
    }
    return candidates, metrics


def primary_reviews_node(state: EstimationState) -> EstimationState:
    issue = state["issue"]
    similar_issues = state.get("similar_issues", [])
    repository_technologies = state.get("repository_technologies", {})
    max_concurrency = int(getattr(settings, "PRIMARY_AGENT_MAX_CONCURRENCY", 4) or 4)
    max_concurrency = max(1, min(max_concurrency, 4))

    def _analogical_call() -> Estimation:
        llm = LLMClient()
        started_at = time.perf_counter()
        res = run_analogical(
            issue_context=issue,
            similar_issues=similar_issues,
            repository_technologies=repository_technologies,
            llm=llm,
        )
        normalized = normalize_estimation(res, fallback_mode="analogical")
        normalized["latency_ms"] = int((time.perf_counter() - started_at) * 1000)
        return normalized

    def _complexity_call() -> Estimation:
        llm = LLMClient()
        started_at = time.perf_counter()
        res = run_complexity_review(issue_context=issue, llm=llm)
        normalized = normalize_estimation(res, fallback_mode="complexity_review")
        normalized["latency_ms"] = int((time.perf_counter() - started_at) * 1000)
        return normalized

    def _agile_guard_call() -> Estimation:
        llm = LLMClient()
        started_at = time.perf_counter()
        res = run_agile_guard(issue_context=issue, llm=llm)
        normalized = normalize_estimation(res, fallback_mode="agile_guard")
        normalized["latency_ms"] = int((time.perf_counter() - started_at) * 1000)
        return normalized

    started_at = time.perf_counter()
    tasks = {
        "analogical": _analogical_call,
        "heuristic_bundle": lambda: _run_heuristic_ensemble(issue),
        "complexity_review": _complexity_call,
        "agile_guard_review": _agile_guard_call,
    }

    out: EstimationState = {}
    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        future_to_name = {executor.submit(task): task_name for task_name, task in tasks.items()}
        for future in as_completed(future_to_name):
            task_name = future_to_name[future]
            if task_name == "heuristic_bundle":
                candidates, metrics = future.result()
                out["heuristic_candidates"] = candidates
                out["heuristic_ensemble_metrics"] = metrics
            else:
                out[task_name] = future.result()

    out["execution_metrics"] = {
        "primary_reviews_latency_ms": int((time.perf_counter() - started_at) * 1000),
        "analogical_latency_ms": int((out.get("analogical") or {}).get("latency_ms") or 0),
        "heuristic_ensemble_latency_ms": int((out.get("heuristic_ensemble_metrics") or {}).get("latency_ms") or 0),
        "complexity_review_latency_ms": int((out.get("complexity_review") or {}).get("latency_ms") or 0),
        "agile_guard_latency_ms": int((out.get("agile_guard_review") or {}).get("latency_ms") or 0),
    }
    return out


def critic_node(state: EstimationState) -> EstimationState:
    llm = LLMClient()
    started_at = time.perf_counter()
    res = run_estimation_critic(
        issue_context=state["issue"],
        analogical=state.get("analogical"),
        heuristic_candidates=state.get("heuristic_candidates", []),
        complexity_review=state.get("complexity_review"),
        agile_guard_review=state.get("agile_guard_review"),
        llm=llm,
    )
    res["latency_ms"] = int((time.perf_counter() - started_at) * 1000)

    execution_metrics = dict(state.get("execution_metrics") or {})
    execution_metrics["critic_latency_ms"] = int(res.get("latency_ms") or 0)
    return {
        "critic_review": normalize_critic(res),
        "execution_metrics": execution_metrics,
    }


def _enrich_heuristic_candidates(
    candidates: List[Estimation],
    profile: Dict[str, Any],
    calibration_mode: str = "standard",
) -> List[Estimation]:
    enriched: List[Estimation] = []
    for candidate in candidates or []:
        rank = candidate.get("bucket_rank")
        if rank in (None, ""):
            rank = 3
        calibrated = calibrate_bucket_rank_to_hours(rank, profile, calibration_mode=calibration_mode)
        merged = dict(candidate)
        merged.setdefault("size_bucket", calibrated.get("size_bucket"))
        merged.setdefault("bucket_rank", calibrated.get("bucket_rank"))
        merged["estimated_hours"] = calibrated.get("estimated_hours")
        merged["min_hours"] = calibrated.get("min_hours")
        merged["max_hours"] = calibrated.get("max_hours")
        merged["calibration_source"] = calibrated.get("calibration_source")
        enriched.append(merged)
    return enriched


def calibration_node(state: EstimationState) -> EstimationState:
    started_at = time.perf_counter()
    issue = state["issue"]
    similar_issues = state.get("similar_issues", [])
    analogical = dict(state.get("analogical") or {})
    heuristic_candidates = list(state.get("heuristic_candidates") or [])
    complexity_review = dict(state.get("complexity_review") or {})
    agile_guard_review = dict(state.get("agile_guard_review") or {})
    critic_review = dict(state.get("critic_review") or {})

    profile = build_calibration_profile(
        issue_context=issue,
        similar_issues=similar_issues,
        useful_score_threshold=0.70,
    )
    analogical_hours = _optional_float(analogical.get("estimated_hours"))
    analogical_conf = float(analogical.get("confidence", 0.0) or 0.0)
    analogical_rank = analogical.get("bucket_rank")
    if analogical_rank in (None, "") and analogical_hours is not None:
        analogical_rank = infer_bucket_rank_from_hours(
            analogical_hours,
            hours_pool=profile.get("selected_hours") or profile.get("all_hours"),
        )

    route = str(analogical.get("retrieval_route") or "analogical_weak")
    heuristic_consensus = aggregate_bucket_consensus(heuristic_candidates)
    heuristic_calibration_mode = "weak_prior" if route in {"analogical_weak", "analogical_soft_signal"} else "standard"
    heuristic_candidates = _enrich_heuristic_candidates(
        heuristic_candidates,
        profile,
        calibration_mode=heuristic_calibration_mode,
    )
    heuristic_calibration = calibrate_bucket_rank_to_hours(
        heuristic_consensus.get("bucket_rank"),
        profile,
        calibration_mode=heuristic_calibration_mode,
    )

    retrieval_stats = dict(analogical.get("retrieval_stats") or {})
    top1_score = float(retrieval_stats.get("top1_score") or 0.0)
    useful_count = int(retrieval_stats.get("useful_count") or 0)
    anchor_overlap = float(retrieval_stats.get("anchor_overlap") or 0.0)
    analogical_rank_value = clamp_bucket_rank(analogical_rank or 3)
    heuristic_rank_value = clamp_bucket_rank(heuristic_consensus.get("bucket_rank") or 3)
    bucket_gap = analogical_rank_value - heuristic_rank_value
    moderate_analogical_signal = (
        route == "analogical_soft_signal"
        or top1_score >= 0.64
        or (top1_score >= 0.60 and float(retrieval_stats.get("top3_avg_score") or 0.0) >= 0.60)
        or useful_count >= 1
        or (anchor_overlap >= 0.18 and top1_score >= 0.58)
    )

    if route == "analogical_primary" and analogical_hours is not None:
        base_hours = float(analogical_hours)
        base_min = _optional_float(analogical.get("min_hours")) or max(1.0, base_hours * 0.82)
        base_max = _optional_float(analogical.get("max_hours")) or max(base_hours, base_hours * 1.18)
        base_rank = clamp_bucket_rank(analogical_rank or heuristic_consensus.get("bucket_rank") or 3)
        finalization_mode = "analogical_calibrated"
        dominant_strategy = "analogical_consensus"
        selected_model = "analogical_calibrated"
        calibration_source = str(analogical.get("calibration_source") or "analogical_neighbors")
        base_confidence = max(analogical_conf, 0.62)
    elif route == "analogical_support" and analogical_hours is not None:
        heuristic_hours = float(heuristic_calibration.get("estimated_hours") or analogical_hours)
        analogical_weight = 0.65 if analogical_conf >= float(heuristic_consensus.get("confidence", 0.0) or 0.0) else 0.55
        base_hours = round((analogical_hours * analogical_weight) + (heuristic_hours * (1.0 - analogical_weight)), 1)
        base_min = min(
            _optional_float(analogical.get("min_hours")) or analogical_hours,
            float(heuristic_calibration.get("min_hours") or heuristic_hours),
        )
        base_max = max(
            _optional_float(analogical.get("max_hours")) or analogical_hours,
            float(heuristic_calibration.get("max_hours") or heuristic_hours),
        )
        weighted_rank = round(
            (clamp_bucket_rank(analogical_rank or 3) * analogical_weight)
            + (clamp_bucket_rank(heuristic_consensus.get("bucket_rank")) * (1.0 - analogical_weight))
        )
        base_rank = clamp_bucket_rank(weighted_rank)
        finalization_mode = "hybrid_calibrated"
        dominant_strategy = "analogical_consensus"
        selected_model = "hybrid_calibrated"
        calibration_source = (
            f"{analogical.get('calibration_source') or 'analogical_neighbors'}"
            f"+{heuristic_calibration.get('calibration_source') or 'heuristic_calibration'}"
        )
        base_confidence = max(0.52, min(0.82, (analogical_conf * analogical_weight) + (float(heuristic_consensus.get("confidence", 0.4)) * (1.0 - analogical_weight))))
    elif route in {"analogical_weak", "analogical_soft_signal"} and analogical_hours is not None and moderate_analogical_signal:
        heuristic_hours = float(heuristic_calibration.get("estimated_hours") or analogical_hours)
        analogical_weight = 0.32
        if top1_score >= 0.72:
            analogical_weight = 0.48
        elif top1_score >= 0.68:
            analogical_weight = 0.42
        elif useful_count >= 1:
            analogical_weight = 0.38
        if bucket_gap >= 2:
            analogical_weight += 0.10
        if analogical_conf >= 0.75:
            analogical_weight += 0.05
        analogical_weight = max(0.28, min(0.58, analogical_weight))

        base_hours = round((analogical_hours * analogical_weight) + (heuristic_hours * (1.0 - analogical_weight)), 1)
        base_min = min(
            _optional_float(analogical.get("min_hours")) or analogical_hours,
            float(heuristic_calibration.get("min_hours") or heuristic_hours),
        )
        base_max = max(
            _optional_float(analogical.get("max_hours")) or analogical_hours,
            float(heuristic_calibration.get("max_hours") or heuristic_hours),
        )
        weighted_rank = round(
            (analogical_rank_value * analogical_weight)
            + (heuristic_rank_value * (1.0 - analogical_weight))
        )
        base_rank = clamp_bucket_rank(weighted_rank)
        finalization_mode = "soft_hybrid_calibrated"
        dominant_strategy = "analogical_consensus"
        selected_model = "soft_hybrid_calibrated"
        calibration_source = (
            f"{analogical.get('calibration_source') or 'analogical_neighbors'}"
            f"+{heuristic_calibration.get('calibration_source') or 'weak_prior'}"
        )
        base_confidence = max(
            0.46,
            min(
                0.76,
                (analogical_conf * analogical_weight)
                + (float(heuristic_consensus.get("confidence", 0.4)) * (1.0 - analogical_weight)),
            ),
        )
    else:
        base_hours = float(heuristic_calibration.get("estimated_hours") or profile.get("median_hours") or 8.0)
        base_min = float(heuristic_calibration.get("min_hours") or max(1.0, base_hours * 0.78))
        base_max = float(heuristic_calibration.get("max_hours") or max(base_hours, base_hours * 1.22))
        base_rank = clamp_bucket_rank(heuristic_consensus.get("bucket_rank") or 3)
        finalization_mode = "heuristic_bucket_calibrated"
        dominant_strategy = "multiagent_heuristic_consensus"
        selected_model = "heuristic_bucket_calibrated"
        calibration_source = str(heuristic_calibration.get("calibration_source") or "heuristic_calibration")
        base_confidence = max(0.35, float(heuristic_consensus.get("confidence", 0.35) or 0.35))

    meta_prediction = predict_meta_calibration(
        build_meta_feature_payload(
            issue_context=issue,
            analogical=analogical,
            heuristic_consensus=heuristic_consensus,
            heuristic_calibration=heuristic_calibration,
            calibration_source=calibration_source,
            retrieval_route=route,
            retrieval_stats=retrieval_stats,
            base_hours=base_hours,
            base_confidence=base_confidence,
            rule_selected_model=selected_model,
            complexity_review=complexity_review,
            agile_guard_review=agile_guard_review,
            rag_stats=state.get("rag_stats"),
        )
    )
    meta_applied = False
    if bool(meta_prediction.get("available")):
        meta_hours = float(meta_prediction.get("estimated_hours") or base_hours)
        meta_min = float(meta_prediction.get("min_hours") or base_min)
        meta_max = float(meta_prediction.get("max_hours") or base_max)
        meta_conf = float(meta_prediction.get("confidence") or 0.0)
        meta_weight = 0.0
        if route == "analogical_primary":
            meta_weight = 0.16
        elif route == "analogical_support":
            meta_weight = 0.30
        elif route == "analogical_soft_signal":
            meta_weight = 0.48
        else:
            meta_weight = 0.56

        prior_source = str(meta_prediction.get("prior_source") or "")
        prior_count = int(meta_prediction.get("prior_count") or 0)
        if prior_source == "project":
            meta_weight -= 0.10
        elif prior_source in {"project_issue", "project_issue_route"}:
            meta_weight += 0.03
        elif prior_source == "project_issue_bucket":
            meta_weight += 0.07
        if prior_count >= 8:
            meta_weight += 0.05
        if meta_conf < 0.5:
            meta_weight -= 0.10
        if route == "analogical_support" and top1_score >= 0.78:
            meta_weight -= 0.05
        meta_weight = max(0.10, min(0.68, meta_weight))

        base_hours = round((base_hours * (1.0 - meta_weight)) + (meta_hours * meta_weight), 1)
        base_min = round(min(base_min, (base_min * (1.0 - meta_weight)) + (meta_min * meta_weight)), 1)
        base_max = round(max(base_max, (base_max * (1.0 - meta_weight)) + (meta_max * meta_weight)), 1)
        base_rank = infer_bucket_rank_from_hours(
            base_hours,
            hours_pool=profile.get("selected_hours") or profile.get("all_hours"),
        )
        base_confidence = max(
            0.32,
            min(0.86, (base_confidence * (1.0 - (meta_weight * 0.55))) + (meta_conf * (meta_weight * 0.55))),
        )
        calibration_source = f"{calibration_source}+{meta_prediction.get('meta_source') or 'meta_linear'}"
        meta_applied = meta_weight >= 0.14

    agile_delta = int(agile_guard_review.get("bucket_delta") or 0)
    if agile_delta:
        agile_rank = clamp_bucket_rank(base_rank + agile_delta)
        agile_hint = calibrate_bucket_rank_to_hours(
            agile_rank,
            profile,
            calibration_mode=heuristic_calibration_mode,
        )
        base_hours = round((base_hours * 0.8) + (float(agile_hint.get("estimated_hours") or base_hours) * 0.2), 1)
        base_min = min(base_min, float(agile_hint.get("min_hours") or base_min))
        base_max = max(base_max, float(agile_hint.get("max_hours") or base_max))
        base_rank = agile_rank

    adjustments = bounded_adjustment_from_reviews(
        base_hours=base_hours,
        complexity_review=complexity_review,
        critic_review=critic_review,
    )
    adjusted_hours = float(adjustments.get("adjusted_hours") or base_hours)
    adjustment_delta = float(adjustments.get("adjustment_delta") or 0.0)
    final_rank = infer_bucket_rank_from_hours(
        adjusted_hours,
        hours_pool=profile.get("selected_hours") or profile.get("all_hours"),
    )
    final_bucket = rank_to_bucket(final_rank)

    critic_under = float(critic_review.get("risk_of_underestimation", 0.0) or 0.0)
    critic_over = float(critic_review.get("risk_of_overestimation", 0.0) or 0.0)
    min_hours = max(1.0, min(base_min, adjusted_hours * (0.88 - (critic_over * 0.08))))
    max_hours = max(base_max, adjusted_hours * (1.14 + (critic_under * 0.18)))

    should_split = (
        bool(agile_guard_review.get("should_split"))
        or bool(complexity_review.get("should_split"))
        or str(agile_guard_review.get("fit_status") or "") == "oversized"
        or adjusted_hours > 40.0
    )
    split_reason = (
        agile_guard_review.get("split_reason")
        or complexity_review.get("split_reason")
    )

    calibrated: CalibratedEstimation = {
        "size_bucket": final_bucket,
        "bucket_rank": final_rank,
        "heuristic_size_bucket": str(heuristic_consensus.get("size_bucket") or "M"),
        "heuristic_bucket_rank": clamp_bucket_rank(heuristic_consensus.get("bucket_rank") or 3),
        "analogical_size_bucket": str(analogical.get("size_bucket") or rank_to_bucket(analogical_rank or 3)),
        "analogical_bucket_rank": clamp_bucket_rank(analogical_rank or 3),
        "base_hours": round(base_hours, 1),
        "adjusted_hours": round(adjusted_hours, 1),
        "adjustment_delta": round(adjustment_delta, 1),
        "min_hours": round(min_hours, 1),
        "max_hours": round(max_hours, 1),
        "calibration_source": calibration_source,
        "finalization_mode": finalization_mode,
        "dominant_strategy": dominant_strategy,
        "selected_model": selected_model,
        "base_confidence": round(base_confidence, 4),
        "retrieval_route": route,
        "retrieval_stats": dict(analogical.get("retrieval_stats") or {}),
        "meta_applied": bool(meta_applied),
        "meta_hours": round(float(meta_prediction.get("estimated_hours") or 0.0), 1) if bool(meta_prediction.get("available")) else 0.0,
        "meta_min_hours": round(float(meta_prediction.get("min_hours") or 0.0), 1) if bool(meta_prediction.get("available")) else 0.0,
        "meta_max_hours": round(float(meta_prediction.get("max_hours") or 0.0), 1) if bool(meta_prediction.get("available")) else 0.0,
        "meta_confidence": round(float(meta_prediction.get("confidence") or 0.0), 4) if bool(meta_prediction.get("available")) else 0.0,
        "meta_source": str(meta_prediction.get("meta_source") or "") if bool(meta_prediction.get("available")) else "",
        "meta_prior_source": str(meta_prediction.get("prior_source") or "") if bool(meta_prediction.get("available")) else "",
        "meta_prior_count": int(meta_prediction.get("prior_count") or 0) if bool(meta_prediction.get("available")) else 0,
        "meta_blend_weight": round(float(meta_weight), 4) if bool(meta_prediction.get("available")) else 0.0,
        "meta_model_version": str(meta_prediction.get("model_version") or "") if bool(meta_prediction.get("available")) else "",
        "should_split": should_split,
        "split_reason": split_reason,
        "evidence": [
            f"base_hours={round(base_hours, 1)}",
            f"adjusted_hours={round(adjusted_hours, 1)}",
            f"heuristic_bucket={heuristic_consensus.get('size_bucket')}",
            f"analogical_route={route}",
            f"meta_applied={str(meta_applied).lower()}",
        ],
    }

    execution_metrics = dict(state.get("execution_metrics") or {})
    execution_metrics["calibration_latency_ms"] = int((time.perf_counter() - started_at) * 1000)
    return {
        "heuristic_candidates": heuristic_candidates,
        "calibrated_estimation": calibrated,
        "execution_metrics": execution_metrics,
    }


def supervisor_node(state: EstimationState) -> EstimationState:
    started_at = time.perf_counter()
    final = combine_multi_agent_estimations(
        issue_context=state["issue"],
        strategy=str(state.get("strategy") or "multiagent_heuristic_consensus"),
        analogical=state.get("analogical"),
        heuristic_candidates=state.get("heuristic_candidates", []),
        complexity_review=state.get("complexity_review"),
        agile_guard_review=state.get("agile_guard_review"),
        critic_review=state.get("critic_review"),
        calibrated_estimation=state.get("calibrated_estimation"),
    )
    supervisor_latency_ms = int((time.perf_counter() - started_at) * 1000)
    final["latency_ms"] = supervisor_latency_ms
    final["estimation_model"] = model_from_finalization(
        final.get("finalization_mode"),
        final.get("dominant_strategy"),
    )
    final["selected_model"] = str(
        final.get("selected_model")
        or final.get("finalization_mode")
        or final.get("dominant_strategy")
        or state.get("strategy")
        or ""
    )

    analogical = dict(state.get("analogical") or {})
    final["retrieval_stats"] = analogical.get("retrieval_stats") or {}
    final["retrieval_route"] = final.get("retrieval_route") or analogical.get("retrieval_route")
    final["calibrated_estimation"] = state.get("calibrated_estimation")
    final["agent_trace"] = final.get("agent_trace") or {
        "analogical": state.get("analogical"),
        "heuristic_candidates": state.get("heuristic_candidates", []),
        "complexity_review": state.get("complexity_review"),
        "agile_guard_review": state.get("agile_guard_review"),
        "critic_review": state.get("critic_review"),
        "calibrated_estimation": state.get("calibrated_estimation"),
    }

    execution_metrics = dict(state.get("execution_metrics") or {})
    execution_metrics["supervisor_latency_ms"] = supervisor_latency_ms
    final["execution_trace"] = execution_metrics

    return {
        "final_estimation": final,
        "execution_metrics": execution_metrics,
    }


graph = StateGraph(EstimationState)
graph.add_node("retriever", retriever_node)
graph.add_node("primary_reviews", primary_reviews_node)
graph.add_node("critic", critic_node)
graph.add_node("calibration", calibration_node)
graph.add_node("supervisor", supervisor_node)

graph.add_edge(START, "retriever")
graph.add_edge("retriever", "primary_reviews")
graph.add_edge("primary_reviews", "critic")
graph.add_edge("critic", "calibration")
graph.add_edge("calibration", "supervisor")
graph.add_edge("supervisor", END)

estimation_graph = graph.compile()


def run_estimation_flow(dto: IssueEstimationDTO) -> EstimationState:
    workflow_started_at = time.perf_counter()
    initial_state: EstimationState = {"issue": dto.model_dump()}
    state: EstimationState = estimation_graph.invoke(initial_state)

    def _sum_llm_usage(usages: List[Dict[str, Any]]) -> TokenUsage:
        total: TokenUsage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        for usage in usages:
            u = coerce_token_usage(usage)
            total = {
                "prompt_tokens": total["prompt_tokens"] + u["prompt_tokens"],
                "completion_tokens": total["completion_tokens"] + u["completion_tokens"],
                "total_tokens": total["total_tokens"] + u["total_tokens"],
            }
        return total

    rag_stats = state.get("rag_stats") or {}
    rag_usage = rag_stats.get("token_usage") or {}
    rag_embedding_tokens = max(0, int(rag_usage.get("embedding_tokens") or 0))

    llm_usages: List[Dict[str, Any]] = []
    for key in ["analogical", "complexity_review", "agile_guard_review", "critic_review", "final_estimation"]:
        value = state.get(key) or {}
        if isinstance(value, dict):
            llm_usages.append(value.get("token_usage") or {})
    for candidate in state.get("heuristic_candidates") or []:
        if isinstance(candidate, dict):
            llm_usages.append(candidate.get("token_usage") or {})

    llm_total = _sum_llm_usage(llm_usages)
    summary = {
        "predicted_llm_prompt_tokens": int(llm_total["prompt_tokens"]),
        "predicted_llm_completion_tokens": int(llm_total["completion_tokens"]),
        "predicted_llm_total_tokens": int(llm_total["total_tokens"]),
        "predicted_rag_embedding_tokens": int(rag_embedding_tokens),
        "predicted_total_tokens": int(llm_total["total_tokens"] + rag_embedding_tokens),
    }
    state["token_usage_summary"] = summary

    execution_metrics = dict(state.get("execution_metrics") or {})
    execution_metrics["workflow_latency_ms"] = int((time.perf_counter() - workflow_started_at) * 1000)
    state["execution_metrics"] = execution_metrics

    final_estimation = state.get("final_estimation")
    if isinstance(final_estimation, dict):
        final_estimation["token_usage_summary"] = summary
        final_execution_trace = dict(final_estimation.get("execution_trace") or {})
        final_execution_trace.update(execution_metrics)
        final_estimation["execution_trace"] = final_execution_trace

    return state
