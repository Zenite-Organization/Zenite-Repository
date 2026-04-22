import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TypedDict, Dict, Any, List, Optional

from langgraph.graph import StateGraph, START, END

from ai.core.effort_calibration import (
    aggregate_range_consensus,
    build_calibration_profile,
    bucket_rank_to_default_range_index,
    clamp_bucket_rank,
    clamp_range_index,
    hours_to_range_payload,
    hours_to_range_index,
    rank_to_bucket,
    range_index_to_bucket_rank,
    range_index_to_payload,
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
    estimated_hours_raw: Optional[float]
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
    range_index: int
    range_label: str
    range_min_hours: int
    range_max_hours: int
    display_hours: int


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
    range_index: int
    range_label: str
    range_min_hours: int
    range_max_hours: int
    display_hours: int
    base_range_index: int
    base_range_label: str
    base_range_min_hours: int
    base_range_max_hours: int
    base_display_hours: int
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
    meta_range_index: int
    meta_range_label: str
    meta_range_min_hours: int
    meta_range_max_hours: int
    meta_display_hours: int
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
    estimated_hours = _optional_float(res.get("estimated_hours"))
    out: Estimation = {
        "estimated_hours": estimated_hours,
        "estimated_hours_raw": _optional_float(res.get("estimated_hours_raw", estimated_hours)),
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
        "range_index",
        "range_label",
        "range_min_hours",
        "range_max_hours",
        "display_hours",
    )
    for key in passthrough_keys:
        if key in res:
            out[key] = res.get(key)
    if estimated_hours is not None:
        for key, value in hours_to_range_payload(estimated_hours).items():
            out.setdefault(key, value)
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


LARGE_SCOPE_TERMS = (
    "integration",
    "integracao",
    "migration",
    "migracao",
    "certificate",
    "certificado",
    "deploy",
    "pipeline",
    "ci/cd",
    "compatibility",
    "compatibilidade",
    "sdk",
    "oauth",
    "auth",
    "autentic",
    "multi",
    "multiple",
    "environments",
    "ambientes",
    "version",
    "versao",
    "rollback",
    "refactor",
    "refator",
)

DISCOVERY_FIX_TERMS = (
    "investigate",
    "investig",
    "analyze",
    "analise",
    "debug",
    "diagnost",
    "root cause",
    "causa raiz",
    "fix",
    "corrig",
    "validate",
    "valid",
    "regression",
    "regress",
)


def _issue_text(issue: Dict[str, Any]) -> str:
    labels = issue.get("labels") or []
    if isinstance(labels, list):
        labels_text = " ".join(str(label or "") for label in labels)
    else:
        labels_text = str(labels or "")
    return " ".join(
        [
            str(issue.get("title") or ""),
            str(issue.get("description") or ""),
            labels_text,
        ]
    ).lower()


def _count_terms(text: str, terms: tuple[str, ...]) -> int:
    return sum(1 for term in terms if term in text)


def _candidate_range_index(candidate: Dict[str, Any] | None, fallback_bucket_rank: int = 3) -> int:
    candidate = dict(candidate or {})
    if candidate.get("range_index") not in (None, ""):
        return clamp_range_index(candidate.get("range_index"))
    estimated_hours = _optional_float(candidate.get("estimated_hours"))
    if estimated_hours is not None:
        return hours_to_range_index(estimated_hours)
    bucket_rank = candidate.get("bucket_rank")
    if bucket_rank in (None, ""):
        size_bucket = str(candidate.get("size_bucket") or "").strip().upper()
        bucket_rank = {
            "XS": 1,
            "S": 2,
            "M": 3,
            "L": 4,
            "XL": 5,
            "XXL": 6,
        }.get(size_bucket, fallback_bucket_rank)
    return bucket_rank_to_default_range_index(bucket_rank or fallback_bucket_rank)


def _blend_range_indices(primary_index: Any, secondary_index: Any, primary_weight: float) -> int:
    primary = clamp_range_index(primary_index)
    secondary = clamp_range_index(secondary_index)
    weight = max(0.0, min(1.0, float(primary_weight)))
    blended = (primary * weight) + (secondary * (1.0 - weight))
    return clamp_range_index(round(blended))


def _large_issue_floor_index(
    issue: Dict[str, Any],
    analogical: Dict[str, Any],
    heuristic_candidates: List[Estimation],
    complexity_review: Dict[str, Any],
    agile_guard_review: Dict[str, Any],
) -> int:
    floor_index = 1
    text = _issue_text(issue)
    large_scope_hits = _count_terms(text, LARGE_SCOPE_TERMS)
    discovery_hits = _count_terms(text, DISCOVERY_FIX_TERMS)

    fit_status = str(agile_guard_review.get("fit_status") or "").strip().lower()
    agile_confidence = float(agile_guard_review.get("confidence") or 0.0)
    if fit_status == "oversized" or bool(agile_guard_review.get("should_split")):
        floor_index = max(floor_index, 8)
    elif fit_status == "borderline" and agile_confidence >= 0.75:
        floor_index = max(floor_index, 6)

    complexity_risk = float(complexity_review.get("risk_hidden_complexity") or 0.0)
    if bool(complexity_review.get("should_split")):
        floor_index = max(floor_index, 8)
    elif int(complexity_review.get("bucket_delta") or 0) > 0 and complexity_risk >= 0.82:
        floor_index = max(floor_index, 6)

    analogical_range_index = _candidate_range_index(analogical, fallback_bucket_rank=3)
    retrieval_stats = dict(analogical.get("retrieval_stats") or {})
    top1_score = float(retrieval_stats.get("top1_score") or 0.0)
    useful_count = int(retrieval_stats.get("useful_count") or 0)
    supporting_hours = list(analogical.get("supporting_hours") or [])
    large_neighbor_count = sum(1 for hour in supporting_hours if hours_to_range_index(hour) >= 8)
    mid_large_neighbor_count = sum(1 for hour in supporting_hours if hours_to_range_index(hour) >= 7)

    if large_neighbor_count >= 2 and (top1_score >= 0.68 or useful_count >= 1):
        floor_index = max(floor_index, max(7, analogical_range_index - 1))
    elif mid_large_neighbor_count >= 2 and (top1_score >= 0.64 or useful_count >= 1):
        floor_index = max(floor_index, min(analogical_range_index, 7))

    heuristic_large_votes = sum(
        1 for candidate in heuristic_candidates or []
        if _candidate_range_index(candidate, fallback_bucket_rank=3) >= 7
    )
    if heuristic_large_votes >= 2 and (large_scope_hits + discovery_hits) >= 2:
        floor_index = max(floor_index, 6)

    if large_scope_hits >= 3 and discovery_hits >= 1:
        floor_index = max(floor_index, 6)
    if large_scope_hits >= 4:
        floor_index = max(floor_index, 7)

    # [B1.2] Quando analogical é primary com strong anchor, o floor não pode subir
    # acima de analogical_range_index + 1. Texto com keywords (LARGE_SCOPE_TERMS)
    # não deve sobrescrever um match com top1 ~0.99. Isso evitava que, para uma
    # issue de 2h corretamente estimada pelo analogical, palavras como "auth" ou
    # "version" no texto disparassem o floor em 7 (18-21h).
    route = str(analogical.get("retrieval_route") or "")
    retrieval_stats = analogical.get("retrieval_stats") or {}
    has_strong_anchor = bool(retrieval_stats.get("has_strong_anchor"))
    if route == "analogical_primary" and has_strong_anchor:
        floor_index = min(floor_index, analogical_range_index + 1)

    return clamp_range_index(floor_index)


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
        merged = dict(candidate)
        range_index = _candidate_range_index(merged, fallback_bucket_rank=3)
        range_payload = range_index_to_payload(range_index)
        bucket_rank = range_index_to_bucket_rank(range_index)
        merged["range_index"] = range_payload["range_index"]
        merged["range_label"] = range_payload["range_label"]
        merged["range_min_hours"] = range_payload["range_min_hours"]
        merged["range_max_hours"] = range_payload["range_max_hours"]
        merged["display_hours"] = range_payload["display_hours"]
        merged["estimated_hours"] = float(range_payload["display_hours"])
        merged["estimated_hours_raw"] = float(range_payload["display_hours"])
        merged["min_hours"] = float(range_payload["range_min_hours"])
        merged["max_hours"] = float(range_payload["range_max_hours"])
        merged["bucket_rank"] = bucket_rank
        merged["size_bucket"] = rank_to_bucket(bucket_rank)
        merged["calibration_source"] = "heuristic_range_vote"
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
    analogical_available = bool(analogical) and (
        analogical.get("range_index") not in (None, "")
        or analogical_hours is not None
    )
    analogical_range_index = _candidate_range_index(
        analogical,
        fallback_bucket_rank=analogical.get("bucket_rank") or 3,
    )
    analogical_rank = range_index_to_bucket_rank(analogical_range_index)

    route = str(analogical.get("retrieval_route") or "analogical_weak")
    heuristic_calibration_mode = "weak_prior" if route in {"analogical_weak", "analogical_soft_signal"} else "standard"
    heuristic_candidates = _enrich_heuristic_candidates(
        heuristic_candidates,
        profile,
        calibration_mode=heuristic_calibration_mode,
    )
    heuristic_consensus = aggregate_range_consensus(heuristic_candidates)
    heuristic_calibration = {
        "size_bucket": heuristic_consensus.get("size_bucket"),
        "bucket_rank": heuristic_consensus.get("bucket_rank"),
        "estimated_hours": float(heuristic_consensus.get("display_hours") or 9.0),
        "estimated_hours_raw": float(heuristic_consensus.get("display_hours") or 9.0),
        "min_hours": float(heuristic_consensus.get("range_min_hours") or 9.0),
        "max_hours": float(heuristic_consensus.get("range_max_hours") or 12.0),
        "calibration_source": "heuristic_range_consensus",
        "range_index": heuristic_consensus.get("range_index"),
        "range_label": heuristic_consensus.get("range_label"),
        "range_min_hours": heuristic_consensus.get("range_min_hours"),
        "range_max_hours": heuristic_consensus.get("range_max_hours"),
        "display_hours": heuristic_consensus.get("display_hours"),
    }
    heuristic_range_index = clamp_range_index(heuristic_consensus.get("range_index") or 4)

    retrieval_stats = dict(analogical.get("retrieval_stats") or {})
    top1_score = float(retrieval_stats.get("top1_score") or 0.0)
    useful_count = int(retrieval_stats.get("useful_count") or 0)
    anchor_overlap = float(retrieval_stats.get("anchor_overlap") or 0.0)
    analogical_rank_value = clamp_bucket_rank(analogical_rank or 3)
    heuristic_rank_value = clamp_bucket_rank(heuristic_consensus.get("bucket_rank") or 3)
    range_gap = analogical_range_index - heuristic_range_index
    moderate_analogical_signal = (
        route == "analogical_soft_signal"
        or top1_score >= 0.64
        or (top1_score >= 0.60 and float(retrieval_stats.get("top3_avg_score") or 0.0) >= 0.60)
        or useful_count >= 1
        or (anchor_overlap >= 0.18 and top1_score >= 0.58)
    )

    if route == "analogical_primary" and analogical_available:
        base_range_index = analogical_range_index
        finalization_mode = "analogical_calibrated"
        dominant_strategy = "analogical_consensus"
        selected_model = "analogical_calibrated"
        calibration_source = str(analogical.get("calibration_source") or "analogical_range_neighbors")
        base_confidence = max(analogical_conf, 0.62)
    elif route == "analogical_support" and analogical_available:
        analogical_weight = 0.65 if analogical_conf >= float(heuristic_consensus.get("confidence", 0.0) or 0.0) else 0.55
        base_range_index = _blend_range_indices(
            analogical_range_index,
            heuristic_range_index,
            analogical_weight,
        )
        finalization_mode = "hybrid_calibrated"
        dominant_strategy = "analogical_consensus"
        selected_model = "hybrid_calibrated"
        calibration_source = (
            f"{analogical.get('calibration_source') or 'analogical_range_neighbors'}"
            f"+{heuristic_calibration.get('calibration_source') or 'heuristic_range_consensus'}"
        )
        base_confidence = max(
            0.52,
            min(
                0.82,
                (analogical_conf * analogical_weight)
                + (float(heuristic_consensus.get("confidence", 0.4)) * (1.0 - analogical_weight)),
            ),
        )
    elif route in {"analogical_weak", "analogical_soft_signal"} and analogical_available and moderate_analogical_signal:
        analogical_weight = 0.32
        if top1_score >= 0.72:
            analogical_weight = 0.48
        elif top1_score >= 0.68:
            analogical_weight = 0.42
        elif useful_count >= 1:
            analogical_weight = 0.38
        if range_gap >= 3:
            analogical_weight += 0.10
        if analogical_conf >= 0.75:
            analogical_weight += 0.05
        analogical_weight = max(0.28, min(0.58, analogical_weight))

        base_range_index = _blend_range_indices(
            analogical_range_index,
            heuristic_range_index,
            analogical_weight,
        )
        finalization_mode = "soft_hybrid_calibrated"
        dominant_strategy = "analogical_consensus"
        selected_model = "soft_hybrid_calibrated"
        calibration_source = (
            f"{analogical.get('calibration_source') or 'analogical_range_neighbors'}"
            f"+{heuristic_calibration.get('calibration_source') or 'heuristic_range_consensus'}"
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
        base_range_index = heuristic_range_index
        finalization_mode = "heuristic_bucket_calibrated"
        dominant_strategy = "multiagent_heuristic_consensus"
        selected_model = "heuristic_bucket_calibrated"
        calibration_source = str(heuristic_calibration.get("calibration_source") or "heuristic_range_consensus")
        base_confidence = max(0.35, float(heuristic_consensus.get("confidence", 0.35) or 0.35))

    base_range_index = clamp_range_index(base_range_index)
    base_range = range_index_to_payload(base_range_index)
    base_hours = float(base_range["display_hours"])
    base_rank = range_index_to_bucket_rank(base_range_index)

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
    meta_weight = 0.0
    if bool(meta_prediction.get("available")):
        meta_hours = float(meta_prediction.get("estimated_hours") or base_hours)
        meta_conf = float(meta_prediction.get("confidence") or 0.0)
        # [META_V4] Pesos aumentados porque o meta agora opera direto em
        # range_index (unidade = 3h). Antes, o meta produzia deltas médios de
        # 0.3h que morriam na quantização mesmo com peso alto; agora cada
        # delta unitário do meta muda uma faixa inteira no output final, então
        # o peso precisa refletir a verdadeira autoridade que damos a ele.
        #
        # Regime fraco (analogical_weak/soft_signal): pesos maiores porque o
        # resto do pipeline tem pouca evidência e o histórico é nossa melhor
        # fonte de correção. Regime forte (analogical_primary): pesos baixos
        # preservando a supremacia do match analógico top1>=0.90.
        min_meta_weight = 0.08
        max_meta_weight = 0.20
        if route == "analogical_primary":
            meta_weight = 0.10
            min_meta_weight = 0.08
            max_meta_weight = 0.18
        elif route == "analogical_support":
            meta_weight = 0.22
            min_meta_weight = 0.18
            max_meta_weight = 0.28
        elif route == "analogical_soft_signal":
            meta_weight = 0.40
            min_meta_weight = 0.30
            max_meta_weight = 0.50
        else:
            # analogical_weak — máxima autoridade para o meta.
            meta_weight = 0.45
            min_meta_weight = 0.35
            max_meta_weight = 0.55

        prior_source = str(meta_prediction.get("prior_source") or "")
        prior_count = int(meta_prediction.get("prior_count") or 0)
        if prior_source == "project":
            meta_weight -= 0.05
        elif prior_source == "project_issue":
            meta_weight += 0.01
        elif prior_source in {"project_issue_route", "project_issue_route_bucket"}:
            meta_weight += 0.02
        elif prior_source == "project_issue_bucket":
            meta_weight += 0.01
        if prior_count >= 8:
            meta_weight += 0.02
        elif prior_count < 5:
            meta_weight -= 0.05
        if meta_conf < 0.5:
            meta_weight -= 0.04
        if route == "analogical_support" and top1_score >= 0.78:
            meta_weight -= 0.03
        meta_weight = max(min_meta_weight, min(max_meta_weight, meta_weight))

        meta_range_index = clamp_range_index(
            meta_prediction.get("range_index") or hours_to_range_index(meta_hours)
        )
        base_range_index = _blend_range_indices(meta_range_index, base_range_index, meta_weight)
        base_range = range_index_to_payload(base_range_index)
        base_hours = float(base_range["display_hours"])
        base_rank = range_index_to_bucket_rank(base_range_index)
        base_confidence = max(
            0.32,
            min(0.86, (base_confidence * (1.0 - (meta_weight * 0.55))) + (meta_conf * (meta_weight * 0.55))),
        )
        calibration_source = f"{calibration_source}+{meta_prediction.get('meta_source') or 'meta_linear'}"
        meta_applied = meta_weight >= 0.08

    # [B1.1] Detecta quando analogical é forte (primary + strong anchor).
    # Nesse regime, os agentes de revisão (agile_guard, complexity, critic) não
    # devem sobrescrever a estimativa. Eles foram projetados como rede de segurança
    # para quando o analogical é fraco; ao disparar em analogical forte, eles
    # inflam estimativas corretas (visto em issues 159410 e 159782 do confserver).
    strong_analogical = (
        route == "analogical_primary"
        and bool(retrieval_stats.get("has_strong_anchor"))
    )

    agile_delta = int(agile_guard_review.get("bucket_delta") or 0)
    # [B1.1a] Ignorar inflação do agile_guard quando analogical é forte.
    # agile_delta negativo (pra baixo) ainda é permitido — analogical pode estar alto demais.
    if strong_analogical and agile_delta > 0:
        agile_delta = 0
    if agile_delta:
        base_range_index = clamp_range_index(base_range_index + agile_delta)
        base_range = range_index_to_payload(base_range_index)
        base_hours = float(base_range["display_hours"])
        base_rank = range_index_to_bucket_rank(base_range_index)

    range_adjustment = 0
    complexity_delta = int(complexity_review.get("bucket_delta") or 0)
    complexity_conf = float(complexity_review.get("confidence") or 0.0)
    complexity_risk = float(complexity_review.get("risk_hidden_complexity") or 0.0)
    # [B1.1b] Complexity review só eleva a estimativa se analogical NÃO é forte.
    if (
        complexity_delta > 0
        and complexity_risk >= 0.78
        and complexity_conf >= 0.74
        and not strong_analogical
    ):
        range_adjustment += min(2, complexity_delta)

    critic_under = float(critic_review.get("risk_of_underestimation", 0.0) or 0.0)
    critic_over = float(critic_review.get("risk_of_overestimation", 0.0) or 0.0)
    # [B1.1c] Critic não sobrescreve analogical forte. O crítico estava votando
    # risk_underestimation=0.80 porque heurístico discordava do analogical, mesmo
    # com top1=0.99. Discordância entre signals não é evidência real de viés
    # quando o analogical tem strong anchor.
    if not strong_analogical:
        if critic_under - critic_over >= 0.42:
            range_adjustment += 1
        elif critic_over - critic_under >= 0.55:
            range_adjustment -= 1

    large_issue_floor = _large_issue_floor_index(
        issue=issue,
        analogical=analogical,
        heuristic_candidates=heuristic_candidates,
        complexity_review=complexity_review,
        agile_guard_review=agile_guard_review,
    )
    final_range_index = clamp_range_index(base_range_index + range_adjustment)
    final_range_index = max(final_range_index, large_issue_floor)
    final_range = range_index_to_payload(final_range_index)
    adjusted_hours = float(final_range["display_hours"])
    adjustment_delta = round(adjusted_hours - base_hours, 1)
    final_rank = range_index_to_bucket_rank(final_range_index)
    final_bucket = rank_to_bucket(final_rank)

    should_split = (
        bool(agile_guard_review.get("should_split"))
        or bool(complexity_review.get("should_split"))
        or str(agile_guard_review.get("fit_status") or "") == "oversized"
        or final_range_index >= 10
    )
    split_reason = (
        agile_guard_review.get("split_reason")
        or complexity_review.get("split_reason")
    )
    if should_split and not split_reason and final_range_index >= 10:
        split_reason = "Final range indicates a large backlog item and likely refinement or split."

    calibrated: CalibratedEstimation = {
        "size_bucket": final_bucket,
        "bucket_rank": final_rank,
        "heuristic_size_bucket": str(heuristic_consensus.get("size_bucket") or "M"),
        "heuristic_bucket_rank": clamp_bucket_rank(heuristic_consensus.get("bucket_rank") or 3),
        "analogical_size_bucket": str(analogical.get("size_bucket") or rank_to_bucket(analogical_rank)),
        "analogical_bucket_rank": clamp_bucket_rank(analogical_rank),
        "base_hours": round(float(base_range["display_hours"]), 1),
        "adjusted_hours": round(adjusted_hours, 1),
        "adjustment_delta": round(adjustment_delta, 1),
        "min_hours": final_range["range_min_hours"],
        "max_hours": final_range["range_max_hours"],
        "calibration_source": calibration_source,
        "finalization_mode": finalization_mode,
        "dominant_strategy": dominant_strategy,
        "selected_model": selected_model,
        "base_confidence": round(base_confidence, 4),
        "retrieval_route": route,
        "retrieval_stats": dict(analogical.get("retrieval_stats") or {}),
        "range_index": final_range["range_index"],
        "range_label": final_range["range_label"],
        "range_min_hours": final_range["range_min_hours"],
        "range_max_hours": final_range["range_max_hours"],
        "display_hours": final_range["display_hours"],
        "base_range_index": base_range["range_index"],
        "base_range_label": base_range["range_label"],
        "base_range_min_hours": base_range["range_min_hours"],
        "base_range_max_hours": base_range["range_max_hours"],
        "base_display_hours": base_range["display_hours"],
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
        "meta_range_index": int(meta_prediction.get("range_index") or 0) if bool(meta_prediction.get("available")) else 0,
        "meta_range_label": str(meta_prediction.get("range_label") or "") if bool(meta_prediction.get("available")) else "",
        "meta_range_min_hours": int(meta_prediction.get("range_min_hours") or 0) if bool(meta_prediction.get("available")) else 0,
        "meta_range_max_hours": int(meta_prediction.get("range_max_hours") or 0) if bool(meta_prediction.get("available")) else 0,
        "meta_display_hours": int(meta_prediction.get("display_hours") or 0) if bool(meta_prediction.get("available")) else 0,
        "should_split": should_split,
        "split_reason": split_reason,
        "evidence": [
            f"base_range={base_range['range_label']}",
            f"adjusted_hours={round(adjusted_hours, 1)}",
            f"range={final_range['range_label']}",
            f"heuristic_bucket={heuristic_consensus.get('size_bucket')}",
            f"analogical_route={route}",
            f"large_issue_floor={range_index_to_payload(large_issue_floor)['range_label']}",
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
