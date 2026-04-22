import os
import sys
import uuid
import time
import asyncio
import json
import re
from collections import Counter
from typing import Optional, Any

from sqlalchemy import (
    create_engine,
    text,
    MetaData,
    Table,
    Column,
    String,
    Text as SA_TEXT,
    Float,
    Integer,
    SmallInteger,
    Numeric,
    func,
    bindparam,
)
from sqlalchemy.dialects.mysql import insert as mysql_insert

# Run from repo root:
# python scripts/validation.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ai.dtos.issues_estimation_dto import IssueEstimationDTO
from ai.core.effort_calibration import hours_to_range_payload
from application.services.estimation_service import EstimationService

VALIDATION_TABLE = "issue_estimation_validation"
VALIDATION_KEY_COLUMNS = {
    "validation_run_id",
    "project_key",
    "issue_id",
    "issue_number",
    "repository",
}
VALIDATION_BASE_COLUMNS = [
    "validation_run_id",
    "project_key",
    "issue_id",
    "issue_number",
    "repository",
    "predicted_hours",
    "confidence",
    "justification",
    "estimation_model",
    "model_version",
    "predicted_llm_prompt_tokens",
    "predicted_llm_completion_tokens",
    "predicted_llm_total_tokens",
    "predicted_rag_embedding_tokens",
    "predicted_total_tokens",
]
VALIDATION_DIAGNOSTIC_COLUMNS = {
    "predicted_hours_raw": "DOUBLE NULL",
    "predicted_range_label": "VARCHAR(16) NULL",
    "predicted_range_index": "INT NULL",
    "predicted_range_min_hours": "INT NULL",
    "predicted_range_max_hours": "INT NULL",
    "actual_hours": "DOUBLE NULL",
    "actual_range_label": "VARCHAR(16) NULL",
    "actual_range_index": "INT NULL",
    "actual_range_min_hours": "INT NULL",
    "actual_range_max_hours": "INT NULL",
    "range_hit": "TINYINT(1) NULL",
    "range_distance": "INT NULL",
    "selected_model": "VARCHAR(64) NULL",
    "dominant_strategy": "VARCHAR(64) NULL",
    "finalization_mode": "VARCHAR(64) NULL",
    "retrieval_route": "VARCHAR(32) NULL",
    "top1_score": "DOUBLE NULL",
    "top3_avg_score": "DOUBLE NULL",
    "useful_count": "INT NULL",
    "hours_spread": "DOUBLE NULL",
    "has_strong_anchor": "TINYINT(1) NULL",
    "anchor_score": "DOUBLE NULL",
    "anchor_overlap": "DOUBLE NULL",
    "rag_context_sufficient": "TINYINT(1) NULL",
    "rag_qualified_hits": "INT NULL",
    "rag_min_hits": "INT NULL",
    "rag_min_score": "DOUBLE NULL",
    "size_bucket": "VARCHAR(8) NULL",
    "bucket_rank": "INT NULL",
    "heuristic_size_bucket": "VARCHAR(8) NULL",
    "heuristic_bucket_rank": "INT NULL",
    "analogical_size_bucket": "VARCHAR(8) NULL",
    "analogical_bucket_rank": "INT NULL",
    "calibration_source": "VARCHAR(128) NULL",
    "base_hours": "DOUBLE NULL",
    "adjusted_hours": "DOUBLE NULL",
    "adjustment_delta": "DOUBLE NULL",
    "meta_applied": "TINYINT(1) NULL",
    "meta_hours": "DOUBLE NULL",
    "meta_min_hours": "DOUBLE NULL",
    "meta_max_hours": "DOUBLE NULL",
    "meta_confidence": "DOUBLE NULL",
    "meta_source": "VARCHAR(128) NULL",
    "meta_prior_source": "VARCHAR(64) NULL",
    "meta_prior_count": "INT NULL",
    "meta_blend_weight": "DOUBLE NULL",
    "meta_model_version": "VARCHAR(64) NULL",
    "final_min_hours": "DOUBLE NULL",
    "final_max_hours": "DOUBLE NULL",
    "final_should_split": "TINYINT(1) NULL",
    "final_split_reason": "LONGTEXT NULL",
    "analogical_hours": "DOUBLE NULL",
    "analogical_min_hours": "DOUBLE NULL",
    "analogical_max_hours": "DOUBLE NULL",
    "analogical_confidence": "DOUBLE NULL",
    "analogical_should_split": "TINYINT(1) NULL",
    "heuristic_scope_hours": "DOUBLE NULL",
    "heuristic_scope_confidence": "DOUBLE NULL",
    "heuristic_complexity_hours": "DOUBLE NULL",
    "heuristic_complexity_confidence": "DOUBLE NULL",
    "heuristic_uncertainty_hours": "DOUBLE NULL",
    "heuristic_uncertainty_confidence": "DOUBLE NULL",
    "heuristic_agile_fit_hours": "DOUBLE NULL",
    "heuristic_agile_fit_confidence": "DOUBLE NULL",
    "complexity_review_hours": "DOUBLE NULL",
    "complexity_review_confidence": "DOUBLE NULL",
    "complexity_review_should_split": "TINYINT(1) NULL",
    "complexity_bucket_delta": "INT NULL",
    "agile_guard_hours": "DOUBLE NULL",
    "agile_guard_confidence": "DOUBLE NULL",
    "agile_guard_should_split": "TINYINT(1) NULL",
    "agile_guard_bucket_delta": "INT NULL",
    "agile_guard_fit_status": "VARCHAR(32) NULL",
    "heuristic_scope_bucket": "VARCHAR(8) NULL",
    "heuristic_complexity_bucket": "VARCHAR(8) NULL",
    "heuristic_uncertainty_bucket": "VARCHAR(8) NULL",
    "heuristic_agile_fit_bucket": "VARCHAR(8) NULL",
    "critic_risk_underestimation": "DOUBLE NULL",
    "critic_risk_overestimation": "DOUBLE NULL",
    "service_latency_ms": "INT NULL",
    "workflow_latency_ms": "INT NULL",
    "primary_reviews_latency_ms": "INT NULL",
    "analogical_latency_ms": "INT NULL",
    "heuristic_ensemble_latency_ms": "INT NULL",
    "complexity_review_latency_ms": "INT NULL",
    "agile_guard_latency_ms": "INT NULL",
    "critic_latency_ms": "INT NULL",
    "calibration_latency_ms": "INT NULL",
    "supervisor_latency_ms": "INT NULL",
    "decision_trace_json": "LONGTEXT NULL",
    "agent_trace_json": "LONGTEXT NULL",
    "execution_trace_json": "LONGTEXT NULL",
}


def normalize_estimation_model(strategy: Optional[str]) -> str:
    if strategy in {
        "analogical",
        "analogical_consensus",
        "analogical_calibrated",
        "hybrid_calibrated",
        "soft_hybrid_calibrated",
    }:
        return "analogical+multiagent_consensus"
    if strategy in {"multiagent_heuristic_consensus", "heuristic_bucket_calibrated"}:
        return "multiagent_heuristic_consensus"
    return "heuristic"


def get_engine():
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("Defina DATABASE_URL (mysql+pymysql://...)")
    return create_engine(db_url, pool_pre_ping=True)


def maybe_float(value: Any, digits: int = 4) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def maybe_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def maybe_bool_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    return 1 if bool(value) else 0


def maybe_text(value: Any) -> Optional[str]:
    text_value = str(value or "").strip()
    return text_value or None


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def safe_json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


class ValidationProgress:
    def __init__(self, total: int, progress_every: int):
        self.total = max(0, int(total))
        self.progress_every = max(1, int(progress_every))
        self.completed = 0
        self.ok = 0
        self.errors = 0
        self.total_service_ms = 0
        self.model_counts: Counter[str] = Counter()
        self.selected_counts: Counter[str] = Counter()
        self._lock = asyncio.Lock()

    @staticmethod
    def _compact_counts(counts: Counter[str], limit: int = 3) -> str:
        if not counts:
            return "-"
        parts = [f"{name}:{count}" for name, count in counts.most_common(limit)]
        return ",".join(parts)

    @staticmethod
    def _fmt_hours(value: Any) -> str:
        try:
            hours = float(value)
            return f"{hours:.1f}"
        except (TypeError, ValueError):
            return "-"

    @staticmethod
    def _fmt_range(value: Any) -> str:
        text_value = str(value or "").strip()
        return text_value or "-"

    def _avg_ms(self) -> int:
        if self.completed <= 0:
            return 0
        return int(round(self.total_service_ms / self.completed))

    async def on_success(self, issue_id: Any, payload: dict) -> None:
        async with self._lock:
            self.completed += 1
            self.ok += 1
            service_ms = int(payload.get("service_latency_ms") or 0)
            self.total_service_ms += service_ms
            model_name = str(payload.get("estimation_model") or "unknown")
            selected_name = str(payload.get("selected_model") or "unknown")
            self.model_counts[model_name] += 1
            self.selected_counts[selected_name] += 1

            print(
                f"[{self.completed}/{self.total}] issue={issue_id} "
                f"model={model_name} selected={selected_name} "
                f"range={self._fmt_range(payload.get('predicted_range_label'))} "
                f"hours={self._fmt_hours(payload.get('predicted_hours'))} "
                f"ms={service_ms} ok"
            )

            if self.completed % self.progress_every == 0 or self.completed == self.total:
                print(
                    f"[RESUMO] done={self.completed}/{self.total} ok={self.ok} erro={self.errors} "
                    f"avg_ms={self._avg_ms()} models={self._compact_counts(self.model_counts)} "
                    f"selected={self._compact_counts(self.selected_counts)}"
                )

    async def on_error(self, issue_id: Any, exc: Exception, service_ms: int) -> None:
        async with self._lock:
            self.completed += 1
            self.errors += 1
            self.total_service_ms += max(0, int(service_ms or 0))
            message = " ".join(str(exc).strip().splitlines())[:180] or exc.__class__.__name__

            print(
                f"[{self.completed}/{self.total}] issue={issue_id} "
                f"ms={max(0, int(service_ms or 0))} erro={message}"
            )

            if self.completed % self.progress_every == 0 or self.completed == self.total:
                print(
                    f"[RESUMO] done={self.completed}/{self.total} ok={self.ok} erro={self.errors} "
                    f"avg_ms={self._avg_ms()} models={self._compact_counts(self.model_counts)} "
                    f"selected={self._compact_counts(self.selected_counts)}"
                )


def fetch_table_columns(engine, table_name: str) -> set[str]:
    sql = text(
        """
        SELECT LOWER(COLUMN_NAME) AS column_name
        FROM information_schema.columns
        WHERE table_schema = DATABASE()
          AND table_name = :table_name
        """
    )
    try:
        with engine.connect() as conn:
            rows = conn.execute(sql, {"table_name": table_name}).mappings().all()
    except Exception as exc:
        print(f"[WARN] Nao foi possivel ler colunas de {table_name}: {exc}")
        return set()
    return {str(r["column_name"]).lower() for r in rows}


def ensure_validation_diagnostic_columns(engine, table_name: str) -> set[str]:
    existing = fetch_table_columns(engine, table_name)
    missing = [name for name in VALIDATION_DIAGNOSTIC_COLUMNS if name not in existing]

    if not missing:
        return existing
    try:
        metadata = MetaData()
        table = Table(table_name, metadata, autoload_with=engine)
        with engine.begin() as conn:
            for column_name in missing:
                ddl = VALIDATION_DIAGNOSTIC_COLUMNS[column_name]
                col = _ddl_to_column(column_name, ddl)
                try:
                    col.create(table, conn)
                except Exception as exc:
                    print(
                        f"[WARN] Nao foi possivel criar coluna {column_name} em {table_name}: {exc}"
                    )
        print(
            "[INFO] Colunas diagnosticas adicionadas em "
            f"{table_name}: {', '.join(missing)}"
        )
    except Exception as exc:
        print(
            "[WARN] Nao foi possivel adicionar todas as colunas diagnosticas em "
            f"{table_name}: {exc}"
        )

    return fetch_table_columns(engine, table_name)


def build_upsert_sql(available_columns: set[str], table_name: str = VALIDATION_TABLE, engine=None):
    if engine is None:
        raise RuntimeError("engine is required to build upsert statement")

    optional_columns = [
        column_name
        for column_name in VALIDATION_DIAGNOSTIC_COLUMNS
        if column_name in available_columns
    ]
    insert_columns = VALIDATION_BASE_COLUMNS + optional_columns

    metadata = MetaData()
    table = Table(table_name, metadata, autoload_with=engine)

    stmt = mysql_insert(table).values({col: bindparam(col) for col in insert_columns})

    update_dict = {
        col: stmt.inserted[col]
        for col in insert_columns
        if col not in VALIDATION_KEY_COLUMNS
    }

    # ensure predicted_at is updated
    update_dict["predicted_at"] = func.now()

    stmt = stmt.on_duplicate_key_update(**update_dict)

    return stmt


def _candidate_map(candidates: list[dict]) -> dict[str, dict]:
    by_mode: dict[str, dict] = {}
    for candidate in candidates or []:
        if not isinstance(candidate, dict):
            continue
        mode = str(candidate.get("mode") or candidate.get("source") or "").strip().lower()
        if mode:
            by_mode[mode] = candidate
    return by_mode


def build_validation_payload(
    row: dict,
    state: dict,
    final_estimation: dict,
    usage: dict,
    run_id: str,
    model_version: Optional[str],
    service_latency_ms: int,
    save_verbose_trace: bool = True,
) -> dict:
    estimation_model = (
        final_estimation.get("estimation_model")
        or normalize_estimation_model(state.get("strategy"))
    )

    agent_trace = dict(final_estimation.get("agent_trace") or {})
    analogical_result = dict(
        agent_trace.get("analogical")
        or state.get("analogical")
        or final_estimation.get("analogical_result")
        or {}
    )
    heuristic_candidates = list(
        agent_trace.get("heuristic_candidates")
        or state.get("heuristic_candidates")
        or []
    )
    complexity_review = dict(
        agent_trace.get("complexity_review")
        or state.get("complexity_review")
        or {}
    )
    agile_guard_review = dict(
        agent_trace.get("agile_guard_review")
        or state.get("agile_guard_review")
        or {}
    )
    critic_review = dict(
        agent_trace.get("critic_review")
        or state.get("critic_review")
        or {}
    )
    calibrated_estimation = dict(
        final_estimation.get("calibrated_estimation")
        or agent_trace.get("calibrated_estimation")
        or state.get("calibrated_estimation")
        or {}
    )

    heuristic_by_mode = _candidate_map(heuristic_candidates)
    rag_stats = dict(state.get("rag_stats") or {})
    retrieval_stats = (
        final_estimation.get("retrieval_stats")
        or analogical_result.get("retrieval_stats")
        or {}
    )
    retrieval_route = (
        final_estimation.get("retrieval_route")
        or analogical_result.get("retrieval_route")
        or retrieval_stats.get("route")
    )
    selected_model = (
        final_estimation.get("selected_model")
        or calibrated_estimation.get("selected_model")
        or final_estimation.get("dominant_strategy")
        or estimation_model
    )
    execution_trace = dict(final_estimation.get("execution_trace") or state.get("execution_metrics") or {})

    project_key = str(row.get("project_key") or row["project_id"]).strip().lower()
    predicted_hours = final_estimation.get("estimated_hours")
    predicted_hours_raw = final_estimation.get("estimated_hours_raw", predicted_hours)
    predicted_range = {
        "range_index": maybe_int(final_estimation.get("range_index")),
        "range_label": maybe_text(final_estimation.get("range_label")),
        "range_min_hours": maybe_int(final_estimation.get("range_min_hours")),
        "range_max_hours": maybe_int(final_estimation.get("range_max_hours")),
    }
    if not predicted_range["range_label"] and predicted_hours_raw not in (None, ""):
        normalized_range = hours_to_range_payload(predicted_hours_raw)
        predicted_range = {
            "range_index": int(normalized_range["range_index"]),
            "range_label": str(normalized_range["range_label"]),
            "range_min_hours": int(normalized_range["range_min_hours"]),
            "range_max_hours": int(normalized_range["range_max_hours"]),
        }

    actual_hours = maybe_float(row.get("actual_hours"))
    actual_range = {"range_index": None, "range_label": None, "range_min_hours": None, "range_max_hours": None}
    if actual_hours is not None:
        actual_payload = hours_to_range_payload(actual_hours)
        actual_range = {
            "range_index": int(actual_payload["range_index"]),
            "range_label": str(actual_payload["range_label"]),
            "range_min_hours": int(actual_payload["range_min_hours"]),
            "range_max_hours": int(actual_payload["range_max_hours"]),
        }

    predicted_range_index = predicted_range["range_index"]
    actual_range_index = actual_range["range_index"]
    range_hit = None
    range_distance = None
    if predicted_range_index is not None and actual_range_index is not None:
        range_hit = 1 if int(predicted_range_index) == int(actual_range_index) else 0
        range_distance = abs(int(predicted_range_index) - int(actual_range_index))

    decision_trace_json = None
    agent_trace_json = None
    execution_trace_json = None

    if save_verbose_trace:
        decision_trace = {
            "estimation_model": estimation_model,
            "selected_model": selected_model,
            "user_justification": final_estimation.get("user_justification") or final_estimation.get("justification"),
            "analysis_justification": final_estimation.get("analysis_justification"),
            "predicted_range": predicted_range,
            "actual_range": actual_range,
            "range_hit": range_hit,
            "range_distance": range_distance,
            "dominant_strategy": final_estimation.get("dominant_strategy") or calibrated_estimation.get("dominant_strategy"),
            "finalization_mode": final_estimation.get("finalization_mode") or calibrated_estimation.get("finalization_mode"),
            "retrieval_route": retrieval_route,
            "retrieval_stats": {
                "top1_score": maybe_float(retrieval_stats.get("top1_score")),
                "top3_avg_score": maybe_float(retrieval_stats.get("top3_avg_score")),
                "useful_count": maybe_int(retrieval_stats.get("useful_count")),
                "hours_spread": maybe_float(retrieval_stats.get("hours_spread")),
                "has_strong_anchor": bool(retrieval_stats.get("has_strong_anchor")),
                "anchor_score": maybe_float(retrieval_stats.get("anchor_score")),
                "anchor_overlap": maybe_float(retrieval_stats.get("anchor_overlap")),
            },
            "rag_stats": {
                "rag_context_sufficient": bool(state.get("rag_context_sufficient")),
                "qualified_hits": maybe_int(rag_stats.get("qualified_hits")),
                "min_hits": maybe_int(rag_stats.get("min_hits")),
                "min_score": maybe_float(rag_stats.get("min_score")),
            },
            "calibrated_estimation": {
                "size_bucket": maybe_text(calibrated_estimation.get("size_bucket")),
                "bucket_rank": maybe_int(calibrated_estimation.get("bucket_rank")),
                "heuristic_size_bucket": maybe_text(calibrated_estimation.get("heuristic_size_bucket")),
                "heuristic_bucket_rank": maybe_int(calibrated_estimation.get("heuristic_bucket_rank")),
                "analogical_size_bucket": maybe_text(calibrated_estimation.get("analogical_size_bucket")),
                "analogical_bucket_rank": maybe_int(calibrated_estimation.get("analogical_bucket_rank")),
                "base_hours": maybe_float(calibrated_estimation.get("base_hours")),
                "adjusted_hours": maybe_float(calibrated_estimation.get("adjusted_hours")),
                "adjustment_delta": maybe_float(calibrated_estimation.get("adjustment_delta")),
                "meta_applied": bool(calibrated_estimation.get("meta_applied")),
                "meta_hours": maybe_float(calibrated_estimation.get("meta_hours")),
                "meta_confidence": maybe_float(calibrated_estimation.get("meta_confidence")),
                "meta_source": maybe_text(calibrated_estimation.get("meta_source")),
                "meta_prior_source": maybe_text(calibrated_estimation.get("meta_prior_source")),
                "meta_prior_count": maybe_int(calibrated_estimation.get("meta_prior_count")),
                "meta_blend_weight": maybe_float(calibrated_estimation.get("meta_blend_weight")),
                "meta_model_version": maybe_text(calibrated_estimation.get("meta_model_version")),
                "calibration_source": maybe_text(calibrated_estimation.get("calibration_source")),
            },
            "analogical": {
                "estimated_hours": maybe_float(analogical_result.get("estimated_hours")),
                "confidence": maybe_float(analogical_result.get("confidence")),
                "size_bucket": maybe_text(analogical_result.get("size_bucket")),
                "bucket_rank": maybe_int(analogical_result.get("bucket_rank")),
                "latency_ms": maybe_int(analogical_result.get("latency_ms")),
            },
            "heuristics": {
                mode_name: {
                    "estimated_hours": maybe_float(candidate.get("estimated_hours")),
                    "size_bucket": maybe_text(candidate.get("size_bucket")),
                    "bucket_rank": maybe_int(candidate.get("bucket_rank")),
                    "confidence": maybe_float(candidate.get("confidence")),
                    "latency_ms": maybe_int(candidate.get("latency_ms")),
                }
                for mode_name, candidate in heuristic_by_mode.items()
            },
            "complexity_review": {
                "bucket_delta": maybe_int(complexity_review.get("bucket_delta")),
                "confidence": maybe_float(complexity_review.get("confidence")),
                "latency_ms": maybe_int(complexity_review.get("latency_ms")),
            },
            "agile_guard_review": {
                "bucket_delta": maybe_int(agile_guard_review.get("bucket_delta")),
                "fit_status": maybe_text(agile_guard_review.get("fit_status")),
                "confidence": maybe_float(agile_guard_review.get("confidence")),
                "latency_ms": maybe_int(agile_guard_review.get("latency_ms")),
            },
            "critic_review": {
                "risk_of_underestimation": maybe_float(critic_review.get("risk_of_underestimation")),
                "risk_of_overestimation": maybe_float(critic_review.get("risk_of_overestimation")),
                "latency_ms": maybe_int(critic_review.get("latency_ms")),
            },
            "execution_trace": execution_trace,
            "service_latency_ms": service_latency_ms,
        }
        decision_trace_json = safe_json_dumps(decision_trace)
        agent_trace_json = safe_json_dumps(agent_trace)
        execution_trace_json = safe_json_dumps(execution_trace)

    return {
        "validation_run_id": run_id,
        "project_key": project_key,
        "issue_id": int(row["id"]),
        "issue_number": int(row["id"]),
        "repository": project_key,
        "predicted_hours": predicted_hours,
        "predicted_hours_raw": maybe_float(predicted_hours_raw),
        "predicted_range_label": predicted_range["range_label"],
        "predicted_range_index": predicted_range["range_index"],
        "predicted_range_min_hours": predicted_range["range_min_hours"],
        "predicted_range_max_hours": predicted_range["range_max_hours"],
        "actual_hours": actual_hours,
        "actual_range_label": actual_range["range_label"],
        "actual_range_index": actual_range["range_index"],
        "actual_range_min_hours": actual_range["range_min_hours"],
        "actual_range_max_hours": actual_range["range_max_hours"],
        "range_hit": range_hit,
        "range_distance": range_distance,
        "confidence": final_estimation.get("confidence"),
        "justification": final_estimation.get("user_justification") or final_estimation.get("justification", ""),
        "estimation_model": estimation_model,
        "model_version": model_version,
        "predicted_llm_prompt_tokens": int(usage.get("predicted_llm_prompt_tokens") or 0),
        "predicted_llm_completion_tokens": int(usage.get("predicted_llm_completion_tokens") or 0),
        "predicted_llm_total_tokens": int(usage.get("predicted_llm_total_tokens") or 0),
        "predicted_rag_embedding_tokens": int(usage.get("predicted_rag_embedding_tokens") or 0),
        "predicted_total_tokens": int(usage.get("predicted_total_tokens") or 0),
        "selected_model": maybe_text(selected_model),
        "dominant_strategy": maybe_text(final_estimation.get("dominant_strategy") or calibrated_estimation.get("dominant_strategy")),
        "finalization_mode": maybe_text(final_estimation.get("finalization_mode") or calibrated_estimation.get("finalization_mode")),
        "retrieval_route": maybe_text(retrieval_route),
        "top1_score": maybe_float(retrieval_stats.get("top1_score")),
        "top3_avg_score": maybe_float(retrieval_stats.get("top3_avg_score")),
        "useful_count": maybe_int(retrieval_stats.get("useful_count")),
        "hours_spread": maybe_float(retrieval_stats.get("hours_spread")),
        "has_strong_anchor": maybe_bool_int(retrieval_stats.get("has_strong_anchor")),
        "anchor_score": maybe_float(retrieval_stats.get("anchor_score")),
        "anchor_overlap": maybe_float(retrieval_stats.get("anchor_overlap")),
        "rag_context_sufficient": maybe_bool_int(state.get("rag_context_sufficient")),
        "rag_qualified_hits": maybe_int(rag_stats.get("qualified_hits")),
        "rag_min_hits": maybe_int(rag_stats.get("min_hits")),
        "rag_min_score": maybe_float(rag_stats.get("min_score")),
        "size_bucket": maybe_text(final_estimation.get("size_bucket") or calibrated_estimation.get("size_bucket")),
        "bucket_rank": maybe_int(final_estimation.get("bucket_rank") or calibrated_estimation.get("bucket_rank")),
        "heuristic_size_bucket": maybe_text(calibrated_estimation.get("heuristic_size_bucket")),
        "heuristic_bucket_rank": maybe_int(calibrated_estimation.get("heuristic_bucket_rank")),
        "analogical_size_bucket": maybe_text(analogical_result.get("size_bucket") or calibrated_estimation.get("analogical_size_bucket")),
        "analogical_bucket_rank": maybe_int(analogical_result.get("bucket_rank") or calibrated_estimation.get("analogical_bucket_rank")),
        "calibration_source": maybe_text(final_estimation.get("calibration_source") or calibrated_estimation.get("calibration_source")),
        "base_hours": maybe_float(final_estimation.get("base_hours") or calibrated_estimation.get("base_hours")),
        "adjusted_hours": maybe_float(final_estimation.get("adjusted_hours") or calibrated_estimation.get("adjusted_hours")),
        "adjustment_delta": maybe_float(final_estimation.get("adjustment_delta") or calibrated_estimation.get("adjustment_delta")),
        "meta_applied": maybe_bool_int(calibrated_estimation.get("meta_applied")),
        "meta_hours": maybe_float(calibrated_estimation.get("meta_hours")),
        "meta_min_hours": maybe_float(calibrated_estimation.get("meta_min_hours")),
        "meta_max_hours": maybe_float(calibrated_estimation.get("meta_max_hours")),
        "meta_confidence": maybe_float(calibrated_estimation.get("meta_confidence")),
        "meta_source": maybe_text(calibrated_estimation.get("meta_source")),
        "meta_prior_source": maybe_text(calibrated_estimation.get("meta_prior_source")),
        "meta_prior_count": maybe_int(calibrated_estimation.get("meta_prior_count")),
        "meta_blend_weight": maybe_float(calibrated_estimation.get("meta_blend_weight")),
        "meta_model_version": maybe_text(calibrated_estimation.get("meta_model_version")),
        "final_min_hours": maybe_float(final_estimation.get("min_hours")),
        "final_max_hours": maybe_float(final_estimation.get("max_hours")),
        "final_should_split": maybe_bool_int(final_estimation.get("should_split")),
        "final_split_reason": maybe_text(final_estimation.get("split_reason")),
        "analogical_hours": maybe_float(analogical_result.get("estimated_hours")),
        "analogical_min_hours": maybe_float(analogical_result.get("min_hours")),
        "analogical_max_hours": maybe_float(analogical_result.get("max_hours")),
        "analogical_confidence": maybe_float(analogical_result.get("confidence")),
        "analogical_should_split": maybe_bool_int(analogical_result.get("should_split")),
        "heuristic_scope_hours": maybe_float((heuristic_by_mode.get("scope") or {}).get("estimated_hours")),
        "heuristic_scope_confidence": maybe_float((heuristic_by_mode.get("scope") or {}).get("confidence")),
        "heuristic_scope_bucket": maybe_text((heuristic_by_mode.get("scope") or {}).get("size_bucket")),
        "heuristic_complexity_hours": maybe_float((heuristic_by_mode.get("complexity") or {}).get("estimated_hours")),
        "heuristic_complexity_confidence": maybe_float((heuristic_by_mode.get("complexity") or {}).get("confidence")),
        "heuristic_complexity_bucket": maybe_text((heuristic_by_mode.get("complexity") or {}).get("size_bucket")),
        "heuristic_uncertainty_hours": maybe_float((heuristic_by_mode.get("uncertainty") or {}).get("estimated_hours")),
        "heuristic_uncertainty_confidence": maybe_float((heuristic_by_mode.get("uncertainty") or {}).get("confidence")),
        "heuristic_uncertainty_bucket": maybe_text((heuristic_by_mode.get("uncertainty") or {}).get("size_bucket")),
        "heuristic_agile_fit_hours": maybe_float((heuristic_by_mode.get("agile_fit") or {}).get("estimated_hours")),
        "heuristic_agile_fit_confidence": maybe_float((heuristic_by_mode.get("agile_fit") or {}).get("confidence")),
        "heuristic_agile_fit_bucket": maybe_text((heuristic_by_mode.get("agile_fit") or {}).get("size_bucket")),
        "complexity_review_hours": maybe_float(complexity_review.get("estimated_hours")),
        "complexity_review_confidence": maybe_float(complexity_review.get("confidence")),
        "complexity_review_should_split": maybe_bool_int(complexity_review.get("should_split")),
        "complexity_bucket_delta": maybe_int(complexity_review.get("bucket_delta")),
        "agile_guard_hours": maybe_float(agile_guard_review.get("estimated_hours")),
        "agile_guard_confidence": maybe_float(agile_guard_review.get("confidence")),
        "agile_guard_should_split": maybe_bool_int(agile_guard_review.get("should_split")),
        "agile_guard_bucket_delta": maybe_int(agile_guard_review.get("bucket_delta")),
        "agile_guard_fit_status": maybe_text(agile_guard_review.get("fit_status")),
        "critic_risk_underestimation": maybe_float(critic_review.get("risk_of_underestimation")),
        "critic_risk_overestimation": maybe_float(critic_review.get("risk_of_overestimation")),
        "service_latency_ms": service_latency_ms,
        "workflow_latency_ms": maybe_int(execution_trace.get("workflow_latency_ms")),
        "primary_reviews_latency_ms": maybe_int(execution_trace.get("primary_reviews_latency_ms")),
        "analogical_latency_ms": maybe_int(execution_trace.get("analogical_latency_ms")),
        "heuristic_ensemble_latency_ms": maybe_int(execution_trace.get("heuristic_ensemble_latency_ms")),
        "complexity_review_latency_ms": maybe_int(execution_trace.get("complexity_review_latency_ms")),
        "agile_guard_latency_ms": maybe_int(execution_trace.get("agile_guard_latency_ms")),
        "critic_latency_ms": maybe_int(execution_trace.get("critic_latency_ms")),
        "calibration_latency_ms": maybe_int(execution_trace.get("calibration_latency_ms")),
        "supervisor_latency_ms": maybe_int(execution_trace.get("supervisor_latency_ms")),
        "decision_trace_json": decision_trace_json,
        "agent_trace_json": agent_trace_json,
        "execution_trace_json": execution_trace_json,
    }


def build_dto_from_row(r: dict) -> IssueEstimationDTO:
    issue_id = int(r["id"])
    project_id = str(r.get("project_id"))
    project_key = str(r.get("project_key") or project_id).strip().lower()

    title = (r.get("title") or "").strip()
    description_text = (r.get("description_text") or "").strip()
    issue_type = (r.get("type") or "").strip()
    assignee_id = r.get("assignee_id")

    extra_context_lines = [
        f"ProjectId: {project_id}",
        f"ProjectKey: {project_key}",
        f"Type: {issue_type}" if issue_type else "Type: (null)",
        f"AssigneeId: {assignee_id}" if assignee_id is not None else "AssigneeId: (null)",
    ]
    extra_context = "\n".join(extra_context_lines)

    final_description = (
        f"{extra_context}\n\n"
        f"Description:\n{description_text if description_text else '(empty)'}"
    )

    return IssueEstimationDTO(
        issue_number=issue_id,
        repository=project_key,
        issue_type=issue_type,
        title=title,
        description=final_description,
        labels=[],
        assignees=[],
        state="open",
        is_open=False,
        comments_count=0,
        age_in_days=0,
        author_login="unknown",
        author_role="NONE",
        repo_language=None,
        repo_size=None,
    )


def fetch_issues_for_validation(engine, project_id: int, limit: int) -> list[dict]:
    sql = text(
        """
        SELECT
          id,
          project_id,
          project_key,
          title,
          description_text,
          type,
          assignee_id,
          ROUND((i.total_effort_minutes / 60.0), 4) AS actual_hours
        FROM (
            SELECT
              x.id,
              x.project_id,
              p.project_key,
              x.title,
              x.description_text,
              x.type,
              x.assignee_id,
              x.total_effort_minutes,
              ROW_NUMBER() OVER (
                PARTITION BY x.project_id
                ORDER BY x.id ASC
              ) AS rn
            FROM issue x
            JOIN Project p ON p.ID = x.project_id
            WHERE
                x.resolution_date IS NOT NULL
                AND x.status IN ('Closed','Done','Resolved','Complete')
                AND x.resolution IN ('Fixed','Done','Complete','Completed','Works as Designed')
                AND (x.total_effort_minutes / 60.0) BETWEEN 1 AND 40
                AND length(x.description_text) >= 100
                AND x.project_id = :project_id
        ) i
        ORDER BY rn
        LIMIT :limit
        """
    )

    with engine.connect() as conn:
        rows = conn.execute(sql, {"project_id": project_id, "limit": limit}).mappings().all()
        return [dict(r) for r in rows]


def persist_validation_payload(engine, upsert_sql, payload: dict) -> None:
    with engine.begin() as conn:
        conn.execute(upsert_sql, payload)


async def main():
    run_started_at = time.perf_counter()
    engine = get_engine()

    project_id = int(os.getenv("PROJECT_ID"))
    limit = int(os.getenv("BATCH_LIMIT"))
    run_id = os.getenv("VALIDATION_RUN_ID", str(uuid.uuid4()))
    model_version = os.getenv("MODEL_VERSION")
    max_concurrency = max(1, int(os.getenv("VALIDATION_MAX_CONCURRENCY", "2")))
    progress_every = max(1, int(os.getenv("VALIDATION_PROGRESS_EVERY", "10")))
    save_verbose_trace = env_bool("VALIDATION_SAVE_VERBOSE_TRACE", False)

    issues = fetch_issues_for_validation(engine, project_id=project_id, limit=limit)
    if not issues:
        print("[FIM] nenhuma issue encontrada com os filtros informados.")
        return

    svc = EstimationService()
    available_columns = ensure_validation_diagnostic_columns(engine, VALIDATION_TABLE)
    upsert_sql = build_upsert_sql(available_columns, VALIDATION_TABLE, engine)
    active_diagnostic_columns = [
        column_name
        for column_name in VALIDATION_DIAGNOSTIC_COLUMNS
        if column_name in available_columns
    ]

    progress = ValidationProgress(total=len(issues), progress_every=progress_every)

    print(
        f"[INICIO] run_id={run_id} project_id={project_id} issues={len(issues)} "
        f"concurrency={max_concurrency} progress_every={progress_every} "
        f"verbose_trace={str(save_verbose_trace).lower()} diagnostic_cols={len(active_diagnostic_columns)}"
    )

    semaphore = asyncio.Semaphore(max_concurrency)

    async def process_issue(row: dict) -> None:
        async with semaphore:
            started_at = time.perf_counter()
            dto = build_dto_from_row(row)
            try:
                state = await svc.run(dto)
                service_latency_ms = int((time.perf_counter() - started_at) * 1000)
                final_estimation = state.get("final_estimation", {}) or {}
                usage = state.get("token_usage_summary", {}) or {}
                payload = build_validation_payload(
                    row=row,
                    state=state,
                    final_estimation=final_estimation,
                    usage=usage,
                    run_id=run_id,
                    model_version=model_version,
                    service_latency_ms=service_latency_ms,
                    save_verbose_trace=save_verbose_trace,
                )
                await asyncio.to_thread(persist_validation_payload, engine, upsert_sql, payload)
                await progress.on_success(row["id"], payload)
            except Exception as exc:
                service_latency_ms = int((time.perf_counter() - started_at) * 1000)
                await progress.on_error(row["id"], exc, service_latency_ms)

    tasks = [asyncio.create_task(process_issue(row)) for row in issues]
    for task in asyncio.as_completed(tasks):
        await task

    total_s = round(time.perf_counter() - run_started_at, 1)
    print(
        f"[FIM] done={progress.completed}/{progress.total} ok={progress.ok} erro={progress.errors} "
        f"total_s={total_s} avg_ms={progress._avg_ms()} "
        f"models={progress._compact_counts(progress.model_counts, limit=5)} "
        f"selected={progress._compact_counts(progress.selected_counts, limit=5)}"
    )


if __name__ == "__main__":
    asyncio.run(main())
