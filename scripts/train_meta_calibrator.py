import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from sqlalchemy import create_engine, MetaData, Table, select, func


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from ai.core.meta_calibrator import (  # noqa: E402
    build_training_row_from_validation,
    get_meta_model_path,
    save_meta_model,
    train_meta_calibrator,
)


def get_engine():
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError(
            "Defina DATABASE_URL (mysql+pymysql://...) no ambiente ou no arquivo .env da raiz do projeto."
        )
    return create_engine(db_url, pool_pre_ping=True)


def _parse_csv_env(name: str) -> list[str]:
    raw = str(os.getenv(name) or "").strip()
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _env_bool(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _dedupe_latest_issue_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest_by_issue: dict[Any, dict[str, Any]] = {}
    ordered = sorted(
        rows,
        key=lambda row: (
            str(row.get("predicted_at") or ""),
            str(row.get("validation_run_id") or ""),
        ),
        reverse=True,
    )
    for row in ordered:
        issue_id = row.get("issue_id")
        if issue_id not in latest_by_issue:
            latest_by_issue[issue_id] = row
    return sorted(
        latest_by_issue.values(),
        key=lambda row: (
            str(row.get("predicted_at") or ""),
            int(row.get("issue_id") or 0),
        ),
    )


def fetch_training_rows(engine) -> list[dict[str, Any]]:
    run_ids = _parse_csv_env("TRAIN_VALIDATION_RUN_IDS")
    project_keys = [item.lower() for item in _parse_csv_env("TRAIN_PROJECT_KEYS")]
    predicted_at_from = str(os.getenv("TRAIN_PREDICTED_AT_FROM") or "").strip()
    latest_per_issue = _env_bool("TRAIN_LATEST_PER_ISSUE", True)
    min_hours = float(os.getenv("TRAIN_MIN_ACTUAL_HOURS", "1") or 1)
    max_hours = float(os.getenv("TRAIN_MAX_ACTUAL_HOURS", "40") or 40)

    # Build a safe SQLAlchemy Core select to avoid raw text SQL construction
    metadata = MetaData()
    v = Table("issue_estimation_validation", metadata, autoload_with=engine)
    x = Table("issue", metadata, autoload_with=engine)

    stmt = select(
        v.c.issue_id,
        v.c.validation_run_id,
        v.c.predicted_at,
        v.c.project_key,
        func.lower(func.coalesce(func.nullif(x.c.type, ""), "unknown")).label("issue_type"),
        v.c.selected_model,
        v.c.finalization_mode,
        v.c.retrieval_route,
        v.c.calibration_source,
        v.c.size_bucket,
        v.c.bucket_rank,
        v.c.heuristic_size_bucket,
        v.c.heuristic_bucket_rank,
        v.c.analogical_bucket_rank,
        v.c.base_hours,
        v.c.adjusted_hours,
        v.c.predicted_hours,
        v.c.analogical_hours,
        v.c.analogical_confidence,
        v.c.heuristic_scope_hours,
        v.c.heuristic_scope_confidence,
        v.c.confidence,
        v.c.top1_score,
        v.c.top3_avg_score,
        v.c.useful_count,
        v.c.hours_spread,
        v.c.rag_context_sufficient,
        v.c.complexity_bucket_delta,
        v.c.agile_guard_bucket_delta,
        func.round(x.c.total_effort_minutes / 60.0, 4).label("actual_hours"),
    ).select_from(v.join(x, x.c.id == v.c.issue_id))

    conditions = [
        v.c.base_hours.isnot(None),
        v.c.predicted_hours.isnot(None),
        x.c.total_effort_minutes.isnot(None),
        (x.c.total_effort_minutes / 60.0).between(min_hours, max_hours),
    ]

    if run_ids:
        conditions.append(v.c.validation_run_id.in_(run_ids))

    if project_keys:
        conditions.append(func.lower(v.c.project_key).in_(project_keys))

    if predicted_at_from:
        conditions.append(v.c.predicted_at >= predicted_at_from)

    stmt = stmt.where(*conditions).order_by(v.c.predicted_at.asc(), v.c.issue_id.asc())

    with engine.connect() as conn:
        rows = conn.execute(stmt).mappings().all()
        normalized = [dict(row) for row in rows]
        fetch_training_rows.last_raw_count = len(normalized)  # type: ignore[attr-defined]
        if latest_per_issue:
            selected = _dedupe_latest_issue_rows(normalized)
            fetch_training_rows.last_selected_count = len(selected)  # type: ignore[attr-defined]
            return selected
        fetch_training_rows.last_selected_count = len(normalized)  # type: ignore[attr-defined]
        return normalized


def main():
    engine = get_engine()
    rows = fetch_training_rows(engine)
    min_rows = int(os.getenv("TRAIN_MIN_ROWS", "50") or 50)
    if len(rows) < min_rows:
        raise RuntimeError(
            f"Linhas insuficientes para treinar o meta-calibrador: {len(rows)} < {min_rows}"
        )

    training_rows = [build_training_row_from_validation(row) for row in rows]
    model = train_meta_calibrator(training_rows)

    output_path = Path(os.getenv("TRAIN_MODEL_OUTPUT") or get_meta_model_path())
    saved_path = save_meta_model(model, output_path)

    summary = model.get("training_summary") or {}
    predicted_at_from = str(os.getenv("TRAIN_PREDICTED_AT_FROM") or "").strip()
    latest_per_issue = _env_bool("TRAIN_LATEST_PER_ISSUE", True)
    raw_rows = int(getattr(fetch_training_rows, "last_raw_count", len(rows)))
    selected_rows = int(getattr(fetch_training_rows, "last_selected_count", len(rows)))
    print(
        f"[META] raw_rows={raw_rows} selected_rows={selected_rows} rows={summary.get('row_count')} "
        f"approx_mae_hours={summary.get('approx_mae_hours')} "
        f"target_mode={summary.get('target_mode', 'unknown')} "
        f"predicted_at_from={predicted_at_from or '-'} "
        f"latest_per_issue={str(latest_per_issue).lower()} "
        f"output={saved_path}"
    )


if __name__ == "__main__":
    main()
