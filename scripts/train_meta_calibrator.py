import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from sqlalchemy import create_engine, text


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

    filters = [
        "v.base_hours IS NOT NULL",
        "v.predicted_hours IS NOT NULL",
        "x.total_effort_minutes IS NOT NULL",
        "(x.total_effort_minutes / 60.0) BETWEEN :min_hours AND :max_hours",
    ]
    params: dict[str, Any] = {"min_hours": min_hours, "max_hours": max_hours}

    if run_ids:
        placeholders = []
        for idx, run_id in enumerate(run_ids):
            key = f"run_id_{idx}"
            placeholders.append(f":{key}")
            params[key] = run_id
        filters.append(f"v.validation_run_id IN ({', '.join(placeholders)})")

    if project_keys:
        placeholders = []
        for idx, project_key in enumerate(project_keys):
            key = f"project_key_{idx}"
            placeholders.append(f":{key}")
            params[key] = project_key
        filters.append(f"LOWER(v.project_key) IN ({', '.join(placeholders)})")

    if predicted_at_from:
        filters.append("v.predicted_at >= :predicted_at_from")
        params["predicted_at_from"] = predicted_at_from

    sql = text(
        f"""
        SELECT
          v.issue_id,
          v.validation_run_id,
          v.predicted_at,
          v.project_key,
          LOWER(COALESCE(NULLIF(x.type, ''), 'unknown')) AS issue_type,
          v.selected_model,
          v.finalization_mode,
          v.retrieval_route,
          v.calibration_source,
          v.size_bucket,
          v.bucket_rank,
          v.heuristic_size_bucket,
          v.heuristic_bucket_rank,
          v.analogical_bucket_rank,
          v.base_hours,
          v.adjusted_hours,
          v.predicted_hours,
          v.analogical_hours,
          v.analogical_confidence,
          v.heuristic_scope_hours,
          v.heuristic_scope_confidence,
          v.confidence,
          v.top1_score,
          v.top3_avg_score,
          v.useful_count,
          v.hours_spread,
          v.rag_context_sufficient,
          v.complexity_bucket_delta,
          v.agile_guard_bucket_delta,
          ROUND((x.total_effort_minutes / 60.0), 4) AS actual_hours
        FROM issue_estimation_validation v
        JOIN issue x
          ON x.id = v.issue_id
        WHERE {' AND '.join(filters)}
        ORDER BY v.predicted_at ASC, v.issue_id ASC
        """
    )

    with engine.connect() as conn:
        rows = conn.execute(sql, params).mappings().all()
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
