from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from config.settings import settings


NUMERIC_FEATURES: tuple[str, ...] = (
    "base_log_hours",
    "analogical_log_hours",
    "heuristic_log_hours",
    "bucket_rank",
    "heuristic_bucket_rank",
    "analogical_bucket_rank",
    "top1_score",
    "top3_avg_score",
    "useful_count",
    "hours_spread",
    "analogical_confidence",
    "heuristic_confidence",
    "base_confidence",
    "bucket_gap",
    "rag_context_sufficient",
    "complexity_bucket_delta",
    "agile_guard_bucket_delta",
)

CATEGORY_FIELDS: tuple[str, ...] = (
    "retrieval_route",
    "issue_type",
    "heuristic_size_bucket",
    "calibration_source",
    "rule_selected_model",
)

_MODEL_CACHE: Dict[str, Any] = {"path": None, "mtime": None, "data": None}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(round(float(value)))
    except Exception:
        return int(default)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _normalize_text(value: Any) -> str:
    return str(value or "").strip().lower()


def _median(values: List[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _quantile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    q = _clamp(q, 0.0, 1.0)
    position = q * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] + ((ordered[upper] - ordered[lower]) * weight)


def _log1p_hours(value: Any) -> float:
    return math.log1p(max(0.0, _safe_float(value, 0.0)))


def _expm1_hours(value: Any) -> float:
    return max(1.0, math.expm1(_safe_float(value, 0.0)))


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def get_meta_model_path() -> Path:
    configured = str(getattr(settings, "META_CALIBRATOR_MODEL_PATH", "") or "").strip()
    if configured:
        path = Path(configured)
        if not path.is_absolute():
            return _repo_root() / path
        return path
    return _repo_root() / "artifacts" / "meta_calibrator.json"


def _solve_linear_system(matrix: List[List[float]], target: List[float]) -> List[float]:
    size = len(target)
    augmented = [list(matrix[row]) + [float(target[row])] for row in range(size)]

    for pivot in range(size):
        best_row = max(range(pivot, size), key=lambda row: abs(augmented[row][pivot]))
        if abs(augmented[best_row][pivot]) < 1e-10:
            continue
        if best_row != pivot:
            augmented[pivot], augmented[best_row] = augmented[best_row], augmented[pivot]

        pivot_value = augmented[pivot][pivot]
        for col in range(pivot, size + 1):
            augmented[pivot][col] /= pivot_value

        for row in range(size):
            if row == pivot:
                continue
            factor = augmented[row][pivot]
            if abs(factor) < 1e-12:
                continue
            for col in range(pivot, size + 1):
                augmented[row][col] -= factor * augmented[pivot][col]

    return [augmented[row][-1] for row in range(size)]


def _fit_ridge_regression(
    rows: List[Dict[str, Any]],
    feature_names: Iterable[str],
    target_name: str,
    ridge_penalty: float = 0.15,
) -> Dict[str, Any]:
    features = list(feature_names)
    means: Dict[str, float] = {}
    scales: Dict[str, float] = {}

    for feature in features:
        values = [_safe_float(row.get(feature), 0.0) for row in rows]
        mean = sum(values) / max(1, len(values))
        variance = sum((value - mean) ** 2 for value in values) / max(1, len(values))
        means[feature] = mean
        scales[feature] = math.sqrt(variance) if variance > 1e-12 else 1.0

    design: List[List[float]] = []
    target: List[float] = []
    for row in rows:
        vector = [1.0]
        for feature in features:
            centered = _safe_float(row.get(feature), 0.0) - means[feature]
            vector.append(centered / scales[feature])
        design.append(vector)
        target.append(_safe_float(row.get(target_name), 0.0))

    width = len(features) + 1
    xtx = [[0.0 for _ in range(width)] for _ in range(width)]
    xty = [0.0 for _ in range(width)]

    for vector, expected in zip(design, target):
        for i in range(width):
            xty[i] += vector[i] * expected
            for j in range(width):
                xtx[i][j] += vector[i] * vector[j]

    for index in range(1, width):
        xtx[index][index] += float(ridge_penalty)

    coeffs = _solve_linear_system(xtx, xty)
    return {
        "bias": coeffs[0] if coeffs else 0.0,
        "weights": {feature: coeffs[idx + 1] for idx, feature in enumerate(features)},
        "means": means,
        "scales": scales,
        "features": features,
    }


def _predict_linear(model: Dict[str, Any], numeric_features: Dict[str, float]) -> float:
    prediction = _safe_float(model.get("bias"), 0.0)
    weights = dict(model.get("weights") or {})
    means = dict(model.get("means") or {})
    scales = dict(model.get("scales") or {})

    for feature in model.get("features") or []:
        scale = _safe_float(scales.get(feature), 1.0)
        if abs(scale) < 1e-12:
            scale = 1.0
        centered = _safe_float(numeric_features.get(feature), 0.0) - _safe_float(means.get(feature), 0.0)
        prediction += _safe_float(weights.get(feature), 0.0) * (centered / scale)

    return prediction


def _category_value(record: Dict[str, Any], field: str) -> str:
    return _normalize_text(record.get(field)) or "unknown"


def _build_category_adjustments(
    rows: List[Dict[str, Any]],
    predicted_logs: List[float],
    target_name: str,
    min_count: int = 5,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    out: Dict[str, Dict[str, Dict[str, float]]] = {}

    for field in CATEGORY_FIELDS:
        grouped: Dict[str, List[float]] = defaultdict(list)
        for row, predicted in zip(rows, predicted_logs):
            actual = _safe_float(row.get(target_name), 0.0)
            grouped[_category_value(row, field)].append(actual - predicted)

        field_adjustments: Dict[str, Dict[str, float]] = {}
        for category, residuals in grouped.items():
            if len(residuals) < min_count:
                continue
            field_adjustments[category] = {
                "count": len(residuals),
                "value": round(_median(residuals), 6),
            }
        out[field] = field_adjustments

    return out


def _make_stats(values: List[float]) -> Dict[str, Any]:
    ordered = sorted(float(v) for v in values if _safe_float(v, 0.0) > 0)
    if not ordered:
        return {"count": 0, "median_hours": 0.0, "q20_hours": 0.0, "q80_hours": 0.0}
    return {
        "count": len(ordered),
        "median_hours": round(_median(ordered), 4),
        "q20_hours": round(_quantile(ordered, 0.20), 4),
        "q80_hours": round(_quantile(ordered, 0.80), 4),
    }


def _build_segment_priors(rows: List[Dict[str, Any]], target_hours_field: str) -> Dict[str, Dict[str, Dict[str, Any]]]:
    grouped: Dict[str, Dict[str, List[float]]] = {
        "project_issue_bucket": defaultdict(list),
        "project_issue_route": defaultdict(list),
        "project_issue": defaultdict(list),
        "project": defaultdict(list),
    }

    for row in rows:
        actual_hours = _safe_float(row.get(target_hours_field), 0.0)
        if actual_hours <= 0:
            continue
        project_key = _normalize_text(row.get("project_key")) or "unknown"
        issue_type = _normalize_text(row.get("issue_type")) or "unknown"
        heuristic_bucket = _normalize_text(row.get("heuristic_size_bucket")) or _normalize_text(row.get("size_bucket")) or "unknown"
        route = _normalize_text(row.get("retrieval_route")) or "unknown"

        grouped["project_issue_bucket"][f"{project_key}|{issue_type}|{heuristic_bucket}"].append(actual_hours)
        grouped["project_issue_route"][f"{project_key}|{issue_type}|{route}"].append(actual_hours)
        grouped["project_issue"][f"{project_key}|{issue_type}"].append(actual_hours)
        grouped["project"][project_key].append(actual_hours)

    priors: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for scope, values_by_key in grouped.items():
        priors[scope] = {key: _make_stats(values) for key, values in values_by_key.items()}
    return priors


def train_meta_calibrator(
    training_rows: List[Dict[str, Any]],
    ridge_penalty: float = 0.15,
    min_category_count: int = 5,
) -> Dict[str, Any]:
    if not training_rows:
        raise ValueError("No training rows provided for meta calibrator.")

    prepared: List[Dict[str, Any]] = []
    for row in training_rows:
        actual_hours = _safe_float(row.get("actual_hours"), 0.0)
        if actual_hours <= 0:
            continue

        enriched = dict(row)
        enriched["target_log_hours"] = _log1p_hours(actual_hours)
        prepared.append(enriched)

    if len(prepared) < 12:
        raise ValueError("Meta calibrator requires at least 12 rows with actual_hours.")

    linear_model = _fit_ridge_regression(
        rows=prepared,
        feature_names=NUMERIC_FEATURES,
        target_name="target_log_hours",
        ridge_penalty=ridge_penalty,
    )
    predicted_logs = [_predict_linear(linear_model, row) for row in prepared]
    category_adjustments = _build_category_adjustments(
        rows=prepared,
        predicted_logs=predicted_logs,
        target_name="target_log_hours",
        min_count=min_category_count,
    )
    segment_priors = _build_segment_priors(prepared, target_hours_field="actual_hours")

    mae_sum = 0.0
    for row, predicted_log in zip(prepared, predicted_logs):
        predicted_hours = _expm1_hours(predicted_log)
        mae_sum += abs(predicted_hours - _safe_float(row.get("actual_hours"), 0.0))

    model = {
        "model_type": "segmented_linear_residual",
        "model_version": "meta_calibrator_v1",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_summary": {
            "row_count": len(prepared),
            "approx_mae_hours": round(mae_sum / max(1, len(prepared)), 4),
            "ridge_penalty": ridge_penalty,
            "min_category_count": min_category_count,
        },
        "linear_model": linear_model,
        "category_adjustments": category_adjustments,
        "segment_priors": segment_priors,
    }
    return model


def save_meta_model(model: Dict[str, Any], path: Path | None = None) -> Path:
    output_path = Path(path or get_meta_model_path())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(model, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def load_meta_model(force_reload: bool = False) -> Optional[Dict[str, Any]]:
    path = get_meta_model_path()
    if not path.exists():
        return None

    mtime = path.stat().st_mtime
    if (
        not force_reload
        and _MODEL_CACHE.get("path") == str(path)
        and _MODEL_CACHE.get("mtime") == mtime
        and _MODEL_CACHE.get("data") is not None
    ):
        return dict(_MODEL_CACHE["data"])

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

    _MODEL_CACHE["path"] = str(path)
    _MODEL_CACHE["mtime"] = mtime
    _MODEL_CACHE["data"] = data
    return dict(data)


def _best_segment_prior(model: Dict[str, Any], feature_payload: Dict[str, Any]) -> tuple[Optional[Dict[str, Any]], Optional[str], Optional[str]]:
    priors = dict(model.get("segment_priors") or {})
    project_key = _normalize_text(feature_payload.get("project_key")) or "unknown"
    issue_type = _normalize_text(feature_payload.get("issue_type")) or "unknown"
    heuristic_bucket = _normalize_text(feature_payload.get("heuristic_size_bucket")) or _normalize_text(feature_payload.get("size_bucket")) or "unknown"
    route = _normalize_text(feature_payload.get("retrieval_route")) or "unknown"

    candidates = [
        ("project_issue_bucket", f"{project_key}|{issue_type}|{heuristic_bucket}"),
        ("project_issue_route", f"{project_key}|{issue_type}|{route}"),
        ("project_issue", f"{project_key}|{issue_type}"),
        ("project", project_key),
    ]

    min_count = max(1, int(getattr(settings, "META_CALIBRATOR_MIN_SEGMENT_COUNT", 3) or 3))
    for scope, key in candidates:
        stats = ((priors.get(scope) or {}).get(key) or {})
        if _safe_int(stats.get("count"), 0) >= min_count:
            return dict(stats), scope, key

    for scope, key in candidates:
        stats = ((priors.get(scope) or {}).get(key) or {})
        if _safe_int(stats.get("count"), 0) > 0:
            return dict(stats), scope, key

    return None, None, None


def build_meta_feature_payload(
    issue_context: Dict[str, Any],
    analogical: Dict[str, Any],
    heuristic_consensus: Dict[str, Any],
    heuristic_calibration: Dict[str, Any],
    calibration_source: str,
    retrieval_route: str,
    retrieval_stats: Dict[str, Any],
    base_hours: float,
    base_confidence: float,
    rule_selected_model: str,
    complexity_review: Optional[Dict[str, Any]] = None,
    agile_guard_review: Optional[Dict[str, Any]] = None,
    rag_stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    analogical_hours = _safe_float(analogical.get("estimated_hours"), base_hours)
    heuristic_hours = _safe_float(heuristic_calibration.get("estimated_hours"), base_hours)
    bucket_rank = _safe_int(heuristic_consensus.get("bucket_rank"), 3)
    analogical_bucket_rank = _safe_int(analogical.get("bucket_rank"), bucket_rank)
    heuristic_bucket_rank = _safe_int(heuristic_calibration.get("bucket_rank"), bucket_rank)

    return {
        "project_key": _normalize_text(issue_context.get("repository")),
        "issue_type": _normalize_text(issue_context.get("issue_type")) or "unknown",
        "retrieval_route": _normalize_text(retrieval_route) or "unknown",
        "heuristic_size_bucket": _normalize_text(heuristic_consensus.get("size_bucket") or heuristic_calibration.get("size_bucket") or "m"),
        "size_bucket": _normalize_text(heuristic_consensus.get("size_bucket") or "m"),
        "calibration_source": _normalize_text(calibration_source) or "unknown",
        "rule_selected_model": _normalize_text(rule_selected_model) or "unknown",
        "base_log_hours": _log1p_hours(base_hours),
        "analogical_log_hours": _log1p_hours(analogical_hours),
        "heuristic_log_hours": _log1p_hours(heuristic_hours),
        "bucket_rank": bucket_rank,
        "heuristic_bucket_rank": heuristic_bucket_rank,
        "analogical_bucket_rank": analogical_bucket_rank,
        "top1_score": _safe_float(retrieval_stats.get("top1_score"), 0.0),
        "top3_avg_score": _safe_float(retrieval_stats.get("top3_avg_score"), 0.0),
        "useful_count": _safe_float(retrieval_stats.get("useful_count"), 0.0),
        "hours_spread": _safe_float(retrieval_stats.get("hours_spread"), 0.0),
        "analogical_confidence": _safe_float(analogical.get("confidence"), 0.0),
        "heuristic_confidence": _safe_float(heuristic_consensus.get("confidence"), 0.0),
        "base_confidence": _safe_float(base_confidence, 0.0),
        "bucket_gap": analogical_bucket_rank - heuristic_bucket_rank,
        "rag_context_sufficient": 1.0 if bool((rag_stats or {}).get("qualified_hits", 0) >= (rag_stats or {}).get("min_hits", 2)) else 0.0,
        "complexity_bucket_delta": _safe_float((complexity_review or {}).get("bucket_delta"), 0.0),
        "agile_guard_bucket_delta": _safe_float((agile_guard_review or {}).get("bucket_delta"), 0.0),
    }


def build_training_row_from_validation(row: Dict[str, Any]) -> Dict[str, Any]:
    issue_type = _normalize_text(row.get("issue_type")) or "unknown"
    retrieval_route = _normalize_text(row.get("retrieval_route")) or "unknown"
    heuristic_bucket = _normalize_text(row.get("heuristic_size_bucket")) or _normalize_text(row.get("size_bucket")) or "m"
    selected_model = _normalize_text(row.get("selected_model")) or _normalize_text(row.get("finalization_mode")) or "unknown"
    calibration_source = _normalize_text(row.get("calibration_source")) or "unknown"
    base_hours = _safe_float(row.get("base_hours"), _safe_float(row.get("predicted_hours"), 8.0))
    analogical_hours = _safe_float(row.get("analogical_hours"), base_hours)
    heuristic_hours = _safe_float(row.get("heuristic_scope_hours"), base_hours)
    if heuristic_hours <= 0:
        heuristic_hours = base_hours

    return {
        "project_key": _normalize_text(row.get("project_key")) or "unknown",
        "issue_type": issue_type,
        "retrieval_route": retrieval_route,
        "heuristic_size_bucket": heuristic_bucket,
        "size_bucket": _normalize_text(row.get("size_bucket")) or heuristic_bucket,
        "calibration_source": calibration_source,
        "rule_selected_model": selected_model,
        "base_log_hours": _log1p_hours(base_hours),
        "analogical_log_hours": _log1p_hours(analogical_hours),
        "heuristic_log_hours": _log1p_hours(heuristic_hours),
        "bucket_rank": _safe_float(row.get("bucket_rank"), 3.0),
        "heuristic_bucket_rank": _safe_float(row.get("heuristic_bucket_rank"), _safe_float(row.get("bucket_rank"), 3.0)),
        "analogical_bucket_rank": _safe_float(row.get("analogical_bucket_rank"), _safe_float(row.get("bucket_rank"), 3.0)),
        "top1_score": _safe_float(row.get("top1_score"), 0.0),
        "top3_avg_score": _safe_float(row.get("top3_avg_score"), 0.0),
        "useful_count": _safe_float(row.get("useful_count"), 0.0),
        "hours_spread": _safe_float(row.get("hours_spread"), 0.0),
        "analogical_confidence": _safe_float(row.get("analogical_confidence"), 0.0),
        "heuristic_confidence": _safe_float(row.get("heuristic_scope_confidence"), _safe_float(row.get("confidence"), 0.0)),
        "base_confidence": _safe_float(row.get("confidence"), 0.0),
        "bucket_gap": _safe_float(row.get("analogical_bucket_rank"), _safe_float(row.get("bucket_rank"), 3.0))
        - _safe_float(row.get("heuristic_bucket_rank"), _safe_float(row.get("bucket_rank"), 3.0)),
        "rag_context_sufficient": 1.0 if bool(_safe_int(row.get("rag_context_sufficient"), 0)) else 0.0,
        "complexity_bucket_delta": _safe_float(row.get("complexity_bucket_delta"), 0.0),
        "agile_guard_bucket_delta": _safe_float(row.get("agile_guard_bucket_delta"), 0.0),
        "actual_hours": _safe_float(row.get("actual_hours"), 0.0),
    }


def predict_meta_calibration(feature_payload: Dict[str, Any]) -> Dict[str, Any]:
    if not bool(getattr(settings, "META_CALIBRATOR_ENABLED", True)):
        return {"available": False, "reason": "disabled"}

    model = load_meta_model()
    if not model:
        return {"available": False, "reason": "model_missing"}

    linear_model = dict(model.get("linear_model") or {})
    if not linear_model:
        return {"available": False, "reason": "invalid_model"}

    predicted_log = _predict_linear(linear_model, feature_payload)

    category_adjustments = dict(model.get("category_adjustments") or {})
    weighted_adjustments: List[tuple[float, float]] = []
    for field in CATEGORY_FIELDS:
        bucket = dict(category_adjustments.get(field) or {})
        entry = bucket.get(_category_value(feature_payload, field))
        if not entry:
            continue
        count = max(1, _safe_int(entry.get("count"), 1))
        weight = min(12.0, float(count))
        weighted_adjustments.append((_safe_float(entry.get("value"), 0.0), weight))

    category_delta = 0.0
    if weighted_adjustments:
        total_weight = sum(weight for _, weight in weighted_adjustments)
        category_delta = sum(value * weight for value, weight in weighted_adjustments) / max(1.0, total_weight)
        predicted_log += category_delta

    prior_stats, prior_scope, prior_key = _best_segment_prior(model, feature_payload)
    prior_weight = 0.0
    if prior_stats:
        count = max(1, _safe_int(prior_stats.get("count"), 1))
        route = _category_value(feature_payload, "retrieval_route")
        prior_weight = min(
            float(getattr(settings, "META_CALIBRATOR_MAX_PRIOR_WEIGHT", 0.55) or 0.55),
            0.15 + (0.04 * min(count, 8)),
        )
        if route in {"analogical_weak", "analogical_soft_signal"}:
            prior_weight += 0.08
        elif route == "analogical_support":
            prior_weight += 0.03
        prior_weight = _clamp(prior_weight, 0.12, 0.60)
        predicted_log = (
            (1.0 - prior_weight) * predicted_log
            + prior_weight * math.log1p(_safe_float(prior_stats.get("median_hours"), 0.0))
        )

    predicted_hours = round(_expm1_hours(predicted_log), 1)
    if prior_stats:
        prior_min = _safe_float(prior_stats.get("q20_hours"), max(1.0, predicted_hours * 0.8))
        prior_max = _safe_float(prior_stats.get("q80_hours"), max(predicted_hours, predicted_hours * 1.2))
        min_hours = round(max(1.0, min(predicted_hours, ((1.0 - prior_weight) * predicted_hours * 0.82) + (prior_weight * prior_min))), 1)
        max_hours = round(max(predicted_hours, ((1.0 - prior_weight) * predicted_hours * 1.18) + (prior_weight * prior_max)), 1)
        support_count = _safe_int(prior_stats.get("count"), 0)
    else:
        min_hours = round(max(1.0, predicted_hours * 0.8), 1)
        max_hours = round(max(predicted_hours, predicted_hours * 1.2), 1)
        support_count = 0

    top1_score = _safe_float(feature_payload.get("top1_score"), 0.0)
    route = _category_value(feature_payload, "retrieval_route")
    confidence = 0.38 + min(0.18, 0.03 * min(support_count, 6))
    confidence += min(0.12, top1_score * 0.15)
    if route == "analogical_support":
        confidence += 0.08
    elif route == "analogical_soft_signal":
        confidence += 0.03
    confidence = _clamp(confidence, 0.28, 0.82)

    meta_source = "meta_linear"
    if prior_scope:
        meta_source = f"meta_linear+{prior_scope}"

    return {
        "available": True,
        "estimated_hours": predicted_hours,
        "min_hours": min_hours,
        "max_hours": max_hours,
        "confidence": round(confidence, 4),
        "meta_source": meta_source,
        "prior_source": prior_scope,
        "prior_key": prior_key,
        "prior_count": support_count,
        "blend_weight": round(prior_weight, 4),
        "model_version": str(model.get("model_version") or "meta_calibrator_v1"),
        "category_adjustment": round(category_delta, 6),
    }


__all__ = [
    "NUMERIC_FEATURES",
    "CATEGORY_FIELDS",
    "build_meta_feature_payload",
    "build_training_row_from_validation",
    "get_meta_model_path",
    "load_meta_model",
    "predict_meta_calibration",
    "save_meta_model",
    "train_meta_calibrator",
]
