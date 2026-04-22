from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ai.core.effort_calibration import hours_to_range_index, hours_to_range_payload, range_index_to_payload, clamp_range_index
from config.settings import settings


NUMERIC_FEATURES: tuple[str, ...] = (
    "base_log_hours",
    "analogical_log_hours",
    "heuristic_log_hours",
    "base_range_index",
    "analogical_range_index",
    "heuristic_range_index",
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
    "range_gap",
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


# [META_V4] — Novas funções para o alvo de treino em range_index.
#
# Motivação: a versão anterior (v3) aprendia delta_log_hours e aplicava bounds
# em % do base_hours (ex.: ±25% para analogical_weak). Com base=5h, isso dava
# no máximo ±1.25h de correção. Como o pipeline quantiza em faixas de 3h, o
# ajuste morria no arredondamento: 72% das predições do meta voltavam para o
# mesmo range_index do base. Na v24 com 100 issues, o meta mudou apenas 24
# predições e a média do delta foi +0.3h.
#
# A v4 aprende direto em unidades de range_index, que é a moeda que o
# pipeline final entende. Uma unidade = uma faixa inteira (3h). Os bounds são
# em deltas inteiros, não em %.


def _range_delta_target(actual_hours: float, base_hours: float) -> float:
    """Alvo de treino: delta em range_index entre a faixa real e a faixa do base.

    Retorna float porque o ridge regression precisa de valores contínuos,
    mas na prática fica em -12..+12.
    """
    if actual_hours <= 0 or base_hours <= 0:
        return 0.0
    actual_range = hours_to_range_index(actual_hours)
    base_range = hours_to_range_index(base_hours)
    return float(actual_range - base_range)


def _route_range_delta_bounds(route: str) -> tuple[float, float]:
    """Bounds para o delta de range_index por rota.

    Valores escolhidos em unidades de range_index (não %). Uma faixa = ~3h.
    - analogical_primary: ±2 faixas (analogical é muito confiável)
    - analogical_support: ±3 faixas
    - analogical_soft_signal: ±5 faixas (analogical moderado, precisa corrigir mais)
    - analogical_weak (default): ±8 faixas (rota fraca, corrige livre)
    """
    route_name = _normalize_text(route)
    if route_name == "analogical_primary":
        return -2.0, 2.0
    if route_name == "analogical_support":
        return -3.0, 3.0
    if route_name == "analogical_soft_signal":
        return -5.0, 5.0
    # analogical_weak + default
    return -8.0, 8.0


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


def _route_correction_bounds(route: str) -> tuple[float, float]:
    route_name = _normalize_text(route)
    if route_name == "analogical_support":
        return 0.85, 1.15
    if route_name == "analogical_soft_signal":
        return 0.70, 1.30
    if route_name == "analogical_primary":
        return 0.90, 1.12
    return 0.75, 1.25


def _make_delta_stats(
    delta_logs: List[float],
    factors: List[float],
    range_deltas: Optional[List[float]] = None,
) -> Dict[str, Any]:
    range_deltas = range_deltas or []
    if not delta_logs or not factors:
        return {
            "count": 0,
            "median_delta_log": 0.0,
            "q20_delta_log": 0.0,
            "q80_delta_log": 0.0,
            "median_factor": 1.0,
            "q20_factor": 1.0,
            "q80_factor": 1.0,
            "median_delta_range": 0.0,
            "q20_delta_range": 0.0,
            "q80_delta_range": 0.0,
        }
    stats = {
        "count": len(delta_logs),
        "median_delta_log": round(_median(delta_logs), 6),
        "q20_delta_log": round(_quantile(delta_logs, 0.20), 6),
        "q80_delta_log": round(_quantile(delta_logs, 0.80), 6),
        "median_factor": round(_median(factors), 6),
        "q20_factor": round(_quantile(factors, 0.20), 6),
        "q80_factor": round(_quantile(factors, 0.80), 6),
    }
    if range_deltas:
        stats["median_delta_range"] = round(_median(range_deltas), 6)
        stats["q20_delta_range"] = round(_quantile(range_deltas, 0.20), 6)
        stats["q80_delta_range"] = round(_quantile(range_deltas, 0.80), 6)
    else:
        stats["median_delta_range"] = 0.0
        stats["q20_delta_range"] = 0.0
        stats["q80_delta_range"] = 0.0
    return stats


def _build_segment_priors(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    grouped: Dict[str, Dict[str, List[float]]] = {
        "project_issue_route_bucket": defaultdict(list),
        "project_issue_route_bucket_factor": defaultdict(list),
        "project_issue_route_bucket_range": defaultdict(list),
        "project_issue_route": defaultdict(list),
        "project_issue_route_factor": defaultdict(list),
        "project_issue_route_range": defaultdict(list),
        "project_issue_bucket": defaultdict(list),
        "project_issue_bucket_factor": defaultdict(list),
        "project_issue_bucket_range": defaultdict(list),
        "project_issue": defaultdict(list),
        "project_issue_factor": defaultdict(list),
        "project_issue_range": defaultdict(list),
        "project": defaultdict(list),
        "project_factor": defaultdict(list),
        "project_range": defaultdict(list),
    }

    for row in rows:
        actual_hours = _safe_float(row.get("actual_hours"), 0.0)
        base_hours = _safe_float(row.get("base_hours"), 0.0)
        if actual_hours <= 0 or base_hours <= 0:
            continue
        project_key = _normalize_text(row.get("project_key")) or "unknown"
        issue_type = _normalize_text(row.get("issue_type")) or "unknown"
        heuristic_bucket = _normalize_text(row.get("heuristic_size_bucket")) or _normalize_text(row.get("size_bucket")) or "unknown"
        route = _normalize_text(row.get("retrieval_route")) or "unknown"
        delta_log = _clamp(_log1p_hours(actual_hours) - _log1p_hours(base_hours), -0.55, 0.55)
        correction_factor = _clamp(actual_hours / max(1.0, base_hours), 0.55, 1.85)
        # [META_V4] Delta em range_index — usado pelo novo predict path.
        delta_range = _range_delta_target(actual_hours, base_hours)

        route_bucket_key = f"{project_key}|{issue_type}|{route}|{heuristic_bucket}"
        route_key = f"{project_key}|{issue_type}|{route}"
        bucket_key = f"{project_key}|{issue_type}|{heuristic_bucket}"
        issue_key = f"{project_key}|{issue_type}"

        grouped["project_issue_route_bucket"][route_bucket_key].append(delta_log)
        grouped["project_issue_route_bucket_factor"][route_bucket_key].append(correction_factor)
        grouped["project_issue_route_bucket_range"][route_bucket_key].append(delta_range)
        grouped["project_issue_route"][route_key].append(delta_log)
        grouped["project_issue_route_factor"][route_key].append(correction_factor)
        grouped["project_issue_route_range"][route_key].append(delta_range)
        grouped["project_issue_bucket"][bucket_key].append(delta_log)
        grouped["project_issue_bucket_factor"][bucket_key].append(correction_factor)
        grouped["project_issue_bucket_range"][bucket_key].append(delta_range)
        grouped["project_issue"][issue_key].append(delta_log)
        grouped["project_issue_factor"][issue_key].append(correction_factor)
        grouped["project_issue_range"][issue_key].append(delta_range)
        grouped["project"][project_key].append(delta_log)
        grouped["project_factor"][project_key].append(correction_factor)
        grouped["project_range"][project_key].append(delta_range)

    priors: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for scope in ("project_issue_route_bucket", "project_issue_route", "project_issue_bucket", "project_issue", "project"):
        factor_scope = f"{scope}_factor"
        range_scope = f"{scope}_range"
        priors[scope] = {}
        for key, delta_logs in grouped[scope].items():
            factors = grouped[factor_scope].get(key) or []
            range_deltas = grouped[range_scope].get(key) or []
            priors[scope][key] = _make_delta_stats(delta_logs, factors, range_deltas)
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
        base_hours = _safe_float(row.get("base_hours"), _expm1_hours(row.get("base_log_hours")))
        if actual_hours <= 0 or base_hours <= 0:
            continue

        enriched = dict(row)
        enriched["base_hours"] = base_hours
        # Legacy target (v3): delta log hours, preservado para backward compat.
        enriched["target_delta_log_hours"] = _clamp(
            _log1p_hours(actual_hours) - _log1p_hours(base_hours),
            -0.55,
            0.55,
        )
        # [META_V4] Novo target: delta em range_index.
        # Esse é o alvo que o pipeline realmente consome — o blend final usa
        # range_index, não horas. Aprender na mesma moeda elimina perda por
        # quantização.
        enriched["target_delta_range_index"] = _range_delta_target(actual_hours, base_hours)
        prepared.append(enriched)

    if len(prepared) < 12:
        raise ValueError("Meta calibrator requires at least 12 rows with actual_hours.")

    # Ridge fit para o alvo NOVO (v4) — range_index direto.
    linear_model_range = _fit_ridge_regression(
        rows=prepared,
        feature_names=NUMERIC_FEATURES,
        target_name="target_delta_range_index",
        ridge_penalty=ridge_penalty,
    )
    predicted_range_deltas = [_predict_linear(linear_model_range, row) for row in prepared]
    category_adjustments_range = _build_category_adjustments(
        rows=prepared,
        predicted_logs=predicted_range_deltas,
        target_name="target_delta_range_index",
        min_count=min_category_count,
    )

    # Ridge fit para o alvo LEGADO (log_hours) — mantido para backward compat
    # caso alguém queira comparar ou fazer rollback pontual.
    linear_model_log = _fit_ridge_regression(
        rows=prepared,
        feature_names=NUMERIC_FEATURES,
        target_name="target_delta_log_hours",
        ridge_penalty=ridge_penalty,
    )
    predicted_deltas = [_predict_linear(linear_model_log, row) for row in prepared]
    category_adjustments = _build_category_adjustments(
        rows=prepared,
        predicted_logs=predicted_deltas,
        target_name="target_delta_log_hours",
        min_count=min_category_count,
    )

    segment_priors = _build_segment_priors(prepared)

    # MAE em horas usando o novo caminho (range_index → payload).
    mae_range_sum = 0.0
    for row, predicted_range_delta in zip(prepared, predicted_range_deltas):
        base_range = hours_to_range_index(row.get("base_hours"))
        route = _normalize_text(row.get("retrieval_route"))
        lower_b, upper_b = _route_range_delta_bounds(route)
        bounded_delta = _clamp(predicted_range_delta, lower_b, upper_b)
        predicted_range = clamp_range_index(int(round(base_range + bounded_delta)))
        predicted_hours = float(range_index_to_payload(predicted_range)["display_hours"])
        mae_range_sum += abs(predicted_hours - _safe_float(row.get("actual_hours"), 0.0))

    # MAE em horas usando o caminho legado (log_hours).
    mae_sum = 0.0
    for row, predicted_delta in zip(prepared, predicted_deltas):
        predicted_hours = _expm1_hours(_log1p_hours(row.get("base_hours")) + predicted_delta)
        mae_sum += abs(predicted_hours - _safe_float(row.get("actual_hours"), 0.0))

    model = {
        "model_type": "segmented_linear_residual",
        "model_version": "meta_calibrator_v4_range_index",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_summary": {
            "row_count": len(prepared),
            "approx_mae_hours": round(mae_range_sum / max(1, len(prepared)), 4),
            "approx_mae_hours_legacy_log": round(mae_sum / max(1, len(prepared)), 4),
            "ridge_penalty": ridge_penalty,
            "min_category_count": min_category_count,
            "target_mode": "delta_range_index",
        },
        # Modelo primário (v4) — usado pelo predict.
        "linear_model": linear_model_range,
        "category_adjustments": category_adjustments_range,
        # Modelo legado (v3) — guardado para backward compat.
        "linear_model_log_hours": linear_model_log,
        "category_adjustments_log_hours": category_adjustments,
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
        ("project_issue_route_bucket", f"{project_key}|{issue_type}|{route}|{heuristic_bucket}"),
        ("project_issue_route", f"{project_key}|{issue_type}|{route}"),
        ("project_issue_bucket", f"{project_key}|{issue_type}|{heuristic_bucket}"),
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
        "base_range_index": hours_to_range_index(base_hours),
        "analogical_range_index": hours_to_range_index(analogical_hours),
        "heuristic_range_index": hours_to_range_index(heuristic_hours),
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
        "range_gap": hours_to_range_index(analogical_hours) - hours_to_range_index(heuristic_hours),
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
    base_hours = _safe_float(
        row.get("base_hours"),
        _safe_float(row.get("predicted_hours_raw"), _safe_float(row.get("predicted_hours"), 8.0)),
    )
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
        "base_hours": base_hours,
        "base_log_hours": _log1p_hours(base_hours),
        "analogical_log_hours": _log1p_hours(analogical_hours),
        "heuristic_log_hours": _log1p_hours(heuristic_hours),
        "base_range_index": float(hours_to_range_index(base_hours)),
        "analogical_range_index": float(hours_to_range_index(analogical_hours)),
        "heuristic_range_index": float(hours_to_range_index(heuristic_hours)),
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
        "range_gap": float(hours_to_range_index(analogical_hours) - hours_to_range_index(heuristic_hours)),
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

    # [META_V4] Dispatch por target_mode. Modelos treinados pela v4 gravam
    # target_mode="delta_range_index" e recebem o caminho novo. Modelos
    # anteriores caem no caminho legado (log_hours).
    target_mode = str((model.get("training_summary") or {}).get("target_mode") or "delta_log_hours_over_base")
    if target_mode == "delta_range_index":
        return _predict_v4_range_index(model, feature_payload)
    return _predict_v3_log_hours(model, feature_payload)


def _predict_v4_range_index(model: Dict[str, Any], feature_payload: Dict[str, Any]) -> Dict[str, Any]:
    """[META_V4] Predict path — opera direto em range_index.

    O pipeline consome range_index no blend final. Aprender e predizer na
    mesma moeda elimina a perda por quantização que existia na v3 (onde o
    meta produzia delta de 0.3h em média e morria no arredondamento para
    faixas de 3h).
    """
    linear_model = dict(model.get("linear_model") or {})
    predicted_range_delta = _predict_linear(linear_model, feature_payload)

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
        predicted_range_delta += category_delta

    prior_stats, prior_scope, prior_key = _best_segment_prior(model, feature_payload)
    route = _category_value(feature_payload, "retrieval_route")
    prior_weight = 0.0
    if prior_stats:
        count = max(1, _safe_int(prior_stats.get("count"), 1))
        prior_weight = min(
            float(getattr(settings, "META_CALIBRATOR_MAX_PRIOR_WEIGHT", 0.30) or 0.30),
            0.05 + (0.025 * min(count, 6)),
        )
        if route == "analogical_weak":
            prior_weight += 0.05
        elif route == "analogical_soft_signal":
            prior_weight += 0.03
        elif route == "analogical_support":
            prior_weight += 0.01
        prior_weight = _clamp(prior_weight, 0.05, float(getattr(settings, "META_CALIBRATOR_MAX_PRIOR_WEIGHT", 0.30) or 0.30))
        # Prior também em delta_range quando disponível
        prior_median_range = _safe_float(prior_stats.get("median_delta_range"), 0.0)
        predicted_range_delta = (
            (1.0 - prior_weight) * predicted_range_delta
            + prior_weight * prior_median_range
        )

    # Clamp em unidades de range_index (não em %). Cada unidade ~3h.
    lower_bound, upper_bound = _route_range_delta_bounds(route)
    bounded_delta = _clamp(predicted_range_delta, lower_bound, upper_bound)

    base_range_index = clamp_range_index(
        _safe_int(feature_payload.get("base_range_index"), 3)
        or hours_to_range_index(_expm1_hours(feature_payload.get("base_log_hours")))
    )
    predicted_range_index = clamp_range_index(int(round(base_range_index + bounded_delta)))
    range_payload = range_index_to_payload(predicted_range_index)
    predicted_hours = float(range_payload["display_hours"])

    # min/max_hours a partir dos quantis q20/q80 do prior, se houver.
    if prior_stats:
        q20_delta = _safe_float(prior_stats.get("q20_delta_range"), bounded_delta)
        q80_delta = _safe_float(prior_stats.get("q80_delta_range"), bounded_delta)
        min_range = clamp_range_index(int(round(base_range_index + _clamp(q20_delta, lower_bound, upper_bound))))
        max_range = clamp_range_index(int(round(base_range_index + _clamp(q80_delta, lower_bound, upper_bound))))
        min_hours = float(range_index_to_payload(min_range)["display_hours"])
        max_hours = float(range_index_to_payload(max_range)["display_hours"])
        if min_hours > predicted_hours:
            min_hours = predicted_hours
        if max_hours < predicted_hours:
            max_hours = predicted_hours
        support_count = _safe_int(prior_stats.get("count"), 0)
    else:
        min_hours = predicted_hours
        max_hours = predicted_hours
        support_count = 0

    top1_score = _safe_float(feature_payload.get("top1_score"), 0.0)
    confidence = 0.38 + min(0.18, 0.03 * min(support_count, 6))
    confidence += min(0.12, top1_score * 0.15)
    if route == "analogical_support":
        confidence += 0.08
    elif route == "analogical_soft_signal":
        confidence += 0.03
    confidence = _clamp(confidence, 0.28, 0.82)

    meta_source = "meta_range_linear"
    if prior_scope:
        meta_source = f"meta_range_linear+{prior_scope}"

    return {
        "available": True,
        "estimated_hours": predicted_hours,
        "min_hours": min_hours,
        "max_hours": max_hours,
        "range_index": range_payload["range_index"],
        "range_label": range_payload["range_label"],
        "range_min_hours": range_payload["range_min_hours"],
        "range_max_hours": range_payload["range_max_hours"],
        "display_hours": range_payload["display_hours"],
        "confidence": round(confidence, 4),
        "meta_source": meta_source,
        "prior_source": prior_scope,
        "prior_key": prior_key,
        "prior_count": support_count,
        "blend_weight": round(prior_weight, 4),
        "model_version": str(model.get("model_version") or "meta_calibrator_v4_range_index"),
        "category_adjustment": round(category_delta, 6),
        # Exposto para depuração: o delta em range_index que saiu do meta.
        "predicted_range_delta": round(bounded_delta, 4),
        "target_mode": "delta_range_index",
    }


def _predict_v3_log_hours(model: Dict[str, Any], feature_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Predict path legado — aprendizado em delta_log_hours.

    Preservado para backward compatibility. Em modelos treinados pela v4,
    o dispatch em predict_meta_calibration não passa por aqui.
    """
    linear_model = dict(model.get("linear_model") or {})
    predicted_delta_log = _predict_linear(linear_model, feature_payload)

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
        predicted_delta_log += category_delta

    prior_stats, prior_scope, prior_key = _best_segment_prior(model, feature_payload)
    route = _category_value(feature_payload, "retrieval_route")
    prior_weight = 0.0
    if prior_stats:
        count = max(1, _safe_int(prior_stats.get("count"), 1))
        prior_weight = min(
            float(getattr(settings, "META_CALIBRATOR_MAX_PRIOR_WEIGHT", 0.30) or 0.30),
            0.05 + (0.025 * min(count, 6)),
        )
        if route == "analogical_weak":
            prior_weight += 0.05
        elif route == "analogical_soft_signal":
            prior_weight += 0.03
        elif route == "analogical_support":
            prior_weight += 0.01
        prior_weight = _clamp(prior_weight, 0.05, float(getattr(settings, "META_CALIBRATOR_MAX_PRIOR_WEIGHT", 0.30) or 0.30))
        predicted_delta_log = (
            (1.0 - prior_weight) * predicted_delta_log
            + prior_weight * _safe_float(prior_stats.get("median_delta_log"), 0.0)
        )

    base_hours = max(1.0, _expm1_hours(feature_payload.get("base_log_hours")))
    lower_factor, upper_factor = _route_correction_bounds(route)
    bounded_min = max(1.0, round(base_hours * lower_factor, 1))
    bounded_max = max(bounded_min, round(base_hours * upper_factor, 1))
    raw_predicted_hours = _expm1_hours(_safe_float(feature_payload.get("base_log_hours"), 0.0) + predicted_delta_log)
    predicted_hours = round(_clamp(raw_predicted_hours, bounded_min, bounded_max), 1)
    range_payload = hours_to_range_payload(predicted_hours)

    if prior_stats:
        prior_min = _expm1_hours(_safe_float(feature_payload.get("base_log_hours"), 0.0) + _safe_float(prior_stats.get("q20_delta_log"), 0.0))
        prior_max = _expm1_hours(_safe_float(feature_payload.get("base_log_hours"), 0.0) + _safe_float(prior_stats.get("q80_delta_log"), 0.0))
        min_hours = round(
            max(
                1.0,
                min(
                    predicted_hours,
                    _clamp(prior_min, bounded_min, bounded_max),
                ),
            ),
            1,
        )
        max_hours = round(
            max(
                predicted_hours,
                _clamp(prior_max, bounded_min, bounded_max),
            ),
            1,
        )
        support_count = _safe_int(prior_stats.get("count"), 0)
    else:
        min_hours = round(max(1.0, min(predicted_hours, base_hours * max(0.82, lower_factor))), 1)
        max_hours = round(max(predicted_hours, min(bounded_max, base_hours * min(1.18, upper_factor))), 1)
        support_count = 0

    top1_score = _safe_float(feature_payload.get("top1_score"), 0.0)
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
        "range_index": range_payload["range_index"],
        "range_label": range_payload["range_label"],
        "range_min_hours": range_payload["range_min_hours"],
        "range_max_hours": range_payload["range_max_hours"],
        "display_hours": range_payload["display_hours"],
        "confidence": round(confidence, 4),
        "meta_source": meta_source,
        "prior_source": prior_scope,
        "prior_key": prior_key,
        "prior_count": support_count,
        "blend_weight": round(prior_weight, 4),
        "model_version": str(model.get("model_version") or "meta_calibrator_v3_range_aware"),
        "category_adjustment": round(category_delta, 6),
        "target_mode": "delta_log_hours_over_base",
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
