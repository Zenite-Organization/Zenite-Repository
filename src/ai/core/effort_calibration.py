from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Sequence


BUCKETS: tuple[str, ...] = ("XS", "S", "M", "L", "XL", "XXL")
# Last-resort prior used only when no historical calibration signal is available.
FALLBACK_BUCKET_HOURS: dict[int, float] = {
    1: 2.5,
    2: 5.0,
    3: 8.0,
    4: 13.0,
    5: 21.0,
    6: 34.0,
}
BUCKET_TARGET_QUANTILES: dict[int, float] = {
    1: 0.12,
    2: 0.28,
    3: 0.45,
    4: 0.62,
    5: 0.80,
    6: 0.92,
}
DEFAULT_BUCKET_BOUNDARIES: tuple[float, ...] = (3.0, 6.0, 10.0, 16.0, 28.0)
MODE_BUCKET_WEIGHTS: dict[str, float] = {
    "scope": 1.2,
    "complexity": 1.15,
    "uncertainty": 0.55,
    "agile_fit": 0.7,
}
HOUR_RANGES: tuple[tuple[int, int], ...] = (
    (1, 3),
    (3, 6),
    (6, 9),
    (9, 12),
    (12, 15),
    (15, 18),
    (18, 21),
    (21, 24),
    (24, 27),
    (27, 30),
    (30, 33),
    (33, 36),
    (36, 40),
)
DEFAULT_BUCKET_RANGE_INDEX: dict[int, int] = {
    1: 1,
    2: 2,
    3: 4,
    4: 7,
    5: 10,
    6: 12,
}


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _round_half_up(value: float) -> int:
    return int(math.floor(float(value) + 0.5))


def clamp_supported_hours(hours: Any) -> float:
    value = safe_float(hours, 1.0)
    return max(1.0, min(40.0, value))


def clamp_range_index(index: Any) -> int:
    try:
        numeric = int(round(float(index)))
    except Exception:
        numeric = 1
    return max(1, min(len(HOUR_RANGES), numeric))


def range_index_to_bounds(index: Any) -> tuple[int, int]:
    return HOUR_RANGES[clamp_range_index(index) - 1]


def range_index_to_label(index: Any) -> str:
    low, high = range_index_to_bounds(index)
    return f"{low}-{high}h"


def range_index_to_midpoint(index: Any) -> int:
    low, high = range_index_to_bounds(index)
    return _round_half_up((float(low) + float(high)) / 2.0)


def range_index_to_payload(index: Any) -> Dict[str, Any]:
    range_index = clamp_range_index(index)
    range_min, range_max = range_index_to_bounds(range_index)
    return {
        "range_index": range_index,
        "range_label": range_index_to_label(range_index),
        "range_min_hours": range_min,
        "range_max_hours": range_max,
        "display_hours": range_index_to_midpoint(range_index),
    }


def hours_to_range_index(hours: Any) -> int:
    value = clamp_supported_hours(hours)
    for idx, (low, high) in enumerate(HOUR_RANGES, start=1):
        if idx == len(HOUR_RANGES):
            if low <= value <= high:
                return idx
        elif low <= value < high:
            return idx
    return len(HOUR_RANGES)


def hours_to_range_payload(hours: Any) -> Dict[str, Any]:
    bounded_hours = clamp_supported_hours(hours)
    out = dict(range_index_to_payload(hours_to_range_index(bounded_hours)))
    out["bounded_hours"] = round(bounded_hours, 1)
    return out


def clamp_bucket_rank(rank: Any) -> int:
    try:
        numeric = int(round(float(rank)))
    except Exception:
        numeric = 3
    return max(1, min(len(BUCKETS), numeric))


def rank_to_bucket(rank: Any) -> str:
    return BUCKETS[clamp_bucket_rank(rank) - 1]


def bucket_rank_to_default_range_index(rank: Any) -> int:
    return clamp_range_index(DEFAULT_BUCKET_RANGE_INDEX[clamp_bucket_rank(rank)])


def range_index_to_bucket_rank(index: Any) -> int:
    range_index = clamp_range_index(index)
    if range_index <= 1:
        return 1
    if range_index <= 3:
        return 2
    if range_index <= 5:
        return 3
    if range_index <= 8:
        return 4
    if range_index <= 11:
        return 5
    return 6


def bucket_to_rank(bucket: Any) -> int:
    normalized = str(bucket or "").strip().upper()
    if normalized in BUCKETS:
        return BUCKETS.index(normalized) + 1
    return 3


def _normalize_labels(raw_labels: Any) -> set[str]:
    if isinstance(raw_labels, list):
        return {
            str(item).strip().lower()
            for item in raw_labels
            if str(item).strip()
        }
    if isinstance(raw_labels, str):
        return {
            item.strip().lower()
            for item in raw_labels.split(",")
            if item.strip()
        }
    return set()


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    q = max(0.0, min(1.0, float(q)))
    position = q * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    left = float(sorted_values[lower])
    right = float(sorted_values[upper])
    return left + (right - left) * weight


def _weighted_percentile(points: Sequence[float], weights: Sequence[float], target: float) -> float:
    if not points or not weights or len(points) != len(weights):
        return 0.0
    paired = sorted(
        (
            (safe_float(point), max(0.0001, safe_float(weight, 0.0001)))
            for point, weight in zip(points, weights)
        ),
        key=lambda item: item[0],
    )
    total_weight = sum(weight for _, weight in paired)
    if total_weight <= 0:
        return float(paired[len(paired) // 2][0])

    threshold = total_weight * max(0.0, min(1.0, float(target)))
    running = 0.0
    for point, weight in paired:
        running += weight
        if running >= threshold:
            return float(point)
    return float(paired[-1][0])


def infer_bucket_rank_from_hours(hours: Any, hours_pool: Iterable[float] | None = None) -> int:
    value = safe_float(hours, 0.0)
    pool = sorted(
        safe_float(item, 0.0)
        for item in (hours_pool or [])
        if safe_float(item, 0.0) > 0
    )
    if len(pool) >= 6:
        boundaries = (
            _quantile(pool, 0.18),
            _quantile(pool, 0.36),
            _quantile(pool, 0.56),
            _quantile(pool, 0.78),
            _quantile(pool, 0.92),
        )
    else:
        boundaries = DEFAULT_BUCKET_BOUNDARIES

    if value <= boundaries[0]:
        return 1
    if value <= boundaries[1]:
        return 2
    if value <= boundaries[2]:
        return 3
    if value <= boundaries[3]:
        return 4
    if value <= boundaries[4]:
        return 5
    return 6


def build_calibration_profile(
    issue_context: Dict[str, Any],
    similar_issues: List[Dict[str, Any]],
    useful_score_threshold: float = 0.68,
) -> Dict[str, Any]:
    issue_type = str(issue_context.get("issue_type") or "").strip().lower()
    issue_labels = _normalize_labels(issue_context.get("labels") or [])

    normalized: List[Dict[str, Any]] = []
    for index, item in enumerate(similar_issues or []):
        hours = safe_float(item.get("total_effort_hours"), 0.0)
        if hours <= 0:
            continue
        score = safe_float(item.get("score"), 0.0)
        candidate_type = str(item.get("issue_type") or "").strip().lower()
        candidate_labels = _normalize_labels(item.get("labels") or [])
        normalized.append(
            {
                "index": index,
                "hours": hours,
                "score": score,
                "same_type": bool(issue_type and candidate_type == issue_type),
                "label_overlap": len(issue_labels & candidate_labels),
                "title": item.get("title"),
            }
        )

    useful = [item for item in normalized if item["score"] >= useful_score_threshold]
    focused = [
        item
        for item in useful
        if item["same_type"] or item["label_overlap"] > 0
    ]
    focused_any = [
        item
        for item in normalized
        if item["same_type"] or item["label_overlap"] > 0
    ]
    type_prior = [item for item in normalized if item["same_type"]]
    label_prior = [item for item in normalized if item["label_overlap"] > 0]
    primary = focused if len(focused) >= 3 else useful if len(useful) >= 3 else normalized

    source = "default_prior"
    if primary is focused and focused:
        source = "focused_neighbors"
    elif primary is useful and useful:
        source = "useful_neighbors"
    elif primary:
        source = "all_neighbors"

    selected_hours = sorted(item["hours"] for item in primary)
    all_hours = sorted(item["hours"] for item in normalized)
    focused_hours = sorted(item["hours"] for item in focused)
    useful_hours = sorted(item["hours"] for item in useful)
    focused_any_hours = sorted(item["hours"] for item in focused_any)
    type_prior_hours = sorted(item["hours"] for item in type_prior)
    label_prior_hours = sorted(item["hours"] for item in label_prior)

    return {
        "source": source,
        "selected_hours": selected_hours,
        "all_hours": all_hours,
        "focused_hours": focused_hours,
        "useful_hours": useful_hours,
        "focused_any_hours": focused_any_hours,
        "type_prior_hours": type_prior_hours,
        "label_prior_hours": label_prior_hours,
        "selected_count": len(selected_hours),
        "all_count": len(all_hours),
        "normalized": normalized,
        "median_hours": _quantile(selected_hours or all_hours, 0.5),
    }


def _hours_for_calibration_mode(profile: Dict[str, Any], calibration_mode: str) -> tuple[List[float], str]:
    if calibration_mode == "weak_prior":
        focused_any_hours = list(profile.get("focused_any_hours") or [])
        type_prior_hours = list(profile.get("type_prior_hours") or [])
        label_prior_hours = list(profile.get("label_prior_hours") or [])
        useful_hours = list(profile.get("useful_hours") or [])

        if len(focused_any_hours) >= 2:
            return focused_any_hours, "focused_prior"
        if len(type_prior_hours) >= 2:
            return type_prior_hours, "type_prior"
        if len(label_prior_hours) >= 2:
            return label_prior_hours, "label_prior"
        if len(useful_hours) >= 2:
            return useful_hours, "useful_prior"
        return [], "default_prior"

    selected_hours = list(profile.get("selected_hours") or [])
    if selected_hours:
        return selected_hours, str(profile.get("source") or "selected_neighbors")
    return [], "default_prior"


def calibrate_bucket_rank_to_hours(
    bucket_rank: Any,
    profile: Dict[str, Any],
    calibration_mode: str = "standard",
) -> Dict[str, Any]:
    rank = clamp_bucket_rank(bucket_rank)
    selected_hours, source_name = _hours_for_calibration_mode(profile, calibration_mode)
    if selected_hours:
        target_q = BUCKET_TARGET_QUANTILES[rank]
        estimated = _quantile(selected_hours, target_q)
        min_hours = _quantile(selected_hours, max(0.05, target_q - 0.12))
        max_hours = _quantile(selected_hours, min(0.98, target_q + 0.12))
        calibration_source = source_name
    else:
        estimated = FALLBACK_BUCKET_HOURS[rank]
        min_hours = max(1.0, estimated * 0.72)
        max_hours = estimated * 1.35
        calibration_source = "default_prior"

    out = {
        "size_bucket": rank_to_bucket(rank),
        "bucket_rank": rank,
        "estimated_hours": round(max(1.0, estimated), 1),
        "min_hours": round(max(1.0, min(min_hours, estimated)), 1),
        "max_hours": round(max(max_hours, estimated), 1),
        "calibration_source": calibration_source,
    }
    out.update(hours_to_range_payload(out["estimated_hours"]))
    return out


def aggregate_bucket_consensus(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    ranks: List[float] = []
    weights: List[float] = []
    core_ranks: List[int] = []

    for candidate in candidates or []:
        if not isinstance(candidate, dict):
            continue
        bucket_rank = candidate.get("bucket_rank")
        if bucket_rank in (None, ""):
            bucket_rank = bucket_to_rank(candidate.get("size_bucket"))
        rank = clamp_bucket_rank(bucket_rank)
        confidence = max(0.05, min(0.95, safe_float(candidate.get("confidence"), 0.5)))
        mode = str(candidate.get("mode") or candidate.get("source") or "").strip().lower()
        mode_weight = MODE_BUCKET_WEIGHTS.get(mode, 1.0)
        ranks.append(float(rank))
        weights.append((0.35 + confidence) * mode_weight)
        if mode in {"scope", "complexity"}:
            core_ranks.append(rank)

    if not ranks:
        return {
            "size_bucket": "M",
            "bucket_rank": 3,
            "confidence": 0.25,
            "candidate_count": 0,
            "spread": 0,
        }

    weighted_rank = _weighted_percentile(ranks, weights, 0.5)
    if len(core_ranks) >= 2:
        core_avg = sum(core_ranks) / len(core_ranks)
        if core_avg >= 3.2 and weighted_rank < 3.0:
            weighted_rank = min(3.2, weighted_rank + 0.55)
        elif core_avg >= 2.7 and weighted_rank < 2.5:
            weighted_rank = min(2.7, weighted_rank + 0.35)
    final_rank = clamp_bucket_rank(weighted_rank)
    spread = int(max(ranks) - min(ranks)) if ranks else 0
    avg_confidence = sum(weights) / len(weights) - 0.35
    confidence = max(0.1, min(0.92, avg_confidence - 0.07 * spread))

    return {
        "size_bucket": rank_to_bucket(final_rank),
        "bucket_rank": final_rank,
        "confidence": round(confidence, 4),
        "candidate_count": len(ranks),
        "spread": spread,
    }


def aggregate_range_consensus(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    range_points: List[float] = []
    weights: List[float] = []
    core_ranges: List[int] = []

    for candidate in candidates or []:
        if not isinstance(candidate, dict):
            continue
        range_index = candidate.get("range_index")
        if range_index in (None, ""):
            estimated_hours = candidate.get("estimated_hours")
            if estimated_hours not in (None, ""):
                range_index = hours_to_range_index(estimated_hours)
            else:
                bucket_rank = candidate.get("bucket_rank")
                if bucket_rank in (None, ""):
                    bucket_rank = bucket_to_rank(candidate.get("size_bucket"))
                range_index = bucket_rank_to_default_range_index(bucket_rank)
        coerced_range = clamp_range_index(range_index)
        confidence = max(0.05, min(0.95, safe_float(candidate.get("confidence"), 0.5)))
        mode = str(candidate.get("mode") or candidate.get("source") or "").strip().lower()
        mode_weight = MODE_BUCKET_WEIGHTS.get(mode, 1.0)
        range_points.append(float(coerced_range))
        weights.append((0.35 + confidence) * mode_weight)
        if mode in {"scope", "complexity"}:
            core_ranges.append(coerced_range)

    if not range_points:
        fallback_range = range_index_to_payload(bucket_rank_to_default_range_index(3))
        return {
            **fallback_range,
            "size_bucket": rank_to_bucket(3),
            "bucket_rank": 3,
            "confidence": 0.25,
            "candidate_count": 0,
            "spread": 0,
        }

    weighted_range = _weighted_percentile(range_points, weights, 0.5)
    if len(core_ranges) >= 2:
        core_avg = sum(core_ranges) / len(core_ranges)
        if core_avg >= 7.0 and weighted_range < 6.0:
            weighted_range = min(7.0, weighted_range + 1.2)
        elif core_avg >= 5.5 and weighted_range < 5.0:
            weighted_range = min(5.5, weighted_range + 0.8)
    final_range_index = clamp_range_index(weighted_range)
    spread = int(max(range_points) - min(range_points)) if range_points else 0
    avg_confidence = sum(weights) / len(weights) - 0.35
    confidence = max(0.1, min(0.92, avg_confidence - 0.05 * spread))
    bucket_rank = range_index_to_bucket_rank(final_range_index)

    return {
        **range_index_to_payload(final_range_index),
        "size_bucket": rank_to_bucket(bucket_rank),
        "bucket_rank": bucket_rank,
        "confidence": round(confidence, 4),
        "candidate_count": len(range_points),
        "spread": spread,
    }


def weighted_neighbor_estimate(
    issue_context: Dict[str, Any],
    similar_issues: List[Dict[str, Any]],
    useful_score_threshold: float = 0.68,
    top_k: int = 5,
) -> Dict[str, Any]:
    profile = build_calibration_profile(
        issue_context=issue_context,
        similar_issues=similar_issues,
        useful_score_threshold=useful_score_threshold,
    )
    normalized = list(profile.get("normalized") or [])
    useful = [item for item in normalized if item["score"] >= useful_score_threshold]
    focused = [item for item in useful if item["same_type"] or item["label_overlap"] > 0]
    focused_any = [item for item in normalized if item["same_type"] or item["label_overlap"] > 0]
    type_prior = [item for item in normalized if item["same_type"]]
    label_prior = [item for item in normalized if item["label_overlap"] > 0]

    candidate_source = "default_prior"
    candidates: List[Dict[str, Any]] = []
    if len(focused) >= 2:
        candidates = focused[:top_k]
        candidate_source = "focused_neighbors"
    elif len(useful) >= 2:
        candidates = useful[:top_k]
        candidate_source = "useful_neighbors"
    elif len(focused_any) >= 2:
        candidates = focused_any[:top_k]
        candidate_source = "focused_prior"
    elif len(type_prior) >= 2:
        candidates = type_prior[:top_k]
        candidate_source = "type_prior"
    elif len(label_prior) >= 2:
        candidates = label_prior[:top_k]
        candidate_source = "label_prior"
    elif len(normalized) >= 2:
        candidates = normalized[: min(top_k, len(normalized))]
        candidate_source = "all_neighbors"

    if not candidates:
        fallback = calibrate_bucket_rank_to_hours(3, profile, calibration_mode="weak_prior")
        fallback.update(
            {
                "confidence": 0.25,
                "neighbor_count": 0,
                "supporting_hours": [],
                "weighted_hours": [],
            }
        )
        return fallback

    if len(candidates) == 1 and safe_float(candidates[0].get("score"), 0.0) < useful_score_threshold:
        fallback = calibrate_bucket_rank_to_hours(3, profile, calibration_mode="weak_prior")
        fallback.update(
            {
                "confidence": 0.28,
                "neighbor_count": 1,
                "supporting_hours": [round(safe_float(candidates[0].get("hours"), 0.0), 1)],
                "weighted_hours": [1.0],
            }
        )
        return fallback

    hours: List[float] = []
    weights: List[float] = []
    for item in candidates:
        score = max(0.01, item["score"])
        weight = (score ** 4) / (1.0 + (0.18 * item["index"]))
        if item["same_type"]:
            weight *= 1.15
        if item["label_overlap"] > 0:
            weight *= 1.0 + min(0.12, 0.04 * item["label_overlap"])
        hours.append(item["hours"])
        weights.append(weight)

    estimated = _weighted_percentile(hours, weights, 0.5)
    min_hours = _weighted_percentile(hours, weights, 0.2)
    max_hours = _weighted_percentile(hours, weights, 0.8)
    bucket_rank = infer_bucket_rank_from_hours(
        estimated,
        hours_pool=profile.get("selected_hours") or profile.get("all_hours"),
    )
    total_weight = sum(weights)
    top_weight_ratio = max(weights) / total_weight if total_weight > 0 else 1.0
    confidence = 0.52 + min(0.28, 0.09 * len(candidates)) + min(0.12, top_weight_ratio * 0.12)
    confidence -= min(0.15, 0.05 * max(0, len(set(int(round(h)) for h in hours)) - 2))

    out = {
        "estimated_hours": round(max(1.0, estimated), 1),
        "min_hours": round(max(1.0, min(min_hours, estimated)), 1),
        "max_hours": round(max(max_hours, estimated), 1),
        "size_bucket": rank_to_bucket(bucket_rank),
        "bucket_rank": bucket_rank,
        "calibration_source": candidate_source,
        "confidence": round(max(0.2, min(0.92, confidence)), 4),
        "neighbor_count": len(candidates),
        "supporting_hours": [round(item, 1) for item in hours],
        "weighted_hours": [round(item, 4) for item in weights],
    }
    out.update(hours_to_range_payload(out["estimated_hours"]))
    return out


def weighted_neighbor_range_estimate(
    issue_context: Dict[str, Any],
    similar_issues: List[Dict[str, Any]],
    useful_score_threshold: float = 0.68,
    top_k: int = 5,
) -> Dict[str, Any]:
    profile = build_calibration_profile(
        issue_context=issue_context,
        similar_issues=similar_issues,
        useful_score_threshold=useful_score_threshold,
    )
    normalized = list(profile.get("normalized") or [])
    useful = [item for item in normalized if item["score"] >= useful_score_threshold]
    focused = [item for item in useful if item["same_type"] or item["label_overlap"] > 0]
    focused_any = [item for item in normalized if item["same_type"] or item["label_overlap"] > 0]
    type_prior = [item for item in normalized if item["same_type"]]
    label_prior = [item for item in normalized if item["label_overlap"] > 0]

    candidate_source = "default_prior"
    candidates: List[Dict[str, Any]] = []
    if len(focused) >= 2:
        candidates = focused[:top_k]
        candidate_source = "focused_neighbors"
    elif len(useful) >= 2:
        candidates = useful[:top_k]
        candidate_source = "useful_neighbors"
    elif len(focused_any) >= 2:
        candidates = focused_any[:top_k]
        candidate_source = "focused_prior"
    elif len(type_prior) >= 2:
        candidates = type_prior[:top_k]
        candidate_source = "type_prior"
    elif len(label_prior) >= 2:
        candidates = label_prior[:top_k]
        candidate_source = "label_prior"
    elif len(normalized) >= 2:
        candidates = normalized[: min(top_k, len(normalized))]
        candidate_source = "all_neighbors"

    if not candidates:
        fallback_range = range_index_to_payload(bucket_rank_to_default_range_index(3))
        fallback = {
            "estimated_hours": float(fallback_range["display_hours"]),
            "estimated_hours_raw": float(fallback_range["display_hours"]),
            "min_hours": float(fallback_range["range_min_hours"]),
            "max_hours": float(fallback_range["range_max_hours"]),
            "size_bucket": rank_to_bucket(range_index_to_bucket_rank(fallback_range["range_index"])),
            "bucket_rank": range_index_to_bucket_rank(fallback_range["range_index"]),
            "calibration_source": "default_prior",
            "confidence": 0.25,
            "neighbor_count": 0,
            "supporting_hours": [],
            "weighted_hours": [],
            "supporting_ranges": [],
        }
        fallback.update(fallback_range)
        return fallback

    range_points: List[float] = []
    weights: List[float] = []
    hours: List[float] = []
    for item in candidates:
        score = max(0.01, item["score"])
        weight = (score ** 4) / (1.0 + (0.18 * item["index"]))
        if item["same_type"]:
            weight *= 1.15
        if item["label_overlap"] > 0:
            weight *= 1.0 + min(0.12, 0.04 * item["label_overlap"])
        hour_value = float(item["hours"])
        hours.append(hour_value)
        weights.append(weight)
        range_points.append(float(hours_to_range_index(hour_value)))

    estimated_range = clamp_range_index(_weighted_percentile(range_points, weights, 0.5))
    min_range = clamp_range_index(_weighted_percentile(range_points, weights, 0.2))
    max_range = clamp_range_index(_weighted_percentile(range_points, weights, 0.8))
    range_payload = range_index_to_payload(estimated_range)
    min_payload = range_index_to_payload(min_range)
    max_payload = range_index_to_payload(max_range)
    bucket_rank = range_index_to_bucket_rank(estimated_range)

    total_weight = sum(weights)
    top_weight_ratio = max(weights) / total_weight if total_weight > 0 else 1.0
    range_spread = max(range_points) - min(range_points) if range_points else 0.0
    confidence = 0.50 + min(0.24, 0.08 * len(candidates)) + min(0.12, top_weight_ratio * 0.12)
    confidence -= min(0.18, 0.04 * range_spread)

    out = {
        "estimated_hours": float(range_payload["display_hours"]),
        "estimated_hours_raw": float(range_payload["display_hours"]),
        "min_hours": float(min_payload["range_min_hours"]),
        "max_hours": float(max_payload["range_max_hours"]),
        "size_bucket": rank_to_bucket(bucket_rank),
        "bucket_rank": bucket_rank,
        "calibration_source": candidate_source,
        "confidence": round(max(0.2, min(0.92, confidence)), 4),
        "neighbor_count": len(candidates),
        "supporting_hours": [round(item, 1) for item in hours],
        "weighted_hours": [round(item, 4) for item in weights],
        "supporting_ranges": [range_index_to_label(item) for item in range_points],
    }
    out.update(range_payload)
    return out


def bounded_adjustment_from_reviews(
    base_hours: Any,
    complexity_review: Dict[str, Any] | None,
    critic_review: Dict[str, Any] | None,
) -> Dict[str, Any]:
    base = max(1.0, safe_float(base_hours, 1.0))
    complexity_delta = int(safe_float((complexity_review or {}).get("bucket_delta"), 0.0))
    complexity_conf = max(0.0, min(1.0, safe_float((complexity_review or {}).get("confidence"), 0.5)))
    complexity_risk = max(0.0, min(1.0, safe_float((complexity_review or {}).get("risk_hidden_complexity"), 0.0)))
    critic_under = max(0.0, min(1.0, safe_float((critic_review or {}).get("risk_of_underestimation"), 0.0)))
    critic_over = max(0.0, min(1.0, safe_float((critic_review or {}).get("risk_of_overestimation"), 0.0)))

    if complexity_delta <= 0 or complexity_risk < 0.74:
        complexity_pct = 0.0
    else:
        risk_factor = min(1.0, max(0.0, (complexity_risk - 0.74) / 0.26))
        confidence_factor = max(0.35, complexity_conf)
        complexity_pct = min(0.12, complexity_delta * 0.10 * risk_factor * confidence_factor)
    critic_pct = max(-0.06, min(0.06, (critic_under - critic_over) * 0.10))
    total_pct = max(-0.08, min(0.18, complexity_pct + critic_pct))
    adjusted = round(max(1.0, base * (1.0 + total_pct)), 1)

    return {
        "adjusted_hours": adjusted,
        "adjustment_delta": round(adjusted - base, 1),
        "complexity_adjustment_pct": round(complexity_pct, 4),
        "critic_adjustment_pct": round(critic_pct, 4),
    }
