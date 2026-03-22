from __future__ import annotations

from typing import Any, Mapping, TypedDict, cast


class TokenUsage(TypedDict):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


def _to_int(value: Any) -> int:
    try:
        if value is None:
            return 0
        if isinstance(value, bool):
            return int(value)
        return int(value)
    except (TypeError, ValueError):
        return 0


def _normalize_usage_dict(usage: Mapping[str, Any]) -> TokenUsage:
    prompt = (
        usage.get("prompt_tokens")
        or usage.get("input_tokens")
        or usage.get("input_token_count")
        or usage.get("inputTokenCount")
    )
    completion = (
        usage.get("completion_tokens")
        or usage.get("output_tokens")
        or usage.get("output_token_count")
        or usage.get("outputTokenCount")
    )
    total = (
        usage.get("total_tokens")
        or usage.get("total_token_count")
        or usage.get("totalTokenCount")
    )

    prompt_i = max(0, _to_int(prompt))
    completion_i = max(0, _to_int(completion))
    total_i = max(0, _to_int(total))
    if total_i <= 0:
        total_i = prompt_i + completion_i

    return {
        "prompt_tokens": prompt_i,
        "completion_tokens": completion_i,
        "total_tokens": total_i,
    }


def _maybe_mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return cast(Mapping[str, Any], value)
    return None


def extract_token_usage(response: Any) -> TokenUsage:
    """
    Best-effort extraction of token usage from LangChain message objects
    (e.g., AIMessage) or provider responses.
    """
    # Common places LangChain stores usage for chat models:
    # - response.usage_metadata (dict)
    # - response.response_metadata (dict, sometimes contains usage)
    # - response.additional_kwargs (dict)
    candidates: list[Mapping[str, Any]] = []

    if isinstance(response, Mapping):
        candidates.append(cast(Mapping[str, Any], response))
    else:
        for attr in ("usage_metadata", "response_metadata", "additional_kwargs", "metadata"):
            try:
                v = getattr(response, attr, None)
            except Exception:
                v = None
            m = _maybe_mapping(v)
            if m is not None:
                candidates.append(m)

    # Add nested common keys (one level deep), and also try "usage" itself.
    expanded: list[Mapping[str, Any]] = []
    for c in candidates:
        expanded.append(c)
        for key in ("usage_metadata", "usage", "token_usage"):
            nested = _maybe_mapping(c.get(key))
            if nested is not None:
                expanded.append(nested)

    # Choose the first candidate that yields a non-zero total or has any known key.
    for c in expanded:
        if any(k in c for k in ("prompt_tokens", "completion_tokens", "total_tokens", "input_tokens", "output_tokens")):
            usage = _normalize_usage_dict(c)
            if usage["total_tokens"] > 0 or usage["prompt_tokens"] > 0 or usage["completion_tokens"] > 0:
                return usage

    # If we only found keys but totals were empty, still return normalized from the first keyed dict.
    for c in expanded:
        if any(k in c for k in ("prompt_tokens", "completion_tokens", "total_tokens", "input_tokens", "output_tokens")):
            return _normalize_usage_dict(c)

    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def coerce_token_usage(value: Any) -> TokenUsage:
    if isinstance(value, Mapping):
        return _normalize_usage_dict(cast(Mapping[str, Any], value))
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def add_token_usages(a: Any, b: Any) -> TokenUsage:
    ua = coerce_token_usage(a)
    ub = coerce_token_usage(b)
    return {
        "prompt_tokens": ua["prompt_tokens"] + ub["prompt_tokens"],
        "completion_tokens": ua["completion_tokens"] + ub["completion_tokens"],
        "total_tokens": ua["total_tokens"] + ub["total_tokens"],
    }

