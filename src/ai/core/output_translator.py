"""Output translator for estimation results.

Translates user-facing free-text fields from the estimation pipeline to the
configured locale (default: Brazilian Portuguese). Runs as a post-processing
step after the LangGraph pipeline finishes — it does NOT alter how the agents
think or estimate.

Design choices
--------------
- Single LLM call per estimation (batched), not one per field. Much cheaper.
- Fail-open: if translation breaks for any reason, returns the original dict
  in English. Never blocks the main flow.
- Whitelist of translatable fields. Avoids accidentally translating IDs,
  enums, labels, model names, route names, etc.
- Skip strings shorter than ``MIN_TRANSLATABLE_LENGTH`` (likely identifiers).
- Skip strings already detected as Portuguese (cheap heuristic on accents +
  common stopwords). Avoids re-translating fallback strings that the agents
  already emit in PT-BR.

Configuration
-------------
- ``settings.OUTPUT_LOCALE``: target locale (default ``"pt-BR"``).
- ``settings.OUTPUT_TRANSLATOR_ENABLED``: toggle on/off (default ``True``).
- ``settings.OUTPUT_TRANSLATOR_MODEL``: LLM model name override (default uses
  the standard Gemini Flash from LLMClient).
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from ai.core.llm_client import LLMClient
from ai.core.json_utils import parse_llm_json_response
from config.settings import settings

logger = logging.getLogger(__name__)


# Field names that may contain user-facing free text. Only fields in this set
# (or any sub-field of these, when the value is a list/dict) are eligible
# for translation. Everything else is left untouched.
TRANSLATABLE_FIELD_NAMES = frozenset({
    "summary",
    "justification",
    "user_justification",
    "recommendation",
    "split_reason",
    "strongest_signal",
    "rationale",
    "explanation",
    "narrative",
    "warnings",
    "evidence",
    "assumptions",
    "contradictions",
    "hidden_complexities",
    "missing_information",
    "risks",
    "notes",
    "review_notes",
    "comments",
    "summary_lines",
})

MIN_TRANSLATABLE_LENGTH = 10
MAX_BATCH_TOKENS_APPROX = 4000  # rough soft cap on prompt size


_PT_HEURISTIC_TOKENS = re.compile(
    r"\b(de|do|da|para|com|que|uma?|os|as|você|não|é|está|estão|"
    r"hora|horas|estimativa|análise|complexidade|escopo|esforço|"
    r"calibrad[ao]|ajuste|risco|incerteza)\b",
    flags=re.IGNORECASE,
)


def _looks_like_portuguese(text: str) -> bool:
    """Cheap heuristic: text is likely Portuguese if it has accented chars
    OR several common PT stopwords. Avoids re-translating already-PT text."""
    if not isinstance(text, str):
        return False
    if any(ch in text for ch in "áàâãéêíïóôõúüçÁÀÂÃÉÊÍÏÓÔÕÚÜÇ"):
        return True
    matches = _PT_HEURISTIC_TOKENS.findall(text)
    return len(matches) >= 2


def _collect_translatable(
    obj: Any,
    path: Tuple[str, ...] = (),
    sink: Optional[Dict[str, str]] = None,
    parent_field: Optional[str] = None,
) -> Dict[str, str]:
    """Walk the dict/list and collect user-facing strings into a flat map
    ``{path -> string}``. Path is a tuple of keys/indices, encoded as a
    JSON-pointer-like string when emitted to the LLM.

    Only descends into entries whose key is in TRANSLATABLE_FIELD_NAMES, OR
    when already inside a translatable subtree (parent_field already valid).
    """
    if sink is None:
        sink = {}

    if isinstance(obj, dict):
        for key, value in obj.items():
            key_str = str(key)
            is_translatable_key = key_str in TRANSLATABLE_FIELD_NAMES
            # Descend into translatable sub-trees, OR continue inside a
            # tree that was already opened by a translatable parent.
            if is_translatable_key or parent_field is not None:
                new_parent = key_str if is_translatable_key else parent_field
                _collect_translatable(value, path + (key_str,), sink, new_parent)
    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            _collect_translatable(item, path + (str(idx),), sink, parent_field)
    elif isinstance(obj, str):
        # Reached a leaf string inside a translatable subtree
        if parent_field is None:
            return sink
        text = obj.strip()
        if len(text) < MIN_TRANSLATABLE_LENGTH:
            return sink
        if _looks_like_portuguese(text):
            return sink
        # Encode path as a stable, parseable string
        path_key = "/".join(path)
        sink[path_key] = obj

    return sink


def _apply_translations(obj: Any, translations: Dict[str, str], path: Tuple[str, ...] = ()) -> Any:
    """In-place walk that replaces strings whose path matches a translation."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = _apply_translations(value, translations, path + (str(key),))
        return obj
    if isinstance(obj, list):
        for idx, item in enumerate(obj):
            obj[idx] = _apply_translations(item, translations, path + (str(idx),))
        return obj
    if isinstance(obj, str):
        path_key = "/".join(path)
        if path_key in translations:
            return translations[path_key]
        return obj
    return obj


def _build_translation_prompt(items: Dict[str, str], target_locale: str) -> str:
    """Builds a prompt that asks the LLM to translate each string, returning
    a JSON map from path -> translated string."""
    locale_label = {
        "pt-BR": "Brazilian Portuguese (pt-BR)",
        "pt": "Portuguese",
        "es": "Spanish",
        "es-ES": "European Spanish",
    }.get(target_locale, target_locale)

    payload = json.dumps(items, ensure_ascii=False, indent=2)
    return f"""You are a precise translator for software engineering UIs.

Translate every value below from English to {locale_label}.

Strict rules:
- Preserve the JSON structure: same keys (paths), same number of entries.
- Translate values as natural, professional UI copy. Keep technical terms in
  English when they are domain conventions (e.g. issue, sprint, backlog,
  pull request, commit, deploy, rollback, OAuth, SDK, CI/CD, Kubernetes).
- Do NOT translate placeholder tokens, code identifiers, file names, or
  numbers. Keep "{{var}}", "<x>", "`code`", URLs, and "MODEL/PROJECT-123" as-is.
- Keep numeric values, percentages, and time spans (e.g. "12h", "3 dias")
  with the right unit in the target language.
- Keep the same approximate length when possible.

Return ONLY a JSON object with the same keys as input, no markdown, no prose.

Input:
{payload}

Output JSON:"""


def translate_estimation_output(
    estimation_state: Dict[str, Any],
    *,
    target_locale: Optional[str] = None,
    llm_client: Optional[LLMClient] = None,
) -> Dict[str, Any]:
    """Translate user-facing strings in the estimation result.

    Parameters
    ----------
    estimation_state : dict
        The dict returned by ``run_estimation_flow`` (or a sub-dict like
        ``final_estimation``). Mutated in-place AND returned.
    target_locale : str, optional
        Override the configured locale. Useful for callers that want
        per-request language selection.
    llm_client : LLMClient, optional
        Reuse an existing client (avoids re-instantiating). If None, creates
        a fresh one with default Gemini Flash settings.

    Returns
    -------
    dict
        The same ``estimation_state`` with translated strings. On any
        failure, returns the original (English) dict unchanged.
    """
    if not isinstance(estimation_state, dict):
        return estimation_state

    enabled = bool(getattr(settings, "OUTPUT_TRANSLATOR_ENABLED", True))
    if not enabled:
        return estimation_state

    locale = (target_locale or getattr(settings, "OUTPUT_LOCALE", "pt-BR") or "pt-BR").strip()
    if locale.lower() in {"en", "en-us", "en-gb"}:
        # No-op when target is English itself
        return estimation_state

    items = _collect_translatable(estimation_state)
    if not items:
        logger.debug("[TRANSLATOR] No translatable strings found, skipping LLM call")
        return estimation_state

    # Soft cap on batch size: if too large, split into chunks. For the volumes
    # we see in this project (one estimation = ~5-15 strings), one batch
    # always suffices, but we keep the safety net.
    chunks: List[Dict[str, str]] = []
    current: Dict[str, str] = {}
    current_size = 0
    for key, value in items.items():
        approx = len(key) + len(value) + 6  # JSON overhead per entry
        if current_size + approx > MAX_BATCH_TOKENS_APPROX and current:
            chunks.append(current)
            current = {}
            current_size = 0
        current[key] = value
        current_size += approx
    if current:
        chunks.append(current)

    client = llm_client
    if client is None:
        try:
            model = getattr(settings, "OUTPUT_TRANSLATOR_MODEL", None) or "gemini-3-flash-preview"
            client = LLMClient(model=model, temperature=0.0)
        except Exception as exc:
            logger.warning(
                "[TRANSLATOR] Failed to instantiate LLMClient, falling back to original output: %s",
                exc,
            )
            return estimation_state

    translations: Dict[str, str] = {}
    for chunk in chunks:
        try:
            prompt = _build_translation_prompt(chunk, locale)
            raw = client.send_prompt(prompt)
            parsed = parse_llm_json_response(raw)
            if not isinstance(parsed, dict):
                logger.warning("[TRANSLATOR] Non-dict response from LLM, skipping chunk")
                continue
            # Only accept entries that match keys we asked for AND have string values
            for key, original in chunk.items():
                value = parsed.get(key)
                if isinstance(value, str) and value.strip():
                    translations[key] = value
        except Exception as exc:
            logger.warning(
                "[TRANSLATOR] Translation chunk failed (size=%d), keeping originals: %s",
                len(chunk),
                exc,
            )
            continue

    if not translations:
        logger.info("[TRANSLATOR] No translations produced, returning original output")
        return estimation_state

    logger.info(
        "[TRANSLATOR] locale=%s translated_fields=%d/%d",
        locale,
        len(translations),
        len(items),
    )
    _apply_translations(estimation_state, translations)
    return estimation_state
