"""Tests for ai.core.output_translator.

Uses a fake LLMClient to avoid real Gemini calls. Validates:
- Walks dict/list and finds translatable strings only inside whitelisted fields
- Skips strings that already look Portuguese
- Skips strings shorter than the minimum length
- Applies translations preserving the dict shape
- Fails open when LLM raises or returns garbage
- Disabled flag short-circuits the call
- Locale "en" short-circuits the call
"""
import json
import unittest
from unittest.mock import MagicMock

from ai.core import output_translator as translator
from ai.core.output_translator import (
    _apply_translations,
    _collect_translatable,
    _looks_like_portuguese,
    translate_estimation_output,
)
from config.settings import settings


class FakeLLMClient:
    """Stand-in for LLMClient used by translate_estimation_output.

    Behavior is controlled by ``responses``: a list of strings that
    ``send_prompt`` returns in order. If the list is exhausted, raises
    StopIteration. If ``raise_on_call`` is set, raises that exception
    on every call."""

    def __init__(self, responses=None, raise_on_call=None):
        self.responses = list(responses or [])
        self.raise_on_call = raise_on_call
        self.calls = []

    def send_prompt(self, prompt, **kwargs):
        self.calls.append(prompt)
        if self.raise_on_call is not None:
            raise self.raise_on_call
        if not self.responses:
            raise RuntimeError("No more fake responses")
        return self.responses.pop(0)


class TestPortugueseHeuristic(unittest.TestCase):
    def test_detects_portuguese_with_accents(self):
        self.assertTrue(_looks_like_portuguese("Faixa final não identificada."))

    def test_detects_portuguese_with_stopwords(self):
        self.assertTrue(_looks_like_portuguese("A estimativa de horas para a issue."))

    def test_does_not_detect_english(self):
        self.assertFalse(_looks_like_portuguese("Final range bounded by historical neighbors."))

    def test_handles_short_strings(self):
        self.assertFalse(_looks_like_portuguese("hi"))


class TestCollectTranslatable(unittest.TestCase):
    def test_only_whitelisted_fields_are_collected(self):
        state = {
            "estimated_hours": 14,  # numeric, ignored
            "model_name": "heuristic_bucket_calibrated",  # not whitelisted
            "justification": "This is the justification text from the agent.",
            "summary": "Final summary explaining the estimation result.",
        }
        items = _collect_translatable(state)
        self.assertEqual(len(items), 2)
        # Paths are encoded as slash-separated keys
        self.assertIn("justification", items)
        self.assertIn("summary", items)

    def test_descends_into_lists_inside_translatable_field(self):
        state = {
            "warnings": [
                "First warning is longer than the minimum length.",
                "Second warning also exceeds the threshold ok.",
            ],
        }
        items = _collect_translatable(state)
        self.assertEqual(len(items), 2)
        self.assertIn("warnings/0", items)
        self.assertIn("warnings/1", items)

    def test_skips_short_strings(self):
        state = {"justification": "ok"}  # too short
        items = _collect_translatable(state)
        self.assertEqual(len(items), 0)

    def test_skips_already_portuguese(self):
        state = {"justification": "Esta é uma justificativa em português."}
        items = _collect_translatable(state)
        self.assertEqual(len(items), 0)

    def test_does_not_descend_into_non_whitelisted_fields(self):
        state = {
            "metadata": {
                # description is not in the whitelist, so even though this is
                # a long English string, it should NOT be collected
                "description": "This is a long description that should not be translated.",
            },
        }
        items = _collect_translatable(state)
        self.assertEqual(len(items), 0)

    def test_nested_dict_inside_translatable(self):
        state = {
            "warnings": {
                "primary": "Primary warning text exceeds minimum length easily.",
                "secondary": "Another warning entry that should be translated.",
            },
        }
        items = _collect_translatable(state)
        self.assertEqual(len(items), 2)
        self.assertIn("warnings/primary", items)
        self.assertIn("warnings/secondary", items)


class TestApplyTranslations(unittest.TestCase):
    def test_replaces_string_at_path(self):
        state = {"justification": "Original English text."}
        translations = {"justification": "Texto original em inglês."}
        _apply_translations(state, translations)
        self.assertEqual(state["justification"], "Texto original em inglês.")

    def test_replaces_inside_list(self):
        state = {"warnings": ["First warning.", "Second warning."]}
        translations = {"warnings/0": "Primeiro aviso.", "warnings/1": "Segundo aviso."}
        _apply_translations(state, translations)
        self.assertEqual(state["warnings"], ["Primeiro aviso.", "Segundo aviso."])

    def test_leaves_unmatched_paths_alone(self):
        state = {"justification": "Untouched.", "summary": "Also untouched."}
        translations = {"some_other_path": "ignored"}
        _apply_translations(state, translations)
        self.assertEqual(state["justification"], "Untouched.")
        self.assertEqual(state["summary"], "Also untouched.")


class TestTranslateEstimationOutput(unittest.TestCase):
    def setUp(self):
        # Save originals
        self._orig_enabled = settings.OUTPUT_TRANSLATOR_ENABLED
        self._orig_locale = settings.OUTPUT_LOCALE

    def tearDown(self):
        settings.OUTPUT_TRANSLATOR_ENABLED = self._orig_enabled
        settings.OUTPUT_LOCALE = self._orig_locale

    def test_disabled_flag_is_noop(self):
        settings.OUTPUT_TRANSLATOR_ENABLED = False
        state = {"justification": "Some long English text here please."}
        client = FakeLLMClient()
        result = translate_estimation_output(state, llm_client=client)
        self.assertEqual(result, {"justification": "Some long English text here please."})
        self.assertEqual(len(client.calls), 0)

    def test_locale_en_is_noop(self):
        settings.OUTPUT_TRANSLATOR_ENABLED = True
        state = {"justification": "Some long English text here please."}
        client = FakeLLMClient()
        result = translate_estimation_output(state, target_locale="en", llm_client=client)
        self.assertEqual(result["justification"], "Some long English text here please.")
        self.assertEqual(len(client.calls), 0)

    def test_translates_whitelisted_fields(self):
        settings.OUTPUT_TRANSLATOR_ENABLED = True
        state = {
            "estimated_hours": 14,
            "justification": "Estimate based on similar past issues.",
            "summary": "Final consolidated estimate at 14 hours.",
            "metadata": {"description": "Should not be translated."},
        }
        # Fake LLM returns a JSON dict with translations for the keys that
        # the prompt requested
        client = FakeLLMClient(
            responses=[
                json.dumps({
                    "justification": "Estimativa baseada em issues passadas similares.",
                    "summary": "Estimativa final consolidada em 14 horas.",
                })
            ]
        )
        result = translate_estimation_output(state, target_locale="pt-BR", llm_client=client)

        self.assertEqual(result["justification"], "Estimativa baseada em issues passadas similares.")
        self.assertEqual(result["summary"], "Estimativa final consolidada em 14 horas.")
        # Numeric fields preserved
        self.assertEqual(result["estimated_hours"], 14)
        # Non-whitelisted field preserved (NOT translated)
        self.assertEqual(result["metadata"]["description"], "Should not be translated.")

    def test_fails_open_when_llm_raises(self):
        settings.OUTPUT_TRANSLATOR_ENABLED = True
        state = {"justification": "English text that we expected to translate."}
        client = FakeLLMClient(raise_on_call=RuntimeError("API timeout"))
        result = translate_estimation_output(state, target_locale="pt-BR", llm_client=client)
        # Original preserved
        self.assertEqual(result["justification"], "English text that we expected to translate.")

    def test_fails_open_when_llm_returns_garbage(self):
        settings.OUTPUT_TRANSLATOR_ENABLED = True
        state = {"justification": "English text that we expected to translate."}
        client = FakeLLMClient(responses=["this is not json at all"])
        result = translate_estimation_output(state, target_locale="pt-BR", llm_client=client)
        self.assertEqual(result["justification"], "English text that we expected to translate.")

    def test_partial_translation_keeps_originals_for_missing_keys(self):
        settings.OUTPUT_TRANSLATOR_ENABLED = True
        state = {
            "justification": "First English string here please.",
            "summary": "Second English string here please.",
        }
        # LLM only translates one of the two requested keys
        client = FakeLLMClient(
            responses=[
                json.dumps({
                    "justification": "Primeira frase em português aqui.",
                    # summary missing — keeps original
                })
            ]
        )
        result = translate_estimation_output(state, target_locale="pt-BR", llm_client=client)
        self.assertEqual(result["justification"], "Primeira frase em português aqui.")
        self.assertEqual(result["summary"], "Second English string here please.")

    def test_no_translatable_strings_skips_llm_call(self):
        settings.OUTPUT_TRANSLATOR_ENABLED = True
        state = {"estimated_hours": 14, "model_name": "heuristic_bucket_calibrated"}
        client = FakeLLMClient()
        result = translate_estimation_output(state, target_locale="pt-BR", llm_client=client)
        self.assertEqual(result, {"estimated_hours": 14, "model_name": "heuristic_bucket_calibrated"})
        self.assertEqual(len(client.calls), 0)


if __name__ == "__main__":
    unittest.main()
