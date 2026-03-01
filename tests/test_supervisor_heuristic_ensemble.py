import unittest

from ai.agents.supervisor_agent import combine_heuristic_estimations


class _FakeLLM:
    def __init__(self, response: str):
        self.response = response

    def send_prompt(self, _prompt: str, **_kwargs) -> str:
        return self.response


class TestSupervisorHeuristicEnsemble(unittest.TestCase):
    def test_combine_heuristic_estimations_llm_path(self):
        estimations = [
            {"source": "heuristic_1", "estimate_hours": 5, "confidence": 0.5, "justification": "a"},
            {"source": "heuristic_2", "estimate_hours": 7, "confidence": 0.7, "justification": "b"},
            {"source": "heuristic_3", "estimate_hours": 9, "confidence": 0.9, "justification": "c"},
        ]
        llm = _FakeLLM('{"estimate_hours": 8, "confidence": 0.81, "justification": "Sintese LLM."}')
        out = combine_heuristic_estimations(estimations, llm=llm)
        self.assertEqual(out["estimate_hours"], 8)
        self.assertEqual(out["confidence"], 0.81)
        self.assertEqual(out["justification"], "Sintese LLM.")

    def test_combine_heuristic_estimations_fallback_median(self):
        estimations = [
            {"source": "heuristic_1", "estimate_hours": 2, "confidence": 0.6, "justification": "a"},
            {"source": "heuristic_2", "estimate_hours": 8, "confidence": 0.7, "justification": "b"},
            {"source": "heuristic_3", "estimate_hours": 14, "confidence": 0.8, "justification": "c"},
        ]
        out = combine_heuristic_estimations(estimations, llm=None)
        self.assertEqual(out["estimate_hours"], 8)
        self.assertLess(out["confidence"], 0.7)
        self.assertIn("intervalo observado 2.0-14.0h", out["justification"])

    def test_confidence_penalized_on_high_spread(self):
        narrow = [
            {"source": "heuristic_1", "estimate_hours": 7, "confidence": 0.8, "justification": "a"},
            {"source": "heuristic_2", "estimate_hours": 8, "confidence": 0.8, "justification": "b"},
            {"source": "heuristic_3", "estimate_hours": 9, "confidence": 0.8, "justification": "c"},
        ]
        wide = [
            {"source": "heuristic_1", "estimate_hours": 2, "confidence": 0.8, "justification": "a"},
            {"source": "heuristic_2", "estimate_hours": 8, "confidence": 0.8, "justification": "b"},
            {"source": "heuristic_3", "estimate_hours": 20, "confidence": 0.8, "justification": "c"},
        ]
        narrow_out = combine_heuristic_estimations(narrow, llm=None)
        wide_out = combine_heuristic_estimations(wide, llm=None)
        self.assertLess(wide_out["confidence"], narrow_out["confidence"])


if __name__ == "__main__":
    unittest.main()
