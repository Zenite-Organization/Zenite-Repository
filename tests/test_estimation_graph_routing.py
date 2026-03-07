import unittest
from unittest.mock import patch

from ai.workflows import estimation_graph as eg
from config.settings import settings


class _FakeRetrieverSufficient:
    def __init__(self, _vs):
        pass

    def get_similar_issues(self, _issue):
        return [
            {"id": "1", "score": 0.9, "total_effort_hours": 5},
            {"id": "2", "score": 0.85, "total_effort_hours": 8},
        ]


class _FakeRetrieverInsufficient:
    def __init__(self, _vs):
        pass

    def get_similar_issues(self, _issue):
        return [{"id": "1", "score": 0.5, "total_effort_hours": 5}]


class _NoTechVectorStore:
    pass


class TestEstimationGraphRouting(unittest.TestCase):
    def test_routes_to_analogical_when_context_sufficient(self):
        prev_min_hits = settings.RAG_MIN_HITS_MAIN
        prev_min_score = settings.RAG_MIN_SCORE_MAIN
        settings.RAG_MIN_HITS_MAIN = 2
        settings.RAG_MIN_SCORE_MAIN = 0.75

        original_vs = eg.vector_store
        eg.vector_store = _NoTechVectorStore()
        try:
            with patch.object(eg, "Retriever", _FakeRetrieverSufficient), patch.object(
                eg,
                "run_analogical",
                return_value={
                    "estimated_hours": 13.0,
                    "confidence": 0.82,
                    "justification": "Baseado em historico similar.",
                },
            ) as analogical_mock, patch.object(eg, "run_heuristic") as heuristic_mock, patch.object(
                eg, "combine_heuristic_estimations"
            ) as combine_mock:
                state = eg.estimation_graph.invoke(
                    {"issue": {"title": "A", "description": "B", "labels": [], "repository": "x/y"}}
                )

                self.assertEqual(state["strategy"], "analogical")
                self.assertIn("final_estimation", state)
                self.assertEqual(state["final_estimation"]["estimated_hours"], 13.0)
                self.assertEqual(state["final_estimation"]["confidence"], 0.82)
                self.assertIn("Rota analogical escolhida", state["final_estimation"]["justification"])
                analogical_mock.assert_called_once()
                heuristic_mock.assert_not_called()
                combine_mock.assert_not_called()
        finally:
            settings.RAG_MIN_HITS_MAIN = prev_min_hits
            settings.RAG_MIN_SCORE_MAIN = prev_min_score
            eg.vector_store = original_vs

    def test_routes_to_heuristic_ensemble_when_context_insufficient(self):
        prev_min_hits = settings.RAG_MIN_HITS_MAIN
        prev_min_score = settings.RAG_MIN_SCORE_MAIN
        prev_temp = settings.HEURISTIC_ENSEMBLE_TEMPERATURE
        prev_concurrency = getattr(settings, "HEURISTIC_ENSEMBLE_MAX_CONCURRENCY", None)
        settings.RAG_MIN_HITS_MAIN = 2
        settings.RAG_MIN_SCORE_MAIN = 0.75
        settings.HEURISTIC_ENSEMBLE_TEMPERATURE = 0.6
        settings.HEURISTIC_ENSEMBLE_MAX_CONCURRENCY = 1

        original_vs = eg.vector_store
        eg.vector_store = _NoTechVectorStore()
        try:
            heuristic_outputs = [
                {"estimated_hours": 6, "confidence": 0.6, "justification": "h1", "percentile": "p25"},
                {"estimated_hours": 8, "confidence": 0.7, "justification": "h2", "percentile": "p50"},
                {"estimated_hours": 10, "confidence": 0.8, "justification": "h3", "percentile": "p75"},
                {"estimated_hours": 12, "confidence": 0.75, "justification": "h4", "percentile": "p100"},
            ]

            with patch.object(eg, "Retriever", _FakeRetrieverInsufficient), patch.object(
                eg, "run_analogical"
            ) as analogical_mock, patch.object(
                eg, "run_heuristic", side_effect=heuristic_outputs
            ) as heuristic_mock, patch.object(
                eg,
                "combine_heuristic_estimations",
                return_value={
                    "estimated_hours": 8,
                    "confidence": 0.77,
                    "justification": "consolidado",
                },
            ) as combine_mock:
                state = eg.estimation_graph.invoke(
                    {"issue": {"title": "A", "description": "B", "labels": [], "repository": "x/y"}}
                )

                self.assertEqual(state["strategy"], "heuristic_ensemble")
                self.assertEqual(len(state["heuristic_candidates"]), 4)
                self.assertEqual(state["final_estimation"]["estimated_hours"], 8)
                self.assertEqual(state["final_estimation"]["confidence"], 0.77)
                self.assertEqual(heuristic_mock.call_count, 4)
                analogical_mock.assert_not_called()
                combine_mock.assert_called_once()
                estimations = combine_mock.call_args.kwargs["estimations"]
                self.assertEqual(
                    [e["source"] for e in estimations],
                    ["heuristic_1", "heuristic_2", "heuristic_3", "heuristic_4"],
                )
        finally:
            settings.RAG_MIN_HITS_MAIN = prev_min_hits
            settings.RAG_MIN_SCORE_MAIN = prev_min_score
            settings.HEURISTIC_ENSEMBLE_TEMPERATURE = prev_temp
            settings.HEURISTIC_ENSEMBLE_MAX_CONCURRENCY = prev_concurrency
            eg.vector_store = original_vs


if __name__ == "__main__":
    unittest.main()
