import unittest
from unittest.mock import patch

from ai.workflows import estimation_graph as eg
from config.settings import settings
from ai.dtos.issues_estimation_dto import IssueEstimationDTO


class _FakeRetrieverSufficient:
    def __init__(self, _vs):
        pass

    def get_similar_issues(self, _issue):
        return [
            {"id": "1", "score": 0.92, "total_effort_hours": 5, "title": "A", "issue_type": "bug"},
            {"id": "2", "score": 0.88, "total_effort_hours": 6, "title": "B", "issue_type": "bug"},
            {"id": "3", "score": 0.84, "total_effort_hours": 8, "title": "C", "issue_type": "bug"},
        ]


class _FakeRetrieverInsufficient:
    def __init__(self, _vs):
        pass

    def get_similar_issues(self, _issue):
        return [{"id": "1", "score": 0.5, "total_effort_hours": 5, "title": "A", "issue_type": "bug"}]


class _FakeRetrieverModerate:
    def __init__(self, _vs):
        pass

    def get_similar_issues(self, _issue):
        return [
            {"id": "1", "score": 0.68, "total_effort_hours": 14, "title": "A", "issue_type": "bug"},
            {"id": "2", "score": 0.63, "total_effort_hours": 12, "title": "B", "issue_type": "bug"},
            {"id": "3", "score": 0.42, "total_effort_hours": 3, "title": "C", "issue_type": "task"},
        ]


class _NoTechVectorStore:
    pass


class TestEstimationGraphRouting(unittest.TestCase):
    def test_routes_to_analogical_calibrated_when_retrieval_is_strong(self):
        prev_min_hits = settings.RAG_MIN_HITS_MAIN
        prev_min_score = settings.RAG_MIN_SCORE_MAIN
        prev_heuristic_concurrency = settings.HEURISTIC_ENSEMBLE_MAX_CONCURRENCY
        prev_primary_concurrency = settings.PRIMARY_AGENT_MAX_CONCURRENCY
        settings.RAG_MIN_HITS_MAIN = 2
        settings.RAG_MIN_SCORE_MAIN = 0.75
        settings.HEURISTIC_ENSEMBLE_MAX_CONCURRENCY = 1
        settings.PRIMARY_AGENT_MAX_CONCURRENCY = 1

        original_vs = eg.vector_store
        eg.vector_store = _NoTechVectorStore()
        try:
            with patch.object(eg, "Retriever", _FakeRetrieverSufficient), patch.object(
                eg,
                "run_analogical",
                return_value={
                    "estimated_hours": 6.0,
                    "min_hours": 5.0,
                    "max_hours": 7.0,
                    "confidence": 0.82,
                    "size_bucket": "S",
                    "bucket_rank": 2,
                    "justification": "Analogical base.",
                    "retrieval_route": "analogical_primary",
                    "retrieval_stats": {"top1_score": 0.92, "route": "analogical_primary"},
                    "calibration_source": "focused_neighbors",
                },
            ) as analogical_mock, patch.object(
                eg,
                "run_heuristic",
                return_value={"size_bucket": "S", "bucket_rank": 2, "confidence": 0.6, "justification": "h"},
            ) as heuristic_mock, patch.object(
                eg,
                "run_complexity_review",
                return_value={"bucket_delta": 0, "confidence": 0.7, "justification": "c"},
            ), patch.object(
                eg,
                "run_agile_guard",
                return_value={"fit_status": "healthy", "bucket_delta": 0, "confidence": 0.75, "justification": "a"},
            ), patch.object(
                eg,
                "run_estimation_critic",
                return_value={
                    "risk_of_underestimation": 0.2,
                    "risk_of_overestimation": 0.1,
                    "contradictions": [],
                    "hidden_complexities": [],
                    "strongest_signal": "analogical",
                    "recommendation": "follow analogical",
                },
            ):
                state = eg.estimation_graph.invoke(
                    {"issue": {"title": "A", "description": "B", "labels": [], "repository": "x/y", "issue_type": "bug"}}
                )

                self.assertEqual(state["calibrated_estimation"]["finalization_mode"], "analogical_calibrated")
                self.assertEqual(state["final_estimation"]["estimation_model"], "analogical+multiagent_consensus")
                self.assertEqual(state["final_estimation"]["selected_model"], "analogical_calibrated")
                analogical_mock.assert_called_once()
                self.assertEqual(heuristic_mock.call_count, 4)
        finally:
            settings.RAG_MIN_HITS_MAIN = prev_min_hits
            settings.RAG_MIN_SCORE_MAIN = prev_min_score
            settings.HEURISTIC_ENSEMBLE_MAX_CONCURRENCY = prev_heuristic_concurrency
            settings.PRIMARY_AGENT_MAX_CONCURRENCY = prev_primary_concurrency
            eg.vector_store = original_vs

    def test_routes_to_heuristic_bucket_calibrated_when_retrieval_is_weak(self):
        prev_min_hits = settings.RAG_MIN_HITS_MAIN
        prev_min_score = settings.RAG_MIN_SCORE_MAIN
        prev_heuristic_concurrency = settings.HEURISTIC_ENSEMBLE_MAX_CONCURRENCY
        prev_primary_concurrency = settings.PRIMARY_AGENT_MAX_CONCURRENCY
        settings.RAG_MIN_HITS_MAIN = 2
        settings.RAG_MIN_SCORE_MAIN = 0.75
        settings.HEURISTIC_ENSEMBLE_MAX_CONCURRENCY = 1
        settings.PRIMARY_AGENT_MAX_CONCURRENCY = 1

        original_vs = eg.vector_store
        eg.vector_store = _NoTechVectorStore()
        try:
            with patch.object(eg, "Retriever", _FakeRetrieverInsufficient), patch.object(
                eg,
                "run_analogical",
                return_value={
                    "estimated_hours": 11.0,
                    "confidence": 0.3,
                    "justification": "Analogical weak.",
                    "retrieval_route": "analogical_weak",
                    "retrieval_stats": {"top1_score": 0.5, "route": "analogical_weak"},
                },
            ) as analogical_mock, patch.object(
                eg,
                "run_heuristic",
                side_effect=[
                    {"size_bucket": "S", "bucket_rank": 2, "confidence": 0.6, "justification": "h1"},
                    {"size_bucket": "M", "bucket_rank": 3, "confidence": 0.7, "justification": "h2"},
                    {"size_bucket": "M", "bucket_rank": 3, "confidence": 0.8, "justification": "h3"},
                    {"size_bucket": "S", "bucket_rank": 2, "confidence": 0.75, "justification": "h4"},
                ],
            ) as heuristic_mock, patch.object(
                eg,
                "run_complexity_review",
                return_value={"bucket_delta": 0, "confidence": 0.7, "justification": "c"},
            ), patch.object(
                eg,
                "run_agile_guard",
                return_value={"fit_status": "healthy", "bucket_delta": 0, "confidence": 0.72, "justification": "a"},
            ), patch.object(
                eg,
                "run_estimation_critic",
                return_value={
                    "risk_of_underestimation": 0.4,
                    "risk_of_overestimation": 0.2,
                    "contradictions": [],
                    "hidden_complexities": [],
                    "strongest_signal": "heuristics",
                    "recommendation": "follow calibrated heuristic",
                },
            ):
                state = eg.estimation_graph.invoke(
                    {"issue": {"title": "A", "description": "B", "labels": [], "repository": "x/y", "issue_type": "bug"}}
                )

                self.assertEqual(state["calibrated_estimation"]["finalization_mode"], "heuristic_bucket_calibrated")
                self.assertEqual(state["final_estimation"]["estimation_model"], "multiagent_heuristic_consensus")
                self.assertEqual(state["final_estimation"]["selected_model"], "heuristic_bucket_calibrated")
                analogical_mock.assert_called_once()
                self.assertEqual(heuristic_mock.call_count, 4)
        finally:
            settings.RAG_MIN_HITS_MAIN = prev_min_hits
            settings.RAG_MIN_SCORE_MAIN = prev_min_score
            settings.HEURISTIC_ENSEMBLE_MAX_CONCURRENCY = prev_heuristic_concurrency
            settings.PRIMARY_AGENT_MAX_CONCURRENCY = prev_primary_concurrency
            eg.vector_store = original_vs

    def test_run_estimation_flow_returns_token_usage_summary(self):
        prev_min_hits = settings.RAG_MIN_HITS_MAIN
        prev_min_score = settings.RAG_MIN_SCORE_MAIN
        prev_heuristic_concurrency = settings.HEURISTIC_ENSEMBLE_MAX_CONCURRENCY
        prev_primary_concurrency = settings.PRIMARY_AGENT_MAX_CONCURRENCY
        settings.RAG_MIN_HITS_MAIN = 2
        settings.RAG_MIN_SCORE_MAIN = 0.75
        settings.HEURISTIC_ENSEMBLE_MAX_CONCURRENCY = 1
        settings.PRIMARY_AGENT_MAX_CONCURRENCY = 1

        original_vs = eg.vector_store
        eg.vector_store = _NoTechVectorStore()
        try:
            dto = IssueEstimationDTO(
                issue_number=1,
                repository="x/y",
                title="A",
                description="B",
                labels=[],
                assignees=[],
                state="open",
                is_open=False,
                comments_count=0,
                age_in_days=0,
                author_login="bot",
                author_role="NONE",
                repo_language=None,
                repo_size=None,
                issue_type="bug",
            )

            with patch.object(eg, "Retriever", _FakeRetrieverInsufficient), patch.object(
                eg,
                "run_analogical",
                return_value={
                    "estimated_hours": 11.0,
                    "confidence": 0.3,
                    "justification": "Analogical weak.",
                    "retrieval_route": "analogical_weak",
                    "token_usage": {"prompt_tokens": 9, "completion_tokens": 4, "total_tokens": 13},
                },
            ), patch.object(
                eg,
                "run_heuristic",
                side_effect=[
                    {"size_bucket": "S", "bucket_rank": 2, "confidence": 0.6, "justification": "h1", "token_usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}},
                    {"size_bucket": "M", "bucket_rank": 3, "confidence": 0.7, "justification": "h2", "token_usage": {"prompt_tokens": 11, "completion_tokens": 6, "total_tokens": 17}},
                    {"size_bucket": "M", "bucket_rank": 3, "confidence": 0.8, "justification": "h3", "token_usage": {"prompt_tokens": 12, "completion_tokens": 7, "total_tokens": 19}},
                    {"size_bucket": "S", "bucket_rank": 2, "confidence": 0.75, "justification": "h4", "token_usage": {"prompt_tokens": 13, "completion_tokens": 8, "total_tokens": 21}},
                ],
            ), patch.object(
                eg,
                "run_complexity_review",
                return_value={"bucket_delta": 0, "confidence": 0.7, "justification": "c", "token_usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}},
            ), patch.object(
                eg,
                "run_agile_guard",
                return_value={"fit_status": "healthy", "bucket_delta": 0, "confidence": 0.72, "justification": "a", "token_usage": {"prompt_tokens": 6, "completion_tokens": 3, "total_tokens": 9}},
            ), patch.object(
                eg,
                "run_estimation_critic",
                return_value={
                    "risk_of_underestimation": 0.4,
                    "risk_of_overestimation": 0.2,
                    "contradictions": [],
                    "hidden_complexities": [],
                    "strongest_signal": "heuristics",
                    "recommendation": "follow calibrated heuristic",
                    "token_usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
                },
            ):
                state = eg.run_estimation_flow(dto)

                usage = state.get("token_usage_summary")
                self.assertIsInstance(usage, dict)
                self.assertEqual(usage["predicted_llm_prompt_tokens"], 9 + 10 + 11 + 12 + 13 + 7 + 6 + 5)
                self.assertEqual(usage["predicted_llm_completion_tokens"], 4 + 5 + 6 + 7 + 8 + 3 + 3 + 2)
                self.assertEqual(usage["predicted_llm_total_tokens"], 13 + 15 + 17 + 19 + 21 + 10 + 9 + 7)
                self.assertEqual(usage["predicted_rag_embedding_tokens"], 0)
                self.assertEqual(usage["predicted_total_tokens"], usage["predicted_llm_total_tokens"])
        finally:
            settings.RAG_MIN_HITS_MAIN = prev_min_hits
            settings.RAG_MIN_SCORE_MAIN = prev_min_score
            settings.HEURISTIC_ENSEMBLE_MAX_CONCURRENCY = prev_heuristic_concurrency
            settings.PRIMARY_AGENT_MAX_CONCURRENCY = prev_primary_concurrency
            eg.vector_store = original_vs

    def test_routes_to_soft_hybrid_when_weak_route_still_has_moderate_signal(self):
        prev_min_hits = settings.RAG_MIN_HITS_MAIN
        prev_min_score = settings.RAG_MIN_SCORE_MAIN
        prev_heuristic_concurrency = settings.HEURISTIC_ENSEMBLE_MAX_CONCURRENCY
        prev_primary_concurrency = settings.PRIMARY_AGENT_MAX_CONCURRENCY
        settings.RAG_MIN_HITS_MAIN = 2
        settings.RAG_MIN_SCORE_MAIN = 0.75
        settings.HEURISTIC_ENSEMBLE_MAX_CONCURRENCY = 1
        settings.PRIMARY_AGENT_MAX_CONCURRENCY = 1

        original_vs = eg.vector_store
        eg.vector_store = _NoTechVectorStore()
        try:
            with patch.object(eg, "Retriever", _FakeRetrieverModerate), patch.object(
                eg,
                "run_analogical",
                return_value={
                    "estimated_hours": 14.0,
                    "min_hours": 12.0,
                    "max_hours": 16.0,
                    "confidence": 0.7,
                    "size_bucket": "L",
                    "bucket_rank": 4,
                    "justification": "Analogical moderate.",
                    "retrieval_route": "analogical_soft_signal",
                    "retrieval_stats": {"top1_score": 0.68, "top3_avg_score": 0.57, "useful_count": 0, "anchor_overlap": 0.24, "route": "analogical_soft_signal"},
                    "calibration_source": "all_neighbors",
                },
            ), patch.object(
                eg,
                "run_heuristic",
                side_effect=[
                    {"size_bucket": "S", "bucket_rank": 2, "confidence": 0.6, "justification": "h1"},
                    {"size_bucket": "M", "bucket_rank": 3, "confidence": 0.8, "justification": "h2"},
                    {"size_bucket": "S", "bucket_rank": 2, "confidence": 0.6, "justification": "h3"},
                    {"size_bucket": "S", "bucket_rank": 2, "confidence": 0.7, "justification": "h4"},
                ],
            ), patch.object(
                eg,
                "run_complexity_review",
                return_value={"bucket_delta": 0, "confidence": 0.8, "risk_hidden_complexity": 0.7, "justification": "c"},
            ), patch.object(
                eg,
                "run_agile_guard",
                return_value={"fit_status": "healthy", "bucket_delta": 0, "confidence": 0.8, "justification": "a"},
            ), patch.object(
                eg,
                "run_estimation_critic",
                return_value={
                    "risk_of_underestimation": 0.4,
                    "risk_of_overestimation": 0.1,
                    "contradictions": [],
                    "hidden_complexities": [],
                    "strongest_signal": "analogical moderate signal",
                    "recommendation": "blend analogical and heuristic",
                },
            ):
                state = eg.estimation_graph.invoke(
                    {"issue": {"title": "A", "description": "B", "labels": ["mobile"], "repository": "x/y", "issue_type": "bug"}}
                )

                self.assertEqual(state["calibrated_estimation"]["finalization_mode"], "soft_hybrid_calibrated")
                self.assertEqual(state["final_estimation"]["estimation_model"], "analogical+multiagent_consensus")
                self.assertEqual(state["final_estimation"]["selected_model"], "soft_hybrid_calibrated")
                self.assertGreater(state["final_estimation"]["estimated_hours"], 8.0)
        finally:
            settings.RAG_MIN_HITS_MAIN = prev_min_hits
            settings.RAG_MIN_SCORE_MAIN = prev_min_score
            settings.HEURISTIC_ENSEMBLE_MAX_CONCURRENCY = prev_heuristic_concurrency
            settings.PRIMARY_AGENT_MAX_CONCURRENCY = prev_primary_concurrency
            eg.vector_store = original_vs

    def test_does_not_route_to_soft_hybrid_when_only_bucket_gap_is_high(self):
        prev_min_hits = settings.RAG_MIN_HITS_MAIN
        prev_min_score = settings.RAG_MIN_SCORE_MAIN
        prev_heuristic_concurrency = settings.HEURISTIC_ENSEMBLE_MAX_CONCURRENCY
        prev_primary_concurrency = settings.PRIMARY_AGENT_MAX_CONCURRENCY
        settings.RAG_MIN_HITS_MAIN = 2
        settings.RAG_MIN_SCORE_MAIN = 0.75
        settings.HEURISTIC_ENSEMBLE_MAX_CONCURRENCY = 1
        settings.PRIMARY_AGENT_MAX_CONCURRENCY = 1

        original_vs = eg.vector_store
        eg.vector_store = _NoTechVectorStore()
        try:
            with patch.object(eg, "Retriever", _FakeRetrieverModerate), patch.object(
                eg,
                "run_analogical",
                return_value={
                    "estimated_hours": 24.0,
                    "min_hours": 18.0,
                    "max_hours": 28.0,
                    "confidence": 0.55,
                    "size_bucket": "XL",
                    "bucket_rank": 5,
                    "justification": "Analogical weak but numerically high.",
                    "retrieval_route": "analogical_weak",
                    "retrieval_stats": {"top1_score": 0.56, "top3_avg_score": 0.55, "useful_count": 0, "anchor_overlap": 0.08, "route": "analogical_weak"},
                    "calibration_source": "all_neighbors",
                },
            ), patch.object(
                eg,
                "run_heuristic",
                side_effect=[
                    {"size_bucket": "S", "bucket_rank": 2, "confidence": 0.8, "justification": "h1"},
                    {"size_bucket": "S", "bucket_rank": 2, "confidence": 0.8, "justification": "h2"},
                    {"size_bucket": "S", "bucket_rank": 2, "confidence": 0.7, "justification": "h3"},
                    {"size_bucket": "S", "bucket_rank": 2, "confidence": 0.75, "justification": "h4"},
                ],
            ), patch.object(
                eg,
                "run_complexity_review",
                return_value={"bucket_delta": 0, "confidence": 0.8, "risk_hidden_complexity": 0.7, "justification": "c"},
            ), patch.object(
                eg,
                "run_agile_guard",
                return_value={"fit_status": "healthy", "bucket_delta": 0, "confidence": 0.8, "justification": "a"},
            ), patch.object(
                eg,
                "run_estimation_critic",
                return_value={
                    "risk_of_underestimation": 0.2,
                    "risk_of_overestimation": 0.2,
                    "contradictions": [],
                    "hidden_complexities": [],
                    "strongest_signal": "heuristic",
                    "recommendation": "stay with heuristic",
                },
            ):
                state = eg.estimation_graph.invoke(
                    {"issue": {"title": "A", "description": "B", "labels": ["mobile"], "repository": "x/y", "issue_type": "bug"}}
                )

                self.assertEqual(state["calibrated_estimation"]["finalization_mode"], "heuristic_bucket_calibrated")
                self.assertEqual(state["final_estimation"]["selected_model"], "heuristic_bucket_calibrated")
        finally:
            settings.RAG_MIN_HITS_MAIN = prev_min_hits
            settings.RAG_MIN_SCORE_MAIN = prev_min_score
            settings.HEURISTIC_ENSEMBLE_MAX_CONCURRENCY = prev_heuristic_concurrency
            settings.PRIMARY_AGENT_MAX_CONCURRENCY = prev_primary_concurrency
            eg.vector_store = original_vs


if __name__ == "__main__":
    unittest.main()
