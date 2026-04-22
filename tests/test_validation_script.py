import json
import unittest

import scripts.validation as validation


class TestValidationScript(unittest.TestCase):
    def test_normalize_estimation_model_includes_soft_hybrid(self):
        self.assertEqual(
            validation.normalize_estimation_model("soft_hybrid_calibrated"),
            "analogical+multiagent_consensus",
        )

    def test_build_validation_payload_includes_calibration_diagnostics(self):
        row = {"id": 123, "project_id": 77, "project_key": "ZEN", "actual_hours": 6.8}
        state = {
            "strategy": "analogical_consensus",
            "rag_context_sufficient": True,
            "rag_stats": {
                "qualified_hits": 3,
                "min_hits": 2,
                "min_score": 0.55,
            },
            "analogical": {
                "estimated_hours": 4.5,
                "min_hours": 3.5,
                "max_hours": 6.0,
                "confidence": 0.61,
                "should_split": False,
                "size_bucket": "S",
                "bucket_rank": 2,
                "calibration_source": "focused_neighbors",
                "retrieval_route": "analogical_primary",
                "retrieval_stats": {
                    "top1_score": 0.96,
                    "top3_avg_score": 0.66,
                    "useful_count": 1,
                    "hours_spread": 4.1,
                    "has_strong_anchor": True,
                    "anchor_score": 0.96,
                    "anchor_overlap": 0.44,
                },
            },
            "heuristic_candidates": [
                {"mode": "scope", "size_bucket": "S", "bucket_rank": 2, "estimated_hours": 5.0, "confidence": 0.7},
                {"mode": "complexity", "size_bucket": "M", "bucket_rank": 3, "estimated_hours": 7.0, "confidence": 0.8},
                {"mode": "uncertainty", "size_bucket": "M", "bucket_rank": 3, "estimated_hours": 6.0, "confidence": 0.6},
                {"mode": "agile_fit", "size_bucket": "S", "bucket_rank": 2, "estimated_hours": 4.0, "confidence": 0.75},
            ],
            "complexity_review": {
                "bucket_delta": 1,
                "confidence": 0.67,
                "should_split": False,
            },
            "agile_guard_review": {
                "bucket_delta": 0,
                "fit_status": "healthy",
                "confidence": 0.73,
                "should_split": False,
            },
            "critic_review": {
                "risk_of_underestimation": 0.41,
                "risk_of_overestimation": 0.22,
            },
            "calibrated_estimation": {
                "size_bucket": "M",
                "bucket_rank": 3,
                "heuristic_size_bucket": "M",
                "heuristic_bucket_rank": 3,
                "analogical_size_bucket": "S",
                "analogical_bucket_rank": 2,
                "base_hours": 5.6,
                "adjusted_hours": 6.2,
                "adjustment_delta": 0.6,
                "min_hours": 4.0,
                "max_hours": 8.0,
                "calibration_source": "focused_neighbors",
                "finalization_mode": "analogical_calibrated",
                "dominant_strategy": "analogical_consensus",
                "selected_model": "analogical_calibrated",
                "meta_applied": True,
                "meta_hours": 5.9,
                "meta_min_hours": 4.4,
                "meta_max_hours": 7.7,
                "meta_confidence": 0.66,
                "meta_source": "meta_linear+project_issue",
                "meta_prior_source": "project_issue",
                "meta_prior_count": 9,
                "meta_blend_weight": 0.34,
                "meta_model_version": "meta_calibrator_v1",
                "range_label": "6-9h",
                "range_index": 3,
                "range_min_hours": 6,
                "range_max_hours": 9,
                "display_hours": 8,
            },
            "execution_metrics": {
                "workflow_latency_ms": 900,
                "primary_reviews_latency_ms": 640,
                "analogical_latency_ms": 180,
                "heuristic_ensemble_latency_ms": 250,
                "complexity_review_latency_ms": 110,
                "agile_guard_latency_ms": 100,
                "critic_latency_ms": 70,
                "calibration_latency_ms": 40,
                "supervisor_latency_ms": 60,
            },
        }
        final_estimation = {
            "estimated_hours": 8,
            "estimated_hours_raw": 6.2,
            "min_hours": 6,
            "max_hours": 9,
            "confidence": 0.7,
            "justification": "justificativa amigavel",
            "user_justification": "justificativa amigavel",
            "analysis_justification": "consensus result",
            "estimation_model": "analogical+multiagent_consensus",
            "selected_model": "analogical_calibrated",
            "dominant_strategy": "analogical_consensus",
            "finalization_mode": "analogical_calibrated",
            "calibration_source": "focused_neighbors",
            "size_bucket": "M",
            "bucket_rank": 3,
            "base_hours": 5.6,
            "adjusted_hours": 6.2,
            "adjustment_delta": 0.6,
            "range_label": "6-9h",
            "range_index": 3,
            "range_min_hours": 6,
            "range_max_hours": 9,
            "display_hours": 8,
            "retrieval_route": "analogical_primary",
            "retrieval_stats": state["analogical"]["retrieval_stats"],
            "calibrated_estimation": state["calibrated_estimation"],
            "agent_trace": {
                "analogical": state["analogical"],
                "heuristic_candidates": state["heuristic_candidates"],
                "complexity_review": state["complexity_review"],
                "agile_guard_review": state["agile_guard_review"],
                "critic_review": state["critic_review"],
                "calibrated_estimation": state["calibrated_estimation"],
            },
            "execution_trace": state["execution_metrics"],
            "should_split": False,
            "split_reason": None,
        }
        usage = {
            "predicted_llm_prompt_tokens": 10,
            "predicted_llm_completion_tokens": 4,
            "predicted_llm_total_tokens": 14,
            "predicted_rag_embedding_tokens": 8,
            "predicted_total_tokens": 22,
        }

        payload = validation.build_validation_payload(
            row=row,
            state=state,
            final_estimation=final_estimation,
            usage=usage,
            run_id="run-1",
            model_version="11.1",
            service_latency_ms=1234,
        )

        self.assertEqual(payload["estimation_model"], "analogical+multiagent_consensus")
        self.assertEqual(payload["selected_model"], "analogical_calibrated")
        self.assertEqual(payload["dominant_strategy"], "analogical_consensus")
        self.assertEqual(payload["finalization_mode"], "analogical_calibrated")
        self.assertEqual(payload["retrieval_route"], "analogical_primary")
        self.assertEqual(payload["top1_score"], 0.96)
        self.assertEqual(payload["anchor_score"], 0.96)
        self.assertEqual(payload["anchor_overlap"], 0.44)
        self.assertEqual(payload["size_bucket"], "M")
        self.assertEqual(payload["bucket_rank"], 3)
        self.assertEqual(payload["predicted_hours"], 8)
        self.assertEqual(payload["predicted_hours_raw"], 6.2)
        self.assertEqual(payload["predicted_range_label"], "6-9h")
        self.assertEqual(payload["predicted_range_index"], 3)
        self.assertEqual(payload["actual_range_label"], "6-9h")
        self.assertEqual(payload["actual_range_index"], 3)
        self.assertEqual(payload["range_hit"], 1)
        self.assertEqual(payload["range_distance"], 0)
        self.assertEqual(payload["base_hours"], 5.6)
        self.assertEqual(payload["adjusted_hours"], 6.2)
        self.assertEqual(payload["adjustment_delta"], 0.6)
        self.assertEqual(payload["meta_applied"], 1)
        self.assertEqual(payload["meta_hours"], 5.9)
        self.assertEqual(payload["meta_source"], "meta_linear+project_issue")
        self.assertEqual(payload["meta_prior_source"], "project_issue")
        self.assertEqual(payload["meta_prior_count"], 9)
        self.assertEqual(payload["calibration_source"], "focused_neighbors")
        self.assertEqual(payload["analogical_hours"], 4.5)
        self.assertEqual(payload["heuristic_scope_hours"], 5.0)
        self.assertEqual(payload["heuristic_scope_bucket"], "S")
        self.assertEqual(payload["complexity_bucket_delta"], 1)
        self.assertEqual(payload["agile_guard_fit_status"], "healthy")
        self.assertEqual(payload["critic_risk_underestimation"], 0.41)
        self.assertEqual(payload["service_latency_ms"], 1234)
        self.assertEqual(payload["workflow_latency_ms"], 900)
        self.assertEqual(payload["calibration_latency_ms"], 40)

        trace = json.loads(payload["decision_trace_json"])
        self.assertEqual(trace["selected_model"], "analogical_calibrated")
        self.assertEqual(trace["user_justification"], "justificativa amigavel")
        self.assertEqual(trace["analysis_justification"], "consensus result")
        self.assertEqual(trace["calibrated_estimation"]["base_hours"], 5.6)
        self.assertEqual(trace["calibrated_estimation"]["adjusted_hours"], 6.2)
        self.assertEqual(trace["calibrated_estimation"]["meta_source"], "meta_linear+project_issue")

    def test_build_upsert_sql_includes_optional_diagnostic_columns(self):
        sql = str(
            validation.build_upsert_sql(
                {
                    "selected_model",
                    "dominant_strategy",
                    "finalization_mode",
                    "retrieval_route",
                    "top1_score",
                    "base_hours",
                    "adjusted_hours",
                    "calibration_source",
                    "execution_trace_json",
                }
            )
        )

        self.assertIn("`selected_model`", sql)
        self.assertIn("`dominant_strategy`", sql)
        self.assertIn("`finalization_mode`", sql)
        self.assertIn("`retrieval_route`", sql)
        self.assertIn("`base_hours`", sql)
        self.assertIn("`adjusted_hours`", sql)
        self.assertIn("`calibration_source`", sql)
        self.assertIn("`execution_trace_json`", sql)

    def test_build_validation_payload_can_disable_verbose_trace_jsons(self):
        payload = validation.build_validation_payload(
            row={"id": 1, "project_id": 1, "project_key": "ZEN"},
            state={"strategy": "multiagent_heuristic_consensus"},
            final_estimation={
                "estimated_hours": 5,
                "estimated_hours_raw": 5.0,
                "range_label": "3-6h",
                "range_index": 2,
                "range_min_hours": 3,
                "range_max_hours": 6,
                "confidence": 0.6,
                "justification": "ok",
                "user_justification": "ok",
                "analysis_justification": "debug ok",
                "estimation_model": "multiagent_heuristic_consensus",
            },
            usage={},
            run_id="run-1",
            model_version="11.1",
            service_latency_ms=100,
            save_verbose_trace=False,
        )

        self.assertIsNone(payload["decision_trace_json"])
        self.assertIsNone(payload["agent_trace_json"])
        self.assertIsNone(payload["execution_trace_json"])


if __name__ == "__main__":
    unittest.main()
