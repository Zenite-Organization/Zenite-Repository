import unittest

from ai.agents.supervisor_agent import combine_multi_agent_estimations


class TestSupervisorHeuristicEnsemble(unittest.TestCase):
    def test_supervisor_uses_calibrated_estimation_as_primary_numeric_signal(self):
        out = combine_multi_agent_estimations(
            issue_context={"title": "A"},
            strategy="analogical_consensus",
            analogical={"estimated_hours": 8, "confidence": 0.8, "retrieval_route": "analogical_primary"},
            heuristic_candidates=[
                {"mode": "scope", "size_bucket": "M", "bucket_rank": 3, "confidence": 0.8},
                {"mode": "complexity", "size_bucket": "M", "bucket_rank": 3, "confidence": 0.8},
            ],
            complexity_review={"bucket_delta": 1, "confidence": 0.8},
            agile_guard_review={"fit_status": "healthy", "bucket_delta": 0, "confidence": 0.8},
            critic_review={"risk_of_underestimation": 0.2, "risk_of_overestimation": 0.1},
            calibrated_estimation={
                "base_hours": 8.0,
                "adjusted_hours": 9.2,
                "adjustment_delta": 1.2,
                "min_hours": 7.0,
                "max_hours": 11.0,
                "range_label": "9-12h",
                "range_index": 4,
                "range_min_hours": 9,
                "range_max_hours": 12,
                "display_hours": 11,
                "size_bucket": "M",
                "bucket_rank": 3,
                "calibration_source": "focused_neighbors",
                "finalization_mode": "analogical_calibrated",
                "dominant_strategy": "analogical_consensus",
                "selected_model": "analogical_calibrated",
                "base_confidence": 0.72,
                "retrieval_route": "analogical_primary",
            },
        )
        self.assertEqual(out["estimated_hours"], 11)
        self.assertEqual(out["estimated_hours_raw"], 9.2)
        self.assertEqual(out["range_label"], "9-12h")
        self.assertEqual(out["base_hours"], 8.0)
        self.assertEqual(out["adjustment_delta"], 1.2)
        self.assertEqual(out["finalization_mode"], "analogical_calibrated")
        self.assertEqual(out["dominant_strategy"], "analogical_consensus")
        self.assertIn("9-12h", out["justification"])
        self.assertIn("analogical_calibrated", out["analysis_justification"])

    def test_confidence_penalized_on_high_bucket_spread(self):
        narrow = combine_multi_agent_estimations(
            issue_context={"title": "A"},
            strategy="multiagent_heuristic_consensus",
            analogical={"estimated_hours": 8, "confidence": 0.8},
            heuristic_candidates=[
                {"mode": "scope", "size_bucket": "M", "bucket_rank": 3, "confidence": 0.8},
                {"mode": "complexity", "size_bucket": "M", "bucket_rank": 3, "confidence": 0.8},
            ],
            complexity_review={"bucket_delta": 0, "confidence": 0.8},
            agile_guard_review={"fit_status": "healthy", "bucket_delta": 0, "confidence": 0.8},
            critic_review=None,
            calibrated_estimation={
                "base_hours": 8.0,
                "adjusted_hours": 8.0,
                "min_hours": 7.0,
                "max_hours": 9.0,
                "range_label": "6-9h",
                "range_index": 3,
                "range_min_hours": 6,
                "range_max_hours": 9,
                "display_hours": 8,
                "size_bucket": "M",
                "bucket_rank": 3,
                "finalization_mode": "heuristic_bucket_calibrated",
                "dominant_strategy": "multiagent_heuristic_consensus",
                "selected_model": "heuristic_bucket_calibrated",
                "base_confidence": 0.6,
                "retrieval_route": "analogical_weak",
            },
        )
        wide = combine_multi_agent_estimations(
            issue_context={"title": "A"},
            strategy="multiagent_heuristic_consensus",
            analogical={"estimated_hours": 8, "confidence": 0.8},
            heuristic_candidates=[
                {"mode": "scope", "size_bucket": "XS", "bucket_rank": 1, "confidence": 0.8},
                {"mode": "complexity", "size_bucket": "XL", "bucket_rank": 5, "confidence": 0.8},
            ],
            complexity_review={"bucket_delta": 0, "confidence": 0.8},
            agile_guard_review={"fit_status": "healthy", "bucket_delta": 0, "confidence": 0.8},
            critic_review=None,
            calibrated_estimation={
                "base_hours": 8.0,
                "adjusted_hours": 8.0,
                "min_hours": 7.0,
                "max_hours": 9.0,
                "range_label": "6-9h",
                "range_index": 3,
                "range_min_hours": 6,
                "range_max_hours": 9,
                "display_hours": 8,
                "size_bucket": "M",
                "bucket_rank": 3,
                "finalization_mode": "heuristic_bucket_calibrated",
                "dominant_strategy": "multiagent_heuristic_consensus",
                "selected_model": "heuristic_bucket_calibrated",
                "base_confidence": 0.6,
                "retrieval_route": "analogical_weak",
            },
        )
        self.assertLess(wide["confidence"], narrow["confidence"])

    def test_split_flag_is_preserved_from_agile_guard(self):
        out = combine_multi_agent_estimations(
            issue_context={"title": "A"},
            strategy="multiagent_heuristic_consensus",
            analogical=None,
            heuristic_candidates=[{"mode": "scope", "size_bucket": "L", "bucket_rank": 4, "confidence": 0.7}],
            complexity_review={"bucket_delta": 0, "confidence": 0.7},
            agile_guard_review={
                "fit_status": "oversized",
                "bucket_delta": 1,
                "confidence": 0.8,
                "should_split": True,
                "split_reason": "Issue groups multiple deliverables.",
            },
            critic_review=None,
            calibrated_estimation={
                "base_hours": 18.0,
                "adjusted_hours": 18.0,
                "min_hours": 14.0,
                "max_hours": 22.0,
                "range_label": "18-21h",
                "range_index": 7,
                "range_min_hours": 18,
                "range_max_hours": 21,
                "display_hours": 20,
                "size_bucket": "L",
                "bucket_rank": 4,
                "finalization_mode": "heuristic_bucket_calibrated",
                "dominant_strategy": "multiagent_heuristic_consensus",
                "selected_model": "heuristic_bucket_calibrated",
                "base_confidence": 0.55,
                "retrieval_route": "analogical_weak",
                "should_split": True,
                "split_reason": "Issue groups multiple deliverables.",
            },
        )
        self.assertTrue(out["should_split"])
        self.assertEqual(out["split_reason"], "A issue agrupa múltiplas entregas.")


if __name__ == "__main__":
    unittest.main()
