import unittest

from ai.core.effort_calibration import (
    aggregate_bucket_consensus,
    bounded_adjustment_from_reviews,
    build_calibration_profile,
    calibrate_bucket_rank_to_hours,
    weighted_neighbor_estimate,
)


class TestEffortCalibration(unittest.TestCase):
    def test_bucket_calibration_prefers_focused_neighbors(self):
        issue = {"issue_type": "bug", "labels": ["ui"]}
        similar = [
            {"score": 0.92, "total_effort_hours": 3, "issue_type": "bug", "labels": ["ui"]},
            {"score": 0.88, "total_effort_hours": 4, "issue_type": "bug", "labels": ["ui"]},
            {"score": 0.84, "total_effort_hours": 5, "issue_type": "bug", "labels": ["ui"]},
            {"score": 0.70, "total_effort_hours": 20, "issue_type": "feature", "labels": ["backend"]},
        ]
        profile = build_calibration_profile(issue, similar)
        calibrated = calibrate_bucket_rank_to_hours(2, profile)

        self.assertEqual(profile["source"], "focused_neighbors")
        self.assertEqual(calibrated["calibration_source"], "focused_neighbors")
        self.assertLessEqual(calibrated["estimated_hours"], 5.0)

    def test_weighted_neighbor_estimate_tracks_high_similarity_neighbors(self):
        issue = {"issue_type": "bug", "labels": ["auth"]}
        similar = [
            {"score": 0.95, "total_effort_hours": 4, "issue_type": "bug", "labels": ["auth"]},
            {"score": 0.91, "total_effort_hours": 5, "issue_type": "bug", "labels": ["auth"]},
            {"score": 0.89, "total_effort_hours": 6, "issue_type": "bug", "labels": ["auth"]},
            {"score": 0.55, "total_effort_hours": 18, "issue_type": "feature", "labels": ["ui"]},
        ]
        result = weighted_neighbor_estimate(issue, similar)
        consensus = aggregate_bucket_consensus(
            [
                {"bucket_rank": 2, "confidence": 0.7},
                {"bucket_rank": 2, "confidence": 0.8},
                {"bucket_rank": 3, "confidence": 0.6},
            ]
        )

        self.assertLessEqual(result["estimated_hours"], 6.0)
        self.assertGreaterEqual(result["confidence"], 0.5)
        self.assertEqual(consensus["size_bucket"], "S")

    def test_weak_prior_prefers_type_or_label_prior_over_all_neighbors(self):
        issue = {"issue_type": "bug", "labels": ["mobile"]}
        similar = [
            {"score": 0.62, "total_effort_hours": 12, "issue_type": "bug", "labels": ["mobile"]},
            {"score": 0.58, "total_effort_hours": 14, "issue_type": "bug", "labels": ["mobile"]},
            {"score": 0.40, "total_effort_hours": 2, "issue_type": "task", "labels": ["docs"]},
            {"score": 0.39, "total_effort_hours": 3, "issue_type": "task", "labels": ["docs"]},
        ]
        profile = build_calibration_profile(issue, similar)

        calibrated = calibrate_bucket_rank_to_hours(3, profile, calibration_mode="weak_prior")

        self.assertEqual(calibrated["calibration_source"], "focused_prior")
        self.assertGreaterEqual(calibrated["estimated_hours"], 12.0)

    def test_weighted_neighbor_estimate_prefers_focused_prior_over_all_neighbors(self):
        issue = {"issue_type": "bug", "labels": ["mobile"]}
        similar = [
            {"score": 0.62, "total_effort_hours": 12, "issue_type": "bug", "labels": ["mobile"]},
            {"score": 0.58, "total_effort_hours": 14, "issue_type": "bug", "labels": ["mobile"]},
            {"score": 0.57, "total_effort_hours": 25, "issue_type": "feature", "labels": ["backend"]},
            {"score": 0.55, "total_effort_hours": 28, "issue_type": "feature", "labels": ["backend"]},
        ]

        result = weighted_neighbor_estimate(issue, similar, useful_score_threshold=0.68)

        self.assertEqual(result["calibration_source"], "focused_prior")
        self.assertLess(result["estimated_hours"], 20.0)

    def test_bucket_consensus_gives_more_weight_to_scope_and_complexity(self):
        consensus = aggregate_bucket_consensus(
            [
                {"mode": "scope", "bucket_rank": 3, "confidence": 0.8},
                {"mode": "complexity", "bucket_rank": 3, "confidence": 0.8},
                {"mode": "uncertainty", "bucket_rank": 2, "confidence": 0.7},
                {"mode": "agile_fit", "bucket_rank": 2, "confidence": 0.7},
            ]
        )

        self.assertEqual(consensus["size_bucket"], "M")

    def test_bounded_adjustment_requires_real_hidden_complexity_signal(self):
        adjusted = bounded_adjustment_from_reviews(
            base_hours=10,
            complexity_review={"bucket_delta": 1, "confidence": 0.85, "risk_hidden_complexity": 0.72},
            critic_review={"risk_of_underestimation": 0.2, "risk_of_overestimation": 0.2},
        )

        self.assertEqual(adjusted["adjustment_delta"], 0.0)


if __name__ == "__main__":
    unittest.main()
