import unittest

from ai.agents.analogical_agent import _compute_retrieval_stats
from ai.workflows.estimation_graph import normalize_estimation


class TestAnalogicalRouting(unittest.TestCase):
    def test_normalize_estimation_preserves_analogical_routing_metadata(self):
        raw = {
            "estimated_hours": 4.0,
            "confidence": 0.77,
            "justification": "Analogical result.",
            "retrieval_route": "analogical_primary",
            "retrieval_stats": {
                "route": "analogical_primary",
                "top1_score": 0.96,
                "anchor_score": 0.96,
                "anchor_overlap": 0.44,
                "useful_count": 1,
            },
            "latency_ms": 321,
        }

        normalized = normalize_estimation(raw, fallback_mode="analogical")

        self.assertEqual(normalized["retrieval_route"], "analogical_primary")
        self.assertEqual(normalized["retrieval_stats"]["top1_score"], 0.96)
        self.assertEqual(normalized["retrieval_stats"]["anchor_score"], 0.96)
        self.assertEqual(normalized["latency_ms"], 321)

    def test_strong_anchor_promotes_analogical_primary_even_with_noisy_tail(self):
        issue = {
            "title": "Fix crash when saving draft",
            "description": "The editor crashes whenever the user saves a draft.",
        }
        similar_issues = [
            {
                "title": "Fix crash when saving draft in editor",
                "description": "Crash reproduced while saving a draft from the editor screen.",
                "score": 0.96,
                "total_effort_hours": 4,
            },
            {
                "title": "Refactor authentication service for SSO rollout",
                "description": "Broader auth refactor with multiple touchpoints.",
                "score": 0.57,
                "total_effort_hours": 18,
            },
            {
                "title": "Build dashboard KPI charts",
                "description": "New feature with unrelated UI and backend changes.",
                "score": 0.51,
                "total_effort_hours": 22,
            },
        ]

        stats = _compute_retrieval_stats(issue, similar_issues)

        self.assertTrue(stats["has_strong_anchor"])
        self.assertEqual(stats["useful_count"], 1)
        self.assertLess(stats["top3_avg_score"], 0.72)
        self.assertGreater(stats["hours_spread"], 3.0)
        self.assertEqual(stats["route"], "analogical_primary")

    def test_soft_signal_moves_borderline_case_out_of_pure_weak(self):
        issue = {
            "title": "Fix certificate notarization failure",
            "description": "The macOS package fails during notarization in CI.",
        }
        similar_issues = [
            {
                "title": "Fix notarization failure in CI pipeline",
                "description": "Build remediation for notarization errors in CI.",
                "score": 0.68,
                "total_effort_hours": 18,
            },
            {
                "title": "Adjust release signing workflow",
                "description": "Release pipeline update for signing changes.",
                "score": 0.63,
                "total_effort_hours": 13,
            },
            {
                "title": "Update docs footer",
                "description": "Unrelated docs task.",
                "score": 0.41,
                "total_effort_hours": 2,
            },
        ]

        stats = _compute_retrieval_stats(issue, similar_issues)

        self.assertEqual(stats["useful_count"], 0)
        self.assertGreaterEqual(stats["top1_score"], 0.65)
        self.assertEqual(stats["route"], "analogical_soft_signal")


if __name__ == "__main__":
    unittest.main()
