import unittest

from ai.core.rag_normalizer import normalize_match


class TestRagNormalizer(unittest.TestCase):
    def test_minutes_to_hours_mapping(self):
        item = normalize_match(
            {
                "id": "issue:1",
                "namespace": "mdl",
                "score": 0.9,
                "metadata": {
                    "doc_type": "issue",
                    "issue_id": 1,
                    "issue_key": "MDL-1",
                    "project_key": "MDL",
                    "title": "Fix lock",
                    "description": "desc",
                    "total_effort_minutes": 120,
                    "resolution_time_minutes": 180,
                },
            }
        )
        self.assertEqual(item["estimated_hours"], 2.0)
        self.assertEqual(item["real_hours"], 3.0)
        self.assertEqual(item["doc_type"], "issue")
        self.assertEqual(item["issue_id"], 1)

    def test_real_fallback_to_timespent(self):
        item = normalize_match(
            {
                "id": "issue:2",
                "namespace": "mdl",
                "score": 0.8,
                "metadata": {
                    "doc_type": "issue",
                    "issue_id": 2,
                    "timespent": 90,
                    "total_effort_minutes": 60,
                },
            }
        )
        self.assertEqual(item["estimated_hours"], 1.0)
        self.assertEqual(item["real_hours"], 1.5)


if __name__ == "__main__":
    unittest.main()
