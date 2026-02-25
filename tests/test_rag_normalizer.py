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
                },
            }
        )
        self.assertEqual(item["estimated_hours"], 2.0)
        self.assertNotIn("real_hours", item)
        self.assertEqual(item["doc_type"], "issue")
        self.assertEqual(item["issue_id"], 1)

    def test_issue_title_has_priority(self):
        item = normalize_match(
            {
                "id": "issue:2",
                "namespace": "mdl",
                "score": 0.8,
                "metadata": {
                    "doc_type": "issue",
                    "issue_id": 2,
                    "issue_title": "Issue Title Priority",
                    "title": "Legacy Title",
                    "total_effort_minutes": 60,
                },
            }
        )
        self.assertEqual(item["estimated_hours"], 1.0)
        self.assertEqual(item["title"], "Issue Title Priority")
        self.assertNotIn("real_hours", item)


if __name__ == "__main__":
    unittest.main()
