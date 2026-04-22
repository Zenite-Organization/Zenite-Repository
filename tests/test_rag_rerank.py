import unittest

from ai.core.rag_ranker import rerank_issue_context


class TestRagRerank(unittest.TestCase):
    def test_rerank_prefers_same_type_and_title_overlap(self):
        issue = {
            "title": "Fix notarization failure in CI pipeline",
            "description": "The macOS package fails during notarization in CI.",
            "issue_type": "bug",
            "labels": ["ci", "release"],
        }
        items = [
            {
                "id": "1",
                "title": "Refactor reporting backend",
                "description": "Broader backend work.",
                "issue_type": "feature",
                "labels": ["backend"],
                "score": 0.74,
                "project_key": "mdl",
                "doc_type": "issue",
            },
            {
                "id": "2",
                "title": "Fix notarization failure in release pipeline",
                "description": "CI release remediation for notarization.",
                "issue_type": "bug",
                "labels": ["ci", "release"],
                "score": 0.71,
                "project_key": "mdl",
                "doc_type": "issue",
            },
        ]

        ranked = rerank_issue_context(items, issue)

        self.assertEqual(ranked[0]["id"], "2")
        self.assertGreater(ranked[0]["score"], ranked[1]["score"])
        self.assertEqual(ranked[0]["semantic_score"], 0.71)


if __name__ == "__main__":
    unittest.main()
