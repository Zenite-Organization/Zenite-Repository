import unittest

from ai.core.rag_ranker import join_issue_context


class TestRagJoiner(unittest.TestCase):
    def test_joins_comment_and_changelog_into_issue_anchor(self):
        joined = join_issue_context(
            [
                {
                    "id": "issue:1",
                    "doc_type": "issue",
                    "issue_id": 1,
                    "score": 0.95,
                    "project_key": "mdl",
                    "description": "Issue base context",
                },
                {
                    "id": "comment:1",
                    "doc_type": "comment",
                    "issue_id": 1,
                    "score": 0.7,
                    "project_key": "mdl",
                    "snippet": "Comment context",
                },
                {
                    "id": "changelog:1",
                    "doc_type": "changelog",
                    "issue_id": 1,
                    "score": 0.6,
                    "project_key": "mdl",
                    "snippet": "Changelog context",
                },
            ]
        )

        self.assertEqual(len(joined), 1)
        self.assertIn("Comment context", joined[0]["description"])
        self.assertIn("Changelog context", joined[0]["description"])


if __name__ == "__main__":
    unittest.main()
