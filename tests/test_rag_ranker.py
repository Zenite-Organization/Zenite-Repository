import unittest

from ai.core.rag_ranker import assemble_context


class TestRagRanker(unittest.TestCase):
    def test_issue_first_and_size_cap(self):
        items = []
        for i in range(8):
            items.append(
                {
                    "id": f"issue:{i}",
                    "doc_type": "issue",
                    "project_key": "mdl",
                    "score": 0.9 - (i * 0.01),
                }
            )
        for i in range(8):
            items.append(
                {
                    "id": f"comment:{i}",
                    "doc_type": "comment",
                    "project_key": "mdl",
                    "score": 0.95 - (i * 0.01),
                }
            )

        context = assemble_context(items, final_size=10)
        self.assertEqual(len(context), 10)
        issue_count = len([it for it in context if it["doc_type"] == "issue"])
        self.assertGreaterEqual(issue_count, 6)


if __name__ == "__main__":
    unittest.main()
