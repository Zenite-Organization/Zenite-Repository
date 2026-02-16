import unittest

from ai.workflows import estimation_graph as eg


class _FakeVectorStore:
    def semantic_search(self, text, namespaces, top_k=8):
        if "mdl" in namespaces:
            return [
                {
                    "id": "issue:321762",
                    "namespace": "mdl",
                    "score": 0.91,
                    "metadata": {
                        "doc_type": "issue",
                        "issue_id": 321762,
                        "issue_key": "MDL-65179",
                        "project_key": "MDL",
                        "title": "Lock fix",
                        "description": "Fix lock contention",
                        "total_effort_minutes": 31844,
                        "resolution_time_minutes": 33303,
                    },
                }
            ]
        return []


class TestEstimationGraphRagIntegration(unittest.TestCase):
    def test_retriever_node_returns_rag_context(self):
        original_vs = eg.vector_store
        eg.vector_store = _FakeVectorStore()
        try:
            state = {
                "issue": {
                    "title": "Investigate lock issue",
                    "description": "Requests wait too long",
                    "labels": ["Estimate"],
                    "repository": "org/repo",
                    "repo_language": "Python",
                }
            }
            out = eg.retriever_node(state)
            self.assertIn("similar_issues", out)
            self.assertGreaterEqual(len(out["similar_issues"]), 1)
            first = out["similar_issues"][0]
            self.assertEqual(first["issue_key"], "MDL-65179")
            self.assertEqual(first["estimated_hours"], round(31844 / 60.0, 2))
        finally:
            eg.vector_store = original_vs


if __name__ == "__main__":
    unittest.main()
