import unittest

from ai.workflows import estimation_graph as eg
from config.settings import settings


class _FakeVectorStore:
    def list_namespaces(self):
        return ["mdl", "mdl_comments", "mdl_changelog"]

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
                    },
                }
            ]
        return []


class TestEstimationGraphRagIntegration(unittest.TestCase):
    def test_retriever_node_returns_rag_context(self):
        original_vs = eg.vector_store
        prev_score = settings.RAG_MIN_SCORE_MAIN
        prev_final = settings.RAG_FINAL_CONTEXT_SIZE
        settings.RAG_MIN_SCORE_MAIN = 0.75
        settings.RAG_FINAL_CONTEXT_SIZE = 10
        eg.vector_store = _FakeVectorStore()
        try:
            state = {
                "issue": {
                    "title": "Investigate lock issue",
                    "description": "Requests wait too long",
                    "labels": ["Estimate"],
                    "repository": "mdl/repo",
                    "repo_language": "Python",
                }
            }
            out = eg.retriever_node(state)
            self.assertIn("similar_issues", out)
            self.assertGreaterEqual(len(out["similar_issues"]), 1)
            self.assertLessEqual(len(out["similar_issues"]), settings.RAG_FINAL_CONTEXT_SIZE)
            first = out["similar_issues"][0]
            self.assertEqual(first["issue_key"], "MDL-65179")
            self.assertEqual(first["estimated_hours"], round(31844 / 60.0, 2))
            self.assertGreaterEqual(float(first["score"]), settings.RAG_MIN_SCORE_MAIN)
            self.assertNotIn("real_hours", first)
        finally:
            settings.RAG_MIN_SCORE_MAIN = prev_score
            settings.RAG_FINAL_CONTEXT_SIZE = prev_final
            eg.vector_store = original_vs


if __name__ == "__main__":
    unittest.main()
