import unittest

from ai.core.retriever import Retriever
from ai.core.rag_namespace_policy import parse_fallback_projects, project_namespaces
from config.settings import settings


class _FakeVectorStore:
    def __init__(self, primary_hits: int):
        self.primary_hits = primary_hits
        self.calls = []

    def semantic_search(self, text, namespaces, top_k=8):
        self.calls.append(list(namespaces))
        ns0 = namespaces[0]
        if ns0 == "mdl":
            return [
                {
                    "id": f"issue:{i}",
                    "namespace": "mdl",
                    "score": 0.9,
                    "metadata": {
                        "doc_type": "issue",
                        "issue_id": 100 + i,
                        "project_key": "MDL",
                        "issue_key": f"MDL-{i}",
                        "title": "t",
                        "description": "d",
                        "total_effort_minutes": 60,
                        "resolution_time_minutes": 120,
                    },
                }
                for i in range(self.primary_hits)
            ]
        return []


class TestRagNamespacePolicy(unittest.TestCase):
    def setUp(self):
        self.previous_min = settings.RAG_MIN_HITS_MAIN
        settings.RAG_MIN_HITS_MAIN = 5

    def tearDown(self):
        settings.RAG_MIN_HITS_MAIN = self.previous_min

    def test_project_namespaces(self):
        self.assertEqual(
            project_namespaces("mdl"),
            ["mdl", "mdl_comments", "mdl_changelog"],
        )

    def test_fallback_order(self):
        projects = parse_fallback_projects("mule,confserver")
        self.assertEqual(projects, ["mule", "confserver"])

    def test_no_fallback_when_primary_hits_are_enough(self):
        vs = _FakeVectorStore(primary_hits=5)
        retriever = Retriever(vs)
        retriever.get_similar_issues({"title": "a", "description": "b"})
        self.assertEqual(len(vs.calls), 1)
        self.assertEqual(vs.calls[0], ["mdl", "mdl_comments", "mdl_changelog"])

    def test_fallback_order_when_primary_hits_are_low(self):
        vs = _FakeVectorStore(primary_hits=2)
        retriever = Retriever(vs)
        retriever.get_similar_issues({"title": "a", "description": "b"})
        self.assertEqual(vs.calls[0], ["mdl", "mdl_comments", "mdl_changelog"])
        self.assertEqual(vs.calls[1], ["mule", "mule_comments", "mule_changelog"])
        self.assertEqual(vs.calls[2], ["confserver", "confserver_comments", "confserver_changelog"])


if __name__ == "__main__":
    unittest.main()
