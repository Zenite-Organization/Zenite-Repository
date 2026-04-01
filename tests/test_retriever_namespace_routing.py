import unittest

from ai.core.retriever import Retriever
from config.settings import settings


def _match(namespace: str, idx: int, score: float, description: str | None = None):
    project_key = namespace.replace("_issues", "")
    return {
        "id": f"{namespace}:{idx}",
        "namespace": namespace,
        "score": score,
        "metadata": {
            "doc_type": "issue",
            "issue_id": idx,
            "project_key": project_key.upper(),
            "issue_key": f"{project_key.upper()}-{idx}",
            "issue_title": f"{namespace} issue {idx}",
            "description": description or ("a" * 100),
            "total_effort_minutes": 60,
        },
    }


class _FakeVectorStore:
    def __init__(self, responses, namespaces):
        self.responses = responses
        self.namespaces = namespaces
        self.calls = []

    def semantic_search(self, text, namespaces, top_k=8):
        self.calls.append(list(namespaces))
        base = namespaces[0]
        return list(self.responses.get(base, []))

    def list_namespaces(self):
        return list(self.namespaces)


class TestRetrieverNamespaceRouting(unittest.TestCase):
    def setUp(self):
        self.prev_score = settings.RAG_MIN_SCORE_MAIN
        self.prev_final = settings.RAG_FINAL_CONTEXT_SIZE
        self.prev_cap = settings.RAG_MAX_FALLBACK_BASES
        settings.RAG_MIN_SCORE_MAIN = 0.75
        settings.RAG_FINAL_CONTEXT_SIZE = 4
        settings.RAG_MAX_FALLBACK_BASES = 3

    def tearDown(self):
        settings.RAG_MIN_SCORE_MAIN = self.prev_score
        settings.RAG_FINAL_CONTEXT_SIZE = self.prev_final
        settings.RAG_MAX_FALLBACK_BASES = self.prev_cap

    def test_continues_to_next_namespaces_until_target_size(self):
        vs = _FakeVectorStore(
            responses={
                "mobile-app_issues": [
                    _match("mobile-app_issues", 1, 0.95),
                    _match("mobile-app_issues", 2, 0.84),
                ],
                "mule_issues": [
                    _match("mule_issues", 3, 0.91),
                    _match("mule_issues", 4, 0.88),
                ],
                "confserver_issues": [_match("confserver_issues", 5, 0.99)],
            },
            namespaces=[
                "mobile-app_issues",
                "mobile-app_comments",
                "mule_issues",
                "confserver_issues",
            ],
        )
        retriever = Retriever(vs)
        result = retriever.get_similar_issues(
            {
                "title": "Issue",
                "description": "Desc",
                "repository": "timob/mobile-app",
            }
        )

        self.assertEqual(vs.calls[0], ["mobile-app_issues"])
        self.assertEqual(vs.calls[1], ["mule_issues"])
        self.assertEqual(len(vs.calls), 2)
        self.assertEqual(len(result), 4)
        scores = [float(item["score"]) for item in result]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_does_not_stop_just_because_first_namespace_has_results(self):
        vs = _FakeVectorStore(
            responses={
                "mobile-app_issues": [_match("mobile-app_issues", 1, 0.96)],
                "mule_issues": [
                    _match("mule_issues", 2, 0.95),
                    _match("mule_issues", 3, 0.94),
                ],
            },
            namespaces=["mobile-app_issues", "mule_issues"],
        )
        retriever = Retriever(vs)
        retriever.get_similar_issues(
            {"title": "Issue", "description": "Desc", "repository": "timob/mobile-app"}
        )
        self.assertEqual(vs.calls[0], ["mobile-app_issues"])
        self.assertEqual(vs.calls[1], ["mule_issues"])
        self.assertEqual(len(vs.calls), 2)

    def test_only_issue_namespaces_are_queried(self):
        vs = _FakeVectorStore(
            responses={"mule_issues": [_match("mule_issues", 1, 0.99)]},
            namespaces=["mule_issues", "mule_comments", "timob_issues", "timob_comments"],
        )
        retriever = Retriever(vs)
        retriever.get_similar_issues({"title": "Issue", "description": "Desc"})
        self.assertEqual(vs.calls[0], ["mule_issues"])
        for call in vs.calls:
            self.assertTrue(all(ns.endswith("_issues") for ns in call))

    def test_low_score_matches_are_excluded(self):
        vs = _FakeVectorStore(
            responses={
                "mobile-app_issues": [
                    _match("mobile-app_issues", 1, 0.3),
                    _match("mobile-app_issues", 2, 0.2),
                ],
                "mule_issues": [_match("mule_issues", 3, 0.8)],
            },
            namespaces=["mobile-app_issues", "mule_issues"],
        )
        retriever = Retriever(vs)
        result = retriever.get_similar_issues(
            {"title": "Issue", "description": "Desc", "repository": "timob/mobile-app"}
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["issue_key"], "MULE-3")


if __name__ == "__main__":
    unittest.main()
