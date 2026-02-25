import unittest

from ai.core.retriever import Retriever
from config.settings import settings


def _match(namespace: str, idx: int, score: float):
    return {
        "id": f"{namespace}:{idx}",
        "namespace": namespace,
        "score": score,
        "metadata": {
            "doc_type": "issue",
            "issue_id": idx,
            "project_key": namespace.upper(),
            "issue_key": f"{namespace.upper()}-{idx}",
            "issue_title": f"{namespace} issue {idx}",
            "description": "desc",
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
                "timob": [_match("timob", 1, 0.95), _match("timob", 2, 0.84)],
                "mule": [_match("mule", 3, 0.91), _match("mule", 4, 0.88)],
                "confserver": [_match("confserver", 5, 0.99)],
            },
            namespaces=["timob", "timob_comments", "mule", "confserver"],
        )
        retriever = Retriever(vs)
        result = retriever.get_similar_issues(
            {
                "title": "Issue",
                "description": "Desc",
                "repository": "timob/mobile-app",
            }
        )

        self.assertEqual(vs.calls[0], ["timob"])
        self.assertEqual(vs.calls[1], ["mule"])
        self.assertEqual(len(vs.calls), 2)
        self.assertEqual(len(result), 4)
        scores = [float(item["score"]) for item in result]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_does_not_stop_just_because_first_namespace_has_results(self):
        vs = _FakeVectorStore(
            responses={
                "timob": [_match("timob", 1, 0.96)],
                "mule": [_match("mule", 2, 0.95), _match("mule", 3, 0.94)],
            },
            namespaces=["timob", "mule"],
        )
        retriever = Retriever(vs)
        retriever.get_similar_issues(
            {"title": "Issue", "description": "Desc", "repository": "timob/mobile-app"}
        )
        self.assertEqual(vs.calls[0], ["timob"])
        self.assertEqual(vs.calls[1], ["mule"])
        self.assertEqual(len(vs.calls), 2)

    def test_only_base_namespaces_are_queried(self):
        vs = _FakeVectorStore(
            responses={"mule": [_match("mule", 1, 0.99)]},
            namespaces=["mule", "mule_comments", "timob", "timob_comments"],
        )
        retriever = Retriever(vs)
        retriever.get_similar_issues({"title": "Issue", "description": "Desc"})
        self.assertEqual(vs.calls[0], ["mule"])
        for call in vs.calls:
            self.assertTrue(all("_" not in ns for ns in call))

    def test_low_score_matches_are_excluded(self):
        vs = _FakeVectorStore(
            responses={
                "timob": [_match("timob", 1, 0.3), _match("timob", 2, 0.2)],
                "mule": [_match("mule", 3, 0.8)],
            },
            namespaces=["timob", "mule"],
        )
        retriever = Retriever(vs)
        result = retriever.get_similar_issues(
            {"title": "Issue", "description": "Desc", "repository": "timob/mobile-app"}
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["issue_key"], "MULE-3")


if __name__ == "__main__":
    unittest.main()
