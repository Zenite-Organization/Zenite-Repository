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

    def semantic_search(self, text, namespaces, top_k=8):
        return list(self.responses.get(namespaces[0], []))

    def list_namespaces(self):
        return list(self.namespaces)


class TestRetrieverScoreFilter(unittest.TestCase):
    def setUp(self):
        self.prev_score = settings.RAG_MIN_SCORE_MAIN
        self.prev_final = settings.RAG_FINAL_CONTEXT_SIZE
        self.prev_cap = settings.RAG_MAX_FALLBACK_BASES
        settings.RAG_MIN_SCORE_MAIN = 0.75
        settings.RAG_FINAL_CONTEXT_SIZE = 10
        settings.RAG_MAX_FALLBACK_BASES = 10

    def tearDown(self):
        settings.RAG_MIN_SCORE_MAIN = self.prev_score
        settings.RAG_FINAL_CONTEXT_SIZE = self.prev_final
        settings.RAG_MAX_FALLBACK_BASES = self.prev_cap

    def test_all_below_threshold_returns_empty(self):
        vs = _FakeVectorStore(
            responses={
                "mobile-app_issues": [_match("mobile-app_issues", 1, 0.2)],
                "mule_issues": [_match("mule_issues", 2, 0.5)],
            },
            namespaces=["mobile-app_issues", "mule_issues"],
        )
        retriever = Retriever(vs)
        result = retriever.get_similar_issues(
            {"title": "Issue", "description": "Desc", "repository": "timob/mobile-app"}
        )
        self.assertEqual(result, [])

    def test_mixed_scores_returns_only_qualified(self):
        vs = _FakeVectorStore(
            responses={
                "mobile-app_issues": [
                    _match("mobile-app_issues", 1, 0.9),
                    _match("mobile-app_issues", 2, 0.2),
                ],
                "mule_issues": [_match("mule_issues", 3, 0.85), _match("mule_issues", 4, 0.1)],
            },
            namespaces=["mobile-app_issues", "mule_issues"],
        )
        retriever = Retriever(vs)
        result = retriever.get_similar_issues(
            {"title": "Issue", "description": "Desc", "repository": "timob/mobile-app"}
        )
        self.assertEqual(len(result), 2)
        self.assertTrue(all(float(it["score"]) >= 0.75 for it in result))
        scores = [float(it["score"]) for it in result]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_strict_threshold_returns_less_than_target_when_exhausted(self):
        settings.RAG_MIN_SCORE_MAIN = 0.95
        settings.RAG_FINAL_CONTEXT_SIZE = 5
        vs = _FakeVectorStore(
            responses={
                "mobile-app_issues": [
                    _match("mobile-app_issues", 1, 0.96),
                    _match("mobile-app_issues", 2, 0.94),
                ],
                "mule_issues": [_match("mule_issues", 3, 0.92)],
            },
            namespaces=["mobile-app_issues", "mule_issues"],
        )
        retriever = Retriever(vs)
        result = retriever.get_similar_issues(
            {"title": "Issue", "description": "Desc", "repository": "timob/mobile-app"}
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["issue_key"], "MOBILE-APP-1")

    def test_short_descriptions_are_excluded(self):
        vs = _FakeVectorStore(
            responses={
                "mobile-app_issues": [
                    _match("mobile-app_issues", 1, 0.98, description="curta demais"),
                    _match("mobile-app_issues", 2, 0.97, description="b" * 100),
                ],
            },
            namespaces=["mobile-app_issues"],
        )
        retriever = Retriever(vs)
        result = retriever.get_similar_issues(
            {"title": "Issue", "description": "Desc", "repository": "timob/mobile-app"}
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["issue_key"], "MOBILE-APP-2")


if __name__ == "__main__":
    unittest.main()
