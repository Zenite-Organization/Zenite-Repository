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
    """Simula o pinecone_vector_store: aceita lista de namespaces e retorna
    matches de todos eles (preservando namespace em cada match)."""

    def __init__(self, responses, namespaces):
        self.responses = responses
        self.namespaces = namespaces
        self.calls = []

    def semantic_search(self, text, namespaces, top_k=8, where=None):
        self.calls.append(list(namespaces))
        matches = []
        for ns in namespaces:
            matches.extend(self.responses.get(ns, []))
        return matches

    def list_namespaces(self):
        return list(self.namespaces)


class TestRetrieverGlobalSearch(unittest.TestCase):
    """Testes do comportamento pós busca global (Opção 1).

    Mudanças de semântica vs versão anterior:
    - Uma única chamada de semantic_search com todos os namespaces _issues.
    - Ranking global por score — melhor match vence sempre.
    - Dedup por issue_id preserva o match de maior score.
    """

    def setUp(self):
        self.prev_score = settings.RAG_MIN_SCORE_MAIN
        self.prev_final = settings.RAG_FINAL_CONTEXT_SIZE
        settings.RAG_MIN_SCORE_MAIN = 0.75
        settings.RAG_FINAL_CONTEXT_SIZE = 4

    def tearDown(self):
        settings.RAG_MIN_SCORE_MAIN = self.prev_score
        settings.RAG_FINAL_CONTEXT_SIZE = self.prev_final

    def test_single_call_with_all_issue_namespaces(self):
        """Opção 1: UMA chamada com todos os _issues namespaces de uma vez."""
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

        # UMA única chamada com todos os _issues (comments excluídos)
        self.assertEqual(len(vs.calls), 1)
        called_ns = set(vs.calls[0])
        self.assertEqual(
            called_ns,
            {"mobile-app_issues", "mule_issues", "confserver_issues"},
        )
        # Resultados ranqueados globalmente por score
        self.assertEqual(len(result), 4)
        scores = [float(item["score"]) for item in result]
        self.assertEqual(scores, sorted(scores, reverse=True))
        # O melhor match (confserver 0.99) deve aparecer primeiro mesmo que o
        # repo seja mobile-app — é a essência da busca global.
        self.assertEqual(result[0]["issue_key"], "CONFSERVER-5")

    def test_only_issue_namespaces_are_queried(self):
        """Namespaces de comments/changelog são filtrados."""
        vs = _FakeVectorStore(
            responses={"mule_issues": [_match("mule_issues", 1, 0.99)]},
            namespaces=["mule_issues", "mule_comments", "timob_issues", "timob_comments"],
        )
        retriever = Retriever(vs)
        retriever.get_similar_issues({"title": "Issue", "description": "Desc"})
        self.assertEqual(len(vs.calls), 1)
        for ns in vs.calls[0]:
            self.assertTrue(ns.endswith("_issues"))

    def test_low_score_matches_are_excluded(self):
        """Matches abaixo do min_score são descartados mesmo em busca global."""
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

    def test_best_match_wins_even_from_other_project(self):
        """Cross-project: se o match mais forte está em outro namespace, ele vence.
        Esse é o caso de uso que motivou a Opção 1 — uma issue de login no
        projeto A pode ter match perfeito numa issue de login do projeto B.
        """
        vs = _FakeVectorStore(
            responses={
                "projecta_issues": [_match("projecta_issues", 1, 0.78)],
                "projectb_issues": [_match("projectb_issues", 2, 0.97)],
            },
            namespaces=["projecta_issues", "projectb_issues"],
        )
        retriever = Retriever(vs)
        result = retriever.get_similar_issues(
            {"title": "Tela de login", "description": "Desc", "repository": "org/projecta"}
        )
        self.assertGreaterEqual(len(result), 1)
        # O match do projectB (score 0.97) vence o do projectA (0.78)
        self.assertEqual(result[0]["issue_key"], "PROJECTB-2")


if __name__ == "__main__":
    unittest.main()
