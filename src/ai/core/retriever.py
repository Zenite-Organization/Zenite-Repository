# src/ai/core/retriever.py
from typing import List, Dict, Any


class Retriever:
    def __init__(self, vector_store):
        self.vs = vector_store

    def get_similar_issues(
        self,
        new_issue_text: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retorna issues similares.
        - Mock: retorna TODAS
        - Real: semantic_search
        """
        if hasattr(self.vs, "issues"):
            return list(self.vs.issues)

        return self.vs.semantic_search(new_issue_text, top_k=top_k)
