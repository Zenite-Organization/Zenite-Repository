# src/core/retriever.py
from typing import List, Dict, Any
from src.core.vector_store import VectorStoreClient

class Retriever:
    def __init__(self, vector_store: VectorStoreClient):
        self.vs = vector_store

    def get_similar_issues(self, new_issue_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retorna top_k issues similares do banco vetorizado.
        """
        return self.vs.semantic_search(new_issue_text, top_k=top_k)
