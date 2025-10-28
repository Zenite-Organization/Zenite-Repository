# src/core/vector_store.py
from typing import List, Dict, Any

class VectorStoreClient:
    """
    Interface simples para o banco vetorial.
    Substitua com FAISS / Chroma / LanceDB implementations.
    """

    def __init__(self):
        # TODO: inicializar conexão ao vector DB
        pass

    def upsert(self, docs: List[Dict[str, Any]]) -> None:
        """
        docs: lista de dicts com fields (id, title, description, metadata...)
        """
        raise NotImplementedError

    def semantic_search(self, text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retorna uma lista de issues similares com seus campos (incluindo estimated_hours, real_hours).
        Cada item deve ser um dict representando a issue.
        """
        raise NotImplementedError
