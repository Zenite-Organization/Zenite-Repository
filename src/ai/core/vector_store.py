from typing import Any, Dict, List


class VectorStoreClient:
    """Read-only vector store contract used by RAG retrieval."""

    def upsert(self, docs: List[Dict[str, Any]]) -> None:
        # RAG v1 for Zenite is retrieval-only.
        raise NotImplementedError("This service does not write vectors in RAG v1.")

    def semantic_search(
        self,
        text: str,
        namespaces: List[str],
        top_k: int = 8,
    ) -> List[Dict[str, Any]]:
        """Return raw matches from multiple namespaces."""
        raise NotImplementedError
