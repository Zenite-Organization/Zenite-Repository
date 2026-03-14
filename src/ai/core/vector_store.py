from typing import Any, Dict, List


class VectorStoreClient:
    """Vector store contract used by RAG retrieval (and optional ingestion)."""

    def upsert(self, docs: List[Dict[str, Any]]) -> None:
        """Optionally write documents to the backing store.

        Implementations may ignore this (read-only) by raising NotImplementedError.
        """
        raise NotImplementedError("VectorStoreClient.upsert is not implemented for this backend.")

    def semantic_search(
        self,
        text: str,
        namespaces: List[str],
        top_k: int = 8,
        where: Dict[str, Any] | None = None,
    ) -> List[Dict[str, Any]]:
        """Return raw matches from multiple namespaces."""
        raise NotImplementedError

    def list_namespaces(self) -> List[str]:
        """Return available namespaces in the backing store."""
        raise NotImplementedError
