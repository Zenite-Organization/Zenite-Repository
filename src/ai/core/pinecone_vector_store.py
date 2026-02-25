import logging
from typing import Any, Dict, List

from ai.core.vector_store import VectorStoreClient
from config.settings import settings


logger = logging.getLogger(__name__)


class PineconeVectorStoreClient(VectorStoreClient):
    def __init__(self):
        self._ready = False
        self._pc = None
        self._index = None
        self._openai = None
        self._bootstrap()

    def _bootstrap(self) -> None:
        api_key = settings.PINECONE_API_KEY
        index_name = settings.PINECONE_INDEX_NAME
        openai_key = settings.OPENAI_API_KEY_RAG

        if not api_key or not index_name or not openai_key:
            logger.warning("RAG disabled: missing Pinecone/OpenAI settings.")
            return

        try:
            from pinecone import Pinecone
            from openai import OpenAI
        except Exception as exc:
            logger.exception("RAG disabled: dependency import error: %s", exc)
            return

        try:
            self._pc = Pinecone(api_key=api_key)
            self._index = self._pc.Index(index_name)
            self._openai = OpenAI(api_key=openai_key)
            self._ready = True
        except Exception as exc:
            logger.exception("RAG disabled: Pinecone/OpenAI client init failed: %s", exc)

    def _embed_query(self, text: str) -> List[float]:
        response = self._openai.embeddings.create(
            model=settings.RAG_EMBEDDING_MODEL,
            input=text,
        )
        return response.data[0].embedding

    def semantic_search(
        self,
        text: str,
        namespaces: List[str],
        top_k: int = 8,
    ) -> List[Dict[str, Any]]:
        if not self._ready:
            return []
        if not text.strip() or not namespaces:
            return []

        query_vector = self._embed_query(text)
        all_matches: List[Dict[str, Any]] = []

        for namespace in namespaces:
            try:
                response = self._index.query(
                    vector=query_vector,
                    top_k=top_k,
                    namespace=namespace,
                    include_metadata=True,
                )

                matches = getattr(response, "matches", None) or response.get("matches", [])
                for match in matches:
                    if isinstance(match, dict):
                        record = match
                    else:
                        record = {
                            "id": getattr(match, "id", ""),
                            "score": getattr(match, "score", 0),
                            "metadata": getattr(match, "metadata", {}) or {},
                        }
                    record["namespace"] = namespace
                    print(record)
                    all_matches.append(record)
            except Exception as exc:
                logger.exception("Pinecone query failed for namespace=%s: %s", namespace, exc)
                print(f"[RAG][Pinecone] erro no namespace={namespace}: {exc}")

        return all_matches

    def list_namespaces(self) -> List[str]:
        if not self._ready:
            return []
        try:
            stats = self._index.describe_index_stats()
            namespaces = getattr(stats, "namespaces", None)
            if namespaces is None and isinstance(stats, dict):
                namespaces = stats.get("namespaces", {})
            if isinstance(namespaces, dict):
                return list(namespaces.keys())
        except Exception as exc:
            logger.exception("Pinecone namespace discovery failed: %s", exc)
        return []
