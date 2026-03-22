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
        self.last_embedding_tokens: int = 0
        self._bootstrap()

    def _update_last_embedding_tokens(self, response: Any) -> None:
        tokens = 0
        usage = getattr(response, "usage", None)
        if usage is None and isinstance(response, dict):
            usage = response.get("usage")
        if usage is not None:
            try:
                tokens = int(getattr(usage, "total_tokens", None) or getattr(usage, "prompt_tokens", None) or 0)
            except Exception:
                try:
                    tokens = int((usage.get("total_tokens") or usage.get("prompt_tokens") or 0) if isinstance(usage, dict) else 0)
                except Exception:
                    tokens = 0
        self.last_embedding_tokens = max(0, int(tokens or 0))

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
        self._update_last_embedding_tokens(response)
        return response.data[0].embedding

    def upsert(self, docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not self._ready:
            return {"skipped": True, "reason": "rag_disabled", "upserted": 0}
        if not docs:
            return {"skipped": False, "reason": None, "upserted": 0}

        prepared: List[Dict[str, Any]] = []
        texts: List[str] = []
        for doc in docs:
            if not isinstance(doc, dict):
                continue
            doc_id = str(doc.get("id") or "").strip()
            namespace = str(doc.get("namespace") or "").strip().lower()
            text = str(doc.get("text") or "").strip()
            metadata = doc.get("metadata") or {}
            if not doc_id or not namespace or not text:
                continue
            if not isinstance(metadata, dict):
                metadata = {"value": metadata}
            prepared.append({"id": doc_id, "namespace": namespace, "text": text, "metadata": metadata})
            texts.append(text)

        if not prepared:
            return {"skipped": False, "reason": "no_valid_docs", "upserted": 0}

        response = self._openai.embeddings.create(
            model=settings.RAG_EMBEDDING_MODEL,
            input=texts,
        )
        self._update_last_embedding_tokens(response)
        vectors = [item.embedding for item in response.data]
        if len(vectors) != len(prepared):
            raise RuntimeError(f"Embedding count mismatch: {len(vectors)} != {len(prepared)}")

        by_namespace: Dict[str, List[Dict[str, Any]]] = {}
        for doc, values in zip(prepared, vectors):
            ns = doc["namespace"]
            by_namespace.setdefault(ns, []).append(
                {"id": doc["id"], "values": values, "metadata": doc["metadata"]}
            )

        upserted = 0
        namespace_counts: Dict[str, int] = {}
        for ns, vecs in by_namespace.items():
            self._index.upsert(vectors=vecs, namespace=ns)
            upserted += len(vecs)
            namespace_counts[ns] = len(vecs)

        return {"skipped": False, "reason": None, "upserted": upserted, "namespaces": namespace_counts}

    def semantic_search(
        self,
        text: str,
        namespaces: List[str],
        top_k: int = 8,
        where: Dict[str, Any] | None = None,
    ) -> List[Dict[str, Any]]:
        # Reset so callers don't accidentally read stale values when no embedding happens.
        self.last_embedding_tokens = 0
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
                    filter=where or None,
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
