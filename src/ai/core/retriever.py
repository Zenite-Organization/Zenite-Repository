import logging
from collections import Counter
from typing import Any, Dict, List

from config.settings import settings

from ai.core.rag_namespace_policy import (
    extract_project_issue_namespace,
    group_issue_namespaces,
)
from ai.core.rag_normalizer import normalize_match
from ai.core.rag_ranker import join_issue_context, rerank_issue_context


logger = logging.getLogger(__name__)


class Retriever:
    MIN_DESCRIPTION_LENGTH = 100

    def __init__(self, vector_store):
        self.vs = vector_store
        self.last_rag_usage: Dict[str, Any] = {"embedding_tokens": 0, "embedding_calls": 0}

    @staticmethod
    def _build_query(issue_payload: Dict[str, Any]) -> str:
  
        def norm(x: Any) -> str:
            return str(x).strip() if x else ""

        labels_raw = issue_payload.get("labels") or []
        if isinstance(labels_raw, list):
            labels = ", ".join(str(item).strip() for item in labels_raw if str(item).strip())
        else:
            labels = norm(labels_raw)

        parts = [
            f"[Title] {norm(issue_payload.get('title'))}",
            f"[Type] {norm(issue_payload.get('issue_type'))}",
            f"[Labels] {labels}",
            f"[Description] {norm(issue_payload.get('description'))}",
        ]

        full_text = "\n".join([p for p in parts if p]).strip()
        return full_text

    @staticmethod
    def _best_score(matches: List[Dict[str, Any]]) -> float:
        best = 0.0
        for item in matches:
            try:
                score = float(item.get("score") or 0.0)
            except (TypeError, ValueError):
                score = 0.0
            if score > best:
                best = score
        return best

    @staticmethod
    def _safe_score(match: Dict[str, Any]) -> float:
        try:
            return float(match.get("score") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def _filter_score_threshold(
        self,
        matches: List[Dict[str, Any]],
        min_score: float,
    ) -> List[Dict[str, Any]]:
        return [m for m in matches if self._safe_score(m) >= float(min_score)]

    @classmethod
    def _filter_description_length(cls, matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        filtered: List[Dict[str, Any]] = []
        for match in matches:
            metadata = match.get("metadata") or {}
            description = metadata.get("description")
            if isinstance(description, str) and len(description.strip()) >= cls.MIN_DESCRIPTION_LENGTH:
                filtered.append(match)
        return filtered

    @staticmethod
    def _dedupe_key(match: Dict[str, Any]) -> str:
        metadata = match.get("metadata") or {}
        issue_id = metadata.get("issue_id")
        if issue_id is not None:
            return f"issue:{issue_id}"
        return str(match.get("id") or "")

    @staticmethod
    def _pinecone_where_filter() -> Dict[str, Any]:
        return {
            "$and": [
                {"total_effort_hours": {"$gte": 1, "$lte": 40}},
                {"description": {"$exists": True, "$ne": ""}},
            ]
        }

    def get_similar_issues(
        self,
        issue_payload: Dict[str, Any],
        top_k: int | None = None,
    ) -> List[Dict[str, Any]]:
        # Backward compatibility for tests/mocks that expose in-memory issues.
        if hasattr(self.vs, "issues"):
            return self._filter_description_length(list(self.vs.issues))

        query_text = self._build_query(issue_payload)
        if not query_text:
            logger.debug("[RAG] query vazia, retornando contexto vazio")
            return []

        target_size = settings.RAG_FINAL_CONTEXT_SIZE
        top_k = top_k or settings.RAG_TOPK_PER_NAMESPACE
        min_score = settings.RAG_MIN_SCORE_MAIN
        primary_namespace = extract_project_issue_namespace(issue_payload.get("repository") or "")

        # [GLOBAL SEARCH] Sempre buscamos em todos os _issues namespaces disponíveis
        # e ranqueamos globalmente por score. Isso elimina a dependência de ordem
        # arbitrária do list_namespaces() e garante que o melhor match sempre
        # vença, mesmo que esteja em um projeto diferente do atual.
        #
        # Custo: uma única chamada semantic_search com lista de namespaces. O
        # embedding da query é feito uma só vez e reusado internamente. Pinecone
        # cobra min 0.25 RU por namespace — para os 37 namespaces pequenos do
        # Zenite isso é < 4% do custo total da estimativa (o LLM é ~99%).
        discovered_issue_namespaces: List[str] = []
        if hasattr(self.vs, "list_namespaces"):
            try:
                discovered = self.vs.list_namespaces()
            except Exception:
                discovered = []
            discovered_issue_namespaces = group_issue_namespaces(discovered)

        # Garante que o primário esteja na lista mesmo se list_namespaces falhar
        namespaces_to_query: List[str] = list(discovered_issue_namespaces)
        if primary_namespace and primary_namespace not in namespaces_to_query:
            namespaces_to_query.insert(0, primary_namespace)

        # Fallback: sem discovery, ao menos tenta o namespace primário
        if not namespaces_to_query and primary_namespace:
            namespaces_to_query = [primary_namespace]

        if not namespaces_to_query:
            logger.warning("[RAG] nenhum namespace disponível para busca")
            return []

        logger.debug(
            "[RAG] busca global — primary=%s total_namespaces=%d",
            primary_namespace,
            len(namespaces_to_query),
        )

        where = self._pinecone_where_filter()
        embedding_tokens_total = 0

        # Uma única chamada que consulta todos os namespaces. O Pinecone client
        # itera internamente e reusa o embedding da query. Se o client não
        # aceitar lista (mocks antigos), faz fallback para loop sequencial.
        try:
            raw_matches = self.vs.semantic_search(
                query_text,
                namespaces=namespaces_to_query,
                top_k=top_k,
                where=where,
            )
        except TypeError:
            # Backward-compatibility com mocks/clients antigos que não aceitam
            # `where` ou múltiplos namespaces de uma vez.
            raw_matches = []
            for ns in namespaces_to_query:
                try:
                    raw_matches.extend(
                        self.vs.semantic_search(
                            query_text,
                            namespaces=[ns],
                            top_k=top_k,
                        )
                    )
                except Exception as exc:
                    logger.exception("semantic_search fallback falhou para %s: %s", ns, exc)

        try:
            embedding_tokens_total = int(getattr(self.vs, "last_embedding_tokens", 0) or 0)
        except Exception:
            embedding_tokens_total = 0

        self.last_rag_usage = {
            "embedding_tokens": max(0, int(embedding_tokens_total)),
            # Uma única chamada de embedding mesmo consultando N namespaces
            "embedding_calls": 1 if raw_matches is not None else 0,
        }

        # Filtros por score e descrição mínima
        filtered = self._filter_score_threshold(raw_matches, min_score=min_score)
        filtered_out_low_score_total = max(0, len(raw_matches) - len(filtered))
        filtered = self._filter_description_length(filtered)

        # Dedup preservando o maior score quando há duplicatas
        seen_keys: Dict[str, Dict[str, Any]] = {}
        for match in sorted(filtered, key=self._safe_score, reverse=True):
            key = self._dedupe_key(match)
            if key not in seen_keys:
                seen_keys[key] = match
        qualified_raw_matches = list(seen_keys.values())

        # Ordena globalmente por score e trunca em target_size. Aqui está a
        # diferença chave vs o comportamento antigo: o melhor match sempre
        # vence, independentemente de qual namespace ele veio.
        qualified_raw_matches = sorted(
            qualified_raw_matches,
            key=self._safe_score,
            reverse=True,
        )[:target_size]

        normalized = [normalize_match(match) for match in qualified_raw_matches]
        normalized = rerank_issue_context(normalized, issue_payload)
        joined = join_issue_context(normalized)
        context = sorted(joined, key=lambda it: float(it.get("score") or 0.0), reverse=True)[
            :target_size
        ]

        mix = Counter(item.get("doc_type") for item in context)
        namespaces_hit = Counter(item.get("namespace") for item in context)
        logger.info(
            "[RAG] primary_namespace=%s total_namespaces=%d qualified_collected=%s "
            "filtered_out_low_score_total=%s total_raw=%s final_context=%s "
            "best_score=%.4f mix=%s namespaces_hit=%s",
            primary_namespace,
            len(namespaces_to_query),
            len(qualified_raw_matches),
            filtered_out_low_score_total,
            len(raw_matches),
            len(context),
            self._best_score(qualified_raw_matches),
            dict(mix),
            dict(namespaces_hit),
        )

        # Keep structure expected by prompt_utils/analogical agent.
        result = [
            {
                "id": item.get("id"),
                "title": item.get("title"),
                "description": item.get("description") or item.get("snippet") or "",
                "total_effort_hours": item.get("total_effort_hours"),
                "issue_type": item.get("issue_type") or "unknown",
                "doc_type": item.get("doc_type"),
                "issue_id": item.get("issue_id"),
                "issue_key": item.get("issue_key"),
                "project_key": item.get("project_key"),
                "score": item.get("score"),
                "namespace": item.get("namespace"),
                "metadata": item.get("metadata"),
            }
            for item in context
        ]

        return result
