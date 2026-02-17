import logging
from collections import Counter
from typing import Any, Dict, List

from config.settings import settings

from ai.core.rag_namespace_policy import parse_fallback_projects, project_namespaces
from ai.core.rag_normalizer import normalize_match
from ai.core.rag_ranker import assemble_context, join_issue_context


logger = logging.getLogger(__name__)


class Retriever:
    def __init__(self, vector_store):
        self.vs = vector_store

    @staticmethod
    def _build_query(issue_payload: Dict[str, Any]) -> str:
        labels = " ".join(issue_payload.get("labels") or [])
        return "\n".join(
            [
                str(issue_payload.get("title") or ""),
                str(issue_payload.get("description") or ""),
                labels,
                str(issue_payload.get("repository") or ""),
                str(issue_payload.get("repo_language") or ""),
            ]
        ).strip()

    @staticmethod
    def _namespace_hit_count(raw_matches: List[Dict[str, Any]], namespaces: List[str]) -> int:
        ns = set(namespaces)
        return sum(1 for match in raw_matches if match.get("namespace") in ns)

    def get_similar_issues(
        self,
        issue_payload: Dict[str, Any],
        top_k: int | None = None,
    ) -> List[Dict[str, Any]]:
        # Backward compatibility for tests/mocks that expose in-memory issues.
        if hasattr(self.vs, "issues"):
            return list(self.vs.issues)

        query_text = self._build_query(issue_payload)
        if not query_text:
            print("[RAG] query vazia, retornando contexto vazio")
            return []

        top_k = top_k or settings.RAG_TOPK_PER_NAMESPACE
        primary_project = settings.RAG_PRIMARY_PROJECT.strip().lower()
        primary_namespaces = project_namespaces(primary_project)

        all_raw_matches = self.vs.semantic_search(
            query_text,
            namespaces=primary_namespaces,
            top_k=top_k,
        )
        primary_hits = self._namespace_hit_count(all_raw_matches, primary_namespaces)

        fallback_triggered = primary_hits < settings.RAG_MIN_HITS_MAIN
        if fallback_triggered:
            for project in parse_fallback_projects(settings.RAG_FALLBACK_PROJECTS):
                fallback_namespaces = project_namespaces(project)
                all_raw_matches.extend(
                    self.vs.semantic_search(
                        query_text,
                        namespaces=fallback_namespaces,
                        top_k=top_k,
                    )
                )

        normalized = [normalize_match(match) for match in all_raw_matches]
        joined = join_issue_context(normalized)
        context = assemble_context(joined, final_size=settings.RAG_FINAL_CONTEXT_SIZE)

        mix = Counter(item.get("doc_type") for item in context)
        logger.info(
            "[RAG] primary_namespaces=%s primary_hits=%s fallback=%s total_raw=%s final_context=%s mix=%s",
            primary_namespaces,
            primary_hits,
            fallback_triggered,
            len(all_raw_matches),
            len(context),
            dict(mix),
        )
        print(
            "[RAG] primary_hits=%s fallback=%s total_raw=%s final_context=%s mix=%s"
            % (primary_hits, fallback_triggered, len(all_raw_matches), len(context), dict(mix))
        )

        # Keep structure expected by prompt_utils/analogical agent.
        result = [
            {
                "id": item.get("id"),
                "title": item.get("title"),
                "description": item.get("description") or item.get("snippet") or "",
                "estimated_hours": item.get("estimated_hours"),
                "real_hours": item.get("real_hours"),
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
        print("[RAG] exemplos retornados:", [it.get("id") for it in result[:3]])
        return result
