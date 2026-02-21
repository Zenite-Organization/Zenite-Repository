import logging
from collections import Counter
from typing import Any, Dict, List

from config.settings import settings

from ai.core.rag_namespace_policy import (
    extract_org_namespace,
    group_namespaces_by_base,
)
from ai.core.rag_normalizer import normalize_match
from ai.core.rag_ranker import join_issue_context


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

    @staticmethod
    def _dedupe_key(match: Dict[str, Any]) -> str:
        metadata = match.get("metadata") or {}
        issue_id = metadata.get("issue_id")
        if issue_id is not None:
            return f"issue:{issue_id}"
        return str(match.get("id") or "")

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

        target_size = settings.RAG_FINAL_CONTEXT_SIZE
        top_k = top_k or settings.RAG_TOPK_PER_NAMESPACE
        min_score = settings.RAG_MIN_SCORE_MAIN
        max_fallback_bases = settings.RAG_MAX_FALLBACK_BASES
        primary_base = extract_org_namespace(issue_payload.get("repository") or "")
        namespace_order: List[str] = [primary_base] if primary_base else []

        queried_namespaces: List[str] = []
        fallback_bases_tried: List[str] = []
        discovered_bases: List[str] = []
        if hasattr(self.vs, "list_namespaces"):
            try:
                discovered = self.vs.list_namespaces()
            except Exception:
                discovered = []
            discovered_bases = group_namespaces_by_base(discovered)
            for base in discovered_bases:
                if base == primary_base:
                    continue
                if len(fallback_bases_tried) >= max_fallback_bases:
                    break
                namespace_order.append(base)
                fallback_bases_tried.append(base)

        all_raw_matches: List[Dict[str, Any]] = []
        qualified_raw_matches: List[Dict[str, Any]] = []
        seen_keys = set()
        filtered_out_low_score_total = 0
        stop_reason = "no_more_namespaces"

        for namespace in namespace_order:
            raw = self.vs.semantic_search(
                query_text,
                namespaces=[namespace],
                top_k=top_k,
            )
            queried_namespaces.append(namespace)
            all_raw_matches.extend(raw)

            filtered = self._filter_score_threshold(raw, min_score=min_score)
            filtered_out_low_score_total += max(0, len(raw) - len(filtered))

            for match in filtered:
                key = self._dedupe_key(match)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                qualified_raw_matches.append(match)
                if len(qualified_raw_matches) >= target_size:
                    stop_reason = "filled_target"
                    break

            if len(qualified_raw_matches) >= target_size:
                break

        if (
            stop_reason != "filled_target"
            and len(fallback_bases_tried) >= max_fallback_bases
            and len(discovered_bases) > len(fallback_bases_tried) + (1 if primary_base else 0)
        ):
            stop_reason = "fallback_cap_reached"

        qualified_raw_matches = sorted(
            qualified_raw_matches,
            key=self._safe_score,
            reverse=True,
        )[:target_size]

        normalized = [normalize_match(match) for match in qualified_raw_matches]
        joined = join_issue_context(normalized)
        context = sorted(joined, key=lambda it: float(it.get("score") or 0.0), reverse=True)[
            :target_size
        ]

        mix = Counter(item.get("doc_type") for item in context)
        logger.info(
            "[RAG] primary_base=%s target_size=%s qualified_collected=%s namespaces_queried=%s fallback_bases_tried=%s filtered_out_low_score_total=%s stop_reason=%s total_raw=%s final_context=%s best_score=%.4f mix=%s",
            primary_base,
            target_size,
            len(qualified_raw_matches),
            queried_namespaces,
            fallback_bases_tried,
            filtered_out_low_score_total,
            stop_reason,
            len(all_raw_matches),
            len(context),
            self._best_score(qualified_raw_matches),
            dict(mix),
        )

        # Keep structure expected by prompt_utils/analogical agent.
        result = [
            {
                "id": item.get("id"),
                "title": item.get("title"),
                "description": item.get("description") or item.get("snippet") or "",
                "estimated_hours": item.get("estimated_hours"),
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
