import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from ai.core.pinecone_vector_store import PineconeVectorStoreClient
from ai.core.rag_namespace_policy import extract_project_issue_namespace, extract_project_name
from config.settings import settings
from web.schemas.github_payloads import GitHubIssuesWebhookPayload


logger = logging.getLogger(__name__)


def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    raw = str(ts).strip()
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


@dataclass
class IndexClosedIssueResult:
    skipped: bool
    reason: str | None
    namespace: str | None
    vector_id: str | None
    total_effort_hours: float | None
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skipped": self.skipped,
            "reason": self.reason,
            "namespace": self.namespace,
            "vector_id": self.vector_id,
            "total_effort_hours": self.total_effort_hours,
            "metadata": self.metadata,
        }


class IndexClosedIssueUseCase:
    def __init__(self, vector_store: Any | None = None):
        self.vector_store = vector_store or PineconeVectorStoreClient()

    async def execute(self, payload: GitHubIssuesWebhookPayload) -> Dict[str, Any]:
        if not payload.issue:
            return IndexClosedIssueResult(
                skipped=True,
                reason="missing_issue",
                namespace=None,
                vector_id=None,
                total_effort_hours=None,
                metadata={},
            ).to_dict()

        repo_full_name = payload.repository.full_name
        issue_number = payload.issue.number
        title = (payload.issue.title or "").strip()
        body = (payload.issue.body or "").strip()

        if not title and not body:
            return IndexClosedIssueResult(
                skipped=True,
                reason="empty_issue_text",
                namespace=None,
                vector_id=None,
                total_effort_hours=None,
                metadata={},
            ).to_dict()

        namespace = extract_project_issue_namespace(repo_full_name)
        if not namespace:
            return IndexClosedIssueResult(
                skipped=True,
                reason="missing_namespace",
                namespace=None,
                vector_id=None,
                total_effort_hours=None,
                metadata={},
            ).to_dict()

        vector_id = f"{repo_full_name}#{issue_number}"
        project_key = extract_project_name(repo_full_name)

        created_at = getattr(getattr(payload.issue, "timestamps", None), "created_at", None)
        closed_at = getattr(getattr(payload.issue, "timestamps", None), "closed_at", None)
        created_dt = _parse_iso(created_at)
        closed_dt = _parse_iso(closed_at)

        effort_hours: float = 1.0
        effort_fallback = False
        if created_dt and closed_dt and closed_dt >= created_dt:
            delta_days = (closed_dt - created_dt).total_seconds() / 86400.0
            effort_hours = delta_days * float(settings.WORK_HOURS_PER_DAY)
            effort_hours = round(_clamp(effort_hours, 1.0, 300.0), 2)
        else:
            effort_fallback = True

        labels = [lbl.name for lbl in (payload.issue.labels or []) if getattr(lbl, "name", None)]

        metadata: Dict[str, Any] = {
            "doc_type": "issue",
            "issue_id": issue_number,
            "issue_key": vector_id,
            "issue_title": title,
            "repository": repo_full_name,
            "project_key": project_key,
            "state": "closed",
            "description": body,
            "total_effort_hours": effort_hours,
            "created_at": created_at,
            "closed_at": closed_at,
            "labels": labels,
            "effort_fallback": effort_fallback,
        }

        # If Pinecone is not configured, do a no-op to avoid GitHub retries.
        if hasattr(self.vector_store, "_ready") and not bool(getattr(self.vector_store, "_ready")):
            return IndexClosedIssueResult(
                skipped=True,
                reason="rag_disabled",
                namespace=namespace,
                vector_id=vector_id,
                total_effort_hours=effort_hours,
                metadata=metadata,
            ).to_dict()

        text = "\n\n".join([p for p in [title, body] if p]).strip()
        doc = {"id": vector_id, "namespace": namespace, "text": text, "metadata": metadata}

        try:
            print(
                "[index_closed_issue] pinecone upsert start "
                f"repo={repo_full_name} issue={issue_number} namespace={namespace} "
                f"vector_id={vector_id} text_len={len(text)}",
                flush=True,
            )
            result = self.vector_store.upsert([doc])
            print(
                "[index_closed_issue] pinecone upsert ok "
                f"repo={repo_full_name} issue={issue_number} namespace={namespace} vector_id={vector_id}",
                flush=True,
            )
            logger.info(
                "Indexed closed issue repo=%s issue=%s namespace=%s",
                repo_full_name,
                issue_number,
                namespace,
            )
            if isinstance(result, dict):
                metadata = dict(metadata)
                metadata["upsert_result"] = result
        except NotImplementedError:
            print(
                "[index_closed_issue] pinecone upsert skipped (read-only) "
                f"repo={repo_full_name} issue={issue_number} namespace={namespace} vector_id={vector_id}",
                flush=True,
            )
            return IndexClosedIssueResult(
                skipped=True,
                reason="vector_store_read_only",
                namespace=namespace,
                vector_id=vector_id,
                total_effort_hours=effort_hours,
                metadata=metadata,
            ).to_dict()
        except Exception as exc:
            print(
                "[index_closed_issue] pinecone upsert failed "
                f"repo={repo_full_name} issue={issue_number} namespace={namespace} vector_id={vector_id} error={exc}",
                flush=True,
            )
            logger.exception(
                "Failed to index closed issue repo=%s issue=%s namespace=%s: %s",
                repo_full_name,
                issue_number,
                namespace,
                exc,
            )
            metadata = dict(metadata)
            metadata["index_error"] = str(exc)

        return IndexClosedIssueResult(
            skipped=False,
            reason=None,
            namespace=namespace,
            vector_id=vector_id,
            total_effort_hours=effort_hours,
            metadata=metadata,
        ).to_dict()
