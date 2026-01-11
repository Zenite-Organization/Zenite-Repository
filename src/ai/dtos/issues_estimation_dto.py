from pydantic import BaseModel
from typing import List, Optional


class IssueEstimationDTO(BaseModel):
    # Identificação
    issue_number: int
    repository: str

    # Conteúdo
    title: str
    description: str

    # Contexto semântico
    labels: List[str]
    assignees: List[str]

    # Estado
    state: str
    is_open: bool

    # Métricas
    comments_count: int
    age_in_days: int

    # Autor
    author_login: str
    author_role: str  # OWNER | MEMBER | CONTRIBUTOR | NONE

    # Repositório
    repo_language: Optional[str]
    repo_size: Optional[int]

    # Heurísticas úteis
    is_estimation_issue: bool
    has_assignee: bool
    has_description: bool


from datetime import datetime, timezone


def days_between(start: str | None) -> int:
    if not start:
        return 0
    created = datetime.fromisoformat(start.replace("Z", "+00:00"))
    return (datetime.now(timezone.utc) - created).days

def map_issue_to_estimation_dto(payload) -> IssueEstimationDTO:
    issue = payload.issue
    repo = payload.repository

    labels = [label.name for label in issue.labels or []]
    assignees = [a.login for a in issue.assignees or []]

    return IssueEstimationDTO(
        issue_number=issue.number,
        repository=repo.full_name,

        title=issue.title or "",
        description=issue.body or "",

        labels=labels,
        assignees=assignees,

        state=issue.state or "unknown",
        is_open=issue.state == "open",

        comments_count=(getattr(issue, "metrics", None).comments_count if getattr(issue, "metrics", None) and hasattr(issue.metrics, "comments_count") else 0),
        age_in_days=days_between(issue.timestamps.created_at),

        author_login=issue.user.login if issue.user else "unknown",
        author_role=issue.author_association or "NONE",

        repo_language=repo.language,
        repo_size=repo.size,

        is_estimation_issue="estimate" in [l.lower() for l in labels],
        has_assignee=len(assignees) > 0,
        has_description=bool(issue.body),
    )
