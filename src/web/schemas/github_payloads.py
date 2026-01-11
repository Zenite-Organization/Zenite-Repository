from pydantic import BaseModel, model_validator
from typing import Optional, List, Dict, Any

class InstallationPayload(BaseModel):
    id: int

class UserPayload(BaseModel):
    login: str
    id: Optional[int] = None
    type: Optional[str] = None

class LabelPayload(BaseModel):
    id: int
    name: str
    color: Optional[str] = None
    description: Optional[str] = None

class AssigneePayload(BaseModel):
    login: str
    id: Optional[int] = None

class RepositoryPayload(BaseModel):
    full_name: str
    name: Optional[str] = None
    language: Optional[str] = None
    size: Optional[int] = None
    open_issues_count: Optional[int] = None
    default_branch: Optional[str] = None

class IssueTimestampsPayload(BaseModel):
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    closed_at: Optional[str] = None

class IssueMetricsPayload(BaseModel):
    comments_count: Optional[int] = None
    sub_issues_total: Optional[int] = None
    blocked_by: Optional[int] = None
    blocking: Optional[int] = None

class IssueAuthorPayload(BaseModel):
    login: str
    association: Optional[str] = None

class IssuePayload(BaseModel):
    node_id: str
    number: int

    title: Optional[str] = None
    body: Optional[str] = None

    state: Optional[str] = None
    state_reason: Optional[str] = None

    labels: List[LabelPayload] = []
    assignees: List[AssigneePayload] = []

    user: Optional[UserPayload] = None
    author_association: Optional[str] = None

    timestamps: Optional[IssueTimestampsPayload] = None
    metrics: Optional[IssueMetricsPayload] = None

    milestone: Optional[Any] = None
    changes: Optional[Dict[str, Any]] = None

    @model_validator(mode="before")
    @classmethod
    def enrich_nested_fields(cls, data: Dict[str, Any]):
        if not isinstance(data, dict):
            return data

        data["timestamps"] = IssueTimestampsPayload(
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            closed_at=data.get("closed_at"),
        )

        data["metrics"] = IssueMetricsPayload(
            comments_count=data.get("comments"),
            sub_issues_total=(data.get("sub_issues_summary") or {}).get("total"),
            blocked_by=(data.get("issue_dependencies_summary") or {}).get("blocked_by"),
            blocking=(data.get("issue_dependencies_summary") or {}).get("blocking"),
        )

        return data

class CommentPayload(BaseModel):
    node_id: str
    id: int
    body: Optional[str]
    user: Optional[UserPayload]
    created_at: Optional[str]
    updated_at: Optional[str]


class EstimateUpdatePayload(BaseModel):
    project_id: str
    item_id: str
    field_id: str
    estimate_value: float | int


class GitHubIssuesWebhookPayload(BaseModel):
    action: str

    issue: Optional[IssuePayload] = None
    repository: RepositoryPayload
    installation: InstallationPayload

    sender: Optional[UserPayload] = None
    estimate_update: Optional[EstimateUpdatePayload] = None
