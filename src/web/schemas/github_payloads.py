from pydantic import BaseModel
from typing import Optional

class InstallationPayload(BaseModel):
    id: int

class RepositoryPayload(BaseModel):
    full_name: str

class CommentPayload(BaseModel):
    node_id: str
    id: int
    body: Optional[str]
    user: Optional[dict]
    created_at: Optional[str]
    updated_at: Optional[str]

class IssuePayload(BaseModel):
    node_id: str
    number: int
    title: Optional[str]
    body: Optional[str]

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
    sender: Optional[dict] = None
    estimate_update: Optional[EstimateUpdatePayload] = None
