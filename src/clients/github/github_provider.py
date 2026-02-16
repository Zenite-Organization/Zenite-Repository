from __future__ import annotations
import datetime as dt
from typing import Any

from ai.dtos.issues_estimation_dto import IssueEstimationDTO
from config.settings import settings
from clients.github.graphql.comments import ADD_COMMENT_MUTATION
from clients.github.graphql.projects import (
    FULL_RESOLVE_ISSUE_QUERY,
    ISSUE_PROJECTS_BY_NODE_QUERY,
    PROJECT_ISSUES_WITH_FIELDS_QUERY,
    PROJECT_ITERATION_DURATION_QUERY,
    PROJECT_ITERATION_FIELD_QUERY,
    UPDATE_ISSUE_SPRINT_MUTATION,
    UPDATE_ITEM_FIELD_MUTATION,
)

from .base import ProjectProvider
from .github_auth import GitHubAuth
from .github_graphql import GitHubGraphQL

class GitHubProjectProvider(ProjectProvider):
    # ------------------------------------------------------
    # Funções auxiliares para campos customizados
    # ------------------------------------------------------
    @staticmethod
    def extract_custom_fields(item: dict) -> dict:
        fields = {}

        for node in item.get("fieldValues", {}).get("nodes", []):
            field = node.get("field")
            if not field:
                continue  # 🔥 ignora {}

            field_name = field.get("name")
            if not field_name:
                continue

            # SINGLE_SELECT
            # single select value
            if "name" in node:
                fields[field_name] = node["name"]
                continue

            # iteration value (tem 'iteration' com id e title)
            if node.get("iteration"):
                fields[field_name] = node.get("iteration")
                continue

            # fallback: store the node itself
            fields[field_name] = node

        return fields


    @staticmethod
    def get_status(custom_fields: dict) -> str | None:
        return custom_fields.get("Status")


    def __init__(self, app_id: str, private_key: str, installation_id: int):
        self.auth = GitHubAuth(app_id, private_key, installation_id)
        self.graphql = GitHubGraphQL(self.auth)

    # ------------------------------------------------------
    # ADD COMMENT (usando Node ID)
    # ------------------------------------------------------
    async def add_comment(self, subject_id: str, body: str):
        variables = {"input": {"subjectId": subject_id, "body": body}}
        return await self.graphql.query(ADD_COMMENT_MUTATION, variables)


    async def full_resolve_issue(self, issue_node_id: str):
        data = await self.graphql.query(FULL_RESOLVE_ISSUE_QUERY, {"id": issue_node_id})
        node = data.get("data", {}).get("node")
        if not node:
            return {"projects": []}

        projects = (
            node.get("repository", {})
                .get("owner", {})
                .get("projectsV2", {})
                .get("nodes", [])
        )

        results = []

        for proj in projects:
            project_id = proj.get("id")
            project_title = proj.get("title")

            # Encontrar ID do campo Estimate
            fields = proj.get("fields", {}).get("nodes", []) or []
            estimate_field_id = None
            for f in fields:
                name = (f.get("name") or "").lower()
                if "estim" in name or "hours" in name or "horas" in name:
                    estimate_field_id = f.get("id")
                    break

            # Encontrar item_id do Issue dentro do projeto
            items = proj.get("items", {}).get("nodes", []) or []
            issue_item_id = None
            for item in items:
                if item.get("content", {}).get("id") == issue_node_id:
                    issue_item_id = item.get("id")
                    break

            results.append({
                "project_id": project_id,
                "project_title": project_title,
                "estimate_field_id": estimate_field_id,
                "item_id": issue_item_id
            })

        return {"projects": results}


    async def update_estimate(self, issue_node_id: str, estimate_value: float | int):
        resolved = await self.full_resolve_issue(issue_node_id)
        results = []
        for proj in resolved.get("projects", []):
            project_id = proj["project_id"]
            item_id = proj["item_id"]
            field_id = proj["estimate_field_id"]

            if not field_id or not item_id:
                results.append({
                    "project_id": project_id,
                    "item_id": item_id,
                    "skipped": True,
                    "reason": "Missing Estimate field or item_id"
                })
                continue

            variables = {
                "input": {
                    "projectId": project_id,
                    "itemId": item_id,
                    "fieldId": field_id,
                    "value": {"number": estimate_value}
                }
            }

            try:
                resp = await self.graphql.query(UPDATE_ITEM_FIELD_MUTATION, variables)
                results.append({
                    "project_id": project_id,
                    "item_id": item_id,
                    "field_id": field_id,
                    "result": resp
                })
            except Exception as e:
                results.append({
                    "project_id": project_id,
                    "item_id": item_id,
                    "field_id": field_id,
                    "error": str(e)
                })

        return {"updated": results}

    # ------------------------------------------------------
# LIST BACKLOG ISSUES (GraphQL, paginate)
# ------------------------------------------------------

    async def list_backlog_issues(
        self,
        repo_full_name: str,
        project_number: int,
        backlog_status: str = "Backlog",
    ) -> list:
        owner, _ = repo_full_name.split("/")
        results = []
        has_next_page = True
        cursor = None

        while has_next_page:
            variables = {
                "owner": owner,
                "projectNumber": project_number,
                "after": cursor,
            }

            response = await self.graphql.query(
                PROJECT_ISSUES_WITH_FIELDS_QUERY,
                variables
            )

            project = (
                response
                .get("data", {})
                .get("organization", {})
                .get("projectV2", {})
            )

            items = project.get("items", {})
            page_info = items.get("pageInfo", {})
            has_next_page = page_info.get("hasNextPage", False)
            cursor = page_info.get("endCursor")

            for item in items.get("nodes", []):
                issue = item.get("content")
                if not issue:
                    continue

                # ---- Custom fields (Status, Iteration, etc)
                custom_fields = self.extract_custom_fields(item)
                status = self.get_status(custom_fields)
                print("status:", status)

                # 🔥 FILTRO DE BACKLOG (opcional)
                # if status != backlog_status:
                #     continue

                labels = [l["name"] for l in issue["labels"]["nodes"]]
                assignees = [a["login"] for a in issue["assignees"]["nodes"]]

                created_at = issue.get("createdAt")
                created_dt = (
                    dt.datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                    if created_at else None
                )

                age_in_days = (
                    (dt.datetime.now(dt.timezone.utc) - created_dt).days
                    if created_dt else 0
                )

                repo = issue["repository"]

                dto = IssueEstimationDTO(
                    issue_number=issue["number"],
                    repository=repo["nameWithOwner"],
                    title=issue.get("title") or "",
                    description=issue.get("body") or "",
                    labels=labels,
                    assignees=assignees,
                    state=issue.get("state") or "UNKNOWN",
                    is_open=issue.get("state") == "OPEN",
                    comments_count=issue["comments"]["totalCount"],
                    age_in_days=age_in_days,
                    author_login=issue["author"]["login"] if issue.get("author") else "unknown",
                    author_role=issue.get("authorAssociation") or "NONE",
                    repo_language=(
                        repo["primaryLanguage"]["name"]
                        if repo.get("primaryLanguage")
                        else None
                    ),
                    repo_size=repo.get("diskUsage"),
                    is_estimation_issue=any("estimate" in l.lower() for l in labels),
                    has_assignee=len(assignees) > 0,
                    has_description=bool(issue.get("body")),
                )

                results.append(dto)

        return results



    async def get_project_id(self, issue_node_id: str) -> dict:
        variables = {"issueId": issue_node_id}
        response = await self.graphql.query(ISSUE_PROJECTS_BY_NODE_QUERY, variables)

        nodes = (
            response
            .get("data", {})
            .get("node", {})
            .get("projectItems", {})
            .get("nodes", [])
        )

        if not nodes:
            raise ValueError("Issue não pertence a nenhum Project")

        # Find the oldest project by createdAt
        def parse_created_at(n):
            return dt.datetime.fromisoformat(n["createdAt"].replace("Z", "+00:00"))

        oldest = min(nodes, key=parse_created_at)
        oldest_number = oldest["project"]["number"]
        oldest_id = oldest["project"]["id"]

        return {"number": oldest_number, "id": oldest_id}

    async def get_project_iterations(self, project_id: str) -> list:
        variables = {"projectId": project_id}
        response = await self.graphql.query(PROJECT_ITERATION_DURATION_QUERY, variables)

        fields = (
            response
            .get("data", {})
            .get("node", {})
            .get("fields", {})
            .get("nodes", [])
        )

        iteration_field = next(
            (f for f in fields if f.get("configuration", {}).get("iterations")),
            None
        )

        if not iteration_field:
            raise ValueError("Project não possui campo Iteration")

        iterations = iteration_field["configuration"]["iterations"]

        return iterations

    async def get_iteration_field(self, project_id: str) -> dict:
        """Return the iteration field id and the iterations list for a project node id.

        Returns: {"field_id": str|None, "iterations": list}
        """
        variables = {"projectId": project_id}
        response = await self.graphql.query(PROJECT_ITERATION_FIELD_QUERY, variables)

        fields = (
            response
            .get("data", {})
            .get("node", {})
            .get("fields", {})
            .get("nodes", [])
        )

        for f in fields:
            config = f.get("configuration")
            if config and config.get("iterations") is not None:
                return {"field_id": f.get("id"), "iterations": config.get("iterations", [])}

        return {"field_id": None, "iterations": []}


    async def move_issue_to_sprint(self, project_id: str, item_id: str, field_id: str, iteration_id: str) -> dict:
        """Execute the GraphQL mutation to set the iteration field value for a project item.

        Expects caller to supply correct `project_id` (node id), `item_id`, `field_id` and
        `iteration_id` (the iteration id string). Returns the raw GraphQL response.
        """
        variables = {
            "projectId": project_id,
            "itemId": item_id,
            "fieldId": field_id,
            "iterationId": iteration_id,
        }

        resp = await self.graphql.query(UPDATE_ISSUE_SPRINT_MUTATION, variables)
        return resp


def get_provider_for_installation(installation_id: int) -> GitHubProjectProvider:
    """Create a GitHubProjectProvider using `settings.APP_ID` and `settings.APP_PRIVATE_KEY`.
    If `APP_PRIVATE_KEY` is not set but `APP_PRIVATE_KEY_path` is, the key is loaded from the file.
    """
    app_id = settings.APP_ID
    private_key = settings.APP_PRIVATE_KEY
    if not private_key and settings.APP_PRIVATE_KEY_path:
        try:
            with open(settings.APP_PRIVATE_KEY_path, "r", encoding="utf-8") as f:
                private_key = f.read()
        except Exception:
            private_key = None

    if not app_id or not private_key:
        raise ValueError("GitHub App credentials are not configured")

    return GitHubProjectProvider(
        app_id=app_id,
        private_key=private_key,
        installation_id=installation_id,
    )
