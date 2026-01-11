from __future__ import annotations
from typing import Any, Dict, Optional, List
import httpx
import asyncio
import time
import datetime
import jwt
import os
from dotenv import load_dotenv

from .base import ProjectProvider
from .github_auth import GitHubAuth
from .github_graphql import GitHubGraphQL
import requests
from config.settings import settings

class GitHubProjectProvider(ProjectProvider):
    def __init__(self, app_id: str, private_key: str, installation_id: int):
        self.auth = GitHubAuth(app_id, private_key, installation_id)
        self.graphql = GitHubGraphQL(self.auth)

    # ------------------------------------------------------
    # ADD COMMENT (usando Node ID)
    # ------------------------------------------------------
    async def add_comment(self, subject_id: str, body: str):
        mutation = """
        mutation AddComment($input: AddCommentInput!) {
            addComment(input: $input) {
                clientMutationId
                commentEdge {
                    node {
                        id
                        body
                        createdAt
                        author {
                            login
                        }
                    }
                }
                subject {
                    ... on Issue {
                        id
                        title
                        body
                    }
                    ... on PullRequest {
                        id
                        title
                        body
                    }
                }
            }
        }
        """

        variables = {"input": {"subjectId": subject_id, "body": body}}
        return await self.graphql.query(mutation, variables)


    async def full_resolve_issue(self, issue_node_id: str):
        query = """
        query FullResolve($id: ID!) {
            node(id: $id) {
                ... on Issue {
                    id
                    title
                    repository {
                        name
                        owner {
                            ... on Organization {
                                projectsV2(first: 20) {
                                    nodes {
                                        id
                                        title
                                        fields(first: 50) {
                                            nodes {
                                                __typename
                                                ... on ProjectV2SingleSelectField {
                                                    id
                                                    name
                                                    dataType
                                                    options {
                                                        id
                                                        name
                                                        color
                                                    }
                                                }
                                                ... on ProjectV2IterationField {
                                                    id
                                                    name
                                                    dataType
                                                }
                                                ... on ProjectV2FieldCommon {
                                                    id
                                                    name
                                                    dataType
                                                }
                                            }
                                        }
                                        items(first: 50) {
                                            nodes {
                                                id
                                                content {
                                                    ... on Issue {
                                                        id
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        """

        data = await self.graphql.query(query, {"id": issue_node_id})
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

            mutation = """
            mutation UpdateItemField($input: UpdateProjectV2ItemFieldValueInput!) {
                updateProjectV2ItemFieldValue(input: $input) {
                    projectV2Item { id }
                }
            }
            """

            variables = {
                "input": {
                    "projectId": project_id,
                    "itemId": item_id,
                    "fieldId": field_id,
                    "value": {"number": estimate_value}
                }
            }

            try:
                resp = await self.graphql.query(mutation, variables)
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
    # ASSIGN ITERATION BY NAME (ProjectV2 Iteration field)
    # ------------------------------------------------------
    async def assign_iteration_by_name(self, issue_node_id: str, iteration_name: str):
        """Try to assign an Iteration (ProjectV2) with given name to the issue across projects.
        If the iteration with that name exists in the project, update the item's iteration field.
        Returns a list of results per project.
        """
        resolved = await self.full_resolve_issue(issue_node_id)
        results = []

        for proj in resolved.get("projects", []):
            project_id = proj.get("project_id")
            item_id = proj.get("item_id")
            # find iteration field id from full_resolve_issue fields if available
            fields = proj.get("fields") or []
            iteration_field_id = None
            # Note: full_resolve_issue currently returns only estimate_field_id and item_id;
            # but it also fetched fields earlier — fallback to querying project for iteration field below
            for f in fields:
                if f.get("__typename") == "ProjectV2IterationField":
                    iteration_field_id = f.get("id")
                    break

            # If we don't have item_id or project_id, skip
            if not project_id or not item_id:
                results.append({"project_id": project_id, "item_id": item_id, "skipped": True, "reason": "missing project or item id"})
                continue

            # Query project iterations to find matching title
            query = """
            query ProjectIterations($id: ID!) {
                node(id: $id) {
                    ... on ProjectV2 {
                        id
                        title
                        iterations(first:100) {
                            nodes {
                                id
                                title
                            }
                        }
                        fields(first:50) {
                            nodes {
                                __typename
                                ... on ProjectV2IterationField { id name dataType }
                                ... on ProjectV2FieldCommon { id name dataType }
                            }
                        }
                    }
                }
            }
            """

            try:
                data = await self.graphql.query(query, {"id": project_id})
                node = data.get("data", {}).get("node") or {}
                proj_node = node
                iterations = (proj_node.get("iterations", {}).get("nodes") or [])
                fields_nodes = (proj_node.get("fields", {}).get("nodes") or [])

                # find iteration field id if missing
                if not iteration_field_id:
                    for f in fields_nodes:
                        if f.get("__typename") == "ProjectV2IterationField":
                            iteration_field_id = f.get("id")
                            break

                found_iter = None
                for it in iterations:
                    if str(it.get("title")).strip() == str(iteration_name).strip():
                        found_iter = it
                        break

                if not found_iter:
                    results.append({"project_id": project_id, "item_id": item_id, "skipped": True, "reason": "iteration_not_found"})
                    continue

                if not iteration_field_id:
                    results.append({"project_id": project_id, "item_id": item_id, "skipped": True, "reason": "iteration_field_not_found"})
                    continue

                mutation = """
                mutation UpdateItemField($input: UpdateProjectV2ItemFieldValueInput!) {
                    updateProjectV2ItemFieldValue(input: $input) {
                        projectV2Item { id }
                    }
                }
                """

                variables = {
                    "input": {
                        "projectId": project_id,
                        "itemId": item_id,
                        "fieldId": iteration_field_id,
                        "value": {"iterationId": found_iter.get("id")}
                    }
                }

                resp = await self.graphql.query(mutation, variables)
                results.append({"project_id": project_id, "item_id": item_id, "iteration_id": found_iter.get("id"), "result": resp})

            except Exception as e:
                results.append({"project_id": project_id, "item_id": item_id, "error": str(e)})

        return {"assigned": results}

    # ------------------------------------------------------
    # LIST BACKLOG ISSUES (REST, paginate)
    # ------------------------------------------------------
    async def list_backlog_issues(self, repo_full_name: str, label: str = "Backlog"):
        """Return a list of open issues that have the given label.
        Each item is a dict with at least: node_id, number, title, body, labels, created_at
        """
        owner, repo = repo_full_name.split("/")
        per_page = 100
        page = 1
        issues: List[Dict[str, Any]] = []
        token = await self.auth.ensure_token()
        headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github+json"}
        async with httpx.AsyncClient() as client:
            while True:
                url = f"https://api.github.com/repos/{owner}/{repo}/issues"
                params = {"state": "open", "labels": label, "per_page": per_page, "page": page}
                resp = await client.get(url, headers=headers, params=params)
                if resp.status_code >= 400:
                    resp.raise_for_status()
                data = resp.json()
                if not data:
                    break
                for it in data:
                    issues.append({
                        "node_id": it.get("node_id"),
                        "number": it.get("number"),
                        "title": it.get("title"),
                        "body": it.get("body"),
                        "labels": it.get("labels"),
                        "created_at": it.get("created_at"),
                    })
                if len(data) < per_page:
                    break
                page += 1

        return issues

    # ------------------------------------------------------
    # GET SPRINT CAPACITY
    # ------------------------------------------------------
    async def get_sprint_capacity(self, repo_full_name: str, trigger_label: str | None = None) -> int | None:
        """Try to determine sprint capacity in hours.
        Strategies:
        - parse a duration from trigger_label (e.g., '2w', '1w', '10d')
        - look for a milestone with title == trigger_label and compute days until due_on
        - return None if unknown
        """
        owner, repo = repo_full_name.split("/")
        token = await self.auth.ensure_token()
        headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github+json"}

        # try parse label like '2w' or '10d'
        if trigger_label:
            import re

            m = re.search(r"(\d+)\s*(w|d)", trigger_label.lower())
            if m:
                n = int(m.group(1))
                unit = m.group(2)
                days = n * 7 if unit == "w" else n
                return days * settings.WORK_HOURS_PER_DAY

            # try parse plain integer as days
            if trigger_label.strip().isdigit():
                days = int(trigger_label.strip())
                return days * settings.WORK_HOURS_PER_DAY

        # fallback: try to find a milestone with title == trigger_label
        if trigger_label:
            async with httpx.AsyncClient() as client:
                url = f"https://api.github.com/repos/{owner}/{repo}/milestones"
                params = {"state": "open", "per_page": 100}
                resp = await client.get(url, headers=headers, params=params)
                if resp.status_code == 200:
                    for m in resp.json():
                        if str(m.get("title")) == str(trigger_label):
                            due_on = m.get("due_on")
                            if due_on:
                                try:
                                    from dateutil.parser import isoparse

                                    dt = isoparse(due_on)
                                except Exception:
                                    dt = datetime.datetime.fromisoformat(due_on.replace("Z", "+00:00"))
                                days = (dt - datetime.datetime.utcnow()).days
                                if days < 0:
                                    days = 0
                                return days * settings.WORK_HOURS_PER_DAY

        return None

    # ------------------------------------------------------
    # CREATE SPRINT (milestone)
    # ------------------------------------------------------
    async def create_sprint(self, repo_full_name: str, title: str, days: int = 14) -> Dict[str, Any]:
        owner, repo = repo_full_name.split("/")
        token = await self.auth.ensure_token()
        headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github+json"}
        due_date = (datetime.datetime.utcnow() + datetime.timedelta(days=days)).isoformat() + "Z"
        payload = {"title": title, "due_on": due_date}
        url = f"https://api.github.com/repos/{owner}/{repo}/milestones"
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            return resp.json()

    # ------------------------------------------------------
    # MOVE ISSUE TO SPRINT (assign milestone)
    # ------------------------------------------------------
    async def move_issue_to_sprint(self, issue_number: int, repo_full_name: str, milestone_number: int):
        owner, repo = repo_full_name.split("/")
        token = await self.auth.ensure_token()
        headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github+json"}
        url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}"
        payload = {"milestone": milestone_number}
        async with httpx.AsyncClient() as client:
            resp = await client.patch(url, headers=headers, json=payload)
            resp.raise_for_status()
            return resp.json()


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

    return GitHubProjectProvider(app_id=app_id, private_key=private_key, installation_id=installation_id)