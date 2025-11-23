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