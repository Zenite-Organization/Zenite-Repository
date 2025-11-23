from __future__ import annotations
from typing import Any, Dict, Optional, List
import httpx
import asyncio
import time
import datetime
import jwt
import os
from dotenv import load_dotenv

load_dotenv("C:/projetos/zenite/.env")
from .base import ProjectProvider
import requests
from config.settings import settings

class GitHubProjectProvider(ProjectProvider):

    def __init__(self, api_url="https://api.github.com/graphql", installation_id: Optional[int] = None):
        self.api_url = api_url

        self.app_id = os.getenv("GITHUB_APP_ID")
        private_key_path = "C:\\projetos\\zenite\\private_key.pem"
        try:
            with open(private_key_path, "r", encoding="utf-8") as f:
                self.private_key = f.read()
        except Exception as e:
            raise RuntimeError(f"Falha ao ler a chave privada em {private_key_path}: {e}")

        if not self.app_id or not self.private_key:
            raise RuntimeError("GITHUB_APP_ID e private_key são obrigatórios")

        # installation ID recebido via request
        if not installation_id:
            raise RuntimeError("installation_id é obrigatório ao instanciar GitHubProjectProvider")

        self.installation_id = installation_id

        # cache de token
        self.installation_token: Optional[str] = None
        self.installation_expires_at: Optional[float] = None

    # ------------------------------------------------------
    # JWT para autenticar como GitHub App (10 minutos)
    # ------------------------------------------------------
    def _generate_jwt(self) -> str:
        now = int(time.time())
        payload = {
            "iat": now - 10,
            "exp": now + 600,  # máximo 10 minutos
            "iss": self.app_id
        }
        token = jwt.encode(payload, self.private_key, algorithm="RS256")
        # PyJWT 2.x já retorna string, mas garantir
        if isinstance(token, bytes):
            token = token.decode("utf-8")
        return token

    # Gera Installation Access Token usando installation_id
    # ------------------------------------------------------
    async def _generate_installation_token(self) -> str:
        jwt_token = self._generate_jwt()

        url = f"https://api.github.com/app/installations/{self.installation_id}/access_tokens"

        headers = {
            "Authorization": f"Bearer {jwt_token}",
            "Accept": "application/vnd.github+json",
        }
        response = requests.post(url, headers=headers)

        max_retries = 3
        delay = 2  # segundos
        last_exception = None
        for attempt in range(1, max_retries + 1):
            async with httpx.AsyncClient() as client:
                try:
                    resp = await client.post(url, headers=headers)
                    resp.raise_for_status()
                    data = resp.json()
                    token = data.get("token")
                    expires_at = data.get("expires_at")
                    if not token:
                        raise RuntimeError("Falha ao gerar Installation Access Token")
                    # converter expires_at → timestamp
                    dt = datetime.datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
                    self.installation_expires_at = dt.timestamp()
                    self.installation_token = token
                    return token
                except httpx.HTTPStatusError as e:
                    print(f"Tentativa {attempt} - Erro ao obter installation token: {e.response.status_code} {e.response.text}")
                    last_exception = e
                except Exception as e:
                    print(f"Tentativa {attempt} - Erro inesperado ao obter installation token: {e}")
                    last_exception = e
            if attempt < max_retries:
                print(f"Aguardando {delay} segundos para tentar novamente...")
                await asyncio.sleep(delay)
        # Se chegou aqui, todas as tentativas falharam
        print("Todas as tentativas de obter installation token falharam.")
        if last_exception:
            raise last_exception
        raise RuntimeError("Falha ao gerar Installation Access Token após múltiplas tentativas.")

    # ------------------------------------------------------
    # Garante token válido em cache
    # ------------------------------------------------------
    async def _ensure_token(self) -> str:
        # token válido em cache?
        # if self.installation_token and self.installation_expires_at:
        #     if time.time() < (self.installation_expires_at - 30):
        #         return self.installation_token
       
        # renova token
        return await self._generate_installation_token()

    # ------------------------------------------------------
    # GraphQL call
    # ------------------------------------------------------
    async def _graphql(self, query: str, variables: Dict[str, Any]) -> Dict[str, Any]:
        token = await self._ensure_token()

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/vnd.github+json"
        }

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                self.api_url,
                json={"query": query, "variables": variables},
                headers=headers
            )
            resp.raise_for_status()
            data = resp.json()

            if "errors" in data:
                raise RuntimeError(f"Erro GraphQL: {data['errors']}")

            return data

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

        variables = {
            "input": {
                "subjectId": subject_id,
                "body": body
            }
        }

        return await self._graphql(mutation, variables)

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

        data = await self._graphql(query, {"id": issue_node_id})
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
                    projectV2Item {
                        id
                    }
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
                resp = await self._graphql(mutation, variables)
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

    def _find_estimate_field(self, fields: List[Dict[str, Any]]) -> Optional[str]:
        """
        Procura o field de estimate:
            1) se `settings.github_project_field_estimate_id` estiver setado, retorna ele;
            2) fallback: tenta achar por nome (contendo 'estim', 'estimate', 'horas', etc).
        """
        cfg = getattr(settings, "github_project_field_estimate_id", None)
        if cfg:
            return cfg

        for f in fields:
            name = (f.get("name") or "").lower()
            if any(k in name for k in ("estim", "estimate", "horas", "hours", "tempo")):
                return f.get("id")
        return None