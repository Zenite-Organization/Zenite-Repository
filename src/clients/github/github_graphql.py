import httpx
from typing import Any, Dict
from .github_auth import GitHubAuth

class GitHubGraphQL:
    def __init__(self, auth: GitHubAuth, api_url="https://api.github.com/graphql"):
        self.auth = auth
        self.api_url = api_url

    async def query(self, query: str, variables: Dict[str, Any]) -> Dict[str, Any]:
        token = await self.auth.ensure_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/vnd.github+json",
        }
        async with httpx.AsyncClient() as client:
            resp = await client.post(self.api_url, json={"query": query, "variables": variables}, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            if "errors" in data:
                raise RuntimeError(f"Erro GraphQL: {data['errors']}")
            return data
