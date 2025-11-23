from __future__ import annotations
import time
from typing import Optional
import httpx
import jwt
import os


def generate_jwt(app_id: str, private_key_pem: str, expire_seconds: int = 600) -> str:
    """Generate a JWT for GitHub App authentication (RS256).

    Args:
        app_id: numeric app id as string
        private_key_pem: private key in PEM format
        expire_seconds: expiration window in seconds (default 10 minutes)

    Returns:
        JWT string
    """
    now = int(time.time())
    payload = {"iat": now - 60, "exp": now + expire_seconds, "iss": str(app_id)}
    token = jwt.encode(payload, private_key_pem, algorithm="RS256")
    # PyJWT returns str in v2.x
    return token


async def get_installation_id_for_repo(jwt_token: str, owner: str, repo: str, api_base: str = "https://api.github.com") -> Optional[int]:
    url = f"{api_base}/repos/{owner}/{repo}/installation"
    headers = {"Authorization": f"Bearer {jwt_token}", "Accept": "application/vnd.github+json"}
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        return data.get("id")


async def create_installation_access_token(jwt_token: str, installation_id: int, api_base: str = "https://api.github.com") -> dict:
    url = f"{api_base}/app/installations/{installation_id}/access_tokens"
    headers = {"Authorization": f"Bearer {jwt_token}", "Accept": "application/vnd.github+json"}
    async with httpx.AsyncClient() as client:
        resp = await client.post(url, headers=headers)
        resp.raise_for_status()
        return resp.json()


async def get_installation_access_token_for_repo(app_id: str, private_key_pem: str, owner: str, repo: str, api_base: str = "https://api.github.com") -> dict:
    """Convenience: generate JWT, find installation for repo and create installation token.

    Returns the JSON response from the access token endpoint which includes `token` and `expires_at`.
    """
    jwt_token = generate_jwt(app_id, private_key_pem)
    installation_id = await get_installation_id_for_repo(jwt_token, owner, repo, api_base=api_base)
    if installation_id is None:
        raise RuntimeError("Installation id not found for repo")
    return await create_installation_access_token(jwt_token, installation_id, api_base=api_base)


def load_private_key_from_env_or_path(env_var: str = "GITHUB_APP_PRIVATE_KEY", path_var: Optional[str] = None) -> Optional[str]:
    """Helper: load private key PEM either directly from env var or from file path.

    If `env_var` exists in env, it's returned; otherwise if `path_var` is provided and exists, load file.
    """
    pem = os.getenv(env_var)
    if pem:
        return pem
    if path_var and os.path.exists(path_var):
        with open(path_var, "r", encoding="utf-8") as f:
            return f.read()
    return None
