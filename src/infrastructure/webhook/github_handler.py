import hmac
import hashlib
import json
import httpx
from fastapi import Request
from typing import Any

from config.settings import settings 

async def verify_signature(request: Request) -> bool:
    """Valida assinatura HMAC enviada pelo GitHub"""
    signature = request.headers.get("x-hub-signature-256")
    if not signature or not settings.github_webhook_secret:
        return False

    body = await request.body()
    digest = hmac.new(
        key=settings.github_webhook_secret.encode(),
        msg=body,
        digestmod=hashlib.sha256
    ).hexdigest()
    expected = f"sha256={digest}"
    return hmac.compare_digest(signature, expected)


async def handle_github_event(payload: dict[str, Any], event: str, action: str):
    """Processa eventos do GitHub"""
    if event == "issues" and action in ["opened", "edited", "labeled"]:
        issue = payload.get("issue", {})
        labels = [l["name"].lower() for l in issue.get("labels", [])]
        if "estimativa" in labels:
            data = {
                "repo": payload["repository"]["full_name"],
                "issue_number": issue["number"],
                "issue_title": issue["title"],
                "issue_body": issue["body"],
                "labels": labels,
                "html_url": issue["html_url"],
            }
            async with httpx.AsyncClient() as client:
                resp = await client.post(settings.ia_api_url, json=data)
                print(f"[Webhook] Enviado para IA_API_URL ({resp.status_code})")

