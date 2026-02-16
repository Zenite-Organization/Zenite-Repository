import json
import logging

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from application.use_cases.handle_github_webhook import HandleGithubWebhookUseCase
from clients.github.github_auth import verify_signature
from web.schemas.github_payloads import GitHubIssuesWebhookPayload


router = APIRouter()
logger = logging.getLogger(__name__)
use_case = HandleGithubWebhookUseCase()


@router.post("/webhook/github")
async def handle_github_issues(
    request: Request,
    x_github_event: str = Header(...),
    x_github_delivery: str = Header(...),
    x_hub_signature_256: str | None = Header(None, alias="X-Hub-Signature-256"),
):
    body_bytes = await request.body()
    verify_signature(body_bytes, x_hub_signature_256)

    try:
        payload_dict = json.loads(body_bytes.decode("utf-8"))
        payload = GitHubIssuesWebhookPayload(**payload_dict)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid payload: {e}")

    try:
        result = await use_case.handle(
            payload=payload,
            event=x_github_event,
            delivery_id=x_github_delivery,
        )
        return result.to_dict()
    except Exception as e:
        logger.exception(
            "Failed to process webhook delivery_id=%s event=%s",
            x_github_delivery,
            x_github_event,
        )
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "event": x_github_event,
                "action": payload.action if "payload" in locals() else "unknown",
                "flow": "none",
                "details": {"error": str(e), "delivery_id": x_github_delivery},
            },
        )
