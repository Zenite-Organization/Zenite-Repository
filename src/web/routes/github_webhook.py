import json
import logging
import os

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from application.use_cases.handle_github_webhook import HandleGithubWebhookUseCase
from clients.github.github_auth import verify_signature
from web.idempotency import InMemoryIdempotencyStore
from web.schemas.github_payloads import GitHubIssuesWebhookPayload


router = APIRouter()
logger = logging.getLogger(__name__)
use_case = HandleGithubWebhookUseCase()
idempotency = InMemoryIdempotencyStore(
    ttl_seconds=int(os.getenv("GITHUB_WEBHOOK_DEDUP_TTL_SECONDS", "600")),
    inflight_ttl_seconds=int(os.getenv("GITHUB_WEBHOOK_INFLIGHT_TTL_SECONDS", "300")),
)


@router.post("/webhook/github")
async def handle_github_issues(
    request: Request,
    x_github_event: str = Header(...),
    x_github_delivery: str = Header(...),
    x_hub_signature_256: str | None = Header(None, alias="X-Hub-Signature-256"),
):
    body_bytes = await request.body()
    verify_signature(body_bytes, x_hub_signature_256)

    state, cached = await idempotency.reserve(x_github_delivery)
    if state == "done" and cached is not None:
        logger.info(
            "Duplicate webhook delivery_id=%s event=%s (cached)",
            x_github_delivery,
            x_github_event,
        )
        return cached

    if state == "in_progress":
        logger.info(
            "Duplicate webhook delivery_id=%s event=%s (in_progress)",
            x_github_delivery,
            x_github_event,
        )
        return JSONResponse(
            status_code=202,
            content={
                "status": "accepted",
                "event": x_github_event,
                "action": "unknown",
                "flow": "none",
                "details": {"reason": "duplicate_in_progress", "delivery_id": x_github_delivery},
            },
        )

    try:
        payload_dict = json.loads(body_bytes.decode("utf-8"))

        payload = GitHubIssuesWebhookPayload(**payload_dict)
    except Exception as e:
        await idempotency.release(x_github_delivery)
        raise HTTPException(status_code=400, detail=f"Invalid payload: {e}")

    try:
        result = await use_case.handle(
            payload=payload,
            event=x_github_event,
            delivery_id=x_github_delivery,
        )
        response = result.to_dict()
        await idempotency.mark_done(x_github_delivery, response)
        return response
    except Exception as e:
        await idempotency.release(x_github_delivery)
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
