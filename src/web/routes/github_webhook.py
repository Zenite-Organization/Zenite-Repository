import json
import logging
import os
import time
from datetime import datetime, timezone

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from application.use_cases.handle_github_webhook import HandleGithubWebhookUseCase
from clients.github.github_provider import get_provider_for_installation
from clients.github.github_auth import verify_signature
from clients.github.utils import extract_label_names
from domain.webhook_rules import decide_flow
from domain.webhook_models import WebhookFlow
from web.idempotency import InMemoryIdempotencyStore
from web.rate_limit import InMemoryDailyRateLimiter
from web.schemas.github_payloads import GitHubIssuesWebhookPayload


router = APIRouter()
logger = logging.getLogger(__name__)
use_case = HandleGithubWebhookUseCase()
idempotency = InMemoryIdempotencyStore(
    ttl_seconds=int(os.getenv("GITHUB_WEBHOOK_DEDUP_TTL_SECONDS", "600")),
    inflight_ttl_seconds=int(os.getenv("GITHUB_WEBHOOK_INFLIGHT_TTL_SECONDS", "300")),
)
daily_limit = int(os.getenv("GITHUB_WEBHOOK_DAILY_LIMIT", "10"))
rate_limiter = InMemoryDailyRateLimiter(
    limit_default=daily_limit,
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
        # Extract label names from both `issue.labels` and the top-level `label` field
        raw_labels = []
        if payload.issue:
            raw_labels.extend(payload.issue.labels or [])
        if isinstance(payload_dict, dict) and payload_dict.get("label"):
            raw_labels.append(payload_dict.get("label"))

        labels = extract_label_names(raw_labels)

        # Decide flow early: if no applicable flow, delegate to use_case (returns IGNORED dict)
        flow = decide_flow(x_github_event, payload.action, labels)
        if flow == WebhookFlow.NONE:
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
        raise HTTPException(status_code=400, detail=f"Invalid payload: {e}")

    installation_id = payload.installation.id
    utc_day = datetime.fromtimestamp(time.time(), tz=timezone.utc).date().isoformat()
    rate_key = f"installation:{installation_id}:{utc_day}"
    decision = await rate_limiter.check_and_increment(rate_key)
    if not decision.allowed:
        response_dict = {
            "status": "ignored",
            "event": x_github_event,
            "action": payload.action,
            "flow": "none",
            "details": {
                "reason": "rate_limited",
                "limit": daily_limit,
                "remaining": 0,
                "reset_at": decision.reset_at_iso_utc,
                "key": str(installation_id),
            },
        }

        issue_node_id = payload.issue.node_id if payload.issue else None
        if issue_node_id:
            notify_key = f"notify:{installation_id}:{utc_day}:{issue_node_id}"
            if await rate_limiter.should_notify_once(notify_key):
                reset_at = decision.reset_at_iso_utc
                comment_template = os.getenv(
                    "GITHUB_WEBHOOK_RATE_LIMIT_COMMENT_TEXT",
                    "Zenite (Free): limite diário de {limit} execuções atingido para esta instalação. "
                    "Tente novamente após {reset_at} (UTC).",
                )
                try:
                    provider = get_provider_for_installation(installation_id)
                    await provider.auth.ensure_token()
                    await provider.add_comment(
                        subject_id=issue_node_id,
                        body=comment_template.format(limit=daily_limit, reset_at=reset_at),
                    )
                except Exception:
                    logger.exception(
                        "Failed to comment rate limit notice installation_id=%s issue_node_id=%s",
                        installation_id,
                        issue_node_id,
                    )

        response = JSONResponse(status_code=202, content=response_dict)
        await idempotency.mark_done(x_github_delivery, response)
        return response

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
