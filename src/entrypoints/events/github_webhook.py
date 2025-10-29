from fastapi import APIRouter, Request, HTTPException
from infrastructure.webhook.github_handler import verify_signature, handle_github_event

router = APIRouter()

@router.post("/webhook")
async def github_webhook(request: Request):
    if not await verify_signature(request):
        raise HTTPException(status_code=401, detail="Invalid signature")

    payload = await request.json()
    event = request.headers.get("x-github-event")
    action = payload.get("action")

    await handle_github_event(payload, event, action)
    return {"status": "ok"}

