from fastapi import APIRouter, Header, Request, HTTPException
import hmac
import hashlib
import json
from web.schemas.github_payloads import GitHubIssuesWebhookPayload
from clients.github.github_provider import GitHubProjectProvider
from ai.workflows.estimation_graph import run_estimation_flow
from config.settings import settings
router = APIRouter()

@router.post("/webhook/github/issues")
async def handle_github_issues(
    request: Request,
    x_github_event: str = Header(...),
    x_github_delivery: str = Header(...),
    x_hub_signature_256: str | None = Header(None, alias="X-Hub-Signature-256"),
):
    body_bytes = await request.body()
    secret = settings.WEBHOOK_SECRET
    if secret:
        if not x_hub_signature_256:
            raise HTTPException(status_code=401, detail="Missing X-Hub-Signature-256 header")
        try:
            expected = "sha256=" + hmac.new(secret.encode("utf-8"), body_bytes, hashlib.sha256).hexdigest()
        except Exception:
            raise HTTPException(status_code=500, detail="Error computing HMAC")
        if not hmac.compare_digest(expected, x_hub_signature_256):
            raise HTTPException(status_code=401, detail="Invalid signature")

    try:
        payload_dict = json.loads(body_bytes)
        payload = GitHubIssuesWebhookPayload(**payload_dict)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid payload: {e}")
    if x_github_event != "issues":
        return {"message": "Evento não tratado", "event": x_github_event}

    action = payload.action
    # Aceita também quando uma label for adicionada
    if action not in ("opened", "edited", "labeled"):
        return {"message": f"Ação {action} ignorada"}

    # Só prossegue se a issue possuir a label chamada "Estimate"
    labels = payload_dict.get("issue", {}).get("labels", [])
    has_estimate_label = False
    for lbl in labels:
        if isinstance(lbl, dict):
            name = lbl.get("name")
        else:
            # em alguns casos labels podem ser strings
            name = lbl
        if name == "Estimate":
            has_estimate_label = True
            break
    if not has_estimate_label:
        # retorna 200 vazio quando não há a label desejada
        return None

    installation_id = payload.installation.id
    issue_node_id = payload.issue.node_id if payload.issue else None
    issue_title = payload.issue.title if payload.issue else None
    issue_body = payload.issue.body if payload.issue else ""

    # Fluxo de estimativa da IA
    estimation_state = run_estimation_flow(issue_body)
    estimation = estimation_state.get("final_estimation", {})
    estimate_value = estimation.get("estimate_hours", 0)
    confidence = estimation.get("confidence", 0)
    justification = estimation.get("justification", "")

    estimation_text = (
        f"Estimativa automática: **{estimate_value} horas**.\n\n"
        f"Confiança: {confidence}\n\n"
        f"Justificativa: {justification}"
    )

    app_id = settings.APP_ID
    private_key = settings.APP_PRIVATE_KEY

    if not private_key and settings.APP_PRIVATE_KEY_path:
        try:
            with open(settings.APP_PRIVATE_KEY_path, "r", encoding="utf-8") as f:
                private_key = f.read()
        except Exception:
            private_key = None

    provider = GitHubProjectProvider(app_id=app_id, private_key=private_key, installation_id=installation_id)

    await provider.auth.ensure_token()

    try:
        await provider.add_comment(issue_node_id, estimation_text)
        await provider.update_estimate(issue_node_id, estimate_value=estimate_value)
    except Exception as e:
        print(f"Erro ao interagir com GitHub: {e}")

    return None
