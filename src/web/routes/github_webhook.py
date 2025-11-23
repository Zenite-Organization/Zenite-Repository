from fastapi import APIRouter, Header
from web.schemas.github_payloads import GitHubIssuesWebhookPayload
from clients.github.github_provider import GitHubProjectProvider
from ai.workflows.estimation_graph import run_estimation_flow
from config.settings import settings
router = APIRouter()

@router.post("/webhook/github/issues")
async def handle_github_issues(
    payload: GitHubIssuesWebhookPayload,
    x_github_event: str = Header(...),
    x_github_delivery: str = Header(...),
):
    if x_github_event != "issues":
        return {"message": "Evento não tratado", "event": x_github_event}

    action = payload.action
    if action not in ("opened", "edited"):
        return {"message": f"Ação {action} ignorada"}

    installation_id = payload.installation.id
    issue_node_id = payload.issue.node_id
    issue_title = payload.issue.title
    issue_body = payload.issue.body

    # Fluxo de estimativa da IA
    estimation_state = run_estimation_flow(issue_body)
    estimation = estimation_state.get("final_estimation", {})
    estimate_value = estimation.get("estimate_hours", 0)
    confidence = estimation.get("confidence", 0)
    justification = estimation.get("justification", "")

    estimation_text = (
        f"Obrigado por abrir a issue **{issue_title}**!\n\n"
        f"Sua descrição da issue:\n{issue_body}\n\n"
        f"Aqui vai minha estimativa automática: **{estimate_value} horas**.\n\n"
        f"Confiança: {confidence}\n\n"
        f"Justificativa: {justification}"
    )

    # Load GitHub App credentials from settings (env/.env)
    app_id = settings.github_app_id
    private_key = settings.github_app_private_key
    # If private key is provided as a file path, load it
    if not private_key and settings.github_app_private_key_path:
        try:
            with open(settings.github_app_private_key_path, "r", encoding="utf-8") as f:
                private_key = f.read()
        except Exception:
            private_key = None

    provider = GitHubProjectProvider(app_id=app_id, private_key=private_key, installation_id=installation_id)
    # Ensure installation token is available
    await provider.auth.ensure_token()

    try:
        await provider.add_comment(issue_node_id, estimation_text)
        await provider.update_estimate(issue_node_id, estimate_value=estimate_value)
    except Exception as e:
        print(f"Erro ao interagir com GitHub: {e}")

    return {
        "message": "Comentário criado",
        "issue_node_id": issue_node_id,
        "estimate": estimate_value,
        "confidence": confidence,
        "justification": justification,
    }
