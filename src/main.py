from fastapi import FastAPI, HTTPException, Header, Request
from pydantic import BaseModel
from typing import Optional
import uvicorn
from ai.workflows.estimation_graph import run_estimation_flow

from infrastructure.api_clients.github_client import GitHubProjectProvider

app = FastAPI()


# ---------- SCHEMAS WEBHOOK ----------
class InstallationPayload(BaseModel):
    id: int


# --- Repository info ---
class RepositoryPayload(BaseModel):
    full_name: str  # ex: "org/repo"


# --- Comment payload (somente no evento issue_comment) ---
class CommentPayload(BaseModel):
    node_id: str
    id: int
    body: Optional[str]
    user: Optional[dict]
    created_at: Optional[str]
    updated_at: Optional[str]


# --- Issue payload ---
class IssuePayload(BaseModel):
    node_id: str
    number: int
    title: Optional[str]
    body: Optional[str]  # descrição da issue

# --- Payload para atualização de campos do ProjectV2 ---
class EstimateUpdatePayload(BaseModel):
    project_id: str
    item_id: str
    field_id: str
    estimate_value: float | int


# --- Payload principal do Webhook ---
class GitHubIssuesWebhookPayload(BaseModel):
    action: str                          # "opened", "edited", etc
    issue: Optional[IssuePayload] = None
    repository: RepositoryPayload
    installation: InstallationPayload
    sender: Optional[dict] = None
    estimate_update: Optional[EstimateUpdatePayload] = None


# ---------- ROTA WEBHOOK ----------
@app.post("/webhook/github/issues")
async def handle_github_issues(
    payload: GitHubIssuesWebhookPayload,
    x_github_event: str = Header(...),
    x_github_delivery: str = Header(...),
):
    # Só processar evento "issues"
    if x_github_event != "issues":
        return {"message": "Evento não tratado", "event": x_github_event}

    action = payload.action
    if action not in ("opened", "edited"):
        return {"message": f"Ação {action} ignorada"}

    installation_id = payload.installation.id
    issue_node_id = payload.issue.node_id
    issue_title = payload.issue.title
    issue_body = payload.issue.body

    # Chama o fluxo de estimativa da IA
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

    provider = GitHubProjectProvider(installation_id=installation_id)
    await provider._ensure_token()

    # Comentar na issue
    try:
        await provider.add_comment(issue_node_id, estimation_text)
    except Exception as e:
        print(f"Erro ao adicionar comentário: {e}")

    try:
        await provider.update_estimate(
            issue_node_id=issue_node_id,
            estimate_value=estimate_value,
        )
    except Exception as e:
        print(f"Erro ao atualizar estimate: {e}")

    # Sempre retornar 200
    return {
        "message": "Comentário criado",
        "issue_node_id": issue_node_id,
        "estimate": estimate_value,
        "confidence": confidence,
        "justification": justification,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
