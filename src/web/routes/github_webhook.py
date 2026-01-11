from fastapi import APIRouter, Header, Request, HTTPException
import hmac
import hashlib
import json
from typing import List
from web.schemas.github_payloads import GitHubIssuesWebhookPayload
from clients.github.github_provider import GitHubProjectProvider, get_provider_for_installation
from clients.github.utils import extract_label_names
from clients.github.github_auth import verify_signature
from ai.workflows.sprint_planning_graph import planning_graph
from ai.workflows.estimation_graph import run_estimation_flow
import asyncio
from config.settings import settings
from ai.dtos.issues_estimation_dto import map_issue_to_estimation_dto
router = APIRouter()


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
        payload_dict = json.loads(body_bytes)
        print(f"Payload recebido: {payload_dict}")
        payload = GitHubIssuesWebhookPayload(**payload_dict)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid payload: {e}")

    if x_github_event != "issues":
        return {"message": "Evento não tratado", "event": x_github_event}

    action = payload.action
    if action not in ("opened", "edited", "labeled"):
        return {"message": f"Ação {action} ignorada"}

    # 🔹 Labels
    labels = payload_dict.get("issue", {}).get("labels", [])
    label_names = extract_label_names(labels)

    has_estimate_label = "Estimate" in label_names
    is_planning_label = settings.PLANNING_TRIGGER_LABEL in label_names

    if not has_estimate_label and not is_planning_label:
        return None

    installation_id = payload.installation.id
    issue_node_id = payload.issue.node_id
    repo_full_name = payload.repository.full_name


    # Se for planning, executar fluxo de planning usando sprint_planning_graph
    if is_planning_label:
        provider = get_provider_for_installation(installation_id)
        await provider.auth.ensure_token()

        initial_state = {
            "repo_full_name": repo_full_name,
            "installation_id": installation_id,
            "trigger_issue_node_id": issue_node_id,
            "options": {"backlog_label": settings.SPRINT_BACKLOG_LABEL, "trigger_label": settings.PLANNING_TRIGGER_LABEL},
            "provider": provider,
        }

        try:
            summary = await asyncio.to_thread(planning_graph.invoke, initial_state)
        except Exception as e:
            print(f"Erro no fluxo de planning: {e}")
            raise HTTPException(status_code=500, detail="Planning flow failed")
        return summary

    print(f"IssueEstimationDTO gerado: {payload}")
    dto = map_issue_to_estimation_dto(payload)

    estimation_state = await asyncio.to_thread(
        run_estimation_flow,
        dto
    )

    estimation = estimation_state.get("final_estimation", {})
    estimate_value = estimation.get("estimate_hours", 0)
    confidence = estimation.get("confidence", 0)
    justification = estimation.get("justification", "")

    estimation_text = (
        f"Estimativa automática: **{estimate_value} horas**.\n\n"
        f"Confiança: {confidence}\n\n"
        f"Justificativa: {justification}"
    )

    provider = get_provider_for_installation(installation_id)
    await provider.auth.ensure_token()

    try:
        await provider.add_comment(issue_node_id, estimation_text)
        await provider.update_estimate(issue_node_id, estimate_value=estimate_value)
    except Exception as e:
        print(f"Erro ao interagir com GitHub: {e}")

    return None
