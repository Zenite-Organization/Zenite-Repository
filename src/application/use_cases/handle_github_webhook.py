import logging

from ai.dtos.issues_estimation_dto import map_issue_to_estimation_dto
from clients.github.github_provider import get_provider_for_installation
from clients.github.utils import extract_label_names
from domain.webhook_models import WebhookFlow, WebhookResult, WebhookStatus
from domain.webhook_rules import SUPPORTED_ACTIONS, SUPPORTED_EVENT, decide_flow
from web.schemas.github_payloads import GitHubIssuesWebhookPayload

from application.use_cases.run_issue_estimation import RunIssueEstimationUseCase
from application.use_cases.run_sprint_planning import RunSprintPlanningUseCase


logger = logging.getLogger(__name__)


class HandleGithubWebhookUseCase:
    def __init__(
        self,
        issue_estimation_use_case: RunIssueEstimationUseCase | None = None,
        sprint_planning_use_case: RunSprintPlanningUseCase | None = None,
    ):
        self.issue_estimation_use_case = issue_estimation_use_case or RunIssueEstimationUseCase()
        self.sprint_planning_use_case = sprint_planning_use_case or RunSprintPlanningUseCase()

    async def handle(
        self,
        payload: GitHubIssuesWebhookPayload,
        event: str,
        delivery_id: str,
    ) -> WebhookResult:
        action = payload.action

        if event != SUPPORTED_EVENT:
            return WebhookResult(
                status=WebhookStatus.IGNORED,
                event=event,
                action=action,
                flow=WebhookFlow.NONE,
                details={"reason": "unsupported_event", "supported_event": SUPPORTED_EVENT},
            )

        if action not in SUPPORTED_ACTIONS:
            return WebhookResult(
                status=WebhookStatus.IGNORED,
                event=event,
                action=action,
                flow=WebhookFlow.NONE,
                details={"reason": "unsupported_action", "supported_actions": sorted(SUPPORTED_ACTIONS)},
            )

        labels = extract_label_names(payload.issue.labels if payload.issue else [])
        flow = decide_flow(event, action, labels)
        if flow == WebhookFlow.NONE:
            return WebhookResult(
                status=WebhookStatus.IGNORED,
                event=event,
                action=action,
                flow=WebhookFlow.NONE,
                details={"reason": "missing_control_labels", "labels": labels},
            )

        dto = map_issue_to_estimation_dto(payload)
        installation_id = payload.installation.id
        issue_node_id = payload.issue.node_id
        repo_full_name = payload.repository.full_name

        provider = get_provider_for_installation(installation_id)
        await provider.auth.ensure_token()

        logger.info(
            "Processing webhook delivery_id=%s flow=%s issue_node_id=%s",
            delivery_id,
            flow.value,
            issue_node_id,
        )

        if flow == WebhookFlow.PLANNING:
            details = await self.sprint_planning_use_case.execute(
                dto=dto,
                provider=provider,
                repo_full_name=repo_full_name,
                issue_node_id=issue_node_id,
            )
            return WebhookResult(
                status=WebhookStatus.PROCESSED,
                event=event,
                action=action,
                flow=WebhookFlow.PLANNING,
                details=details,
            )

        details = await self.issue_estimation_use_case.execute(
            dto=dto,
            provider=provider,
            issue_node_id=issue_node_id,
        )
        return WebhookResult(
            status=WebhookStatus.PROCESSED,
            event=event,
            action=action,
            flow=WebhookFlow.ESTIMATION,
            details=details,
        )
