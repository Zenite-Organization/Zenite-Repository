import logging
from typing import Any, Dict

from ai.dtos.issues_estimation_dto import IssueEstimationDTO
from clients.github.github_provider import GitHubProjectProvider
from domain.webhook_rules import ESTIMATION_LABEL

from application.services.estimation_service import EstimationService


logger = logging.getLogger(__name__)


class RunIssueEstimationUseCase:
    def __init__(self, estimation_service: EstimationService | None = None):
        self.estimation_service = estimation_service or EstimationService()

    async def execute(
        self,
        dto: IssueEstimationDTO,
        provider: GitHubProjectProvider,
        issue_node_id: str,
    ) -> Dict[str, Any]:
        estimation_state = await self.estimation_service.run(dto)
        estimation = estimation_state.get("final_estimation", {})

        estimate_value = estimation.get("estimated_hours", 0)
        confidence = estimation.get("confidence", 0)
        justification = estimation.get("justification", "")

        estimation_text = (
            f"Estimativa automática: **{estimate_value} horas**.\n\n"
            f"Confiança: {confidence}\n\n"
            f"Justificativa: {justification}"
        )

        await provider.add_comment(issue_node_id, estimation_text)
        await provider.update_estimate(issue_node_id, estimate_value=estimate_value)

        estimate_label_removed = False
        estimate_label_remove_reason: str | None = None
        estimate_label_remove_error: str | None = None

        try:
            resp = await provider.remove_issue_label(
                repo_full_name=dto.repository,
                issue_number=dto.issue_number,
                label=ESTIMATION_LABEL,
            )
            estimate_label_removed = bool(resp.get("removed"))
            estimate_label_remove_reason = resp.get("reason")
        except Exception as e:
            logger.warning(
                "Failed to remove estimate label repo=%s issue_number=%s label=%s",
                dto.repository,
                dto.issue_number,
                ESTIMATION_LABEL,
                exc_info=True,
            )
            estimate_label_remove_error = str(e)

        return {
            "estimation": estimation,
            "comment_posted": True,
            "estimate_updated": True,
            "estimate_label_removed": estimate_label_removed,
            "estimate_label_remove_reason": estimate_label_remove_reason,
            "estimate_label_remove_error": estimate_label_remove_error,
        }
