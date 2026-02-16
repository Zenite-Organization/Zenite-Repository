from typing import Any, Dict

from ai.dtos.issues_estimation_dto import IssueEstimationDTO
from clients.github.github_provider import GitHubProjectProvider

from application.services.estimation_service import EstimationService


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

        estimate_value = estimation.get("estimate_hours", 0)
        confidence = estimation.get("confidence", 0)
        justification = estimation.get("justification", "")

        estimation_text = (
            f"Estimativa automática: **{estimate_value} horas**.\n\n"
            f"Confiança: {confidence}\n\n"
            f"Justificativa: {justification}"
        )

        await provider.add_comment(issue_node_id, estimation_text)
        await provider.update_estimate(issue_node_id, estimate_value=estimate_value)

        return {
            "estimation": estimation,
            "comment_posted": True,
            "estimate_updated": True,
        }
