import logging
from typing import Any, Dict

from ai.dtos.issues_estimation_dto import IssueEstimationDTO
from ai.core.estimation_localization import normalize_split_reason
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

        estimate_value = float(estimation.get("estimated_hours", 0) or 0)
        confidence = float(estimation.get("confidence", 0) or 0)
        min_hours = estimation.get("min_hours")
        max_hours = estimation.get("max_hours")
        justification = (
            estimation.get("user_justification")
            or estimation.get("justification", "")
        )
        should_split = bool(estimation.get("should_split", False))
        split_reason = normalize_split_reason(estimation.get("split_reason"))

        interval_text = ""
        if min_hours is not None and max_hours is not None:
            interval_text = f"Intervalo sugerido: **{min_hours}h a {max_hours}h**.\n\n"

        split_text = ""
        if should_split:
            split_text = "⚠️ A issue aparenta precisar de quebra/refinamento."
            if split_reason:
                split_text = f"{split_text} {split_reason}"
            split_text = f"{split_text}\n\n"

        estimation_text = (
            f"Estimativa automática: **{estimate_value} horas**.\n\n"
            f"{interval_text}"
            f"Confiança: **{confidence:.2f}**\n\n"
            f"{split_text}"
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
