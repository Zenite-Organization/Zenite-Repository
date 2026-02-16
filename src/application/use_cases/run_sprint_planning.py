from typing import Any, Dict

from ai.dtos.issues_estimation_dto import IssueEstimationDTO
from clients.github.github_provider import GitHubProjectProvider

from application.services.sprint_planning_service import SprintPlanningService


class RunSprintPlanningUseCase:
    def __init__(self, planning_service: SprintPlanningService | None = None):
        self.planning_service = planning_service or SprintPlanningService()

    async def execute(
        self,
        dto: IssueEstimationDTO,
        provider: GitHubProjectProvider,
        repo_full_name: str,
        issue_node_id: str,
    ) -> Dict[str, Any]:
        result = await self.planning_service.run(
            dto=dto,
            provider=provider,
            repo_full_name=repo_full_name,
            issue_node_id=issue_node_id,
        )
        return result
