import asyncio
import logging
from typing import Any, Dict, List

from ai.dtos.issues_estimation_dto import IssueEstimationDTO
from ai.workflows.sprint_planning_graph import run_sprint_planning_flow
from clients.github.github_provider import GitHubProjectProvider


logger = logging.getLogger(__name__)


class SprintPlanningService:
    async def run(
        self,
        dto: IssueEstimationDTO,
        provider: GitHubProjectProvider,
        repo_full_name: str,
        issue_node_id: str,
    ) -> Dict[str, Any]:
        project = await provider.get_project_id(issue_node_id)
        backlog: List[IssueEstimationDTO] = await provider.list_backlog_issues(
            repo_full_name,
            project["number"],
            None,
        )

        capacity_hours = 40.0
        try:
            iterations = await provider.get_project_iterations(project["id"])
            if iterations:
                duration_days = iterations[-1].get("duration")
                if duration_days:
                    capacity_hours = float(duration_days) * 8.0
        except Exception:
            logger.exception("Failed to load project iterations")

        summary = await asyncio.to_thread(
            run_sprint_planning_flow,
            dto,
            backlog,
            capacity_hours,
        )

        return {
            "project_id": project["id"],
            "project_number": project["number"],
            "capacity_hours": capacity_hours,
            "summary": summary,
            "backlog_count": len(backlog),
        }
