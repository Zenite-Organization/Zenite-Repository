import asyncio
from typing import Any, Dict

from ai.dtos.issues_estimation_dto import IssueEstimationDTO
from ai.workflows.estimation_graph import run_estimation_flow


class EstimationService:
    async def run(self, dto: IssueEstimationDTO) -> Dict[str, Any]:
        return await asyncio.to_thread(run_estimation_flow, dto)
