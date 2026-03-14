import asyncio
import os
import sys
import unittest
from unittest.mock import AsyncMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from ai.dtos.issues_estimation_dto import IssueEstimationDTO
from application.use_cases.run_issue_estimation import RunIssueEstimationUseCase
from domain.webhook_rules import ESTIMATION_LABEL


class _FakeEstimationService:
    def __init__(self, estimation: dict):
        self._estimation = estimation

    async def run(self, dto: IssueEstimationDTO) -> dict:  # noqa: ARG002
        return {"final_estimation": self._estimation}


class TestRunIssueEstimationUseCase(unittest.TestCase):
    def _dto(self) -> IssueEstimationDTO:
        return IssueEstimationDTO(
            issue_number=123,
            repository="org/repo",
            title="T",
            description="D",
            labels=[ESTIMATION_LABEL],
            assignees=[],
            state="open",
            is_open=True,
            comments_count=0,
            age_in_days=0,
            author_login="a",
            author_role="MEMBER",
            repo_language=None,
            repo_size=None,
        )

    def test_removes_estimate_label_best_effort_success(self):
        use_case = RunIssueEstimationUseCase(
            estimation_service=_FakeEstimationService(
                {"estimated_hours": 5, "confidence": 0.8, "justification": "ok"}
            )
        )

        provider = AsyncMock()
        provider.add_comment = AsyncMock()
        provider.update_estimate = AsyncMock()
        provider.remove_issue_label = AsyncMock(return_value={"removed": True, "status_code": 204})

        async def run():
            result = await use_case.execute(
                dto=self._dto(),
                provider=provider,
                issue_node_id="ISSUE_NODE_ID",
            )
            self.assertTrue(result["comment_posted"])
            self.assertTrue(result["estimate_updated"])
            self.assertTrue(result["estimate_label_removed"])
            self.assertIsNone(result["estimate_label_remove_reason"])
            self.assertIsNone(result["estimate_label_remove_error"])

            provider.remove_issue_label.assert_awaited_once_with(
                repo_full_name="org/repo",
                issue_number=123,
                label=ESTIMATION_LABEL,
            )

        asyncio.run(run())

    def test_removes_estimate_label_best_effort_failure(self):
        use_case = RunIssueEstimationUseCase(
            estimation_service=_FakeEstimationService(
                {"estimated_hours": 1, "confidence": 0.1, "justification": "x"}
            )
        )

        provider = AsyncMock()
        provider.add_comment = AsyncMock()
        provider.update_estimate = AsyncMock()
        provider.remove_issue_label = AsyncMock(side_effect=RuntimeError("boom"))

        async def run():
            result = await use_case.execute(
                dto=self._dto(),
                provider=provider,
                issue_node_id="ISSUE_NODE_ID",
            )
            self.assertFalse(result["estimate_label_removed"])
            self.assertIsNone(result["estimate_label_remove_reason"])
            self.assertEqual(result["estimate_label_remove_error"], "boom")

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()

