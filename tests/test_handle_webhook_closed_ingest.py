import asyncio
import os
import sys
import unittest
from unittest.mock import AsyncMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from application.use_cases.handle_github_webhook import HandleGithubWebhookUseCase
from domain.webhook_models import WebhookFlow, WebhookStatus
from web.schemas.github_payloads import GitHubIssuesWebhookPayload


class TestHandleGithubWebhookClosedIngest(unittest.TestCase):
    def _payload(self) -> GitHubIssuesWebhookPayload:
        return GitHubIssuesWebhookPayload(
            action="closed",
            issue={
                "node_id": "ISSUE_NODE",
                "number": 10,
                "title": "T",
                "body": "B",
                "labels": [],
                "created_at": "2026-03-10T00:00:00Z",
                "closed_at": "2026-03-12T00:00:00Z",
            },
            repository={"full_name": "org/repo"},
            installation={"id": 1},
        )

    def test_closed_action_indexes_without_labels(self):
        index_uc = AsyncMock()
        index_uc.execute = AsyncMock(return_value={"ok": True})

        estimation_uc = AsyncMock()
        planning_uc = AsyncMock()

        use_case = HandleGithubWebhookUseCase(
            issue_estimation_use_case=estimation_uc,
            sprint_planning_use_case=planning_uc,
            index_closed_issue_use_case=index_uc,
        )

        async def run():
            res = await use_case.handle(
                payload=self._payload(),
                event="issues",
                delivery_id="d1",
            )
            self.assertEqual(res.status, WebhookStatus.PROCESSED)
            self.assertEqual(res.flow, WebhookFlow.NONE)
            self.assertEqual(res.details, {"ok": True})
            index_uc.execute.assert_awaited_once()
            estimation_uc.execute.assert_not_called()
            planning_uc.execute.assert_not_called()

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()

