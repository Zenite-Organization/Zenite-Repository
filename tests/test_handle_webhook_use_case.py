import asyncio
import unittest

from application.use_cases.handle_github_webhook import HandleGithubWebhookUseCase
from domain.webhook_models import WebhookFlow, WebhookStatus
from web.schemas.github_payloads import GitHubIssuesWebhookPayload


class TestHandleGithubWebhookUseCase(unittest.TestCase):
    def setUp(self):
        self.use_case = HandleGithubWebhookUseCase()

    def _payload(self, action: str) -> GitHubIssuesWebhookPayload:
        return GitHubIssuesWebhookPayload(
            action=action,
            issue=None,
            repository={"full_name": "org/repo"},
            installation={"id": 123},
        )

    def test_ignores_non_issues_event(self):
        result = asyncio.run(
            self.use_case.handle(
                payload=self._payload("opened"),
                event="push",
                delivery_id="d1",
            )
        )
        self.assertEqual(result.status, WebhookStatus.IGNORED)
        self.assertEqual(result.flow, WebhookFlow.NONE)

    def test_ignores_unsupported_action(self):
        result = asyncio.run(
            self.use_case.handle(
                payload=self._payload("closed"),
                event="issues",
                delivery_id="d2",
            )
        )
        self.assertEqual(result.status, WebhookStatus.IGNORED)
        self.assertEqual(result.flow, WebhookFlow.NONE)

    def test_ignores_when_no_control_labels(self):
        result = asyncio.run(
            self.use_case.handle(
                payload=self._payload("opened"),
                event="issues",
                delivery_id="d3",
            )
        )
        self.assertEqual(result.status, WebhookStatus.IGNORED)
        self.assertEqual(result.flow, WebhookFlow.NONE)


if __name__ == "__main__":
    unittest.main()
