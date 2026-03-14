import asyncio
import os
import sys
import unittest
from unittest.mock import Mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from application.use_cases.index_closed_issue import IndexClosedIssueUseCase
from web.schemas.github_payloads import GitHubIssuesWebhookPayload


class TestIndexClosedIssueUseCase(unittest.TestCase):
    def _payload(self, issue_overrides: dict | None = None) -> GitHubIssuesWebhookPayload:
        issue = {
            "node_id": "ISSUE_NODE",
            "number": 123,
            "title": "Minha issue",
            "body": "Descricao da issue",
            "labels": [{"id": 1, "name": "bug"}],
            "created_at": "2026-03-10T00:00:00Z",
            "closed_at": "2026-03-12T00:00:00Z",
        }
        if issue_overrides:
            issue.update(issue_overrides)

        return GitHubIssuesWebhookPayload(
            action="closed",
            issue=issue,
            repository={"full_name": "org/repo"},
            installation={"id": 1},
        )

    def test_indexes_closed_issue(self):
        vs = Mock()
        vs.upsert = Mock(return_value={"upserted": 1})
        use_case = IndexClosedIssueUseCase(vector_store=vs)

        async def run():
            res = await use_case.execute(self._payload())
            self.assertFalse(res["skipped"])
            self.assertEqual(res["namespace"], "repo_issues")
            self.assertEqual(res["vector_id"], "org/repo#123")
            self.assertGreaterEqual(res["total_effort_hours"], 1)
            self.assertLessEqual(res["total_effort_hours"], 300)
            self.assertEqual(res["metadata"]["description"], "Descricao da issue")
            self.assertIn("total_effort_hours", res["metadata"])
            vs.upsert.assert_called_once()

        asyncio.run(run())

    def test_effort_fallback_when_missing_timestamps(self):
        vs = Mock()
        vs.upsert = Mock()
        use_case = IndexClosedIssueUseCase(vector_store=vs)

        async def run():
            res = await use_case.execute(self._payload({"closed_at": None}))
            self.assertFalse(res["skipped"])
            self.assertEqual(res["total_effort_hours"], 1.0)
            self.assertTrue(res["metadata"]["effort_fallback"])

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()

