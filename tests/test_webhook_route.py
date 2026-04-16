import unittest
import hashlib
import hmac
import json
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from config.settings import settings
from main import app


class TestWebhookRoute(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        # Reset module-level in-memory rate limiter between tests to avoid state leak.
        from web.rate_limit import InMemoryDailyRateLimiter
        from web.routes import github_webhook

        github_webhook.rate_limiter = InMemoryDailyRateLimiter(limit_default=10)
        github_webhook.daily_limit = 10

    def _signed_headers(self, event: str, delivery: str, body: bytes) -> dict:
        headers = {
            "x-github-event": event,
            "x-github-delivery": delivery,
        }
        if settings.GITHUB_WEBHOOK_SECRET:
            digest = hmac.new(
                settings.GITHUB_WEBHOOK_SECRET.encode("utf-8"),
                body,
                hashlib.sha256,
            ).hexdigest()
            headers["X-Hub-Signature-256"] = f"sha256={digest}"
        return headers

    def test_returns_400_for_invalid_payload(self):
        body = b"{invalid-json"
        response = self.client.post(
            "/webhook/github",
            content=body,
            headers=self._signed_headers("issues", "delivery-1", body),
        )
        self.assertEqual(response.status_code, 400)

    def test_ignores_non_issues_event(self):
        payload = {
            "action": "opened",
            "repository": {"full_name": "org/repo"},
            "installation": {"id": 123},
        }
        body = json.dumps(payload).encode("utf-8")
        response = self.client.post(
            "/webhook/github",
            content=body,
            headers=self._signed_headers("push", "delivery-2", body),
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "ignored")
        self.assertEqual(body["flow"], "none")

    def test_returns_401_when_signature_is_required_and_missing(self):
        previous_secret = settings.GITHUB_WEBHOOK_SECRET
        settings.GITHUB_WEBHOOK_SECRET = "secret"
        try:
            response = self.client.post(
                "/webhook/github",
                json={
                    "action": "opened",
                    "repository": {"full_name": "org/repo"},
                    "installation": {"id": 123},
                },
                headers={
                    "x-github-event": "issues",
                    "x-github-delivery": "delivery-3",
                },
            )
            self.assertEqual(response.status_code, 401)
        finally:
            settings.GITHUB_WEBHOOK_SECRET = previous_secret

    def test_rate_limits_after_10_requests_per_installation_per_day(self):
        payload = {
            "action": "labeled",
            "repository": {"full_name": "org/repo"},
            "installation": {"id": 123},
            "label": {"name": "Estimate"},
        }
        body = json.dumps(payload).encode("utf-8")

        for i in range(10):
            resp = self.client.post(
                "/webhook/github",
                content=body,
                headers=self._signed_headers("issues", f"delivery-rl-{i}", body),
            )
            self.assertEqual(resp.status_code, 200)

        blocked = self.client.post(
            "/webhook/github",
            content=body,
            headers=self._signed_headers("issues", "delivery-rl-10", body),
        )
        self.assertEqual(blocked.status_code, 202)
        data = blocked.json()
        self.assertEqual(data["status"], "ignored")
        self.assertEqual(data["details"]["reason"], "rate_limited")
        self.assertEqual(data["details"]["remaining"], 0)
        self.assertEqual(data["details"]["limit"], 10)

    def test_rate_limit_comments_only_once_per_issue_per_day(self):
        payload = {
            "action": "labeled",
            "issue": {"node_id": "ISSUE_NODE", "number": 1, "labels": []},      
            "repository": {"full_name": "org/repo"},
            "installation": {"id": 123},
            "label": {"name": "Estimate"},
        }
        body = json.dumps(payload).encode("utf-8")

        provider = type("P", (), {})()
        provider.auth = type("A", (), {})()
        provider.auth.ensure_token = AsyncMock(return_value="tok")
        provider.add_comment = AsyncMock(return_value={})

        with patch("web.routes.github_webhook.get_provider_for_installation", return_value=provider):
            # 10 ok + 1 blocked => should comment once
            for i in range(11):
                self.client.post(
                    "/webhook/github",
                    content=body,
                    headers=self._signed_headers("issues", f"delivery-cm-{i}", body),
                )

            # Additional blocked deliveries should not add more comments for same issue/day.
            for i in range(11, 14):
                resp = self.client.post(
                    "/webhook/github",
                    content=body,
                    headers=self._signed_headers("issues", f"delivery-cm-{i}", body),
                )
                self.assertEqual(resp.status_code, 202)

        provider.add_comment.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
