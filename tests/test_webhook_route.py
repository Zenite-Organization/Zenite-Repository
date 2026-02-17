import unittest
import hashlib
import hmac
import json

from fastapi.testclient import TestClient

from config.settings import settings
from main import app


class TestWebhookRoute(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def _signed_headers(self, event: str, delivery: str, body: bytes) -> dict:
        headers = {
            "x-github-event": event,
            "x-github-delivery": delivery,
        }
        if settings.WEBHOOK_SECRET:
            digest = hmac.new(
                settings.WEBHOOK_SECRET.encode("utf-8"),
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
        previous_secret = settings.WEBHOOK_SECRET
        settings.WEBHOOK_SECRET = "secret"
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
            settings.WEBHOOK_SECRET = previous_secret


if __name__ == "__main__":
    unittest.main()
