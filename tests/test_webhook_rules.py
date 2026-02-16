import unittest

from domain.webhook_models import WebhookFlow
from domain.webhook_rules import decide_flow


class TestWebhookRules(unittest.TestCase):
    def test_ignores_non_issues_event(self):
        flow = decide_flow("push", "opened", ["Estimate"])
        self.assertEqual(flow, WebhookFlow.NONE)

    def test_ignores_unsupported_action(self):
        flow = decide_flow("issues", "closed", ["Estimate"])
        self.assertEqual(flow, WebhookFlow.NONE)

    def test_routes_planning(self):
        flow = decide_flow("issues", "labeled", ["Planning"])
        self.assertEqual(flow, WebhookFlow.PLANNING)

    def test_routes_estimation(self):
        flow = decide_flow("issues", "opened", ["Estimate"])
        self.assertEqual(flow, WebhookFlow.ESTIMATION)

    def test_ignores_missing_control_label(self):
        flow = decide_flow("issues", "opened", ["bug", "backend"])
        self.assertEqual(flow, WebhookFlow.NONE)


if __name__ == "__main__":
    unittest.main()
