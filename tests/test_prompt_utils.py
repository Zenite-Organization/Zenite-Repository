import unittest

from ai.core.prompt_utils import format_similar_issues


class TestPromptUtils(unittest.TestCase):
    def test_skips_items_without_title_or_estimate(self):
        text = format_similar_issues(
            [
                {"title": "Valid 1", "total_effort_hours": 2.5},
                {"title": "", "total_effort_hours": 4},
                {"title": "No estimate", "total_effort_hours": None},
                {"title": "Valid 2", "total_effort_hours": 8},
            ]
        )
        self.assertIn("1. Valid 1 | est: 2.5h", text)
        self.assertIn("2. Valid 2 | est: 8h", text)
        self.assertNotIn("No estimate", text)

    def test_returns_message_when_no_valid_items(self):
        text = format_similar_issues(
            [
                {"title": "", "total_effort_hours": None},
                {"title": None, "total_effort_hours": None},
            ]
        )
        self.assertEqual(text, "No valid similar issues with title and estimate.")


if __name__ == "__main__":
    unittest.main()
