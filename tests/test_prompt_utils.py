import unittest

from ai.core.prompt_utils import format_similar_issues


class TestPromptUtils(unittest.TestCase):
    def test_skips_items_without_title_or_estimate(self):
        text = format_similar_issues(
            [
                {
                    "title": "Valid 1",
                    "total_effort_hours": 2.5,
                    "score": 0.91,
                    "issue_type": "bug",
                    "doc_type": "issue",
                    "description": "Small localized fix in one preference page.",
                },
                {"title": "", "total_effort_hours": 4},
                {"title": "No estimate", "total_effort_hours": None},
                {
                    "title": "Valid 2",
                    "total_effort_hours": 8,
                    "score": 0.83,
                    "issue_type": "task",
                },
            ]
        )
        self.assertIn("1. Score: 0.910", text)
        self.assertIn("Tipo: bug", text)
        self.assertIn("Horas: 2.5h", text)
        self.assertIn("Titulo: Valid 1", text)
        self.assertIn("Descricao: Small localized fix in one preference page.", text)
        self.assertIn("2. Score: 0.830", text)
        self.assertIn("Titulo: Valid 2", text)
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
