import unittest

from ai.core.rag_namespace_policy import (
    extract_project_issue_namespace,
    extract_project_name,
    group_issue_namespaces,
    is_issue_namespace,
    namespace_quality,
    project_namespaces,
)


class TestRagNamespacePolicy(unittest.TestCase):
    def test_project_namespaces(self):
        self.assertEqual(
            project_namespaces("mdl"),
            ["mdl_issues"],
        )

    def test_extract_project_name(self):
        self.assertEqual(extract_project_name("timob/mobile-app"), "mobile-app")
        self.assertEqual(extract_project_name("TIMOB/mobile-app"), "mobile-app")
        self.assertEqual(extract_project_name("repo"), "repo")
        self.assertEqual(extract_project_name(""), "")

    def test_extract_project_issue_namespace(self):
        self.assertEqual(
            extract_project_issue_namespace("ORG/Repo-A"),
            "repo-a_issues",
        )

    def test_group_issue_namespaces(self):
        grouped = group_issue_namespaces(
            [
                "timob_issues",
                "timob_comments",
                "timob_changelog",
                "mule_issues",
                "mule_comments",
                "confserver_changelog",
            ]
        )
        self.assertEqual(grouped, ["timob_issues", "mule_issues"])

    def test_is_issue_namespace(self):
        self.assertTrue(is_issue_namespace("timob_issues"))
        self.assertFalse(is_issue_namespace("timob_comments"))
        self.assertFalse(is_issue_namespace("confserver_changelog"))

    def test_namespace_quality_uses_score_and_hits(self):
        matches = [
            {"namespace": "timob_issues", "score": 0.9},
            {"namespace": "timob_comments", "score": 0.8},
            {"namespace": "timob_changelog", "score": 0.2},
        ]
        namespaces = ["timob_issues", "timob_comments", "timob_changelog"]
        self.assertTrue(namespace_quality(matches, namespaces, min_hits=2, min_score=0.75))
        self.assertFalse(namespace_quality(matches, namespaces, min_hits=3, min_score=0.75))
        self.assertFalse(namespace_quality(matches, namespaces, min_hits=2, min_score=0.95))


if __name__ == "__main__":
    unittest.main()
