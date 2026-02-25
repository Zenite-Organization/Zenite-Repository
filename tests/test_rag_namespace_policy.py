import unittest

from ai.core.rag_namespace_policy import (
    extract_org_namespace,
    group_namespaces_by_base,
    is_base_namespace,
    namespace_quality,
    project_namespaces,
)


class TestRagNamespacePolicy(unittest.TestCase):
    def test_project_namespaces(self):
        self.assertEqual(
            project_namespaces("mdl"),
            ["mdl", "mdl_comments", "mdl_changelog"],
        )

    def test_extract_org_namespace(self):
        self.assertEqual(extract_org_namespace("timob/mobile-app"), "timob")
        self.assertEqual(extract_org_namespace("TIMOB/mobile-app"), "timob")
        self.assertEqual(extract_org_namespace("timob"), "timob")
        self.assertEqual(extract_org_namespace(""), "")

    def test_group_namespaces_by_base(self):
        grouped = group_namespaces_by_base(
            [
                "timob",
                "timob_comments",
                "timob_changelog",
                "mule_comments",
                "mule",
                "confserver_changelog",
            ]
        )
        self.assertEqual(grouped, ["timob", "mule"])

    def test_is_base_namespace(self):
        self.assertTrue(is_base_namespace("timob"))
        self.assertFalse(is_base_namespace("timob_comments"))
        self.assertFalse(is_base_namespace("confserver_changelog"))

    def test_namespace_quality_uses_score_and_hits(self):
        matches = [
            {"namespace": "timob", "score": 0.9},
            {"namespace": "timob_comments", "score": 0.8},
            {"namespace": "timob_changelog", "score": 0.2},
        ]
        namespaces = ["timob", "timob_comments", "timob_changelog"]
        self.assertTrue(namespace_quality(matches, namespaces, min_hits=2, min_score=0.75))
        self.assertFalse(namespace_quality(matches, namespaces, min_hits=3, min_score=0.75))
        self.assertFalse(namespace_quality(matches, namespaces, min_hits=2, min_score=0.95))


if __name__ == "__main__":
    unittest.main()
