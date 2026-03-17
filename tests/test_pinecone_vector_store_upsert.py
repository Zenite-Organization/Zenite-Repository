import os
import sys
import unittest
from unittest.mock import Mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from ai.core.pinecone_vector_store import PineconeVectorStoreClient


class _EmbItem:
    def __init__(self, embedding):
        self.embedding = embedding


class _EmbResp:
    def __init__(self, embeddings):
        self.data = [_EmbItem(e) for e in embeddings]


class TestPineconeVectorStoreUpsert(unittest.TestCase):
    def test_upsert_groups_by_namespace_and_calls_index(self):
        client = PineconeVectorStoreClient.__new__(PineconeVectorStoreClient)
        client._ready = True
        client._index = Mock()

        openai = Mock()
        openai.embeddings.create = Mock(return_value=_EmbResp([[0.1, 0.2], [0.3, 0.4]]))
        client._openai = openai

        docs = [
            {"id": "org/repo#1", "namespace": "Repo_Issues", "text": "t1", "metadata": {"a": 1}},
            {"id": "org/repo#2", "namespace": "repo_issues", "text": "t2", "metadata": {"b": 2}},
        ]

        res = client.upsert(docs)
        self.assertFalse(res["skipped"])
        self.assertEqual(res["upserted"], 2)
        self.assertEqual(res["namespaces"], {"repo_issues": 2})

        openai.embeddings.create.assert_called_once()
        client._index.upsert.assert_called_once()
        args, kwargs = client._index.upsert.call_args
        self.assertEqual(kwargs["namespace"], "repo_issues")
        self.assertEqual(len(kwargs["vectors"]), 2)
        self.assertEqual(kwargs["vectors"][0]["id"], "org/repo#1")
        self.assertIn("values", kwargs["vectors"][0])
        self.assertEqual(kwargs["vectors"][0]["metadata"], {"a": 1})


if __name__ == "__main__":
    unittest.main()

