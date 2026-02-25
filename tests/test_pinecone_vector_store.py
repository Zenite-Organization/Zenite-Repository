import unittest

from ai.core.pinecone_vector_store import PineconeVectorStoreClient


class _StatsIndex:
    def describe_index_stats(self):
        return {"namespaces": {"timob": {}, "timob_comments": {}, "mule": {}}}


class _ObjectStats:
    def __init__(self):
        self.namespaces = {"timob": {}, "confserver": {}}


class _ObjectStatsIndex:
    def describe_index_stats(self):
        return _ObjectStats()


class _BrokenIndex:
    def describe_index_stats(self):
        raise RuntimeError("boom")


class TestPineconeVectorStoreListNamespaces(unittest.TestCase):
    def test_list_namespaces_from_dict_stats(self):
        client = PineconeVectorStoreClient.__new__(PineconeVectorStoreClient)
        client._ready = True
        client._index = _StatsIndex()
        self.assertEqual(
            client.list_namespaces(),
            ["timob", "timob_comments", "mule"],
        )

    def test_list_namespaces_from_object_stats(self):
        client = PineconeVectorStoreClient.__new__(PineconeVectorStoreClient)
        client._ready = True
        client._index = _ObjectStatsIndex()
        self.assertEqual(
            client.list_namespaces(),
            ["timob", "confserver"],
        )

    def test_list_namespaces_error_returns_empty(self):
        client = PineconeVectorStoreClient.__new__(PineconeVectorStoreClient)
        client._ready = True
        client._index = _BrokenIndex()
        self.assertEqual(client.list_namespaces(), [])


if __name__ == "__main__":
    unittest.main()
