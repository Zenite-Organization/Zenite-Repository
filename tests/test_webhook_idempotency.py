import asyncio
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from web.idempotency import InMemoryIdempotencyStore


class TestInMemoryIdempotencyStore(unittest.TestCase):
    def test_reserve_marks_in_progress_then_done(self):
        store = InMemoryIdempotencyStore(ttl_seconds=60, inflight_ttl_seconds=60)

        async def run():
            state1, cached1 = await store.reserve("k1")
            self.assertEqual(state1, "ok")
            self.assertIsNone(cached1)

            state2, cached2 = await store.reserve("k1")
            self.assertEqual(state2, "in_progress")
            self.assertIsNone(cached2)

            await store.mark_done("k1", {"status": "processed"})

            state3, cached3 = await store.reserve("k1")
            self.assertEqual(state3, "done")
            self.assertEqual(cached3, {"status": "processed"})

        asyncio.run(run())

    def test_release_allows_retry_after_error(self):
        store = InMemoryIdempotencyStore(ttl_seconds=60, inflight_ttl_seconds=60)

        async def run():
            state1, _ = await store.reserve("k2")
            self.assertEqual(state1, "ok")

            await store.release("k2")

            state2, _ = await store.reserve("k2")
            self.assertEqual(state2, "ok")

        asyncio.run(run())

    def test_concurrent_reserve_only_one_owner(self):
        store = InMemoryIdempotencyStore(ttl_seconds=60, inflight_ttl_seconds=60)

        async def runner():
            state, _ = await store.reserve("k3")
            return state

        async def run():
            results = await asyncio.gather(*[runner() for _ in range(20)])
            self.assertEqual(results.count("ok"), 1)
            self.assertEqual(results.count("in_progress"), 19)

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
