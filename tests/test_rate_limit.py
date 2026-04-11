import asyncio
import os
import sys
import unittest
from datetime import datetime, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from web.rate_limit import InMemoryDailyRateLimiter


def _epoch(dt: datetime) -> float:
    if dt.tzinfo is None:
        raise ValueError("dt must be timezone-aware")
    return dt.timestamp()


class TestInMemoryDailyRateLimiter(unittest.TestCase):
    def test_allows_10_blocks_11_and_sets_reset(self):
        now = _epoch(datetime(2026, 4, 11, 12, 0, 0, tzinfo=timezone.utc))

        def now_fn() -> float:
            return now

        limiter = InMemoryDailyRateLimiter(limit_default=10, now_fn=now_fn)

        async def run():
            for _ in range(10):
                d = await limiter.check_and_increment("installation:1:2026-04-11")
                self.assertTrue(d.allowed)
            blocked = await limiter.check_and_increment("installation:1:2026-04-11")
            self.assertFalse(blocked.allowed)
            self.assertEqual(blocked.remaining, 0)
            self.assertEqual(blocked.reset_at_iso_utc, "2026-04-12T00:00:00Z")

        asyncio.run(run())

    def test_resets_after_midnight_utc(self):
        clock = {"now": _epoch(datetime(2026, 4, 11, 23, 59, 0, tzinfo=timezone.utc))}

        def now_fn() -> float:
            return clock["now"]

        limiter = InMemoryDailyRateLimiter(limit_default=2, now_fn=now_fn)

        async def run():
            d1 = await limiter.check_and_increment("k")
            self.assertTrue(d1.allowed)
            d2 = await limiter.check_and_increment("k")
            self.assertTrue(d2.allowed)
            d3 = await limiter.check_and_increment("k")
            self.assertFalse(d3.allowed)

            # Advance past midnight UTC
            clock["now"] = _epoch(datetime(2026, 4, 12, 0, 1, 0, tzinfo=timezone.utc))
            d4 = await limiter.check_and_increment("k")
            self.assertTrue(d4.allowed)
            self.assertEqual(d4.count, 1)

        asyncio.run(run())

    def test_should_notify_once_is_once_per_key(self):
        now = _epoch(datetime(2026, 4, 11, 12, 0, 0, tzinfo=timezone.utc))

        def now_fn() -> float:
            return now

        limiter = InMemoryDailyRateLimiter(limit_default=10, now_fn=now_fn)

        async def run():
            k1 = "notify:1:2026-04-11:ISSUE_NODE"
            self.assertTrue(await limiter.should_notify_once(k1))
            self.assertFalse(await limiter.should_notify_once(k1))

            k2 = "notify:1:2026-04-12:ISSUE_NODE"
            self.assertTrue(await limiter.should_notify_once(k2))

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()

