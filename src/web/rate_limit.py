import asyncio
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict


@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    count: int
    remaining: int
    reset_at_iso_utc: str


@dataclass
class _CounterEntry:
    count: int
    reset_at_epoch: float
    last_seen_epoch: float


class InMemoryDailyRateLimiter:
    """
    Best-effort in-memory daily rate limiter (UTC day boundaries).

    Notes:
    - Works only within a single Python process (no cross-worker / cross-pod limits).
    - Resets at the next 00:00:00 UTC boundary.
    """

    def __init__(
        self,
        *,
        limit_default: int = 10,
        now_fn: Callable[[], float] | None = None,
        max_retention_seconds: int = 48 * 3600,
    ):
        if limit_default <= 0:
            raise ValueError("limit_default must be > 0")
        if max_retention_seconds <= 0:
            raise ValueError("max_retention_seconds must be > 0")

        self._limit_default = limit_default
        self._now = now_fn or time.time
        self._max_retention_seconds = max_retention_seconds

        self._lock = asyncio.Lock()
        self._counters: Dict[str, _CounterEntry] = {}
        self._notify_expiry: Dict[str, float] = {}

    @staticmethod
    def _next_midnight_utc_epoch(now_epoch: float) -> float:
        now_dt = datetime.fromtimestamp(now_epoch, tz=timezone.utc)
        tomorrow = (now_dt.date() + timedelta(days=1))
        reset_dt = datetime(
            year=tomorrow.year,
            month=tomorrow.month,
            day=tomorrow.day,
            tzinfo=timezone.utc,
        )
        return reset_dt.timestamp()

    @staticmethod
    def _iso_utc(epoch_seconds: float) -> str:
        return datetime.fromtimestamp(epoch_seconds, tz=timezone.utc).isoformat().replace("+00:00", "Z")

    async def check_and_increment(self, key: str, limit: int | None = None) -> RateLimitDecision:
        if not key:
            raise ValueError("key must be non-empty")
        effective_limit = limit if limit is not None else self._limit_default
        if effective_limit <= 0:
            raise ValueError("limit must be > 0")

        now = self._now()
        async with self._lock:
            self._cleanup_locked(now)

            entry = self._counters.get(key)
            if entry is None or now >= entry.reset_at_epoch:
                reset_at = self._next_midnight_utc_epoch(now)
                entry = _CounterEntry(count=0, reset_at_epoch=reset_at, last_seen_epoch=now)
                self._counters[key] = entry

            entry.last_seen_epoch = now

            if entry.count >= effective_limit:
                return RateLimitDecision(
                    allowed=False,
                    count=entry.count,
                    remaining=0,
                    reset_at_iso_utc=self._iso_utc(entry.reset_at_epoch),
                )

            entry.count += 1
            remaining = max(0, effective_limit - entry.count)
            return RateLimitDecision(
                allowed=True,
                count=entry.count,
                remaining=remaining,
                reset_at_iso_utc=self._iso_utc(entry.reset_at_epoch),
            )

    async def should_notify_once(self, key: str) -> bool:
        """
        Best-effort "send once" guard for a notification key.

        The caller should include UTC date in the key (e.g., installation + date + issue_id).
        """
        if not key:
            raise ValueError("key must be non-empty")

        now = self._now()
        async with self._lock:
            self._cleanup_locked(now)

            expiry = self._notify_expiry.get(key)
            if expiry is not None and now < expiry:
                return False

            # Keep the marker long enough to survive retries for the same "day".
            self._notify_expiry[key] = now + self._max_retention_seconds
            return True

    def _cleanup_locked(self, now_epoch: float) -> None:
        # Counters: keep only for a bounded time after reset to avoid leaks.
        counter_delete = []
        for key, entry in self._counters.items():
            if now_epoch > (entry.reset_at_epoch + self._max_retention_seconds):
                counter_delete.append(key)
        for key in counter_delete:
            self._counters.pop(key, None)

        notify_delete = []
        for key, expiry in self._notify_expiry.items():
            if now_epoch > expiry:
                notify_delete.append(key)
        for key in notify_delete:
            self._notify_expiry.pop(key, None)

