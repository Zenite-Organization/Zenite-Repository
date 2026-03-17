import asyncio
import time
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple


State = Literal["ok", "in_progress", "done"]


@dataclass
class _Entry:
    state: Literal["in_progress", "done"]
    created_at: float
    updated_at: float
    response: Optional[Dict[str, Any]] = None


class InMemoryIdempotencyStore:
    """
    Best-effort in-memory idempotency store.

    Notes:
    - Works only within a single Python process (no cross-worker / cross-pod dedupe).
    - Uses TTLs to avoid leaks and to recover from stuck in-flight processing.
    """

    def __init__(self, *, ttl_seconds: int = 600, inflight_ttl_seconds: int = 300):
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be > 0")
        if inflight_ttl_seconds <= 0:
            raise ValueError("inflight_ttl_seconds must be > 0")

        self._ttl_seconds = ttl_seconds
        self._inflight_ttl_seconds = inflight_ttl_seconds
        self._lock = asyncio.Lock()
        self._entries: Dict[str, _Entry] = {}

    async def reserve(self, key: str) -> Tuple[State, Optional[Dict[str, Any]]]:
        """
        Reserve a key for processing.

        Returns:
        - ("ok", None): caller owns processing
        - ("in_progress", None): someone else is processing now
        - ("done", cached_response): already processed recently; use cached_response
        """
        now = time.monotonic()
        async with self._lock:
            self._cleanup_locked(now)

            entry = self._entries.get(key)
            if entry is None:
                self._entries[key] = _Entry(
                    state="in_progress",
                    created_at=now,
                    updated_at=now,
                    response=None,
                )
                return "ok", None

            if entry.state == "done" and entry.response is not None:
                return "done", entry.response

            return "in_progress", None

    async def mark_done(self, key: str, response: Dict[str, Any]) -> None:
        now = time.monotonic()
        async with self._lock:
            self._entries[key] = _Entry(
                state="done",
                created_at=now,
                updated_at=now,
                response=response,
            )
            self._cleanup_locked(now)

    async def release(self, key: str) -> None:
        """
        Release an in-flight reservation (e.g., on error) to allow retries.
        """
        async with self._lock:
            entry = self._entries.get(key)
            if entry is not None and entry.state == "in_progress":
                self._entries.pop(key, None)

    def _cleanup_locked(self, now: float) -> None:
        to_delete = []
        for key, entry in self._entries.items():
            age = now - entry.updated_at
            if entry.state == "done":
                if age > self._ttl_seconds:
                    to_delete.append(key)
                continue

            # in_progress: remove if it looks stuck for too long
            inflight_age = now - entry.created_at
            if inflight_age > self._inflight_ttl_seconds:
                to_delete.append(key)

        for key in to_delete:
            self._entries.pop(key, None)

