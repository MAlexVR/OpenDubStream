"""Fixed-capacity scheduling that only evicts unstarted stale work."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from opendubstream.domain.pipeline import Utterance


@dataclass(frozen=True)
class EnqueueResult:
    dropped_identifier: str | None


@dataclass
class SchedulerMetrics:
    overload_dropped: int = 0
    stale_dropped: int = 0


class FixedCapacityScheduler:
    def __init__(self, *, capacity: int, max_age_seconds: float) -> None:
        if capacity < 1 or max_age_seconds <= 0:
            raise ValueError("capacity and maximum age must be positive")
        self.capacity = capacity
        self.max_age_seconds = max_age_seconds
        self._pending: deque[Utterance] = deque()
        self.metrics = SchedulerMetrics()

    @property
    def backlog(self) -> int:
        return len(self._pending)

    def enqueue(self, utterance: Utterance, *, now: float) -> EnqueueResult:
        self._discard_stale(now)
        dropped: str | None = None
        if len(self._pending) == self.capacity:
            dropped = self._pending.popleft().identifier
            self.metrics.overload_dropped += 1
        self._pending.append(utterance)
        return EnqueueResult(dropped)

    def start_next(self, *, now: float) -> Utterance | None:
        self._discard_stale(now)
        return self._pending.popleft() if self._pending else None

    def _discard_stale(self, now: float) -> None:
        retained: deque[Utterance] = deque()
        while self._pending:
            item = self._pending.popleft()
            if now - item.phrase_end_at > self.max_age_seconds:
                self.metrics.stale_dropped += 1
            else:
                retained.append(item)
        self._pending = retained
