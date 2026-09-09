"""Latency metrics that preserve invalid and stale samples as evidence."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LatencyReport:
    sample_count: int
    p50_ms: float | None
    p95_ms: float | None
    invalid_samples: int
    stale_samples: int


@dataclass
class _ModeSamples:
    values_ms: list[float] = field(default_factory=list)
    invalid_samples: int = 0
    stale_samples: int = 0


class LatencyMetrics:
    def __init__(self, *, max_age_seconds: float) -> None:
        self.max_age_seconds = max_age_seconds
        self._by_mode: dict[str, _ModeSamples] = {}

    def record(self, mode: str, phrase_end_at: float | None, playback_started_at: float | None) -> None:
        samples = self._by_mode.setdefault(mode, _ModeSamples())
        if phrase_end_at is None or playback_started_at is None or playback_started_at < phrase_end_at:
            samples.invalid_samples += 1
            return
        seconds = playback_started_at - phrase_end_at
        if seconds > self.max_age_seconds:
            samples.stale_samples += 1
            return
        samples.values_ms.append(round(seconds * 1000, 6))

    def report(self, mode: str) -> LatencyReport:
        samples = self._by_mode.get(mode, _ModeSamples())
        ordered = sorted(samples.values_ms)
        return LatencyReport(
            sample_count=len(ordered),
            p50_ms=self._percentile(ordered, 50),
            p95_ms=self._percentile(ordered, 95),
            invalid_samples=samples.invalid_samples,
            stale_samples=samples.stale_samples,
        )

    @staticmethod
    def _percentile(values: list[float], percentile: int) -> float | None:
        if not values:
            return None
        index = max(0, (len(values) * percentile + 99) // 100 - 1)
        return values[index]
