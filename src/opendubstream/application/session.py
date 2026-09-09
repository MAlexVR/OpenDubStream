"""Ties a capture/playback failure to exactly one route recovery attempt."""

from __future__ import annotations

from typing import Callable, TypeVar

from opendubstream.domain.contracts import AudioRouter

T = TypeVar("T")


def run_protected(router: AudioRouter, work: Callable[[], T]) -> T:
    """Run `work`; on any failure, recover the route exactly once, then re-raise the original error.

    If recovery itself fails, that error propagates instead and the journal is retained
    (unreconciled) rather than silently cleared or retried.
    """
    try:
        return work()
    except Exception:
        router.recover()
        raise
