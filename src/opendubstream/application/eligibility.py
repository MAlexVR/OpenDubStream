"""Pre-Start eligibility gate: `ExecutionMode` resolution plus Chrome-stream-count check.

Pure, no I/O, no routing mutation -- matches the `ui-session-orchestration` spec's
"Eligibility query surface" requirement. Per design.md's "Eligibility gate" decision, Start
is enabled only for `ExecutionMode.CUDA`; no measured CPU latency is ever admitted at UI
runtime (`cpu_latency_ms` stays `None`), so `choose_execution_mode` resolves only `CUDA` or
`CPU_UNSUPPORTED`/`CpuFallbackStatus.UNSUPPORTED` in practice. `execution_mode_permits_start`
still checks against the full `ExecutionMode` enum so a future caller that supplies a
measured `CPU_QUALIFIED`/`CPU_DEGRADED` mode still fails closed rather than being silently
admitted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from opendubstream.domain.pipeline import ExecutionMode, choose_execution_mode
from opendubstream.ui.resources import MessageKey

# Dead in practice: `cpu_latency_ms` is always `None` at UI runtime, so `choose_execution_mode`
# never reaches the threshold comparison. Kept as an explicit, named constant rather than a
# magic number passed inline.
_CPU_LATENCY_THRESHOLD_MS = 250.0


class CudaProbe(Protocol):
    """Zero-argument callable returning whether CUDA is currently available. Probed at UI
    startup and re-evaluated immediately before every Start, per design.md."""

    def __call__(self) -> bool: ...


@dataclass(frozen=True)
class Eligibility:
    startable: bool
    reason: MessageKey | None = None


def execution_mode_permits_start(mode: ExecutionMode) -> bool:
    """Only `ExecutionMode.CUDA` permits Start; any CPU fallback (qualified, degraded, or
    unsupported) is refused."""
    return mode is ExecutionMode.CUDA


def evaluate_eligibility(*, cuda_available: bool, active_chrome_streams: int) -> Eligibility:
    """Structured eligible/ineligible result with a plain-language `MessageKey` reason.
    Performs no I/O and no routing mutation; `active_chrome_streams` is the count of
    actively playing, non-corked Chrome streams already discovered by the caller -- exactly
    one is required, matching the `bilingual-desktop-control` spec's "Pre-Start eligibility
    gate" requirement. The GPU-unavailable reason takes priority when both preconditions
    fail, since it is the more fundamental gate."""
    mode, _status = choose_execution_mode(
        cuda_available=cuda_available, cpu_latency_ms=None, cpu_threshold_ms=_CPU_LATENCY_THRESHOLD_MS,
    )
    if not execution_mode_permits_start(mode):
        return Eligibility(startable=False, reason=MessageKey.INELIGIBLE_GPU_UNAVAILABLE)
    if active_chrome_streams != 1:
        return Eligibility(startable=False, reason=MessageKey.INELIGIBLE_NO_CHROME_STREAM)
    return Eligibility(startable=True, reason=None)
