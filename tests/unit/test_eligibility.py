"""RED-first pure coverage for the pre-Start eligibility gate (task 1.5): all four
`ExecutionMode` values, plus 0/1/>1 eligible Chrome streams."""

from __future__ import annotations

import pytest

from opendubstream.application.eligibility import Eligibility, evaluate_eligibility, execution_mode_permits_start
from opendubstream.domain.pipeline import ExecutionMode
from opendubstream.ui.resources import MessageKey


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (ExecutionMode.CUDA, True),
        (ExecutionMode.CPU_QUALIFIED, False),
        (ExecutionMode.CPU_DEGRADED, False),
        (ExecutionMode.CPU_UNSUPPORTED, False),
    ],
)
def test_execution_mode_permits_start_only_for_cuda(mode: ExecutionMode, expected: bool) -> None:
    assert execution_mode_permits_start(mode) is expected


def test_evaluate_eligibility_refuses_start_when_cuda_is_unavailable() -> None:
    result = evaluate_eligibility(cuda_available=False, active_chrome_streams=1)

    assert result == Eligibility(startable=False, reason=MessageKey.INELIGIBLE_GPU_UNAVAILABLE)


def test_evaluate_eligibility_allows_start_with_cuda_and_exactly_one_chrome_stream() -> None:
    result = evaluate_eligibility(cuda_available=True, active_chrome_streams=1)

    assert result == Eligibility(startable=True, reason=None)


@pytest.mark.parametrize("active_chrome_streams", [0, 2, 3])
def test_evaluate_eligibility_refuses_start_without_exactly_one_chrome_stream(active_chrome_streams: int) -> None:
    result = evaluate_eligibility(cuda_available=True, active_chrome_streams=active_chrome_streams)

    assert result == Eligibility(startable=False, reason=MessageKey.INELIGIBLE_NO_CHROME_STREAM)


def test_evaluate_eligibility_prioritizes_the_gpu_reason_over_the_chrome_stream_reason() -> None:
    # GIVEN both preconditions fail, the more fundamental GPU gate reason MUST win.
    result = evaluate_eligibility(cuda_available=False, active_chrome_streams=0)

    assert result.reason == MessageKey.INELIGIBLE_GPU_UNAVAILABLE
    assert result.startable is False
