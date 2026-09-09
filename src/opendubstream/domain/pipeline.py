"""Pure local-dubbing domain records and execution qualification contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


class ExecutionMode(StrEnum):
    CUDA = "cuda"
    CPU_QUALIFIED = "cpu-qualified"
    CPU_DEGRADED = "cpu-degraded"
    CPU_UNSUPPORTED = "cpu-unsupported"


class CpuFallbackStatus(StrEnum):
    NOT_USED = "not-used"
    QUALIFIED = "qualified"
    DEGRADED = "degraded"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class Utterance:
    identifier: str
    audio: bytes
    captured_at: float
    phrase_end_at: float


@dataclass(frozen=True)
class ProcessedUtterance:
    source: Utterance
    transcript: str
    translation: str
    synthesized_audio: bytes
    timestamps: dict[str, float]


class Vad(Protocol):
    def accepts(self, audio: bytes) -> bool: ...


class Asr(Protocol):
    def transcribe(self, audio: bytes) -> str: ...


class Translator(Protocol):
    def translate(self, text: str) -> str: ...


class Tts(Protocol):
    def synthesize(self, text: str) -> bytes: ...


def choose_execution_mode(
    *, cuda_available: bool, cpu_latency_ms: float | None, cpu_threshold_ms: float
) -> tuple[ExecutionMode, CpuFallbackStatus]:
    """Prefer CUDA; CPU claims are permitted only with measured qualification."""
    if cuda_available:
        return ExecutionMode.CUDA, CpuFallbackStatus.NOT_USED
    if cpu_latency_ms is None:
        return ExecutionMode.CPU_UNSUPPORTED, CpuFallbackStatus.UNSUPPORTED
    if cpu_latency_ms <= cpu_threshold_ms:
        return ExecutionMode.CPU_QUALIFIED, CpuFallbackStatus.QUALIFIED
    return ExecutionMode.CPU_DEGRADED, CpuFallbackStatus.DEGRADED
