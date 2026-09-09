"""Mandatory local Silero VAD voice-activity detection adapter contract."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from opendubstream.infrastructure.models._pcm import decode_pcm16le
from opendubstream.infrastructure.models.assets import LocalAssets

_WINDOW_SAMPLES = 512
_VOICE_THRESHOLD = 0.5


def create_silero_vad_session(model_path: str, session_factory: Callable[..., Any]) -> Any:
    """Create the local VAD session, retrying CPU-only after a CUDA provider failure.

    Provider construction, rather than device enumeration, is the authoritative check:
    a visible NVIDIA device can still lack the ONNX Runtime CUDA dependencies. The retry
    deliberately uses a CPU-only provider list so a failed CUDA provider cannot remain in
    the fallback session.
    """
    try:
        return session_factory(model_path, providers=["CUDAExecutionProvider"])
    except Exception:
        return session_factory(model_path, providers=["CPUExecutionProvider"])


class SileroVadDetector:
    """Runs an injected local Silero VAD scoring function over fixed 512-sample windows;
    it never performs HTTP calls and never imports `onnxruntime` itself. `infer` performs
    exactly one raw model call per window and returns that window's voice score -- any
    model-specific tensor packing/RNN state threading is the injected callable's concern
    (built at the real call site in `ui/app.py`, the one place `onnxruntime` is imported),
    not this class's. All windowing, silence-padding of the final partial window, and the
    early-return threshold decision live here, matching the exact contract
    `run_confirmed()`'s inline `has_voice()` established (same window size, same
    threshold)."""

    asset_name = "silero-vad"

    def __init__(self, assets: LocalAssets, infer: Callable[[Sequence[float]], float]) -> None:
        assets.require(self.asset_name)
        self._infer = infer

    def accepts(self, audio: bytes) -> bool:
        samples = decode_pcm16le(audio)
        voiced = False
        for start in range(0, len(samples), _WINDOW_SAMPLES):
            chunk = samples[start : start + _WINDOW_SAMPLES]
            if len(chunk) < _WINDOW_SAMPLES:
                chunk = chunk + [0.0] * (_WINDOW_SAMPLES - len(chunk))
            if self._infer(chunk) >= _VOICE_THRESHOLD:
                voiced = True
        return voiced

    def reset(self) -> None:
        reset = getattr(self._infer, "reset", None)
        if reset is not None:
            reset()
