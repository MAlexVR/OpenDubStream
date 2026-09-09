"""Mandatory local Faster-Whisper English ASR adapter contract."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

from opendubstream.infrastructure.models._pcm import decode_pcm16le
from opendubstream.infrastructure.models.assets import LocalAssets


def create_faster_whisper_model(model_path: str, model_factory: Callable[..., Any]) -> Any:
    """Create the local ASR backend, preferring CUDA but recovering from a broken runtime.

    A visible CUDA device is not enough evidence that CTranslate2 can load its CUDA/cuDNN
    libraries.  Construction is the first authoritative capability check, so retry exactly
    once with the documented local CPU/int8 configuration when that attempt fails.
    """
    try:
        return model_factory(model_path, device="cuda", compute_type="float16", local_files_only=True)
    except Exception:
        return model_factory(model_path, device="cpu", compute_type="int8", local_files_only=True)


class FasterWhisperTranscriber:
    """Runs an injected local Faster-Whisper transcription function; it never performs
    HTTP calls and never imports `faster_whisper` itself. `infer` receives normalized
    float samples and returns the model's segment objects (each exposing `.text`,
    matching `faster_whisper.WhisperModel.transcribe()`'s own segment shape) -- joining
    those segment texts and stripping the result is real, non-Qt business logic that
    belongs here, not duplicated inside `ui/app.py`."""

    asset_name = "faster-whisper-distil-large-v3"

    def __init__(self, assets: LocalAssets, infer: Callable[[Sequence[float]], Iterable[object]]) -> None:
        assets.require(self.asset_name)
        self._infer = infer

    def transcribe(self, audio: bytes) -> str:
        samples = decode_pcm16le(audio)
        segments = self._infer(samples)
        return "".join(segment.text for segment in segments).strip()
