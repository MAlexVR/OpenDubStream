"""RED-first coverage for the Faster-Whisper ASR adapter extracted from `ui/app.py`'s former
`_LazyFasterWhisperAsr` -- segment-joining is real, non-Qt business logic per design.md."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from opendubstream.infrastructure.models.assets import AssetIntegrityError, AssetManifest, AssetPin, LocalAssets
from opendubstream.infrastructure.models.asr import FasterWhisperTranscriber, create_faster_whisper_model


class _FakeSegment:
    def __init__(self, text: str) -> None:
        self.text = text


def _assets(pin_present: bool = True) -> LocalAssets:
    pins = (AssetPin("faster-whisper-distil-large-v3", "faster-whisper-distil-large-v3", "digest"),) if pin_present else ()
    return LocalAssets(root=Path("/models"), manifest=AssetManifest(pins))


def test_transcribe_joins_multiple_segment_texts_and_strips_surrounding_whitespace() -> None:
    transcriber = FasterWhisperTranscriber(_assets(), lambda samples: [_FakeSegment(" Hello"), _FakeSegment(" world ")])

    assert transcriber.transcribe(struct.pack("<1h", 0)) == "Hello world"


def test_transcribe_returns_empty_string_when_no_segments_are_returned() -> None:
    transcriber = FasterWhisperTranscriber(_assets(), lambda samples: [])

    assert transcriber.transcribe(struct.pack("<1h", 0)) == ""


def test_transcribe_decodes_full_scale_pcm16_samples_to_normalized_floats_before_inferring() -> None:
    captured: list[list[float]] = []

    def infer(samples: list[float]) -> list[_FakeSegment]:
        captured.append(samples)
        return [_FakeSegment("ok")]

    transcriber = FasterWhisperTranscriber(_assets(), infer)
    audio = struct.pack("<2h", 32767, -32768)

    transcriber.transcribe(audio)

    assert captured[0][0] == pytest.approx(32767 / 32768.0)
    assert captured[0][1] == pytest.approx(-1.0)


def test_missing_asset_pin_raises_asset_integrity_error_before_any_inference() -> None:
    calls: list[object] = []

    with pytest.raises(AssetIntegrityError):
        FasterWhisperTranscriber(_assets(pin_present=False), lambda samples: calls.append(samples) or [])

    assert calls == []


def test_cuda_model_initialization_failure_retries_once_with_cpu_int8() -> None:
    calls: list[tuple[str, str, bool]] = []
    cpu_model = object()

    def create_model(_path: str, *, device: str, compute_type: str, local_files_only: bool) -> object:
        calls.append((device, compute_type, local_files_only))
        if device == "cuda":
            raise RuntimeError("libcudnn_cnn.so.9: cannot open shared object file")
        return cpu_model

    assert create_faster_whisper_model("/models/whisper", create_model) is cpu_model
    assert calls == [("cuda", "float16", True), ("cpu", "int8", True)]


def test_successful_cuda_model_initialization_does_not_construct_a_cpu_model() -> None:
    calls: list[tuple[str, str]] = []
    cuda_model = object()

    def create_model(_path: str, *, device: str, compute_type: str, local_files_only: bool) -> object:
        assert local_files_only is True
        calls.append((device, compute_type))
        return cuda_model

    assert create_faster_whisper_model("/models/whisper", create_model) is cuda_model
    assert calls == [("cuda", "float16")]
