"""RED-first coverage for the Silero VAD adapter extracted from `ui/app.py`'s former
`_LazySileroVad` (see `openspec/changes/bilingual-desktop-control-ui/design.md`'s "thin Qt
shell" requirement -- windowing/threshold logic must live in a testable module, not `app.py`).
"""

from __future__ import annotations

import struct

import pytest

from opendubstream.infrastructure.models.assets import AssetIntegrityError, AssetManifest, AssetPin, LocalAssets
from opendubstream.infrastructure.models.vad import SileroVadDetector, create_silero_vad_session


def _assets(pin_present: bool = True) -> LocalAssets:
    pins = (AssetPin("silero-vad", "silero-vad/silero_vad.onnx", "digest"),) if pin_present else ()
    return LocalAssets(root=__import__("pathlib").Path("/models"), manifest=AssetManifest(pins))


def _silence(sample_count: int) -> bytes:
    return struct.pack(f"<{sample_count}h", *([0] * sample_count))


def test_accepts_returns_false_when_every_window_score_is_below_threshold() -> None:
    calls: list[list[float]] = []

    def infer(chunk: list[float]) -> float:
        calls.append(chunk)
        return 0.1

    vad = SileroVadDetector(_assets(), infer)

    assert vad.accepts(_silence(1024)) is False
    assert len(calls) == 2


def test_accepts_returns_true_and_preserves_state_by_visiting_every_window() -> None:
    calls: list[list[float]] = []

    def infer(chunk: list[float]) -> float:
        calls.append(chunk)
        return 0.9

    vad = SileroVadDetector(_assets(), infer)

    assert vad.accepts(_silence(1024)) is True
    assert len(calls) == 2


def test_accepts_pads_the_final_partial_window_with_silence() -> None:
    captured: list[list[float]] = []

    def infer(chunk: list[float]) -> float:
        captured.append(chunk)
        return 0.0

    vad = SileroVadDetector(_assets(), infer)
    vad.accepts(_silence(556))

    assert len(captured) == 2
    assert len(captured[-1]) == 512
    assert captured[-1][44:] == [0.0] * (512 - 44)


def test_missing_asset_pin_raises_asset_integrity_error_before_any_inference() -> None:
    calls: list[object] = []

    with pytest.raises(AssetIntegrityError):
        SileroVadDetector(_assets(pin_present=False), lambda chunk: calls.append(chunk) or 1.0)

    assert calls == []


def test_cuda_vad_session_initialization_failure_retries_with_cpu_only_provider() -> None:
    calls: list[tuple[str, list[str]]] = []
    cpu_session = object()

    def create_session(model_path: str, *, providers: list[str]) -> object:
        calls.append((model_path, providers))
        if providers == ["CUDAExecutionProvider"]:
            raise RuntimeError("libcudnn_cnn.so.9: cannot open shared object file")
        return cpu_session

    assert create_silero_vad_session("/models/silero.onnx", create_session) is cpu_session
    assert calls == [
        ("/models/silero.onnx", ["CUDAExecutionProvider"]),
        ("/models/silero.onnx", ["CPUExecutionProvider"]),
    ]


def test_successful_cuda_vad_session_initialization_does_not_construct_cpu_session() -> None:
    calls: list[list[str]] = []
    cuda_session = object()

    def create_session(_model_path: str, *, providers: list[str]) -> object:
        calls.append(providers)
        return cuda_session

    assert create_silero_vad_session("/models/silero.onnx", create_session) is cuda_session
    assert calls == [["CUDAExecutionProvider"]]
