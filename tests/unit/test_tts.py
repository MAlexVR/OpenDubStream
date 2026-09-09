"""RED-first coverage for the Kokoro Spanish TTS adapter extracted from `ui/app.py`'s former
`_LazyKokoroTts` -- resample math is real, non-Qt business logic per design.md."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from opendubstream.infrastructure.models.assets import AssetIntegrityError, AssetManifest, AssetPin, LocalAssets
from opendubstream.infrastructure.models.tts import KokoroSpanishSynthesizer


def _assets(pin_present: bool = True) -> LocalAssets:
    pins = (AssetPin("kokoro-onnx-v1", "kokoro-onnx-v1", "digest"),) if pin_present else ()
    return LocalAssets(root=Path("/models"), manifest=AssetManifest(pins))


def test_synthesize_resamples_from_the_source_rate_to_16khz_and_returns_pcm16_bytes() -> None:
    tts = KokoroSpanishSynthesizer(_assets(), lambda text: ([0.0, 0.5, 1.0, 0.5], 24_000))

    result = tts.synthesize("hola")

    expected_length = max(1, int(4 * 16_000 / 24_000))
    assert len(result) == expected_length * 2
    values = struct.unpack(f"<{expected_length}h", result)
    assert values[0] == 0
    assert values[-1] == pytest.approx(16383, abs=1)


def test_synthesize_clips_out_of_range_samples_to_the_full_scale_range() -> None:
    tts = KokoroSpanishSynthesizer(_assets(), lambda text: ([2.0, -3.0], 16_000))

    result = tts.synthesize("hola")

    values = struct.unpack(f"<{len(result) // 2}h", result)
    assert max(values) <= 32767
    assert min(values) >= -32767


def test_synthesize_produces_a_different_length_for_a_different_source_sample_rate() -> None:
    tts = KokoroSpanishSynthesizer(_assets(), lambda text: ([0.0, 1.0, 0.0, 1.0], 24_000))
    baseline = KokoroSpanishSynthesizer(_assets(), lambda text: ([0.0, 1.0, 0.0, 1.0], 16_000))

    assert len(tts.synthesize("hola")) != len(baseline.synthesize("hola"))


def test_missing_asset_pin_raises_asset_integrity_error_before_any_inference() -> None:
    calls: list[str] = []

    with pytest.raises(AssetIntegrityError):
        KokoroSpanishSynthesizer(_assets(pin_present=False), lambda text: calls.append(text) or ([], 16_000))

    assert calls == []
