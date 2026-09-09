"""RED-first pure coverage for the PCM16/resample math shared by the model adapters
(`vad.py`, `asr.py`, `tts.py`). Deliberately numpy-free: this project's lightweight
`.venv` (see `pyproject.toml`'s empty `dependencies = []`) has no numpy, so every model
adapter under `infrastructure/models/` must stay importable and testable without it."""

from __future__ import annotations

import struct

import pytest

from opendubstream.infrastructure.models._pcm import decode_pcm16le, encode_pcm16le, linear_resample


def test_decode_pcm16le_normalizes_full_scale_and_zero_samples() -> None:
    audio = struct.pack("<3h", 32767, -32768, 0)

    samples = decode_pcm16le(audio)

    assert samples[0] == pytest.approx(32767 / 32768.0)
    assert samples[1] == pytest.approx(-1.0)
    assert samples[2] == pytest.approx(0.0)


def test_decode_pcm16le_truncates_a_trailing_odd_byte() -> None:
    audio = struct.pack("<2h", 100, -100) + b"\x01"

    samples = decode_pcm16le(audio)

    assert len(samples) == 2


def test_encode_pcm16le_clips_values_outside_the_full_scale_range() -> None:
    encoded = encode_pcm16le([2.0, -3.0, 0.0])

    values = struct.unpack("<3h", encoded)

    assert values == (32767, -32767, 0)


def test_encode_pcm16le_round_trips_representable_values_through_decode_pcm16le() -> None:
    original = [0.5, -0.5, 0.25]

    round_tripped = decode_pcm16le(encode_pcm16le(original))

    for expected, actual in zip(original, round_tripped):
        assert actual == pytest.approx(expected, abs=1e-3)


def test_linear_resample_upsamples_preserving_endpoint_values() -> None:
    resampled = linear_resample([0.0, 1.0], source_rate=1, target_rate=2)

    assert len(resampled) == 4
    assert resampled[0] == pytest.approx(0.0)
    assert resampled[-1] == pytest.approx(1.0)


def test_linear_resample_downsamples_to_a_shorter_target_length() -> None:
    resampled = linear_resample([0.0, 1.0, 2.0, 3.0], source_rate=4, target_rate=2)

    assert len(resampled) == 2
    assert resampled[0] == pytest.approx(0.0)
    assert resampled[-1] == pytest.approx(3.0)


def test_linear_resample_returns_a_single_sample_when_the_computed_target_length_is_zero() -> None:
    resampled = linear_resample([0.0, 1.0, 2.0, 3.0], source_rate=100, target_rate=1)

    assert resampled == [0.0]


def test_linear_resample_interpolates_between_two_source_points() -> None:
    resampled = linear_resample([0.0, 10.0], source_rate=1, target_rate=3)

    assert len(resampled) == 6
    assert resampled[0] == pytest.approx(0.0)
    assert resampled[-1] == pytest.approx(10.0)
    assert 0.0 < resampled[2] < 10.0
