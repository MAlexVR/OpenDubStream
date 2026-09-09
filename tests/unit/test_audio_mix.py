from __future__ import annotations

import pytest

from opendubstream.application.audio_mix import apply_pcm_gain


def test_apply_pcm_gain_scales_signed_samples_and_clamps_without_changing_format() -> None:
    pcm = (-20_000).to_bytes(2, "little", signed=True) + (20_000).to_bytes(2, "little", signed=True)

    assert apply_pcm_gain(pcm, 50) == (-10_000).to_bytes(2, "little", signed=True) + (10_000).to_bytes(2, "little", signed=True)
    assert apply_pcm_gain(pcm, 200) == (-32_768).to_bytes(2, "little", signed=True) + (32_767).to_bytes(2, "little", signed=True)


def test_apply_pcm_gain_rejects_invalid_percent_or_incomplete_pcm() -> None:
    with pytest.raises(ValueError, match="percentage"):
        apply_pcm_gain(b"\x00\x00", -1)
    with pytest.raises(ValueError, match="complete"):
        apply_pcm_gain(b"\x00", 100)
