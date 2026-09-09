"""Pure per-session PCM gain; never changes a PipeWire default or route."""

from __future__ import annotations


def apply_pcm_gain(pcm: bytes, volume_percent: int) -> bytes:
    """Scale signed 16-bit little-endian PCM with deterministic saturation."""
    if not isinstance(volume_percent, int) or volume_percent < 0:
        raise ValueError("volume percentage must be a non-negative integer")
    if len(pcm) % 2:
        raise ValueError("PCM must contain complete s16le samples")
    scaled = bytearray()
    for offset in range(0, len(pcm), 2):
        sample = int.from_bytes(pcm[offset : offset + 2], "little", signed=True)
        value = round(sample * volume_percent / 100)
        scaled.extend(max(-32_768, min(32_767, value)).to_bytes(2, "little", signed=True))
    return bytes(scaled)
