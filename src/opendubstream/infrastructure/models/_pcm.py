"""Pure-Python PCM16/resample math shared by the model adapters in this package.

Deliberately free of numpy (or any third-party dependency): this project's lightweight
`.venv` installs zero third-party packages (`pyproject.toml`'s `dependencies = []`), so
every adapter under `infrastructure/models/` must stay importable and unit-testable
there. The real, numpy-backed model calls live in the injected `infer` callables built
at the one real call site (`ui/app.py`, under `.venv-inference`), never in these helpers.
"""

from __future__ import annotations

import struct
from collections.abc import Sequence

_PCM16_FULL_SCALE = 32768.0
_PCM16_MAX_SAMPLE_VALUE = 32767


def decode_pcm16le(audio: bytes) -> list[float]:
    """Decodes little-endian signed 16-bit PCM bytes into floats normalized to [-1.0, 1.0].
    A trailing odd byte (not enough to form one more sample) is silently dropped."""
    sample_count = len(audio) // 2
    raw = struct.unpack(f"<{sample_count}h", audio[: sample_count * 2])
    return [value / _PCM16_FULL_SCALE for value in raw]


def encode_pcm16le(samples: Sequence[float]) -> bytes:
    """Clips floats to [-1.0, 1.0], scales to full-scale 16-bit range, and packs them as
    little-endian signed 16-bit PCM bytes."""
    scaled = [int(max(-1.0, min(1.0, value)) * _PCM16_MAX_SAMPLE_VALUE) for value in samples]
    if not scaled:
        return b""
    return struct.pack(f"<{len(scaled)}h", *scaled)


def linear_resample(audio: Sequence[float], source_rate: int, target_rate: int) -> list[float]:
    """Linear-interpolation resample from `source_rate` to `target_rate` Hz, matching
    `np.interp(np.linspace(0, len(audio) - 1, target_length), np.arange(len(audio)), audio)`
    sample for sample."""
    source_length = len(audio)
    if source_length == 0:
        return []
    target_length = max(1, int(source_length * target_rate / source_rate))
    if target_length == 1:
        return [audio[0]]
    step = (source_length - 1) / (target_length - 1)
    resampled: list[float] = []
    for index in range(target_length):
        position = index * step
        lower = int(position)
        upper = min(lower + 1, source_length - 1)
        fraction = position - lower
        resampled.append(audio[lower] * (1 - fraction) + audio[upper] * fraction)
    return resampled
