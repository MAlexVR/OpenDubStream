"""Mandatory local Kokoro Spanish text-to-speech adapter contract."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from opendubstream.infrastructure.models._pcm import encode_pcm16le, linear_resample
from opendubstream.infrastructure.models.assets import LocalAssets

_TARGET_SAMPLE_RATE_HZ = 16_000


class KokoroSpanishSynthesizer:
    """Runs an injected local Kokoro synthesis function; it never performs HTTP calls and
    never imports `kokoro_onnx` itself. `infer` receives text and returns `(audio,
    source_rate)` exactly as `kokoro_onnx.Kokoro.create()` does -- resampling that audio
    to the fixed 16 kHz playback rate and packing it as PCM16 bytes is real, non-Qt
    business logic that belongs here, not duplicated inside `ui/app.py`. The voice/
    language selection is baked into the injected callable at construction time in
    `ui/app.py` (mirroring `OpusMtEnglishSpanishTranslator`'s `infer: Callable[[str], str]`
    shape exactly), so this class's own contract stays a single `text -> bytes` call."""

    asset_name = "kokoro-onnx-v1"

    def __init__(self, assets: LocalAssets, infer: Callable[[str], tuple[Sequence[float], int]]) -> None:
        assets.require(self.asset_name)
        self._infer = infer

    def synthesize(self, text: str) -> bytes:
        audio, source_rate = self._infer(text)
        resampled = linear_resample(audio, source_rate, _TARGET_SAMPLE_RATE_HZ)
        return encode_pcm16le(resampled)
