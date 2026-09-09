"""Bounded energy endpointing for 16-kHz mono s16le capture.

Energy only proposes boundaries; Silero remains the speech classifier. Frames stay in
order, with a short pre-roll, a pause endpoint and a hard cap for continuous speech.
"""
from __future__ import annotations

import struct
import time


class PhraseSegmenter:
    frame_bytes = 640  # 20 ms

    def __init__(self, *, max_seconds: float = 6.0) -> None:
        if max_seconds < 0.1:
            raise ValueError("phrase bound must be at least 100 ms")
        self._limit = int(max_seconds * 32000)
        self._pending = bytearray()
        self._phrase = bytearray()
        self._preroll = bytearray()
        self._silence = 0

    def feed(self, pcm: bytes) -> list[bytes]:
        if len(pcm) % 2:
            raise ValueError("PCM requires complete samples")
        out = []
        # Consume incrementally so even a large read cannot become a persistent backlog.
        for offset in range(0, len(pcm), self.frame_bytes):
            self._pending.extend(pcm[offset:offset + self.frame_bytes])
            while len(self._pending) >= self.frame_bytes:
                frame = bytes(self._pending[:self.frame_bytes])
                del self._pending[:self.frame_bytes]
                values = struct.unpack('<320h', frame)
                voiced = sum(v * v for v in values) / 320 >= 260 ** 2
                if not self._phrase and not voiced:
                    self._preroll.extend(frame)
                    del self._preroll[:-6400]
                    continue
                if not self._phrase:
                    self._phrase.extend(self._preroll)
                    self._preroll.clear()
                self._phrase.extend(frame)
                self._silence = 0 if voiced else self._silence + len(frame)
                # Prefer a speaker pause, including shorter breaths after three seconds.
                # Six seconds remains a safety cap, not a semantic sentence guarantee.
                pause_bytes = 3840 if len(self._phrase) >= 96000 else 10240
                if len(self._phrase) >= self._limit or self._silence >= pause_bytes:
                    out.append(bytes(self._phrase))
                    self._phrase.clear()
                    self._silence = 0
        return out

    def flush(self) -> bytes:
        pcm = bytes(self._phrase + self._pending) if self._phrase else bytes(self._preroll + self._pending)
        self._phrase.clear()
        self._pending.clear()
        self._preroll.clear()
        self._silence = 0
        return pcm


class PhraseCapture:
    """Endpoint production PCM without changing the raw capture/probe contracts."""

    def __init__(self, capture) -> None:
        self._capture = capture
        self._segmenter = PhraseSegmenter()
        self._ready: list[bytes] = []
        self._stopped = False
        self._last_pcm_at: float | None = None

    def start(self, lease):
        self._stopped = False
        self._segmenter = PhraseSegmenter()
        self._ready.clear()
        self._last_pcm_at = None
        return self._capture.start(lease)

    def read_phrase(self, deadline: float) -> bytes:
        until = time.monotonic() + deadline
        while not self._stopped and not self._ready:
            remaining = until - time.monotonic()
            if remaining <= 0:
                return b''  # Ordinary silence is not a failed session.
            try:
                pcm = self._capture.read_phrase(remaining)
            except TimeoutError:
                # Budget expiry between normal packets is not a speaker pause.
                # Flush only after sustained no-data, e.g. a corked Chrome stream.
                if self._last_pcm_at is not None and time.monotonic() - self._last_pcm_at >= 0.32:
                    self._last_pcm_at = None
                    return self._segmenter.flush()
                return b""
            if pcm:
                self._last_pcm_at = time.monotonic()
            self._ready.extend(self._segmenter.feed(pcm))
        return self._ready.pop(0) if self._ready else b''

    def stop(self) -> None:
        self._stopped = True
        self._capture.stop()
