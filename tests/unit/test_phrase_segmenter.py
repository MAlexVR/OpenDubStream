"""Streaming PCM endpointing never discards voiced tails or grows without a bound."""
import struct


def pcm(value, seconds):
    return struct.pack('<h', value) * round(16000 * seconds)


def test_phrase_ends_after_pause_without_cutting_each_second():
    from opendubstream.infrastructure.audio.phrases import PhraseSegmenter
    segmenter = PhraseSegmenter()
    out = []
    for _ in range(20):
        out += segmenter.feed(pcm(3000, .1))
    assert out == []
    for _ in range(4):
        out += segmenter.feed(pcm(0, .1))
    assert len(out) == 1
    assert pcm(3000, 2) in out[0]


def test_continuous_voice_is_bounded_and_no_samples_are_lost():
    from opendubstream.infrastructure.audio.phrases import PhraseSegmenter
    segmenter = PhraseSegmenter(max_seconds=3)
    source = pcm(3000, 8)
    out = segmenter.feed(source)
    out.append(segmenter.flush())
    assert b''.join(out) == source
    assert all(len(chunk) <= 96000 for chunk in out)


def test_long_silence_is_bounded_and_does_not_emit_empty_phrases():
    from opendubstream.infrastructure.audio.phrases import PhraseSegmenter
    segmenter = PhraseSegmenter()
    assert segmenter.feed(pcm(0, 30)) == []
    assert len(segmenter.flush()) <= 6400


def test_default_preserves_four_second_clause_until_speaker_pause():
    from opendubstream.infrastructure.audio.phrases import PhraseSegmenter
    segmenter = PhraseSegmenter()
    source = pcm(3000, 4)
    assert segmenter.feed(source) == []
    out = segmenter.feed(pcm(0, .16))
    assert len(out) == 1
    assert out[0].startswith(source)


def test_default_uninterrupted_voice_retains_six_second_safety_bound():
    from opendubstream.infrastructure.audio.phrases import PhraseSegmenter
    segmenter = PhraseSegmenter()
    source = pcm(3000, 13)
    out = segmenter.feed(source) + [segmenter.flush()]
    assert b''.join(out) == source
    assert len(out[0]) == 6 * 32000


def test_phrase_read_budget_does_not_cut_ongoing_speech(monkeypatch):
    from opendubstream.infrastructure.audio.phrases import PhraseCapture
    now = [0.0]
    monkeypatch.setattr('opendubstream.infrastructure.audio.phrases.time.monotonic', lambda: now[0])
    class Capture:
        calls = 0
        def start(self, lease): pass
        def stop(self): pass
        def read_phrase(self, deadline):
            self.calls += 1
            now[0] += deadline
            if self.calls == 1:
                now[0] -= .01
                return pcm(3000, 4)
            if self.calls == 2:
                raise TimeoutError()
            return pcm(3000, 1) + pcm(0, .16)
    capture = PhraseCapture(Capture())
    capture.start(None)
    assert capture.read_phrase(5) == b''
    assert capture.read_phrase(5).startswith(pcm(3000, 5))
