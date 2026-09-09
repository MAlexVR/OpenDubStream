"""Pinned Silero 16-kHz frame/context contract; no real model required."""
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_silero_uses_512_sample_frames_and_visits_all_frames():
    from opendubstream.infrastructure.models.vad import SileroVadDetector
    from test_vad import _assets, _silence
    seen = []
    detector = SileroVadDetector(_assets(), lambda frame: seen.append(frame) or 0.9)
    assert detector.accepts(_silence(1100))
    assert [len(frame) for frame in seen] == [512, 512, 512]


def test_raw_silero_prepends_context_and_reset_clears_state(monkeypatch):
    pytest.importorskip("PySide6")
    np = pytest.importorskip("numpy")
    from opendubstream.ui.app import _make_vad_infer
    seen = []
    class Session:
        def run(self, outputs, feed):
            seen.append(feed)
            return np.array([[0.8]]), np.ones((2, 1, 128), dtype="float32")
    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(InferenceSession=lambda *a, **kw: Session()))
    infer = _make_vad_infer(Path("/models"))
    infer([0.25] * 512)
    infer([0.5] * 512)
    assert seen[0]["input"].shape == (1, 576)
    assert np.all(seen[0]["input"][:, :64] == 0)
    assert np.all(seen[1]["input"][:, :64] == 0.25)
    infer.reset()
    infer([0.5] * 512)
    assert np.all(seen[2]["input"][:, :64] == 0)
    assert np.all(seen[2]["state"] == 0)


def test_tts_receives_current_speech_speed_without_reloading_model(monkeypatch):
    pytest.importorskip('PySide6')
    from opendubstream.ui.app import _make_tts_infer
    from opendubstream.application.live_controls import LiveControls, LiveSettings
    calls = []
    loads = []
    class Engine:
        def __init__(self, *args): loads.append(args)
        def create(self, text, **kwargs): calls.append(kwargs); return [], 16000
    monkeypatch.setitem(sys.modules, 'kokoro_onnx', SimpleNamespace(Kokoro=Engine))
    controls = LiveControls()
    infer = _make_tts_infer(Path('/models'), 'ef_dora', controls)
    infer('hola')
    controls.update(LiveSettings(speed=1.25))
    infer('mundo')
    assert [call['speed'] for call in calls] == [1.0, 1.25]
    assert len(loads) == 1
