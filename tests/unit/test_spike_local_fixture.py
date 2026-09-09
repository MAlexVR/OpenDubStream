"""Tests for the fixture-only, offline vertical-spike entrypoint."""
from __future__ import annotations

import hashlib
import importlib.util
import os
import wave
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "spike-local-fixture.py"


def load_module():
    spec = importlib.util.spec_from_file_location("spike_local_fixture", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def wav(path: Path, *, channels: int = 1, rate: int = 16_000) -> None:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(b"\0\0" * rate)


def model_root(tmp_path: Path) -> Path:
    root = tmp_path / "models"
    for relative, payload in {
        "silero-vad/silero_vad.onnx": b"vad",
        "faster-whisper-distil-large-v3/model.bin": b"asr",
        "opus-mt-en-es/pytorch_model.bin": b"mt",
        "kokoro-onnx-v1/kokoro-v1.0.onnx": b"tts",
        "kokoro-onnx-v1/voices-v1.0.bin": b"voices",
    }.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    root.joinpath("assets.sha256").write_text(
        "".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(root)}\n" for p in root.rglob("*") if p.is_file()),
        encoding="utf-8",
    )
    return root


def test_rejects_non_mono_16khz_pcm_wav(tmp_path: Path) -> None:
    module = load_module()
    invalid = tmp_path / "stereo.wav"
    wav(invalid, channels=2)
    with pytest.raises(module.SpikeError, match="mono PCM 16 kHz"):
        module.read_pcm_wav(invalid)


def test_rejects_missing_or_tampered_manifest_before_inference(tmp_path: Path) -> None:
    module = load_module()
    root = model_root(tmp_path)
    root.joinpath("silero-vad/silero_vad.onnx").write_bytes(b"tampered")
    with pytest.raises(module.SpikeError, match="hash mismatch"):
        module.verify_assets(root, root / "assets.sha256")


def test_run_prefers_cuda_then_falls_back_once_and_stays_offline(tmp_path: Path, monkeypatch, capsys) -> None:
    module = load_module()
    audio = tmp_path / "input.wav"
    wav(audio)
    root = model_root(tmp_path)
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(module, "read_pcm_wav", lambda _: b"audio")
    monkeypatch.setattr(module, "verify_assets", lambda *_: None)
    monkeypatch.setattr(module, "has_voice", lambda *_: True)
    def fake_transcribe(_audio, _root, device, compute):
        calls.append((device, compute))
        if device == "cuda":
            raise RuntimeError("CUDA unavailable")
        return "hello"
    monkeypatch.setattr(module, "transcribe", fake_transcribe)
    monkeypatch.setattr(module, "translate", lambda text, _root: f"ES:{text}")
    monkeypatch.setattr(module, "synthesize", lambda text, _root, voice: (text, voice))
    monkeypatch.setattr(module, "ensure_phonemizer", lambda: None)
    times = iter([1.0, 1.1, 1.3, 1.6, 2.0])
    monkeypatch.setattr(module.time, "monotonic", lambda: next(times))

    result = module.run(audio, root, root / "assets.sha256", "ef_dora")

    assert calls == [("cuda", "float16"), ("cpu", "int8")]
    assert result.mode == "CPU/int8 fallback"
    assert result.english == "hello"
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
    module.print_result(result)
    output = capsys.readouterr().out
    assert "EN: hello" in output and "ES: ES:hello" in output and "MODE: CPU/int8 fallback" in output


def test_run_fails_without_voice_without_asr_or_playback(tmp_path: Path, monkeypatch) -> None:
    module = load_module()
    audio = tmp_path / "silence.wav"
    wav(audio)
    root = model_root(tmp_path)
    monkeypatch.setattr(module, "read_pcm_wav", lambda _: b"silence")
    monkeypatch.setattr(module, "verify_assets", lambda *_: None)
    monkeypatch.setattr(module, "has_voice", lambda *_: False)
    monkeypatch.setattr(module, "transcribe", lambda *_: pytest.fail("ASR must not run for no-voice input"))

    with pytest.raises(module.SpikeError, match="no speech"):
        module.run(audio, root, root / "assets.sha256", "ef_dora")
    source = SCRIPT.read_text(encoding="utf-8")
    assert "sounddevice" not in source and "pyaudio" not in source and "requests" not in source
