#!/usr/bin/env python3
"""Run one offline WAV fixture through VAD → ASR → MT → Kokoro without playback.

This is deliberately a fixture-only feasibility harness: it never captures a
PipeWire/Chrome stream, writes synthesized audio, starts playback, or downloads
assets. Run it from the GNOME desktop session after local provisioning.
"""

import argparse
import hashlib
import importlib.util
import os
import shutil
import subprocess
import time
import wave
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = Path(os.environ.get("OPENDUBSTREAM_MODEL_ROOT", Path.home() / ".local/share/opendubstream/models"))
REQUIRED_ASSETS = (
    "silero-vad/silero_vad.onnx",
    "faster-whisper-distil-large-v3",
    "opus-mt-en-es",
    "kokoro-onnx-v1/kokoro-v1.0.onnx",
    "kokoro-onnx-v1/voices-v1.0.bin",
)


class SpikeError(RuntimeError):
    """The local fixture cannot safely proceed."""


@dataclass(frozen=True)
class SpikeResult:
    english: str
    spanish: str
    mode: str
    asr_seconds: float
    translation_seconds: float
    tts_seconds: float
    total_seconds: float


def read_pcm_wav(path: Path) -> bytes:
    try:
        with wave.open(str(path), "rb") as input_file:
            valid = input_file.getnchannels() == 1 and input_file.getframerate() == 16_000 and input_file.getsampwidth() == 2 and input_file.getcomptype() == "NONE"
            if not valid:
                raise SpikeError("--input must be a mono PCM 16 kHz WAV")
            frames = input_file.readframes(input_file.getnframes())
    except (wave.Error, OSError) as error:
        raise SpikeError(f"cannot read --input WAV: {error}") from error
    if not frames:
        raise SpikeError("--input WAV contains no PCM frames")
    return frames


def verify_assets(root: Path, manifest: Path) -> None:
    if not root.is_dir() or not manifest.is_file():
        raise SpikeError("local model root and assets.sha256 manifest are required")
    try:
        entries = [line.split(maxsplit=1) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    except OSError as error:
        raise SpikeError(f"cannot read asset manifest: {error}") from error
    if not entries:
        raise SpikeError("asset manifest is empty")
    for entry in entries:
        if len(entry) != 2:
            raise SpikeError("asset manifest has an invalid line")
        digest, relative = entry
        target = (root / relative.strip()).resolve()
        if root.resolve() not in target.parents or not target.is_file():
            raise SpikeError(f"manifest asset is missing: {relative}")
        if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise SpikeError(f"asset hash mismatch: {relative}")
    missing = [item for item in REQUIRED_ASSETS if not (root / item).exists()]
    if missing:
        raise SpikeError(f"required local assets are missing: {', '.join(missing)}")


def has_voice(pcm: bytes, root: Path) -> bool:
    """Use the external ONNX Silero model, not the bundled Python loader."""
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError as error:
        raise SpikeError("Silero ONNX runtime is unavailable in the local inference venv") from error
    try:
        session = ort.InferenceSession(str(root / "silero-vad/silero_vad.onnx"), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        samples = np.frombuffer(pcm, dtype="<i2").astype("float32") / 32768.0
        state = np.zeros((2, 1, 128), dtype="float32")
        # This pinned ONNX export only accepts a 256-sample window: 512+ raises an
        # LSTM shape error internally, and 512 itself silently returns near-zero scores.
        window = 256
        for start in range(0, len(samples), window):
            chunk = samples[start : start + window]
            if len(chunk) < window:
                chunk = np.pad(chunk, (0, window - len(chunk)))
            score, state = session.run(None, {"input": chunk[None, :], "state": state, "sr": np.array(16000, dtype="int64")})
            if float(score[0][0]) >= 0.5:
                return True
    except Exception as error:
        raise SpikeError(f"external Silero ONNX VAD failed: {error}") from error
    return False


def transcribe(pcm: bytes, root: Path, device: str, compute: str) -> str:
    try:
        import numpy as np
        from faster_whisper import WhisperModel
    except ImportError as error:
        raise SpikeError("faster-whisper is unavailable in the local inference venv") from error
    model = WhisperModel(str(root / "faster-whisper-distil-large-v3"), device=device, compute_type=compute, local_files_only=True)
    segments, _ = model.transcribe(np.frombuffer(pcm, dtype="<i2").astype("float32") / 32768.0, language="en", task="transcribe", vad_filter=False)
    text = "".join(segment.text for segment in segments).strip()
    if not text:
        raise SpikeError("ASR returned no English transcription")
    return text


def translate(text: str, root: Path) -> str:
    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as error:
        raise SpikeError("transformers OPUS-MT runtime is unavailable") from error
    model_path = str(root / "opus-mt-en-es")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True).to(device)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    generated = model.generate(**inputs)
    output = tokenizer.decode(generated[0], skip_special_tokens=True).strip()
    if not output:
        raise SpikeError("OPUS-MT returned no Spanish translation")
    return output


def ensure_phonemizer() -> None:
    if importlib.util.find_spec("misaki") is not None:
        return
    executable = shutil.which("espeak-ng") or shutil.which("espeak")
    if executable:
        completed = subprocess.run([executable, "--version"], check=False, capture_output=True, text=True)
        if completed.returncode == 0:
            return
    raise SpikeError("Kokoro phonemization requires local Misaki or espeak-ng; install it locally and retry (no online fallback is used)")


def synthesize(text: str, root: Path, voice: str) -> object:
    try:
        from kokoro_onnx import Kokoro
    except ImportError as error:
        raise SpikeError("kokoro-onnx is unavailable in the local inference venv") from error
    engine = Kokoro(str(root / "kokoro-onnx-v1/kokoro-v1.0.onnx"), str(root / "kokoro-onnx-v1/voices-v1.0.bin"))
    return engine.create(text, voice=voice, lang="es")


def run(input_path: Path, root: Path, manifest: Path, voice: str) -> SpikeResult:
    os.environ["HF_HUB_OFFLINE"] = "1"
    # kokoro_onnx detects GPU by checking find_spec("onnxruntime-gpu"), which never
    # matches the "onnxruntime" import name, so it silently defaults to CPU otherwise.
    os.environ.setdefault("ONNX_PROVIDER", "CUDAExecutionProvider")
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    started = time.monotonic()
    pcm = read_pcm_wav(input_path)
    verify_assets(root, manifest)
    if not has_voice(pcm, root):
        raise SpikeError("external Silero VAD found no speech; no ASR, translation, or synthesis was run")
    try:
        english = transcribe(pcm, root, "cuda", "float16")
        mode = "CUDA/float16"
    except Exception as cuda_error:
        try:
            english = transcribe(pcm, root, "cpu", "int8")
            mode = "CPU/int8 fallback"
        except Exception as cpu_error:
            raise SpikeError(f"ASR failed on CUDA and the one allowed CPU fallback: {cpu_error}") from cuda_error
    after_asr = time.monotonic()
    spanish = translate(english, root)
    after_translation = time.monotonic()
    ensure_phonemizer()
    synthesize(spanish, root, voice)
    after_tts = time.monotonic()
    return SpikeResult(english, spanish, mode, after_asr - started, after_translation - after_asr, after_tts - after_translation, after_tts - started)


def print_result(result: SpikeResult) -> None:
    print(f"EN: {result.english}\nES: {result.spanish}\nMODE: {result.mode}")
    print(f"ASR: {result.asr_seconds:.3f}s\nTR: {result.translation_seconds:.3f}s\nTTS: {result.tts_seconds:.3f}s\nTOTAL: {result.total_seconds:.3f}s")


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline WAV fixture-only local dubbing spike; no capture or playback.")
    parser.add_argument("--input", required=True, type=Path, help="mono PCM 16 kHz WAV fixture")
    parser.add_argument("--model-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--voice", default="ef_dora", help="local Spanish Kokoro voice")
    args = parser.parse_args()
    try:
        print_result(run(args.input, args.model_root, args.manifest or args.model_root / "assets.sha256", args.voice))
        return 0
    except SpikeError as error:
        print(f"ERROR: {error}", file=__import__("sys").stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
