#!/usr/bin/env python3
"""Measure per-phrase latency with models loaded once, as the running app would.

scripts/spike-local-fixture.py loads every model fresh per invocation (correct for
a one-shot correctness check, but its timings are dominated by model-load cost, not
inference). This script loads each model once and processes multiple WAV fixtures
against the warm models, reporting P50/P95 per stage. Read-only, offline, no
playback, no PipeWire mutation.
"""

import argparse
import os
import time
import wave
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = Path(os.environ.get("OPENDUBSTREAM_MODEL_ROOT", Path.home() / ".local/share/opendubstream/models"))


def load_pcm(path: Path):
    with wave.open(str(path), "rb") as f:
        return f.readframes(f.getnframes())


def main() -> int:
    parser = argparse.ArgumentParser(description="Warm-model per-phrase latency benchmark; no capture or playback.")
    parser.add_argument("--inputs", required=True, type=Path, nargs="+", help="mono PCM 16 kHz WAV fixtures")
    parser.add_argument("--model-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--voice", default="ef_dora")
    args = parser.parse_args()

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("ONNX_PROVIDER", "CUDAExecutionProvider")

    import numpy as np
    import onnxruntime as ort
    import torch
    from faster_whisper import WhisperModel
    from kokoro_onnx import Kokoro
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    root = args.model_root
    device = "cuda" if torch.cuda.is_available() else "cpu"

    load_start = time.monotonic()
    vad_session = ort.InferenceSession(str(root / "silero-vad/silero_vad.onnx"), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    asr_model = WhisperModel(str(root / "faster-whisper-distil-large-v3"), device=device, compute_type="float16" if device == "cuda" else "int8", local_files_only=True)
    mt_tokenizer = AutoTokenizer.from_pretrained(str(root / "opus-mt-en-es"), local_files_only=True)
    mt_model = AutoModelForSeq2SeqLM.from_pretrained(str(root / "opus-mt-en-es"), local_files_only=True).to(device)
    tts_engine = Kokoro(str(root / "kokoro-onnx-v1/kokoro-v1.0.onnx"), str(root / "kokoro-onnx-v1/voices-v1.0.bin"))
    tts_engine.create("Hola", voice=args.voice, lang="es")  # pay one-time CUDA kernel warmup outside measurement
    load_seconds = time.monotonic() - load_start
    print(f"LOAD: {load_seconds:.2f}s (paid once at startup, not per phrase)")
    print(f"TTS providers: {tts_engine.sess.get_providers()}")

    def has_voice(pcm: bytes) -> bool:
        samples = np.frombuffer(pcm, dtype="<i2").astype("float32") / 32768.0
        state = np.zeros((2, 1, 128), dtype="float32")
        window = 256
        for start in range(0, len(samples), window):
            chunk = samples[start : start + window]
            if len(chunk) < window:
                chunk = np.pad(chunk, (0, window - len(chunk)))
            score, state = vad_session.run(None, {"input": chunk[None, :], "state": state, "sr": np.array(16000, dtype="int64")})
            if float(score[0][0]) >= 0.5:
                return True
        return False

    rows = []
    for path in args.inputs:
        pcm = load_pcm(path)
        samples = np.frombuffer(pcm, dtype="<i2").astype("float32") / 32768.0
        t0 = time.monotonic()
        if not has_voice(pcm):
            print(f"{path.name}: no voice detected, skipped")
            continue
        t1 = time.monotonic()
        segments, _ = asr_model.transcribe(samples, language="en", task="transcribe", vad_filter=False)
        english = "".join(s.text for s in segments)
        t2 = time.monotonic()
        inputs = mt_tokenizer(english, return_tensors="pt").to(device)
        generated = mt_model.generate(**inputs)
        spanish = mt_tokenizer.decode(generated[0], skip_special_tokens=True)
        t3 = time.monotonic()
        tts_engine.create(spanish, voice=args.voice, lang="es")
        t4 = time.monotonic()
        rows.append((t1 - t0, t2 - t1, t3 - t2, t4 - t3, t4 - t0))
        print(f"{path.name}: vad={t1-t0:.3f}s asr={t2-t1:.3f}s tr={t3-t2:.3f}s tts={t4-t3:.3f}s total={t4-t0:.3f}s")

    if not rows:
        print("No trials had detectable voice.")
        return 2

    data = np.array(rows)
    for idx, name in enumerate(("VAD", "ASR", "TR", "TTS", "TOTAL")):
        col = data[:, idx]
        print(f"{name}: P50={np.percentile(col, 50):.3f}s P95={np.percentile(col, 95):.3f}s min={col.min():.3f}s max={col.max():.3f}s n={len(col)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
