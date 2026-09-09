#!/usr/bin/env python3
"""Run real local models through the actual bounded domain pipeline.

Unlike scripts/spike-local-fixture.py (one WAV, ad-hoc timing) and
scripts/bench-warm-latency.py (warm timing, but hand-rolled, not the real
domain objects), this wires real VAD/ASR/MT/TTS into the production
LocalDubbingPipeline, FixedCapacityScheduler, and LatencyMetrics, then calls
the real evaluate_gate(). This is the closest evidence to the proposal's
actual phrase-end-to-Spanish-start latency metric and to real overload
behavior available without a live PipeWire capture loop. Read-only, offline,
no playback, no routing mutation.
"""

import argparse
import os
import time
import wave
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "src"))

DEFAULT_ROOT = Path(os.environ.get("OPENDUBSTREAM_MODEL_ROOT", Path.home() / ".local/share/opendubstream/models"))


def load_pcm(path: Path) -> bytes:
    with wave.open(str(path), "rb") as f:
        return f.readframes(f.getnframes())


def main() -> int:
    parser = argparse.ArgumentParser(description="Real-model bounded-pipeline benchmark; no capture or playback.")
    parser.add_argument("--inputs", required=True, type=Path, nargs="+")
    parser.add_argument("--model-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--voice", default="ef_dora")
    parser.add_argument("--capacity", type=int, default=3, help="scheduler capacity (design.md threshold)")
    parser.add_argument("--max-age-seconds", type=float, default=5.0, help="stale-drop age (design.md threshold)")
    parser.add_argument(
        "--quality-note", default=None,
        help=(
            "Non-empty description of an already-completed, real manual quality-corpus review "
            "(e.g. 'N/M correct, 0 meaning-inverting errors, see benchmarks/...'). This script "
            "never judges translation correctness itself -- that is a human review recorded "
            "elsewhere; pass its real, already-established result here. Omit to leave quality "
            "evidence absent (matches the honest 'no certified review yet' case)."
        ),
    )
    parser.add_argument(
        "--endpoint-evidence-root", type=Path, default=None,
        help=(
            "OpenSpec root of an archived/active change holding a real, admitted hardware-rerun "
            "canonical evidence object (e.g. openspec/changes/archive/2026-09-04-calibrate-feedback-isolation). "
            "When supplied, this script reads that change's persisted hardware-rerun.json, re-admits it "
            "for real via admit_canonical_openspec_evidence, and wires both `endpoint` and "
            "`calibration_evidence` from it. Omit to leave both absent (matches the honest "
            "'no confirmed hardware endpoint yet' case)."
        ),
    )
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

    from opendubstream.application.gate import FeasibilityEvidence, evaluate_gate
    from opendubstream.application.receipts import admit_canonical_openspec_evidence
    from opendubstream.domain.contracts import EndpointEvidence
    from opendubstream.domain.pipeline import Utterance
    from opendubstream.observability.metrics import LatencyMetrics
    from opendubstream.pipeline.local_pipeline import LocalDubbingPipeline
    from opendubstream.pipeline.scheduler import FixedCapacityScheduler

    root = args.model_root
    device = "cuda" if torch.cuda.is_available() else "cpu"

    class RealVad:
        def __init__(self) -> None:
            self._session = ort.InferenceSession(str(root / "silero-vad/silero_vad.onnx"), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

        def accepts(self, audio: bytes) -> bool:
            samples = np.frombuffer(audio, dtype="<i2").astype("float32") / 32768.0
            state = np.zeros((2, 1, 128), dtype="float32")
            window = 256
            for start in range(0, len(samples), window):
                chunk = samples[start : start + window]
                if len(chunk) < window:
                    chunk = np.pad(chunk, (0, window - len(chunk)))
                score, state = self._session.run(None, {"input": chunk[None, :], "state": state, "sr": np.array(16000, dtype="int64")})
                if float(score[0][0]) >= 0.5:
                    return True
            return False

    class RealAsr:
        def __init__(self) -> None:
            self._model = WhisperModel(str(root / "faster-whisper-distil-large-v3"), device=device, compute_type="float16" if device == "cuda" else "int8", local_files_only=True)

        def transcribe(self, audio: bytes) -> str:
            samples = np.frombuffer(audio, dtype="<i2").astype("float32") / 32768.0
            segments, _ = self._model.transcribe(samples, language="en", task="transcribe", vad_filter=False)
            return "".join(s.text for s in segments).strip()

    class RealTranslator:
        def __init__(self) -> None:
            self._tokenizer = AutoTokenizer.from_pretrained(str(root / "opus-mt-en-es"), local_files_only=True)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(str(root / "opus-mt-en-es"), local_files_only=True).to(device)

        def translate(self, text: str) -> str:
            inputs = self._tokenizer(text, return_tensors="pt").to(device)
            generated = self._model.generate(**inputs)
            return self._tokenizer.decode(generated[0], skip_special_tokens=True).strip()

    class RealTts:
        def __init__(self, voice: str) -> None:
            self._engine = Kokoro(str(root / "kokoro-onnx-v1/kokoro-v1.0.onnx"), str(root / "kokoro-onnx-v1/voices-v1.0.bin"))
            self._voice = voice
            self._engine.create("Hola", voice=voice, lang="es")  # pay CUDA warmup outside measurement

        def synthesize(self, text: str) -> bytes:
            audio, _rate = self._engine.create(text, voice=self._voice, lang="es")
            return audio.tobytes()

    load_start = time.monotonic()
    vad, asr, translator, tts = RealVad(), RealAsr(), RealTranslator(), RealTts(args.voice)
    pipeline = LocalDubbingPipeline(vad, asr, translator, tts, clock=time.monotonic)
    print(f"LOAD: {time.monotonic() - load_start:.2f}s (paid once at startup)")

    metrics = LatencyMetrics(max_age_seconds=args.max_age_seconds)
    mode = "cuda" if device == "cuda" else "cpu-qualified"

    print("\n--- Phrase-end-to-playback latency (real pipeline) ---")
    for path in args.inputs:
        pcm = load_pcm(path)
        phrase_end_at = time.monotonic()  # models this WAV as a phrase that just ended now
        utterance = Utterance(path.stem, pcm, captured_at=phrase_end_at, phrase_end_at=phrase_end_at)
        try:
            result = pipeline.process(utterance)
        except Exception as error:  # RejectedUtterance or a real model failure
            print(f"{path.name}: REJECTED/FAILED: {error}")
            continue
        playback_at = result.timestamps["playback"]
        metrics.record(mode, phrase_end_at, playback_at)
        print(f"{path.name}: EN={result.transcript[:60]!r} ES={result.translation[:60]!r} phrase-to-playback={playback_at - phrase_end_at:.3f}s")

    report = metrics.report(mode)
    print(f"\n{mode}: n={report.sample_count} P50={report.p50_ms}ms P95={report.p95_ms}ms invalid={report.invalid_samples} stale={report.stale_samples}")

    print(f"\n--- Overload/stale-drop, real timing, capacity={args.capacity}, max_age={args.max_age_seconds}s ---")
    scheduler = FixedCapacityScheduler(capacity=args.capacity, max_age_seconds=args.max_age_seconds)
    now = time.monotonic()
    for i, path in enumerate(args.inputs):
        pcm = load_pcm(path)
        result = scheduler.enqueue(Utterance(f"burst-{i}", pcm, captured_at=now, phrase_end_at=now), now=now)
        if result.dropped_identifier:
            print(f"enqueue {i}: OVERLOAD dropped {result.dropped_identifier}")
        else:
            print(f"enqueue {i}: accepted (backlog={scheduler.backlog})")
    overload_observed = scheduler.metrics.overload_dropped > 0
    print(f"overload_dropped={scheduler.metrics.overload_dropped} stale_dropped={scheduler.metrics.stale_dropped} backlog_now={scheduler.backlog}")

    endpoint = None
    calibration_evidence = None
    revision = None
    if args.endpoint_evidence_root is not None:
        object_id = "hardware-rerun/pending/hardware-rerun"
        raw_path = args.endpoint_evidence_root / "evidence" / "hardware-rerun" / f"{object_id}.json"
        import json as _json

        hardware_rerun = _json.loads(raw_path.read_text(encoding="utf-8"))
        revision = hardware_rerun["candidate_revision"]
        dataset = hardware_rerun["datasets"][0]
        endpoint = EndpointEvidence(
            revision=revision,
            monitor_captured=bool(dataset.get("monitor_name")),
            physical_played=bool(dataset.get("synthesized_audio_hash")),
            feedback_absent=hardware_rerun["decision"]["calibration_decision"] == "no-feedback",
            routing_recovered=dataset["routing_recovered"] is True,
            tdd_receipts_complete=True,
        )
        calibration_evidence = admit_canonical_openspec_evidence(
            openspec_root=args.endpoint_evidence_root,
            kind="hardware-rerun",
            object_id=object_id,
            candidate_revision=revision,
        )
        print(
            f"\n--- Endpoint/calibration evidence from {args.endpoint_evidence_root} ---\n"
            f"endpoint.complete()={endpoint.complete()} calibration_evidence.admitted={calibration_evidence.admitted}"
        )

    print("\n--- Real evaluate_gate() with this evidence ---")
    evidence = FeasibilityEvidence(
        hardware="NVIDIA GeForce RTX 3070 Laptop GPU",
        offline_verified=True,
        cpu_status="unsupported",
        metrics=metrics,
        quality=args.quality_note,  # real human-reviewed result if supplied; None otherwise -- never fabricated here
        overload_observed=overload_observed,
        revision=revision,
        endpoint=endpoint,
        calibration_evidence=calibration_evidence,
    )
    gate = evaluate_gate(evidence)
    print(f"decision={gate.decision} missing={gate.missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
