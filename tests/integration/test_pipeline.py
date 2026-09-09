from __future__ import annotations

import hashlib
from pathlib import Path

from opendubstream.application.gate import FeasibilityEvidence, evaluate_gate
from opendubstream.application.receipts import CanonicalOpenSpecEvidenceAdmission
from opendubstream.domain.contracts import EndpointEvidence
from opendubstream.domain.pipeline import Utterance
from opendubstream.infrastructure.models.assets import AssetManifest, AssetPin, LocalAssetStore
from opendubstream.infrastructure.models.translation import OpusMtEnglishSpanishTranslator
from opendubstream.observability.metrics import LatencyMetrics
from opendubstream.pipeline.local_pipeline import LocalDubbingPipeline


class FakeVad:
    def accepts(self, audio: bytes) -> bool:
        return audio == b"phrase"


class FakeAsr:
    def transcribe(self, audio: bytes) -> str:
        assert audio == b"phrase"
        return "hello"


class FakeTts:
    def synthesize(self, text: str) -> bytes:
        assert text == "hola"
        return b"spanish-audio"


def test_network_disabled_verified_assets_run_local_vad_asr_opus_mt_tts_pipeline(tmp_path: Path) -> None:
    model = tmp_path / "source"
    model.mkdir()
    data = b"offline-opus-mt-en-es"
    (model / "opus-mt-en-es.bin").write_bytes(data)
    pin = AssetPin("opus-mt-en-es", "opus-mt-en-es.bin", hashlib.sha256(data).hexdigest())
    assets = LocalAssetStore(tmp_path / "models").stage(AssetManifest((pin,)), model)
    clock_values = iter((1.1, 1.2, 1.3, 1.4))
    translator = OpusMtEnglishSpanishTranslator(assets, infer=lambda text: {"hello": "hola"}[text])
    pipeline = LocalDubbingPipeline(FakeVad(), FakeAsr(), translator, FakeTts(), clock=lambda: next(clock_values))

    result = pipeline.process(Utterance("u1", b"phrase", captured_at=1.0, phrase_end_at=1.0))

    assert result.transcript == "hello"
    assert result.translation == "hola"
    assert result.synthesized_audio == b"spanish-audio"
    assert result.timestamps == {"asr": 1.1, "translation": 1.2, "tts": 1.3, "playback": 1.4}
    assert translator.model_id == "Helsinki-NLP/opus-mt-en-es"


def complete_endpoint(revision: str = "rev-1") -> EndpointEvidence:
    return EndpointEvidence(revision, monitor_captured=True, physical_played=True, feedback_absent=True, routing_recovered=True, tdd_receipts_complete=True)


def admitted_calibration(revision: str = "rev-1") -> CanonicalOpenSpecEvidenceAdmission:
    return CanonicalOpenSpecEvidenceAdmission(
        admitted=True,
        object_id=f"calibration/{revision}/selected-monitor",
        candidate_revision=revision,
        openspec_path="evidence/calibration/calibration/rev-1/selected-monitor.json",
        canonical_payload_sha256="canonical-digest",
        reason=None,
    )


def test_gate_fails_closed_until_offline_cpu_latency_quality_and_overload_evidence_is_complete() -> None:
    metrics = LatencyMetrics(max_age_seconds=3)
    metrics.record("cuda", 1.0, 1.2)
    metrics.record("cpu-qualified", 1.0, 1.8)
    incomplete = FeasibilityEvidence(hardware=None, offline_verified=True, cpu_status="qualified", metrics=metrics, quality=None, overload_observed=True)
    complete = FeasibilityEvidence(
        hardware="test-host", offline_verified=True, cpu_status="qualified", metrics=metrics, quality="acceptable", overload_observed=True,
        revision="rev-1", endpoint=complete_endpoint(), calibration_evidence=admitted_calibration(),
    )

    assert evaluate_gate(incomplete).decision == "no-go"
    assert "hardware" in evaluate_gate(incomplete).missing
    assert "endpoint evidence" in evaluate_gate(incomplete).missing
    assert evaluate_gate(complete).decision == "go"


def test_gate_does_not_require_cpu_latency_when_cpu_is_ruled_out_as_unsupported() -> None:
    """A CUDA-only host with CPU measured and rejected must still be able to reach go."""
    metrics = LatencyMetrics(max_age_seconds=3)
    metrics.record("cuda", 1.0, 1.2)

    evidence = FeasibilityEvidence(
        hardware="test-host", offline_verified=True, cpu_status="unsupported", metrics=metrics, quality="acceptable", overload_observed=True,
        revision="rev-1", endpoint=complete_endpoint(), calibration_evidence=admitted_calibration(),
    )

    result = evaluate_gate(evidence)

    assert "cpu latency" not in result.missing
    assert result.decision == "go"
