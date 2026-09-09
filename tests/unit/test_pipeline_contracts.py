from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from opendubstream.domain.pipeline import (
    CpuFallbackStatus,
    ExecutionMode,
    Utterance,
    choose_execution_mode,
)
from opendubstream.infrastructure.models.assets import (
    AssetIntegrityError,
    AssetManifest,
    AssetPin,
    LocalAssetStore,
)
from opendubstream.observability.metrics import LatencyMetrics
from opendubstream.pipeline.scheduler import FixedCapacityScheduler


def pin(name: str, content: bytes) -> AssetPin:
    return AssetPin(name=name, filename=f"{name}.bin", sha256=hashlib.sha256(content).hexdigest())


def utterance(identifier: str, phrase_end_at: float) -> Utterance:
    return Utterance(identifier=identifier, audio=b"audio", captured_at=phrase_end_at - 0.1, phrase_end_at=phrase_end_at)


def test_local_assets_reject_missing_or_mismatched_pins_without_activating_partial_set(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "asr.bin").write_bytes(b"actual")
    store = LocalAssetStore(tmp_path / "models")

    with pytest.raises(AssetIntegrityError):
        store.stage(AssetManifest((pin("asr", b"expected"), pin("mt", b"missing"))), source)

    assert not (tmp_path / "models").exists()


def test_local_assets_stage_verified_set_atomically_and_validate_without_network(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    asr, mt = b"asr-v1", b"mt-v1"
    (source / "asr.bin").write_bytes(asr)
    (source / "mt.bin").write_bytes(mt)
    manifest = AssetManifest((pin("asr", asr), pin("mt", mt)))
    store = LocalAssetStore(tmp_path / "models")

    assets = store.stage(manifest, source)

    assert assets.require("asr").read_bytes() == asr
    assert store.require_verified(manifest) == assets
    assert store.network_calls == 0


def test_cuda_is_preferred_and_cpu_requires_measured_qualification() -> None:
    assert choose_execution_mode(cuda_available=True, cpu_latency_ms=None, cpu_threshold_ms=250) == (
        ExecutionMode.CUDA,
        CpuFallbackStatus.NOT_USED,
    )
    assert choose_execution_mode(cuda_available=False, cpu_latency_ms=200, cpu_threshold_ms=250) == (
        ExecutionMode.CPU_QUALIFIED,
        CpuFallbackStatus.QUALIFIED,
    )
    assert choose_execution_mode(cuda_available=False, cpu_latency_ms=251, cpu_threshold_ms=250) == (
        ExecutionMode.CPU_DEGRADED,
        CpuFallbackStatus.DEGRADED,
    )
    assert choose_execution_mode(cuda_available=False, cpu_latency_ms=None, cpu_threshold_ms=250) == (
        ExecutionMode.CPU_UNSUPPORTED,
        CpuFallbackStatus.UNSUPPORTED,
    )


def test_fixed_scheduler_drops_oldest_unstarted_work_and_never_exceeds_capacity() -> None:
    scheduler = FixedCapacityScheduler(capacity=2, max_age_seconds=5)

    scheduler.enqueue(utterance("first", 0), now=0)
    scheduler.enqueue(utterance("second", 1), now=1)
    result = scheduler.enqueue(utterance("third", 2), now=2)

    assert result.dropped_identifier == "first"
    assert scheduler.backlog == 2
    assert scheduler.start_next(now=2).identifier == "second"
    assert scheduler.start_next(now=2).identifier == "third"


def test_scheduler_discards_stale_unstarted_work_and_preserves_running_work() -> None:
    scheduler = FixedCapacityScheduler(capacity=2, max_age_seconds=2)
    scheduler.enqueue(utterance("running", 0), now=0)
    assert scheduler.start_next(now=0).identifier == "running"
    scheduler.enqueue(utterance("stale", 0), now=0)
    scheduler.enqueue(utterance("fresh", 4), now=4)

    assert scheduler.start_next(now=4).identifier == "fresh"
    assert scheduler.metrics.stale_dropped == 1
    assert scheduler.metrics.overload_dropped == 0


def test_latency_metrics_report_percentiles_and_identify_invalid_or_stale_samples() -> None:
    metrics = LatencyMetrics(max_age_seconds=2)
    metrics.record("cuda", phrase_end_at=1.0, playback_started_at=1.1)
    metrics.record("cuda", phrase_end_at=2.0, playback_started_at=3.0)
    metrics.record("cpu-qualified", phrase_end_at=3.0, playback_started_at=2.9)
    metrics.record("cpu-qualified", phrase_end_at=3.0, playback_started_at=None)
    metrics.record("cpu-qualified", phrase_end_at=3.0, playback_started_at=6.1)

    cuda = metrics.report("cuda")
    cpu = metrics.report("cpu-qualified")

    assert (cuda.sample_count, cuda.p50_ms, cuda.p95_ms) == (2, 100.0, 1000.0)
    assert cpu.sample_count == 0
    assert (cpu.invalid_samples, cpu.stale_samples) == (2, 1)
