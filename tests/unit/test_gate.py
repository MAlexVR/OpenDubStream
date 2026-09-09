"""Endpoint-evidence and cross-store receipt requirements for the feasibility gate."""

from __future__ import annotations

import pytest

from opendubstream.application.gate import FeasibilityEvidence, evaluate_gate
from opendubstream.application.receipts import (
    CanonicalOpenSpecEvidenceAdmission,
    HybridEvidenceAdmission,
    ReceiptMismatchError,
    canonical_receipt_hash,
    receipts_match,
)
from opendubstream.domain.contracts import EndpointEvidence
from opendubstream.observability.metrics import LatencyMetrics


def base_evidence(**overrides: object) -> FeasibilityEvidence:
    metrics = LatencyMetrics(max_age_seconds=3)
    metrics.record("cuda", 1.0, 1.2)
    fields: dict[str, object] = dict(
        hardware="test-host", offline_verified=True, cpu_status="unsupported", metrics=metrics, quality="acceptable", overload_observed=True,
    )
    fields.update(overrides)
    return FeasibilityEvidence(**fields)  # type: ignore[arg-type]


def complete_endpoint(revision: str = "rev-1", **overrides: object) -> EndpointEvidence:
    fields: dict[str, object] = dict(
        revision=revision, monitor_captured=True, physical_played=True, feedback_absent=True, routing_recovered=True, tdd_receipts_complete=True,
    )
    fields.update(overrides)
    return EndpointEvidence(**fields)  # type: ignore[arg-type]


def admitted_calibration(revision: str = "rev-1") -> CanonicalOpenSpecEvidenceAdmission:
    return CanonicalOpenSpecEvidenceAdmission(
        admitted=True,
        object_id=f"calibration/{revision}/selected-monitor",
        candidate_revision=revision,
        openspec_path="evidence/calibration/selected-monitor.json",
        canonical_payload_sha256="canonical-digest",
        reason=None,
    )


def test_gate_rejects_missing_endpoint_evidence() -> None:
    result = evaluate_gate(base_evidence(revision="rev-1", endpoint=None))

    assert result.decision == "no-go"
    assert "endpoint evidence" in result.missing


@pytest.mark.parametrize(
    "field", ["monitor_captured", "physical_played", "feedback_absent", "routing_recovered", "tdd_receipts_complete"]
)
def test_gate_rejects_any_single_incomplete_endpoint_condition(field: str) -> None:
    endpoint = complete_endpoint(**{field: False})

    result = evaluate_gate(base_evidence(revision="rev-1", endpoint=endpoint))

    assert result.decision == "no-go"
    assert "incomplete endpoint evidence" in result.missing


def test_gate_rejects_endpoint_evidence_from_a_different_revision() -> None:
    endpoint = complete_endpoint(revision="rev-stale")

    result = evaluate_gate(base_evidence(revision="rev-current", endpoint=endpoint))

    assert result.decision == "no-go"
    assert "endpoint revision mismatch" in result.missing


def test_gate_reaches_go_with_complete_same_revision_endpoint_evidence() -> None:
    result = evaluate_gate(
        base_evidence(revision="rev-1", endpoint=complete_endpoint("rev-1"), calibration_evidence=admitted_calibration())
    )

    assert result.decision == "go"
    assert result.missing == ()


@pytest.mark.parametrize(
    "calibration",
    [
        None,
        CanonicalOpenSpecEvidenceAdmission(False, None, None, None, None, "OpenSpec bytes differ"),
        HybridEvidenceAdmission(False, None, None, None, None, "legacy copies differ"),
    ],
)
def test_gate_rejects_missing_or_rejected_calibration_evidence(
    calibration: CanonicalOpenSpecEvidenceAdmission | HybridEvidenceAdmission | None,
) -> None:
    result = evaluate_gate(
        base_evidence(revision="rev-1", endpoint=complete_endpoint("rev-1"), calibration_evidence=calibration)
    )

    assert result.decision == "no-go"
    assert "admitted canonical OpenSpec calibration evidence" in result.missing


def test_gate_rejects_calibration_evidence_from_a_different_revision() -> None:
    result = evaluate_gate(
        base_evidence(
            revision="rev-1",
            endpoint=complete_endpoint("rev-1"),
            calibration_evidence=admitted_calibration("rev-stale"),
        )
    )

    assert result.decision == "no-go"
    assert "calibration revision mismatch" in result.missing


def test_gate_rejects_legacy_hybrid_admission_even_when_it_claims_success() -> None:
    legacy = HybridEvidenceAdmission(
        admitted=True,
        object_id="calibration/rev-1/selected-monitor",
        candidate_revision="rev-1",
        openspec_path="evidence/calibration/selected-monitor.json",
        engram_topic="sdd/calibrate-feedback-isolation/raw/calibration/rev-1/selected-monitor",
        reason=None,
    )

    result = evaluate_gate(
        base_evidence(revision="rev-1", endpoint=complete_endpoint("rev-1"), calibration_evidence=legacy)
    )

    assert result.decision == "no-go"
    assert "admitted canonical OpenSpec calibration evidence" in result.missing


def test_canonical_receipt_hash_ignores_key_order_and_whitespace() -> None:
    a = '{"work_unit": "x", "candidate_revision": "abc"}'
    b = '{  "candidate_revision":"abc",   "work_unit":"x"}'

    assert canonical_receipt_hash(a) == canonical_receipt_hash(b)


def test_canonical_receipt_hash_rejects_malformed_json() -> None:
    with pytest.raises(ReceiptMismatchError, match="malformed"):
        canonical_receipt_hash("{not json")


def test_receipts_match_detects_real_content_drift() -> None:
    openspec_receipt = '{"work_unit": "02-physical-playback", "candidate_revision": "abc"}'
    stale_engram_receipt = '{"work_unit": "02-physical-playback", "candidate_revision": "different"}'

    assert receipts_match(openspec_receipt, openspec_receipt) is True
    assert receipts_match(openspec_receipt, stale_engram_receipt) is False
