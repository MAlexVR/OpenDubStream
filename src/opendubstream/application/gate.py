"""Fail-closed feasibility gate for the evidence collected by the local spike."""

from __future__ import annotations

from dataclasses import dataclass

from opendubstream.application.receipts import CanonicalOpenSpecEvidenceAdmission
from opendubstream.domain.contracts import EndpointEvidence
from opendubstream.observability.metrics import LatencyMetrics


@dataclass(frozen=True)
class FeasibilityEvidence:
    hardware: str | None
    offline_verified: bool
    cpu_status: str | None
    metrics: LatencyMetrics
    quality: str | None
    overload_observed: bool
    revision: str | None = None
    endpoint: EndpointEvidence | None = None
    calibration_evidence: CanonicalOpenSpecEvidenceAdmission | None = None


@dataclass(frozen=True)
class GateResult:
    decision: str
    missing: tuple[str, ...]


def evaluate_gate(evidence: FeasibilityEvidence) -> GateResult:
    missing: list[str] = []
    if not evidence.hardware:
        missing.append("hardware")
    if not evidence.offline_verified:
        missing.append("offline verification")
    if evidence.cpu_status not in {"qualified", "degraded", "unsupported"}:
        missing.append("cpu qualification")
    if evidence.metrics.report("cuda").sample_count == 0:
        missing.append("cuda latency")
    if evidence.cpu_status == "qualified" and evidence.metrics.report("cpu-qualified").sample_count == 0:
        missing.append("cpu latency")
    if not evidence.quality:
        missing.append("quality")
    if not evidence.overload_observed:
        missing.append("overload")
    if evidence.endpoint is None:
        missing.append("endpoint evidence")
    elif not evidence.endpoint.complete():
        missing.append("incomplete endpoint evidence")
    elif evidence.revision is not None and evidence.endpoint.revision != evidence.revision:
        missing.append("endpoint revision mismatch")
    if not isinstance(evidence.calibration_evidence, CanonicalOpenSpecEvidenceAdmission):
        missing.append("admitted canonical OpenSpec calibration evidence")
    elif not evidence.calibration_evidence.admitted:
        missing.append("admitted canonical OpenSpec calibration evidence")
    elif evidence.revision is not None and evidence.calibration_evidence.candidate_revision != evidence.revision:
        missing.append("calibration revision mismatch")
    return GateResult("go" if not missing else "no-go", tuple(missing))
