"""Strict canonical and cross-store admission tests for calibration evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from opendubstream.application.receipts import (
    CanonicalOpenSpecEvidenceAdmission,
    EvidenceAdmissionError,
    EvidenceLocator,
    EngramRawPayloadUnavailable,
    InMemoryEngramRawJsonAdapter,
    admit_canonical_openspec_evidence,
    admit_hybrid_evidence,
    canonical_raw_json,
    persist_hybrid_evidence,
    require_raw_engram_payload,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PHASE_TWO_OBJECT_ID = "tdd-receipt/85a8d932d6ae9c38b60c807da946dd21ae0ef0e7b82931f5ec926b61b15aab10/02-hybrid-evidence"


def raw_object(**overrides: object) -> str:
    payload: dict[str, object] = {
        "schema": "opendubstream.calibration-evidence/v1",
        "object_id": "calibration/rev-1/selected-monitor",
        "candidate_revision": "rev-1",
        "recorded_at": "2026-09-03T19:00:00Z",
        "baseline_reference": {"sha256": "historical-digest", "decision": "feedback_absent=false"},
        "protocol": {"version": "v1", "comparison_count": 102416},
        "datasets": [{"label": "null-0", "p_value": 0.75}],
        "decision": {"result": "inconclusive", "inputs": ["null-0"]},
    }
    payload.update(overrides)
    return canonical_raw_json(payload)


def persist(tmp_path: Path, raw: str | None = None):  # type: ignore[no-untyped-def]
    adapter = InMemoryEngramRawJsonAdapter()
    locator = persist_hybrid_evidence(
        openspec_root=tmp_path,
        kind="calibration",
        raw_json=raw or raw_object(),
        adapter=adapter,
    )
    return adapter, locator


def test_canonical_raw_json_has_stable_digest_despite_input_key_order() -> None:
    first = raw_object()
    second = canonical_raw_json(json.loads(first))

    assert first == second
    assert json.loads(first)["canonical_payload_sha256"]


def test_canonical_openspec_admission_accepts_only_the_deterministic_raw_object(tmp_path: Path) -> None:
    raw = raw_object()
    object_id = json.loads(raw)["object_id"]
    target = tmp_path / "evidence/calibration" / f"{object_id}.json"
    target.parent.mkdir(parents=True)
    target.write_text(raw, encoding="utf-8")

    admission = admit_canonical_openspec_evidence(
        openspec_root=tmp_path,
        kind="calibration",
        object_id=object_id,
        candidate_revision="rev-1",
    )

    assert isinstance(admission, CanonicalOpenSpecEvidenceAdmission)
    assert admission.admitted is True
    assert admission.object_id == object_id
    assert admission.canonical_payload_sha256 == json.loads(raw)["canonical_payload_sha256"]
    assert admission.openspec_path == f"evidence/calibration/{object_id}.json"


@pytest.mark.parametrize(
    "mutator",
    [
        lambda raw: raw + "\n",
        lambda raw: raw.replace('"candidate_revision":"rev-1"', '"candidate_revision":"rev-stale"'),
        lambda raw: raw.replace('"canonical_payload_sha256":"', '"canonical_payload_sha256":"tampered-'),
    ],
)
def test_canonical_openspec_admission_rejects_drift_stale_or_invalid_self_digest(
    tmp_path: Path, mutator
) -> None:  # type: ignore[no-untyped-def]
    raw = raw_object()
    object_id = json.loads(raw)["object_id"]
    target = tmp_path / "evidence/calibration" / f"{object_id}.json"
    target.parent.mkdir(parents=True)
    target.write_text(mutator(raw), encoding="utf-8")

    admission = admit_canonical_openspec_evidence(
        openspec_root=tmp_path,
        kind="calibration",
        object_id=object_id,
        candidate_revision="rev-1",
    )

    assert admission.admitted is False
    assert admission.reason is not None


def test_canonical_openspec_admission_rejects_missing_evidence_without_an_engram_fallback(tmp_path: Path) -> None:
    admission = admit_canonical_openspec_evidence(
        openspec_root=tmp_path,
        kind="calibration",
        object_id="calibration/rev-1/selected-monitor",
        candidate_revision="rev-1",
    )

    assert admission.admitted is False
    assert admission.reason is not None
    # OSError text is locale-dependent (e.g. Spanish "No existe el fichero o el directorio"
    # instead of "No such file or directory"); the missing object's own path is not.
    assert "calibration/rev-1/selected-monitor" in admission.reason


def test_canonical_openspec_admission_has_no_engram_adapter_input(tmp_path: Path) -> None:
    """Runtime admission cannot read, scrape, or compare Engram content."""
    with pytest.raises(TypeError, match="adapter"):
        admit_canonical_openspec_evidence(  # type: ignore[call-arg]
            openspec_root=tmp_path,
            kind="calibration",
            object_id="calibration/rev-1/selected-monitor",
            candidate_revision="rev-1",
            adapter=InMemoryEngramRawJsonAdapter(),
        )


def test_supported_engram_raw_boundary_accepts_only_the_exact_canonical_payload() -> None:
    raw = raw_object()

    assert require_raw_engram_payload(raw) == raw


def test_mem_get_observation_metadata_envelope_is_not_a_supported_raw_boundary() -> None:
    """The documented MCP read shape must fail closed instead of being scraped."""
    raw = raw_object()
    observed_mem_get_response = json.dumps(
        {
            "project": "opendubstream",
            "project_path": "/workspace/OpenDubStream",
            "project_source": "git_root",
            "result": (
                "#373 [architecture] Persisted selected capture isolation receipt\\n"
                f"{raw}\\n"
                "Session: manual-save-opendubstream\\n"
                "Topic: sdd/calibrate-feedback-isolation/raw/calibration/rev-1/selected-monitor"
            ),
        },
        separators=(",", ":"),
    )

    with pytest.raises(EngramRawPayloadUnavailable, match="raw canonical JSON"):
        require_raw_engram_payload(observed_mem_get_response)


def test_persisted_equal_raw_copies_are_admitted_from_both_stores(tmp_path: Path) -> None:
    adapter, locator = persist(tmp_path)

    admission = admit_hybrid_evidence(openspec_root=tmp_path, locator=locator, adapter=adapter)

    assert admission.admitted is True
    assert admission.object_id == "calibration/rev-1/selected-monitor"
    assert admission.openspec_path == locator.openspec_path
    assert admission.engram_topic == locator.engram_topic


def test_persistence_writes_exact_same_canonical_bytes_to_both_stores(tmp_path: Path) -> None:
    raw = raw_object()
    adapter, locator = persist(tmp_path, raw)

    assert (tmp_path / locator.openspec_path).read_text(encoding="utf-8") == raw
    assert adapter.load(locator.engram_topic, locator.engram_observation_id) == raw


def test_persistence_records_only_the_returned_observation_id_in_the_locator_index(tmp_path: Path) -> None:
    adapter, locator = persist(tmp_path)

    index = json.loads((tmp_path / "evidence/index.json").read_text(encoding="utf-8"))

    assert index["schema"] == "opendubstream.evidence-index/v1"
    assert index["runtime_authority"] == "openspec-canonical-bytes"
    assert index["objects"] == [
        {
            "canonical_payload_sha256": json.loads(raw_object())["canonical_payload_sha256"],
            "engram_locator_role": "non-authoritative-recovery-only",
            "engram_observation_id": locator.engram_observation_id,
            "engram_topic": locator.engram_topic,
            "kind": locator.kind,
            "object_id": locator.object_id,
            "openspec_path": locator.openspec_path,
        }
    ]


@pytest.mark.skipif(
    not (REPOSITORY_ROOT / "openspec/changes/archive/2026-09-04-calibrate-feedback-isolation").is_dir(),
    reason="Private development receipts are not distributed; synthetic admission tests still run",
)
def test_phase_two_checked_in_receipt_is_admitted_from_its_canonical_openspec_path() -> None:
    # calibrate-feedback-isolation was archived 2026-09-04; its evidence now lives
    # permanently under openspec/changes/archive/, not the active changes/ folder.
    root = REPOSITORY_ROOT / "openspec/changes/archive/2026-09-04-calibrate-feedback-isolation"

    admission = admit_canonical_openspec_evidence(
        openspec_root=root,
        kind="tdd-receipt",
        object_id=PHASE_TWO_OBJECT_ID,
        candidate_revision="85a8d932d6ae9c38b60c807da946dd21ae0ef0e7b82931f5ec926b61b15aab10",
    )

    assert admission.admitted is True
    assert admission.canonical_payload_sha256 == "d47b55ff1810099b74c814f698b928e812f47c1d628998ee38de236656b808b7"


@pytest.mark.skipif(
    not (REPOSITORY_ROOT / "openspec/changes/archive/2026-09-04-calibrate-feedback-isolation").is_dir(),
    reason="Private development receipts are not distributed; synthetic admission tests still run",
)
def test_checked_in_evidence_index_digest_matches_every_canonical_openspec_object() -> None:
    # calibrate-feedback-isolation was archived 2026-09-04; its evidence now lives
    # permanently under openspec/changes/archive/, not the active changes/ folder.
    root = REPOSITORY_ROOT / "openspec/changes/archive/2026-09-04-calibrate-feedback-isolation"
    index = json.loads((root / "evidence/index.json").read_text(encoding="utf-8"))

    for entry in index["objects"]:
        raw = (root / entry["openspec_path"]).read_text(encoding="utf-8")
        payload = json.loads(raw)

        assert raw == canonical_raw_json(payload)
        assert entry["canonical_payload_sha256"] == payload["canonical_payload_sha256"]


@pytest.mark.parametrize(
    "mutator",
    [
        lambda raw: raw.replace('"datasets"', '"tampered_datasets"'),
        lambda raw: raw.replace('"candidate_revision":"rev-1"', '"candidate_revision":"rev-2"'),
    ],
)
def test_admission_rejects_tampered_openspec_copy(tmp_path: Path, mutator) -> None:  # type: ignore[no-untyped-def]
    adapter, locator = persist(tmp_path)
    (tmp_path / locator.openspec_path).write_text(mutator(raw_object()), encoding="utf-8")

    admission = admit_hybrid_evidence(openspec_root=tmp_path, locator=locator, adapter=adapter)

    assert admission.admitted is False
    assert admission.reason is not None


def test_admission_rejects_engram_wrapper_instead_of_raw_object(tmp_path: Path) -> None:
    adapter, locator = persist(tmp_path)
    adapter.replace(locator.engram_topic, locator.engram_observation_id, '{"content":{"not":"raw evidence"}}')

    admission = admit_hybrid_evidence(openspec_root=tmp_path, locator=locator, adapter=adapter)

    assert admission.admitted is False
    assert "raw canonical JSON" in admission.reason


def test_admission_rejects_valid_but_different_engram_revision(tmp_path: Path) -> None:
    adapter, locator = persist(tmp_path)
    adapter.replace(
        locator.engram_topic,
        locator.engram_observation_id,
        raw_object(candidate_revision="rev-2"),
    )

    admission = admit_hybrid_evidence(openspec_root=tmp_path, locator=locator, adapter=adapter)

    assert admission.admitted is False
    assert "candidate revision differs" in admission.reason


def test_admission_rejects_unavailable_engram_copy(tmp_path: Path) -> None:
    adapter, locator = persist(tmp_path)
    adapter.remove(locator.engram_topic, locator.engram_observation_id)

    admission = admit_hybrid_evidence(openspec_root=tmp_path, locator=locator, adapter=adapter)

    assert admission.admitted is False
    assert "unavailable" in admission.reason


def test_admission_rejects_unavailable_openspec_copy(tmp_path: Path) -> None:
    adapter, locator = persist(tmp_path)
    (tmp_path / locator.openspec_path).unlink()

    admission = admit_hybrid_evidence(openspec_root=tmp_path, locator=locator, adapter=adapter)

    assert admission.admitted is False
    # OSError text is locale-dependent; the missing file's own path is not.
    assert locator.openspec_path in admission.reason


def test_admission_rejects_openspec_copy_with_a_trailing_newline_even_though_json_is_equal(tmp_path: Path) -> None:
    """Confirmed live: parsing-then-recanonicalizing before comparison silently admitted this drift."""
    adapter, locator = persist(tmp_path)
    target = tmp_path / locator.openspec_path
    target.write_text(target.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    admission = admit_hybrid_evidence(openspec_root=tmp_path, locator=locator, adapter=adapter)

    assert admission.admitted is False
    assert admission.reason is not None


def test_admission_rejects_engram_copy_with_non_canonical_whitespace(tmp_path: Path) -> None:
    """Same drift, mirrored on the Engram side, and with spacing instead of a newline."""
    adapter, locator = persist(tmp_path)
    non_canonical = json.dumps(json.loads(raw_object()), sort_keys=True, separators=(", ", ": "))
    adapter.replace(locator.engram_topic, locator.engram_observation_id, non_canonical)

    admission = admit_hybrid_evidence(openspec_root=tmp_path, locator=locator, adapter=adapter)

    assert admission.admitted is False
    assert admission.reason is not None


def test_admission_rejects_locator_that_does_not_match_deterministic_topic_or_path(tmp_path: Path) -> None:
    adapter, locator = persist(tmp_path)
    hostile = EvidenceLocator(
        kind=locator.kind,
        object_id=locator.object_id,
        openspec_path="evidence/other.json",
        engram_topic=locator.engram_topic,
        engram_observation_id=locator.engram_observation_id,
    )

    admission = admit_hybrid_evidence(openspec_root=tmp_path, locator=hostile, adapter=adapter)

    assert admission.admitted is False
    assert "locator" in admission.reason


def test_canonical_object_rejects_missing_schema_fields() -> None:
    incomplete = {"schema": "opendubstream.calibration-evidence/v1"}

    with pytest.raises(EvidenceAdmissionError, match="required"):
        canonical_raw_json(incomplete)
