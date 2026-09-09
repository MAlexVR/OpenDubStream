"""Canonical, fail-closed persistence for calibration evidence.

OpenSpec compact bytes are the sole runtime authority.  Engram is a recovery
summary boundary only; legacy fixture helpers below are deliberately not used
by the runtime gate.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Mapping, Protocol


CHANGE_NAME = "calibrate-feedback-isolation"
RAW_TOPIC_PREFIX = f"sdd/{CHANGE_NAME}/raw"
REQUIRED_RAW_FIELDS = frozenset(
    {
        "schema", "object_id", "candidate_revision", "recorded_at",
        "baseline_reference", "protocol", "datasets", "decision",
        "canonical_payload_sha256",
    }
)


class ReceiptMismatchError(RuntimeError):
    """Two receipt copies do not canonicalize to the same content."""


class EvidenceAdmissionError(ValueError):
    """A purported raw evidence object violates the canonical contract."""


class EngramRawPayloadUnavailable(EvidenceAdmissionError):
    """Engram exposed observation metadata rather than an exact raw payload."""


class EngramRawJsonAdapter(Protocol):
    """Minimal adapter; implementations must return raw JSON only."""

    def save(self, topic: str, raw_json: str) -> str: ...

    def load(self, topic: str, observation_id: str) -> str: ...


@dataclass(frozen=True)
class EvidenceLocator:
    kind: str
    object_id: str
    openspec_path: str
    engram_topic: str
    engram_observation_id: str


@dataclass(frozen=True)
class HybridEvidenceAdmission:
    admitted: bool
    object_id: str | None
    candidate_revision: str | None
    openspec_path: str | None
    engram_topic: str | None
    reason: str | None


@dataclass(frozen=True)
class CanonicalOpenSpecEvidenceAdmission:
    """The only evidence shape accepted by production feasibility decisions."""

    admitted: bool
    object_id: str | None
    candidate_revision: str | None
    openspec_path: str | None
    canonical_payload_sha256: str | None
    reason: str | None


class InMemoryEngramRawJsonAdapter:
    """Test adapter whose values are deliberately retrievable by both keys."""

    def __init__(self) -> None:
        self._records: dict[tuple[str, str], str] = {}
        self._sequence = 0

    def save(self, topic: str, raw_json: str) -> str:
        self._sequence += 1
        observation_id = f"observation-{self._sequence}"
        self._records[(topic, observation_id)] = raw_json
        return observation_id

    def load(self, topic: str, observation_id: str) -> str:
        try:
            return self._records[(topic, observation_id)]
        except KeyError as error:
            raise LookupError("Engram raw JSON is unavailable") from error

    def replace(self, topic: str, observation_id: str, raw_json: str) -> None:
        self._records[(topic, observation_id)] = raw_json

    def remove(self, topic: str, observation_id: str) -> None:
        self._records.pop((topic, observation_id), None)


def canonical_receipt_hash(receipt_json: str) -> str:
    """Sort keys and use compact separators so formatting differences don't matter."""
    try:
        parsed = json.loads(receipt_json)
    except json.JSONDecodeError as error:
        raise ReceiptMismatchError(f"malformed receipt JSON: {error}") from error
    canonical = json.dumps(parsed, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def receipts_match(openspec_receipt_json: str, engram_receipt_json: str) -> bool:
    return canonical_receipt_hash(openspec_receipt_json) == canonical_receipt_hash(engram_receipt_json)


def canonical_raw_json(payload: Mapping[str, object]) -> str:
    """Create the only persisted representation of a new evidence object."""
    value = dict(payload)
    value.pop("canonical_payload_sha256", None)
    _validate_raw_fields(value, digest_required=False)
    value["canonical_payload_sha256"] = _digest_without_self(value)
    return _compact_json(value)


def require_raw_engram_payload(raw_json: str) -> str:
    """Accept only a direct, canonical raw JSON payload from an Engram adapter.

    `mem_get_observation` currently returns a metadata envelope containing a
    rendered observation, not this payload.  Scraping its title/result/session
    framing would make byte identity depend on presentation formatting, so that
    response shape is deliberately not an adapter boundary.
    """
    if not isinstance(raw_json, str):
        raise EngramRawPayloadUnavailable("Engram did not return raw canonical JSON")
    try:
        payload = _parse_raw_object(raw_json)
    except EvidenceAdmissionError as error:
        raise EngramRawPayloadUnavailable("Engram did not return raw canonical JSON") from error
    if raw_json != _compact_json(payload):
        raise EngramRawPayloadUnavailable("Engram did not return raw canonical JSON")
    return raw_json


def persist_hybrid_evidence(
    *, openspec_root: Path, kind: str, raw_json: str, adapter: EngramRawJsonAdapter
) -> EvidenceLocator:
    """Persist identical canonical bytes to OpenSpec first and then Engram."""
    payload = _parse_raw_object(raw_json)
    object_id = _required_text(payload, "object_id")
    path = _deterministic_path(kind, object_id)
    topic = _deterministic_topic(object_id)
    canonical = _compact_json(payload)
    target = openspec_root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(canonical, encoding="utf-8")
    observation_id = adapter.save(topic, canonical)
    if not observation_id:
        raise EvidenceAdmissionError("Engram returned an empty observation ID")
    locator = EvidenceLocator(kind, object_id, path.as_posix(), topic, observation_id)
    _write_locator_index(openspec_root, locator, _required_text(payload, "canonical_payload_sha256"))
    return locator


def admit_canonical_openspec_evidence(
    *,
    openspec_root: Path,
    kind: str,
    object_id: str,
    candidate_revision: str,
) -> CanonicalOpenSpecEvidenceAdmission:
    """Admit exactly one canonical OpenSpec object, without an Engram read.

    The deterministic location, compact UTF-8 bytes, declared identity and
    revision, and self-digest are all independently checked.  A missing or
    hostile object produces an inadmissible result rather than an exception so
    callers cannot accidentally turn evidence I/O failures into a go decision.
    """
    try:
        path = _deterministic_path(kind, object_id)
        raw_bytes = (openspec_root / path).read_bytes()
        raw_json = raw_bytes.decode("utf-8")
        payload = _parse_raw_object(raw_json)
        if raw_json != _compact_json(payload):
            raise EvidenceAdmissionError("OpenSpec raw evidence is not canonical compact JSON")
        if _required_text(payload, "object_id") != object_id:
            raise EvidenceAdmissionError("OpenSpec object ID does not match requested object")
        if _required_text(payload, "candidate_revision") != candidate_revision:
            raise EvidenceAdmissionError("OpenSpec candidate revision does not match requested revision")
        digest = _required_text(payload, "canonical_payload_sha256")
        return CanonicalOpenSpecEvidenceAdmission(
            True, object_id, candidate_revision, path.as_posix(), digest, None
        )
    except (EvidenceAdmissionError, OSError, UnicodeError) as error:
        return CanonicalOpenSpecEvidenceAdmission(False, None, None, None, None, str(error))


def admit_hybrid_evidence(
    *, openspec_root: Path, locator: EvidenceLocator, adapter: EngramRawJsonAdapter
) -> HybridEvidenceAdmission:
    """Read each copy independently and reject every unavailable or drift case."""
    try:
        expected_path = _deterministic_path(locator.kind, locator.object_id).as_posix()
        expected_topic = _deterministic_topic(locator.object_id)
        if locator.openspec_path != expected_path or locator.engram_topic != expected_topic:
            raise EvidenceAdmissionError("locator does not match deterministic object locations")

        openspec_raw = (openspec_root / locator.openspec_path).read_text(encoding="utf-8")
        openspec = _parse_raw_object(openspec_raw)
        if openspec_raw != _compact_json(openspec):
            raise EvidenceAdmissionError("OpenSpec raw copy is not canonical compact JSON")
        try:
            engram_raw = adapter.load(locator.engram_topic, locator.engram_observation_id)
        except Exception as error:
            raise EvidenceAdmissionError("Engram raw JSON is unavailable") from error
        engram_raw = require_raw_engram_payload(engram_raw)
        engram = _parse_raw_object(engram_raw)

        if _required_text(openspec, "object_id") != locator.object_id or _required_text(engram, "object_id") != locator.object_id:
            raise EvidenceAdmissionError("object ID does not match locator")
        if _required_text(openspec, "candidate_revision") != _required_text(engram, "candidate_revision"):
            raise EvidenceAdmissionError("candidate revision differs between stores")
        if _required_text(openspec, "canonical_payload_sha256") != _required_text(engram, "canonical_payload_sha256"):
            raise EvidenceAdmissionError("canonical digest differs between stores")
        # Compare the exact retrieved bytes, not a re-parsed/recanonicalized form: the latter
        # silently admitted whitespace/newline drift (confirmed live, 2026-09-03) because two
        # byte-different raw copies can parse to identical Python objects.
        if openspec_raw != engram_raw:
            raise EvidenceAdmissionError("raw bytes differ between stores")

        return HybridEvidenceAdmission(
            True, locator.object_id, _required_text(openspec, "candidate_revision"),
            locator.openspec_path, locator.engram_topic, None,
        )
    except (EvidenceAdmissionError, OSError, UnicodeError) as error:
        return HybridEvidenceAdmission(False, None, None, None, None, str(error))


def _parse_raw_object(raw_json: str) -> dict[str, object]:
    try:
        payload = json.loads(raw_json)
    except (TypeError, json.JSONDecodeError) as error:
        raise EvidenceAdmissionError("raw evidence JSON is malformed") from error
    if not isinstance(payload, dict):
        raise EvidenceAdmissionError("raw evidence must be exactly one JSON object")
    _validate_raw_fields(payload, digest_required=True)
    declared = _required_text(payload, "canonical_payload_sha256")
    without_digest = dict(payload)
    without_digest.pop("canonical_payload_sha256")
    if declared != _digest_without_self(without_digest):
        raise EvidenceAdmissionError("canonical payload digest is invalid")
    return payload


def _validate_raw_fields(payload: Mapping[str, object], *, digest_required: bool) -> None:
    required = REQUIRED_RAW_FIELDS if digest_required else REQUIRED_RAW_FIELDS - {"canonical_payload_sha256"}
    missing = sorted(required - payload.keys())
    if missing:
        raise EvidenceAdmissionError(f"required evidence fields are missing: {', '.join(missing)}")
    for field in ("schema", "object_id", "candidate_revision", "recorded_at"):
        _required_text(payload, field)
    if not isinstance(payload["baseline_reference"], dict):
        raise EvidenceAdmissionError("baseline_reference must be an object")
    if not isinstance(payload["protocol"], dict):
        raise EvidenceAdmissionError("protocol must be an object")
    if not isinstance(payload["datasets"], list):
        raise EvidenceAdmissionError("datasets must be an array")
    if not isinstance(payload["decision"], dict):
        raise EvidenceAdmissionError("decision must be an object")
    if digest_required:
        _required_text(payload, "canonical_payload_sha256")


def _required_text(payload: Mapping[str, object], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value:
        raise EvidenceAdmissionError(f"required evidence field {name} must be non-empty text")
    return value


def _digest_without_self(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(_compact_json(payload).encode("utf-8")).hexdigest()


def _compact_json(payload: Mapping[str, object]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _deterministic_path(kind: str, object_id: str) -> PurePosixPath:
    if not kind or "/" in kind or not object_id or object_id.startswith("/") or ".." in PurePosixPath(object_id).parts:
        raise EvidenceAdmissionError("unsafe deterministic evidence locator")
    return PurePosixPath("evidence") / kind / f"{object_id}.json"


def _deterministic_topic(object_id: str) -> str:
    if object_id.startswith("/") or ".." in PurePosixPath(object_id).parts:
        raise EvidenceAdmissionError("unsafe deterministic evidence topic")
    return f"{RAW_TOPIC_PREFIX}/{object_id}"


def _write_locator_index(openspec_root: Path, locator: EvidenceLocator, digest: str) -> None:
    """Keep locators separate from raw evidence and bind the returned ID."""
    target = openspec_root / "evidence/index.json"
    if target.exists():
        try:
            index = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise EvidenceAdmissionError("evidence locator index is unreadable") from error
        if not isinstance(index, dict) or index.get("schema") != "opendubstream.evidence-index/v1":
            raise EvidenceAdmissionError("evidence locator index schema is invalid")
        entries = index.get("objects")
        if not isinstance(entries, list):
            raise EvidenceAdmissionError("evidence locator index objects must be an array")
        index.setdefault("runtime_authority", "openspec-canonical-bytes")
    else:
        index = {
            "schema": "opendubstream.evidence-index/v1",
            "runtime_authority": "openspec-canonical-bytes",
            "objects": [],
        }
        entries = index["objects"]
    entry = {
        "canonical_payload_sha256": digest,
        "engram_locator_role": "non-authoritative-recovery-only",
        "engram_observation_id": locator.engram_observation_id,
        "engram_topic": locator.engram_topic,
        "kind": locator.kind,
        "object_id": locator.object_id,
        "openspec_path": locator.openspec_path,
    }
    entries[:] = [item for item in entries if not isinstance(item, dict) or item.get("object_id") != locator.object_id]
    entries.append(entry)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(_compact_json(index), encoding="utf-8")
