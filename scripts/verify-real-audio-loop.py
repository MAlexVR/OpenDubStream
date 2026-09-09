#!/usr/bin/env python3
"""Confirmed hardware E2E harness: real routing -> capture -> pipeline -> playback -> recovery.

Importing or unit-testing this module never touches PipeWire, a model, or a device: every
live action is gated behind `main()` actually being invoked with `--confirm`, exactly like
`scripts/provision-local-models.py`'s `--apply`. No global default is ever changed; evidence
records default sink/source before and after to prove it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from opendubstream.application import receipts  # noqa: E402
from opendubstream.application.chrome_route import select_active_chrome_stream  # noqa: E402
from opendubstream.application.session import run_protected  # noqa: E402
from opendubstream.domain.contracts import EndpointEvidence, ReferenceLease, RouteLease, SinkRef, StreamSelector  # noqa: E402
from opendubstream.infrastructure.audio.capture import PipeWireMonitorCapture  # noqa: E402
from opendubstream.infrastructure.audio.discovery import (  # noqa: E402
    OwnedMonitorDiscovery,
    OwnedReferenceMonitorDiscovery,
    PipeWireStreamDiscovery,
)
from opendubstream.infrastructure.audio.journal import RecoveryJournal  # noqa: E402
from opendubstream.infrastructure.audio.playback import (  # noqa: E402
    CalibrationDecision,
    PcmPlaybackError,
    PcmPlayer,
    ProtocolConfig,
    PwPlayPcmPlayer,
    best_tag_correlation_score,
    crop_tagged_monitor_window,
    deterministic_tag,
    emit_tagged_probe_pcm,
    evaluate_calibration,
    locate_best_tag_position,
    resolve_physical_target,
    score_pcm_dataset,
)
from opendubstream.infrastructure.audio.process import SafePactlRunner  # noqa: E402
from opendubstream.infrastructure.audio.router import PipeWirePulseRouter  # noqa: E402

DEFAULT_ROOT = Path(os.environ.get("OPENDUBSTREAM_MODEL_ROOT", Path.home() / ".local/share/opendubstream/models"))
# Flat, human-readable "latest run" dump only -- same shape as `build_hardware_e2e_evidence`
# produces today. This is NOT the canonical admitted evidence object: that is the
# `opendubstream.hardware-rerun/v1` object built by `build_hardware_rerun_raw_payload` and
# persisted/admitted by `persist_and_admit_hardware_rerun_evidence` at its own deterministic
# `evidence/hardware-rerun/...` path via `receipts.admit_canonical_openspec_evidence`. This
# path previously (incorrectly) pointed at the historical, immutable `complete-real-audio-loop`
# change's evidence folder; it now targets the active `calibrate-feedback-isolation` change.
EVIDENCE_PATH = REPO_ROOT / "openspec/changes/calibrate-feedback-isolation/evidence/hardware-rerun.json"
_CALIBRATE_FEEDBACK_ISOLATION_OPENSPEC_ROOT = REPO_ROOT / "openspec/changes/calibrate-feedback-isolation"


class ConfirmationRequired(RuntimeError):
    """The reference-hardware run was attempted without explicit user confirmation."""


class BoundedOutputError(RuntimeError):
    """A recorded output exceeded its declared size bound."""


class EvidenceSchemaError(RuntimeError):
    """Hardware E2E evidence is incomplete or internally inconsistent."""


class TransportOffsetRequired(RuntimeError):
    """The predeclared transport-offset sample count was not supplied before a rerun."""


def require_confirmation(confirmed: bool, *, reason: str) -> None:
    if not confirmed:
        raise ConfirmationRequired(f"user confirmation required: {reason}")


def bounded_output_hash(data: bytes, *, max_bytes: int) -> str:
    if len(data) > max_bytes:
        raise BoundedOutputError(f"output exceeds bound: {len(data)} > {max_bytes} bytes")
    return hashlib.sha256(data).hexdigest()


REQUIRED_EVIDENCE_FIELDS = (
    "revision",
    "default_sink_before", "default_sink_after",
    "default_source_before", "default_source_after",
    "stream_identifier", "virtual_sink", "virtual_module_id",
    "monitor_name", "monitor_serial", "physical_sink_name", "physical_sink_serial",
    "asr_seconds", "translation_seconds", "tts_seconds", "total_seconds",
    "calibration_decision", "null_p_values", "positive_p_value", "selected_p_value", "baseline_digest",
    "selected_capture_plan", "selected_topology_audit",
    "routing_recovered",
    "transcript_hash", "translation_hash", "synthesized_audio_hash",
)


def require_unchanged_default_devices(*, sink_before: object, sink_after: object, source_before: object, source_after: object) -> None:
    """Reject any run that ends with a different default sink or source than it started with."""
    if sink_before != sink_after:
        raise EvidenceSchemaError("default sink changed during the run")
    if source_before != source_after:
        raise EvidenceSchemaError("default source changed during the run")


def build_hardware_e2e_evidence(**fields: object) -> dict[str, object]:
    missing = [name for name in REQUIRED_EVIDENCE_FIELDS if name not in fields]
    if missing:
        raise EvidenceSchemaError(f"missing evidence fields: {', '.join(missing)}")
    require_unchanged_default_devices(
        sink_before=fields["default_sink_before"], sink_after=fields["default_sink_after"],
        source_before=fields["default_source_before"], source_after=fields["default_source_after"],
    )
    # `feedback_absent` is derived from the calibrated max-T decision, never supplied
    # by the caller: only an explicit "no-feedback" verdict counts, so any inconclusive
    # or feedback-present result fails closed to False.
    feedback_absent = fields["calibration_decision"] == "no-feedback"
    return {"schema": "opendubstream.hardware-e2e/v1", "feedback_absent": feedback_absent, **fields}


_SETTLE_SECONDS = 6.0
_IMMUTABLE_FAILED_BASELINE_DIGEST = "db82ba31a54ab1a70dc0de1e1a3256c2c9993217ea05eeba4c6a35ae1bd6bc72"


def _iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_hardware_rerun_raw_payload(evidence: dict[str, object], *, now: Callable[[], str] = _iso_now) -> dict[str, object]:
    """Map the already-built, already-validated `build_hardware_e2e_evidence` output into
    `receipts.REQUIRED_RAW_FIELDS` canonical shape. Nests the entire flat evidence dict as the
    sole dataset entry rather than re-deriving/duplicating any of its fields. Never sets
    `canonical_payload_sha256` -- `receipts.canonical_raw_json` computes and injects that."""
    candidate_revision = evidence["revision"]
    protocol_config = ProtocolConfig()
    return {
        "schema": "opendubstream.hardware-rerun/v1",
        "object_id": f"hardware-rerun/{candidate_revision}/hardware-rerun",
        "candidate_revision": candidate_revision,
        "recorded_at": now(),
        "baseline_reference": {"sha256": evidence["baseline_digest"]},
        "protocol": {
            # Not an importable module constant as of the 2026-09-04 guard-band fix (task
            # 1.4): the fix bumped this value in the persisted receipt/design.md wording only.
            # Recorded here literally, matching that same value.
            "permutation_version": "circular-shift-guarded-v1",
            "alpha": protocol_config.alpha,
            "code_count": protocol_config.code_count,
            "tag_samples": protocol_config.tag_samples,
            "analysis_samples": protocol_config.analysis_samples,
            "permutation_count": protocol_config.permutation_count,
            "max_lag": protocol_config.max_lag,
        },
        "datasets": [{"kind": "hardware-rerun-observation", **evidence}],
        "decision": {
            "calibration_decision": evidence["calibration_decision"],
            "feedback_absent": evidence["feedback_absent"],
        },
    }


def persist_and_admit_hardware_rerun_evidence(
    evidence: dict[str, object],
    *,
    openspec_root: Path = _CALIBRATE_FEEDBACK_ISOLATION_OPENSPEC_ROOT,
) -> receipts.CanonicalOpenSpecEvidenceAdmission:
    """Build the canonical `hardware-rerun` raw object, write it as compact canonical JSON at
    its deterministic OpenSpec path, and admit it via OpenSpec bytes plus their internal
    SHA-256 -- exactly `receipts`' own canonical-OpenSpec-only admission machinery, with no
    Engram adapter of any kind in this real-hardware code path (the script process has no
    MCP/Engram tool access; only the orchestrating session may mirror a summary later)."""
    payload = build_hardware_rerun_raw_payload(evidence)
    raw_json = receipts.canonical_raw_json(payload)
    object_id = payload["object_id"]
    candidate_revision = payload["candidate_revision"]
    # Reuse `receipts`' own deterministic path convention so this write and
    # `admit_canonical_openspec_evidence`'s independently-computed path agree byte-for-byte.
    target = openspec_root / receipts._deterministic_path("hardware-rerun", object_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(raw_json, encoding="utf-8")
    return receipts.admit_canonical_openspec_evidence(
        openspec_root=openspec_root,
        kind="hardware-rerun",
        object_id=object_id,
        candidate_revision=candidate_revision,
    )


def require_transport_offset_samples(value: object) -> int:
    """The transport offset MUST be predeclared before capture; it is never fitted after
    the fact from the collected data."""
    if not isinstance(value, int):
        raise TransportOffsetRequired("--transport-offset-samples must be a predeclared integer")
    return value


class SampleCountingReader:
    """Wraps a raw phrase reader; tracks the cumulative sample count and the full
    continuous byte stream, so a later crop can be measured from one absolute
    byte-clock anchor shared with `reader_sample_count_at_pw_play`."""

    def __init__(self, read_phrase: Callable[[float], bytes]) -> None:
        self._read_phrase = read_phrase
        self.sample_count = 0
        self._chunks: list[bytes] = []

    def __call__(self, deadline: float) -> bytes:
        chunk = self._read_phrase(deadline)
        self._chunks.append(chunk)
        self.sample_count += len(chunk) // 2
        return chunk

    def captured_bytes(self) -> bytes:
        return b"".join(self._chunks)


def advance_reader_to(reader: SampleCountingReader, target_sample_count: int, *, deadline: float = 5.0, max_reads: int = 60) -> bool:
    """Keep reading the continuous stream until it holds at least `target_sample_count`
    samples. Fails closed (returns False) if the reader stalls or an empty chunk arrives."""
    for _ in range(max_reads):
        if reader.sample_count >= target_sample_count:
            return True
        if not reader(deadline):
            return False
    return reader.sample_count >= target_sample_count


def settle_and_collect_null_controls(
    reader: SampleCountingReader,
    config: ProtocolConfig,
    *,
    settle_seconds: float = _SETTLE_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
    deadline: float = 5.0,
) -> tuple[bytes, bytes, bytes] | None:
    """Wait the predeclared reader-settle period, then read three independent
    analysis-length null datasets from the same continuous stream, with no probe
    playback. Fails closed to None on any short or stalled read."""
    sleep(settle_seconds)
    target = 3 * config.analysis_samples
    if not advance_reader_to(reader, target, deadline=deadline):
        return None
    captured = reader.captured_bytes()
    step = config.analysis_samples * 2
    return (captured[0:step], captured[step : 2 * step], captured[2 * step : 3 * step])


def build_positive_control(null_zero: bytes, tag: bytes, config: ProtocolConfig) -> bytes:
    """A byte-identical copy of null[0] with the active code overwritten at samples
    `[tag_insertion_samples, tag_insertion_samples + tag_samples)` -- never a fresh
    hardware capture, so the positive control adds no extra playback confound."""
    if len(null_zero) != config.analysis_samples * 2 or len(tag) != config.tag_samples * 2:
        raise EvidenceSchemaError("positive control requires one full null dataset and one complete tag")
    start = config.tag_insertion_samples * 2
    end = start + config.tag_samples * 2
    if end > len(null_zero):
        raise EvidenceSchemaError("tag insertion window exceeds the null dataset")
    return null_zero[:start] + tag + null_zero[end:]


def play_predeclared_probe(
    player: PcmPlayer,
    inventory: Callable[[], str],
    probe_pcm: bytes,
    target: SinkRef,
    lease: RouteLease,
    *,
    deadline: float,
) -> SinkRef:
    """Resolve the physical target fresh and play exactly the predeclared probe PCM;
    nothing is appended after the fact (unlike `PhysicalPcmPlayback`, which always
    appends the legacy correlation tag used by `complete-real-audio-loop`)."""
    resolved = resolve_physical_target(inventory(), target, lease)
    print(f"probe playback target: name={resolved.name} serial={resolved.serial}", file=sys.stderr)
    try:
        player.play(resolved, probe_pcm, deadline)
    except TimeoutError as error:
        player.stop()
        raise PcmPlaybackError("PCM player timed out") from error
    except Exception:
        player.stop()
        raise
    return resolved


class ReferenceTargetError(RuntimeError):
    """A reference-sink playback target could not be resolved to the owned lease."""


def resolve_reference_target(sinks_json: str, desired: SinkRef, reference: ReferenceLease) -> SinkRef:
    """Re-resolve the owned reference null-sink fresh before every reference-measurement
    playback. Mirrors `resolve_physical_target`'s fresh-inventory discipline, but validates
    identity against a tool-owned `ReferenceLease` instead of real playback hardware -- a
    null-sink has no `device.bus` property, so `resolve_physical_target`'s physical-hardware
    check does not (and must not) apply here. `resolve_physical_target` itself is untouched:
    this is an additive, structurally parallel function for a different playback target
    class."""
    if desired.is_physical:
        raise ReferenceTargetError("reference-sink playback requires a non-physical owned null-sink target")
    if desired.name != reference.reference_sink:
        raise ReferenceTargetError("target is not the owned reference sink for this lease")
    try:
        entries = json.loads(sinks_json)
    except json.JSONDecodeError as error:
        raise ReferenceTargetError(f"malformed sink inventory: {error}") from error
    if not isinstance(entries, list) or not all(isinstance(entry, dict) for entry in entries):
        raise ReferenceTargetError("malformed sink inventory")
    matches = [entry for entry in entries if entry.get("name") == desired.name]
    if len(matches) != 1:
        raise ReferenceTargetError(f"reference sink identity matched {len(matches)} entries; exactly one is required")
    return desired


def play_predeclared_probe_to_reference(
    player: PcmPlayer,
    inventory: Callable[[], str],
    probe_pcm: bytes,
    target: SinkRef,
    reference: ReferenceLease,
    *,
    deadline: float,
) -> SinkRef:
    """Mirrors `play_predeclared_probe` exactly, except it resolves and validates a
    tool-owned reference null-sink (via `resolve_reference_target`) instead of a physical
    playback sink. Used only by the transport-offset measurement's reference-sink routing
    path; `play_predeclared_probe` itself, and the Chrome-stream capture path, are
    unmodified."""
    resolved = resolve_reference_target(inventory(), target, reference)
    print(f"probe playback target: name={resolved.name} serial={resolved.serial}", file=sys.stderr)
    try:
        player.play(resolved, probe_pcm, deadline)
    except TimeoutError as error:
        player.stop()
        raise PcmPlaybackError("PCM player timed out") from error
    except Exception:
        player.stop()
        raise
    return resolved


class SelectedCapturePlan:
    """Immutable fixed-clock contract for the real selected-monitor observation."""

    __slots__ = (
        "schema", "sample_rate_hz", "sample_width_bytes", "pre_roll_samples", "analysis_samples",
        "post_roll_samples", "owned_monitor_name", "owned_monitor_serial", "physical_target_name",
        "physical_target_serial", "virtual_sink_name", "virtual_sink_serial", "virtual_sink_module_id",
        "candidate_revision", "_frozen",
    )

    def __init__(self, **values: object) -> None:
        for name in self.__slots__[:-1]:
            object.__setattr__(self, name, values[name])
        object.__setattr__(self, "_frozen", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_frozen", False):
            raise AttributeError("SelectedCapturePlan is immutable")
        object.__setattr__(self, name, value)

    @classmethod
    def create(cls, **identities: str) -> "SelectedCapturePlan":
        return cls(
            schema="opendubstream.selected-capture-plan/v1", sample_rate_hz=16_000,
            sample_width_bytes=2, pre_roll_samples=32_000, analysis_samples=64_000,
            post_roll_samples=32_000, **identities,
        )

    @property
    def total_samples(self) -> int:
        return self.pre_roll_samples + self.analysis_samples + self.post_roll_samples

    def completion_target(self, reader_count_before_player: int) -> int:
        return reader_count_before_player + self.analysis_samples + self.post_roll_samples

    def extract_selected_window(
        self, *, captured_pcm: bytes, reader_count_before_player: int,
        reader_count_after_completion: int, capture_started_monotonic_ns: int,
        player_invoked_monotonic_ns: int, capture_completed_monotonic_ns: int,
    ) -> bytes | None:
        """Return exactly [before,before+64000), or fail closed on coverage drift."""
        if (
            reader_count_before_player < self.pre_roll_samples
            or reader_count_after_completion < self.completion_target(reader_count_before_player)
            or len(captured_pcm) % self.sample_width_bytes
            or not (capture_started_monotonic_ns <= player_invoked_monotonic_ns <= capture_completed_monotonic_ns)
        ):
            return None
        start = reader_count_before_player * self.sample_width_bytes
        end = (reader_count_before_player + self.analysis_samples) * self.sample_width_bytes
        if len(captured_pcm) < (reader_count_after_completion * self.sample_width_bytes) or len(captured_pcm) < end:
            return None
        return captured_pcm[start:end]


class SelectedTopologyAudit:
    """Pure, fail-closed audit of pre/during injected PipeWire graph snapshots."""

    def __init__(self, schema: str, pre_snapshot_sha256: str, during_snapshot_sha256: str,
                 valid: bool, no_physical_to_selected_path: bool, reason: str = "") -> None:
        self.schema = schema
        self.pre_snapshot_sha256 = pre_snapshot_sha256
        self.during_snapshot_sha256 = during_snapshot_sha256
        self.valid = valid
        self.no_physical_to_selected_path = no_physical_to_selected_path
        self.reason = reason

    @classmethod
    def from_snapshots(
        cls, plan: SelectedCapturePlan, pre_snapshot: object, during_snapshot: object,
    ) -> "SelectedTopologyAudit":
        pre_digest = _graph_digest(pre_snapshot)
        during_digest = _graph_digest(during_snapshot)
        try:
            pre = _parse_graph(pre_snapshot)
            during = _parse_graph(during_snapshot)
            pre_ids = _validate_graph_identities(pre, plan, require_player=False)
            during_ids = _validate_graph_identities(during, plan)
            if any(pre_ids[name] != during_ids[name] for name in ("physical", "virtual", "monitor")):
                raise ValueError("graph identities changed between snapshots")
            if any(node.get("role") == "physical-player" for node in pre["nodes"]):
                pre_player = _validate_graph_identities(pre, plan, require_player=True)
                if (
                    pre_player["player"] != during_ids["player"]
                    or pre_player["player_serial"] != during_ids["player_serial"]
                ):
                    raise ValueError("physical player identity changed between snapshots")
            has_path = any(
                _has_directed_path(during["links"], origin, target)
                for origin in (during_ids["player"], during_ids["physical"])
                for target in (during_ids["virtual"], during_ids["monitor"])
            ) or any(
                _has_directed_path(pre["links"], origin, target)
                for origin in (pre_ids["physical"],)
                for target in (pre_ids["virtual"], pre_ids["monitor"])
            )
            if has_path:
                raise ValueError("physical playback graph reaches selected capture")
        except (TypeError, ValueError, KeyError):
            return cls("opendubstream.selected-topology-audit/v1", pre_digest, during_digest, False, False, "inconclusive")
        return cls("opendubstream.selected-topology-audit/v1", pre_digest, during_digest, True, True)


class SelectedTopologyAuditRecorder:
    """Collect exactly one graph snapshot before and one during physical pw-play.

    ``on_started`` (wired to ``capture_during_playback``) fires immediately after
    ``subprocess.Popen()`` returns, with zero wait for the spawned ``pw-play`` process
    to actually connect to PipeWire and register its stream node in the graph -- the
    same class of registration race already fixed once for ``parec`` (see
    `4.1c-fix3`'s warm-up-read fix in this file). A single immediate snapshot can
    therefore miss the physical-player node even though playback genuinely started.
    To stay tolerant of that race without weakening what counts as a *valid* audit,
    the during-snapshot capture retries a small, bounded number of times with a short
    sleep between attempts, re-taking the snapshot each time, until a snapshot with a
    ``physical-player`` role is found or the retry budget is exhausted -- in which case
    it falls back to the last snapshot taken. `SelectedTopologyAudit.from_snapshots`'s
    existing fail-closed logic already reports `inconclusive` for a snapshot that never
    finds the player node, so no behavior change is needed there.
    """

    def __init__(
        self,
        plan: SelectedCapturePlan,
        snapshot: Callable[[], object],
        *,
        sleep: Callable[[float], None] = time.sleep,
        retry_attempts: int = 6,
        retry_interval_seconds: float = 0.05,
    ) -> None:
        self._plan = plan
        self._snapshot = snapshot
        self._sleep = sleep
        self._retry_attempts = max(1, retry_attempts)
        self._retry_interval_seconds = retry_interval_seconds
        self._before: object | None = None
        self._during: object | None = None

    def capture_before_playback(self) -> None:
        if self._before is not None:
            raise ValueError("before-playback topology snapshot already captured")
        self._before = self._snapshot()

    def capture_during_playback(self) -> None:
        if self._during is not None:
            raise ValueError("during-playback topology snapshot already captured")
        snapshot = self._snapshot()
        for _ in range(self._retry_attempts - 1):
            if _snapshot_has_physical_player(snapshot):
                break
            self._sleep(self._retry_interval_seconds)
            snapshot = self._snapshot()
        self._during = snapshot

    def finish(self) -> SelectedTopologyAudit | None:
        if self._before is None or self._during is None:
            return None
        return SelectedTopologyAudit.from_snapshots(self._plan, self._before, self._during)



def admit_selected_decision(decision: CalibrationDecision, audit: SelectedTopologyAudit | None) -> CalibrationDecision:
    """A selected feedback verdict is never publishable without two valid no-path snapshots."""
    if audit is not None and audit.valid and audit.no_physical_to_selected_path:
        return decision
    return CalibrationDecision("inconclusive", decision.null_p_values, decision.positive_p_value,
                               decision.selected_p_value, decision.baseline_digest)


def _graph_digest(snapshot: object) -> str:
    try:
        raw = json.dumps(snapshot, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    except (TypeError, ValueError):
        return ""
    return hashlib.sha256(raw).hexdigest()


def _snapshot_has_physical_player(snapshot: object) -> bool:
    """Cheap, exception-free check used only to decide whether to retry a during-snapshot."""
    if not isinstance(snapshot, dict):
        return False
    nodes = snapshot.get("nodes")
    if not isinstance(nodes, list):
        return False
    return any(isinstance(node, dict) and node.get("role") == "physical-player" for node in nodes)


def _parse_graph(snapshot: object) -> dict[str, list[dict[str, object]]]:
    if not isinstance(snapshot, dict):
        raise ValueError("graph snapshot is not an object")
    nodes, links = snapshot.get("nodes"), snapshot.get("links")
    if not isinstance(nodes, list) or not isinstance(links, list):
        raise ValueError("graph snapshot is malformed")
    if not all(isinstance(item, dict) for item in nodes + links):
        raise ValueError("graph entries are malformed")
    return {"nodes": nodes, "links": links}


def capture_selected_topology_snapshot(plan: SelectedCapturePlan) -> dict[str, object]:
    """Reduce one real ``pw-dump`` read to the strict audit graph shape.

    This is intentionally a direct literal command, outside the Pactl allowlist:
    PipeWire's graph cannot be reconstructed from Pulse sink inventories. Any
    unsupported pw-dump shape becomes an invalid graph and is admitted only as
    ``inconclusive`` by the caller.
    """
    try:
        completed = subprocess.run(("pw-dump",), shell=False, check=False, capture_output=True, text=True, timeout=2.0)
        if completed.returncode != 0:
            return {"nodes": [], "links": []}
        raw = json.loads(completed.stdout)
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return {"nodes": [], "links": []}
    if not isinstance(raw, list):
        return {"nodes": [], "links": []}

    def properties(item: object) -> dict[str, object]:
        if not isinstance(item, dict):
            return {}
        info = item.get("info")
        values = info.get("props") if isinstance(info, dict) else None
        return values if isinstance(values, dict) else {}

    def prop(values: dict[str, object], *names: str) -> str:
        for name in names:
            value = values.get(name)
            if isinstance(value, (str, int)) and str(value):
                return str(value)
        return ""

    nodes: list[dict[str, object]] = []
    links: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        kind = item.get("type")
        info = item.get("info")
        if not isinstance(kind, str) or not isinstance(info, dict):
            continue
        if kind.endswith(":Node"):
            values = properties(item)
            node_id = item.get("id")
            if not isinstance(node_id, int):
                continue
            name = prop(values, "node.name")
            serial = prop(values, "object.serial")
            module_id = prop(values, "pulse.module.id", "module.id")
            role = ""
            if name == plan.physical_target_name and serial == plan.physical_target_serial:
                role = "physical-target"
            elif name == plan.virtual_sink_name and serial == plan.virtual_sink_serial and module_id == plan.virtual_sink_module_id:
                # This node's PipeWire monitor is not a separate Node (only monitor
                # Ports on this same Node), so there is no distinct "selected-monitor"
                # role to match here -- `_validate_graph_identities` aliases the
                # "monitor" identity to this "virtual-sink" node instead.
                role = "virtual-sink"
            elif prop(values, "application.name", "media.name", "node.name") == "pw-play":
                role = "physical-player"
            if role:
                nodes.append({"id": str(node_id), "serial": serial, "name": name, "module_id": module_id, "role": role})
        elif kind.endswith(":Link"):
            output_id = info.get("output-node-id")
            input_id = info.get("input-node-id")
            if isinstance(output_id, int) and isinstance(input_id, int):
                links.append({"from": str(output_id), "to": str(input_id)})
    return {"nodes": nodes, "links": links}


def selected_capture_plan_trace(plan: SelectedCapturePlan) -> dict[str, object]:
    return {
        "schema": plan.schema,
        "sample_rate_hz": plan.sample_rate_hz,
        "sample_width_bytes": plan.sample_width_bytes,
        "pre_roll_samples": plan.pre_roll_samples,
        "analysis_samples": plan.analysis_samples,
        "post_roll_samples": plan.post_roll_samples,
        "owned_monitor_name": plan.owned_monitor_name,
        "owned_monitor_serial": plan.owned_monitor_serial,
        "physical_target_name": plan.physical_target_name,
        "physical_target_serial": plan.physical_target_serial,
        "virtual_sink_name": plan.virtual_sink_name,
        "virtual_sink_serial": plan.virtual_sink_serial,
        "virtual_sink_module_id": plan.virtual_sink_module_id,
        "candidate_revision": plan.candidate_revision,
    }


def selected_topology_audit_trace(audit: SelectedTopologyAudit | None) -> dict[str, object]:
    if audit is None:
        return {"schema": "opendubstream.selected-topology-audit/v1", "valid": False, "reason": "inconclusive"}
    return {
        "schema": audit.schema,
        "pre_snapshot_sha256": audit.pre_snapshot_sha256,
        "during_snapshot_sha256": audit.during_snapshot_sha256,
        "valid": audit.valid,
        "no_physical_to_selected_path": audit.no_physical_to_selected_path,
        "reason": audit.reason,
    }


def _unique_node_id(nodes: list[dict[str, object]], **expected: str) -> str:
    matches = [node for node in nodes if all(node.get(key) == value for key, value in expected.items())]
    if len(matches) != 1 or not isinstance(matches[0].get("id"), str) or not matches[0]["id"]:
        raise ValueError("graph identity is absent or ambiguous")
    return str(matches[0]["id"])


def _validate_graph_identities(
    graph: dict[str, list[dict[str, object]]], plan: SelectedCapturePlan, *, require_player: bool = True,
) -> dict[str, str]:
    """A sink's PipeWire monitor is not a separately addressable Node -- it is only a
    pair of monitor Ports (`port.monitor: true`) attached to the *same* Node as the sink
    itself. Confirmed by a throwaway `pactl`/`pw-dump` diagnostic and by both real 4.1d-c
    hardware runs' own recorded evidence, where `owned_monitor_serial` and
    `virtual_sink_serial` were always equal ("1348"=="1348", "1434"=="1434"). Searching
    for a distinct `role="selected-monitor"` node therefore always failed, silently
    downgrading every clean statistical result to `inconclusive`
    (see design.md "Amendment: Monitor Identity Is Not a Separate PipeWire Node",
    superseding the "selected-monitor node ID/serial" wording in the earlier "Exact
    Selected-Capture Contract" amendment). The invariant is checked explicitly -- not
    silently assumed -- so a future/differently-configured PipeWire that ever violates it
    still fails closed instead of fabricating validity.
    """
    if plan.owned_monitor_serial != plan.virtual_sink_serial:
        raise ValueError("owned monitor serial does not match the virtual sink serial")
    nodes = graph["nodes"]
    virtual_id = _unique_node_id(
        nodes, role="virtual-sink", name=plan.virtual_sink_name, serial=plan.virtual_sink_serial,
        module_id=plan.virtual_sink_module_id,
    )
    identities = {
        "physical": _unique_node_id(nodes, role="physical-target", name=plan.physical_target_name, serial=plan.physical_target_serial),
        "virtual": virtual_id,
        # The monitor is an alias of the virtual-sink node, not a separately matched node.
        "monitor": virtual_id,
    }
    if require_player:
        identities["player"] = _unique_node_id(nodes, role="physical-player")
        identities["player_serial"] = _unique_node_serial(nodes, role="physical-player")
    return identities


def _unique_node_serial(nodes: list[dict[str, object]], **expected: str) -> str:
    matches = [node for node in nodes if all(node.get(key) == value for key, value in expected.items())]
    if len(matches) != 1 or not isinstance(matches[0].get("serial"), str) or not matches[0]["serial"]:
        raise ValueError("graph identity has no stable serial")
    return str(matches[0]["serial"])


def _has_directed_path(links: list[dict[str, object]], origin: str, target: str) -> bool:
    edges: dict[str, list[str]] = {}
    for link in links:
        source, destination = link.get("from"), link.get("to")
        if not isinstance(source, str) or not isinstance(destination, str):
            raise ValueError("graph link is malformed")
        edges.setdefault(source, []).append(destination)
    todo, visited = [origin], set()
    while todo:
        node = todo.pop()
        if node == target:
            return True
        if node in visited:
            continue
        visited.add(node)
        todo.extend(edges.get(node, ()))
    return False


def run_calibrated_observation(
    *,
    router: object,
    read_phrase: Callable[[float], bytes],
    player: PcmPlayer,
    inventory: Callable[[], str],
    synthesized_pcm: bytes,
    target: SinkRef,
    lease: RouteLease,
    candidate_revision: str,
    selected_plan: SelectedCapturePlan | None = None,
    config: ProtocolConfig = ProtocolConfig(),
    settle_seconds: float = _SETTLE_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
    reader_deadline: float = 5.0,
    play_deadline: float = 15.0,
) -> CalibrationDecision:
    """Collect labelled controls, emit only the predeclared probe PCM, crop by the
    byte-clock formula, and evaluate the calibrated decision. The route is recovered
    exactly once via `run_protected`; any collection, admission, or playback failure
    yields an `inconclusive` decision instead of propagating. Selected observations always
    play only to the validated physical target; the reference sink is transport-measurement
    only and is deliberately not accepted by this API."""
    reader = SampleCountingReader(read_phrase)

    def work() -> tuple[tuple[bytes, bytes, bytes], bytes, bytes] | None:
        # Warm up the underlying reader (trigger its first real read, hence `parec`'s lazy
        # launch) before the settle sleep -- same class of fix as
        # `measure_transport_offset_samples` (4.1c-fix3): without this, the settle sleep can
        # fully elapse before anything has started flowing. Calls `read_phrase` directly
        # (bypassing `reader`) and discards the result, rather than going through `reader`
        # itself: `settle_and_collect_null_controls` (Phase 3, untouched) slices its captured
        # bytes from absolute position 0, so anything recorded into `reader` before it runs
        # would shift every null/positive/selected byte offset by the warm-up chunk's length.
        # A raised exception here propagates exactly like a failure inside
        # `settle_and_collect_null_controls`'s own reads: `run_protected` recovers the route
        # once and re-raises, converted to `inconclusive` by the outer except below.
        read_phrase(reader_deadline)
        nulls = settle_and_collect_null_controls(reader, config, settle_seconds=settle_seconds, sleep=sleep, deadline=reader_deadline)
        if nulls is None:
            return None
        tag = deterministic_tag(config, 0)
        positive = build_positive_control(nulls[0], tag, config)
        probe = emit_tagged_probe_pcm(synthesized_pcm, tag, config)
        if selected_plan is not None:
            # This branch deliberately has no acoustic/transport offset: its fixed window
            # is defined solely by the reader clock immediately before pw-play.
            if not advance_reader_to(reader, reader.sample_count + selected_plan.pre_roll_samples, deadline=reader_deadline):
                return None
            reader_sample_count_at_pw_play = reader.sample_count
            capture_started = time.monotonic_ns()
            player_invoked = time.monotonic_ns()
            play_predeclared_probe(player, inventory, probe, target, lease, deadline=play_deadline)
            if not advance_reader_to(reader, selected_plan.completion_target(reader_sample_count_at_pw_play), deadline=reader_deadline):
                return None
            selected = selected_plan.extract_selected_window(
                captured_pcm=reader.captured_bytes(), reader_count_before_player=reader_sample_count_at_pw_play,
                reader_count_after_completion=reader.sample_count, capture_started_monotonic_ns=capture_started,
                player_invoked_monotonic_ns=player_invoked, capture_completed_monotonic_ns=time.monotonic_ns(),
            )
        else:
            # Compatibility path for the earlier controlled-harness receipt only. The live
            # selected path always supplies SelectedCapturePlan and never uses this crop.
            reader_sample_count_at_pw_play = reader.sample_count
            play_predeclared_probe(player, inventory, probe, target, lease, deadline=play_deadline)
            expected = reader_sample_count_at_pw_play + config.tag_insertion_samples
            if not advance_reader_to(reader, expected + config.analysis_samples, deadline=reader_deadline):
                return None
            selected = crop_tagged_monitor_window(reader.captured_bytes(), reader_sample_count_at_pw_play, 0, config)
        if selected is None:
            return None
        return nulls, positive, selected

    try:
        collected = run_protected(router, work)
    except Exception:
        collected = None

    if collected is None:
        return CalibrationDecision("inconclusive", (), None, None, _IMMUTABLE_FAILED_BASELINE_DIGEST)

    nulls, positive, selected = collected
    null_scores = tuple(
        score_pcm_dataset(sample, candidate_revision, f"null-{index}", config) for index, sample in enumerate(nulls)
    )
    positive_score = score_pcm_dataset(positive, candidate_revision, "positive", config)
    selected_score = score_pcm_dataset(selected, candidate_revision, "selected", config)
    return evaluate_calibration(
        null_scores=null_scores,
        positive_score=positive_score,
        selected_score=selected_score,
        baseline_digest=_IMMUTABLE_FAILED_BASELINE_DIGEST,
        config=config,
    )


def build_calibration_observation_call(
    *,
    router: object,
    capture: PipeWireMonitorCapture,
    player: PcmPlayer,
    pactl: SafePactlRunner,
    synthesized_pcm: bytes,
    physical_target: SinkRef,
    lease: RouteLease,
    candidate_revision: str,
    selected_plan: SelectedCapturePlan,
) -> dict[str, object]:
    """Pure assembly of `run_calibrated_observation`'s keyword arguments from the
    already-live objects `run_confirmed` holds at this point in its flow. Kept
    separate from `run_confirmed` itself so the wiring can be proven correct with
    fakes: `run_confirmed` stays real-hardware-only and is never called by tests."""
    return {
        "router": router,
        "read_phrase": capture.read_phrase,
        "player": player,
        "inventory": lambda: pactl.run(("pactl", "-f", "json", "list", "sinks")),
        "synthesized_pcm": synthesized_pcm,
        "target": physical_target,
        "lease": lease,
        "candidate_revision": candidate_revision,
        "selected_plan": selected_plan,
    }


def measure_transport_offset_samples(
    *,
    router: object,
    read_phrase: Callable[[float], bytes],
    player: PcmPlayer,
    inventory: Callable[[], str],
    target: object,
    lease: object,
    config: ProtocolConfig = ProtocolConfig(),
    settle_seconds: float = _SETTLE_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
    reader_deadline: float = 5.0,
    play_deadline: float = 15.0,
    search_range_samples: int = 128_000,
    confidence_floor: float = 0.5,
    reference: ReferenceLease | None = None,
) -> int | None:
    """Measure the one-time-per-environment `transport_offset_samples`: play exactly one
    deterministic tag (no leading silence, no synthesized audio, no probe), then search a
    wide, config-bounded window beyond the naively expected position for its best-scoring
    start. Reuses the same settle period as `settle_and_collect_null_controls`
    (`_SETTLE_SECONDS`) and the same normalized-correlation math as feedback-window
    detection via `locate_best_tag_position`. The route is recovered exactly once via
    `run_protected`; any collection, admission, or playback failure -- or a below-floor
    correlation -- yields `None` instead of a guess.

    When `reference` is provided, the tag is played to the owned reference null-sink
    (`SinkRef(reference.reference_sink, False)` as `target`) via
    `play_predeclared_probe_to_reference` instead of the physical sink via
    `play_predeclared_probe` -- the reference sink is bridged to the physical sink by
    `PipeWirePulseRouter.open_reference_sink`'s loopback, so playback stays audible.
    `reference` defaults to `None`, preserving the original physical-target playback path
    exactly for any caller that does not supply one."""
    reader = SampleCountingReader(read_phrase)

    def work() -> int | None:
        # Warm up the reader (trigger the first real read) BEFORE the settle sleep and the
        # tag playback. `parec` negotiates its connection latency on first use -- confirmed
        # live (2026-09-03) at server-default latency this can take several seconds -- so a
        # blind sleep-then-play-then-read sequence can let the tag finish playing before the
        # reader has even started flowing. Fails closed identically to a later stalled read.
        if not advance_reader_to(reader, 1, deadline=reader_deadline):
            print(
                f"transport-offset measurement: capture stalled during warm-up "
                f"(captured={reader.sample_count} needed=1)",
                file=sys.stderr,
            )
            return None
        sleep(settle_seconds)
        anchor = reader.sample_count
        tag = deterministic_tag(config, 0)
        if reference is not None:
            play_predeclared_probe_to_reference(player, inventory, tag, target, reference, deadline=play_deadline)
        else:
            play_predeclared_probe(player, inventory, tag, target, lease, deadline=play_deadline)
        search_range = config.tag_insertion_samples + search_range_samples
        window_target = anchor + search_range + config.tag_samples
        if not advance_reader_to(reader, window_target, deadline=reader_deadline):
            print(
                f"transport-offset measurement: capture stalled "
                f"(captured={reader.sample_count - anchor} needed={window_target - anchor})",
                file=sys.stderr,
            )
            return None
        window = reader.captured_bytes()[anchor * 2 :]
        best_score = best_tag_correlation_score(tag, window, search_range)
        print(
            f"transport-offset measurement: best_correlation={best_score:.3f} "
            f"confidence_floor={confidence_floor} search_range_samples={search_range_samples} "
            f"captured_bytes={len(window)}",
            file=sys.stderr,
        )
        best_relative = locate_best_tag_position(tag, window, search_range, confidence_floor=confidence_floor)
        if best_relative is None:
            return None
        return best_relative - config.tag_insertion_samples

    try:
        return run_protected(router, work)
    except Exception:
        return None


def build_transport_offset_measurement_call(
    *,
    router: object,
    capture: PipeWireMonitorCapture,
    player: PcmPlayer,
    pactl: SafePactlRunner,
    physical_target: SinkRef,
    lease: RouteLease,
    reference: ReferenceLease | None = None,
) -> dict[str, object]:
    """Pure assembly of `measure_transport_offset_samples`'s keyword arguments from the
    already-live objects a real caller holds at this point in its flow. Kept separate so
    the wiring can be proven correct with fakes: the real caller stays real-hardware-only
    and is never called by tests -- mirrors `build_calibration_observation_call`.

    `reference` defaults to `None` and is passed straight through; a caller that supplies
    it routes `measure_transport_offset_samples`'s playback through the owned reference
    sink instead of the physical target named by `physical_target`/`lease`."""
    return {
        "router": router,
        "read_phrase": capture.read_phrase,
        "player": player,
        "inventory": lambda: pactl.run(("pactl", "-f", "json", "list", "sinks")),
        "target": physical_target,
        "lease": lease,
        "reference": reference,
    }


def read_duration(read_phrase, seconds: float, *, deadline: float = 2.0, max_reads: int = 30) -> bytes:
    """Accumulate mono 16-kHz s16le PCM by calling read_phrase repeatedly until >= seconds worth is captured."""
    target_bytes = int(seconds * 16_000 * 2)
    chunks: list[bytes] = []
    total = 0
    for _ in range(max_reads):
        if total >= target_bytes:
            break
        chunk = read_phrase(deadline)
        chunks.append(chunk)
        total += len(chunk)
    return b"".join(chunks)[:target_bytes]


def normalize_pcm16(pcm: bytes, *, target_peak: float = 0.9) -> bytes:
    """Peak-normalize s16le PCM. Monitor-captured Chrome audio is consistently very quiet
    (confirmed ~2% peak in this session's earlier read-only captures) and fails VAD unamplified."""
    import numpy as np

    samples = np.frombuffer(pcm, dtype="<i2").astype("float32")
    peak = np.max(np.abs(samples))
    if peak <= 0:
        return pcm
    gain = (32768.0 * target_peak) / peak
    return np.clip(samples * gain, -32768, 32767).astype("<i2").tobytes()


def has_voice(pcm: bytes, root: Path) -> bool:
    """Silero VAD gate; window MUST be 256 for this pinned asset (see benchmarks/feasibility-2026-09-03.md)."""
    import numpy as np
    import onnxruntime as ort

    session = ort.InferenceSession(str(root / "silero-vad/silero_vad.onnx"), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    samples = np.frombuffer(pcm, dtype="<i2").astype("float32") / 32768.0
    state = np.zeros((2, 1, 128), dtype="float32")
    window = 256
    for start in range(0, len(samples), window):
        chunk = samples[start : start + window]
        if len(chunk) < window:
            chunk = np.pad(chunk, (0, window - len(chunk)))
        score, state = session.run(None, {"input": chunk[None, :], "state": state, "sr": np.array(16000, dtype="int64")})
        if float(score[0][0]) >= 0.5:
            return True
    return False


def resample_to_pcm16(audio, from_rate: int, to_rate: int) -> bytes:
    """Linear-interpolation resample; Kokoro outputs float32 at 24 kHz, pw-play here expects s16le at 16 kHz."""
    import numpy as np

    target_length = max(1, int(len(audio) * to_rate / from_rate))
    source_indices = np.arange(len(audio))
    target_indices = np.linspace(0, len(audio) - 1, target_length)
    resampled = np.interp(target_indices, source_indices, audio)
    pcm16 = np.clip(resampled, -1.0, 1.0)
    return (pcm16 * 32767).astype("<i2").tobytes()


def sink_serial(pactl: SafePactlRunner, sink_name: str) -> str:
    sinks = json.loads(pactl.run(("pactl", "-f", "json", "list", "sinks")))
    for entry in sinks:
        if entry.get("name") == sink_name:
            serial = entry.get("properties", {}).get("object.serial")
            if isinstance(serial, str) and serial:
                return serial
    raise EvidenceSchemaError(f"sink {sink_name!r} has no discoverable stable serial")


def default_endpoints(pactl: SafePactlRunner) -> tuple[str, str]:
    sinks = json.loads(pactl.run(("pactl", "-f", "json", "list", "sinks")))
    default_sink = next((s["name"] for s in sinks if s.get("state") == "RUNNING"), "")
    return default_sink, ""  # source tracking is out of scope until a real capture-source default exists


def select_active_chrome_route(pactl: SafePactlRunner, router: PipeWirePulseRouter) -> RouteLease:
    """Select the one actively playing (non-corked) Chrome stream and lease its route.
    Shared by `run_confirmed` and `measure_transport_offset` so this route-selection
    logic is defined exactly once. The pure selection itself lives in
    `application/chrome_route.py`; this function only supplies its live pactl I/O and
    performs the router mutation, with byte-identical behavior to the pre-extraction
    inline implementation."""
    streams = PipeWireStreamDiscovery(pactl)()
    raw = json.loads(pactl.run(("pactl", "-f", "json", "list", "sink-inputs")))
    selected_stream = select_active_chrome_stream(streams, raw)
    selected = router.select_unique(StreamSelector(application_name=selected_stream.application_name, serial=selected_stream.serial))
    return router.begin(selected, SinkRef(selected.sink_name, True))


def measure_transport_offset(*, candidate_revision: str = "pending") -> int | None:
    """Real-hardware, one-time-per-environment transport-offset measurement. Reuses
    `select_active_chrome_route` (the same route-selection precondition as `run_confirmed`:
    one actively playing, non-corked Chrome stream), then plays exactly one deterministic
    tag into a fresh tool-owned reference null-sink (`PipeWirePulseRouter.open_reference_sink`,
    bridged to the physical sink with a `module-loopback` so it stays audible) and searches
    for it via `measure_transport_offset_samples`, capturing that reference sink's own
    monitor through `OwnedReferenceMonitorDiscovery` instead of the Chrome virtual-sink
    monitor `OwnedMonitorDiscovery` validates. This keeps the measurement inside the
    monitor-ownership boundary `OwnedMonitorDiscovery` enforces: playing to and capturing
    `lease.physical_sink` directly would require capturing a monitor this tool did not
    create, which that boundary intentionally forbids (see design.md's "Amendment: Owned
    Reference Sink for Transport-Offset Measurement"). Never called by tests; exercised
    only via `main()` with `--measure-transport-offset`, real hardware only. Runs no
    ASR/MT/TTS and writes no evidence -- the result is printed for reuse as the predeclared
    `--transport-offset-samples` input to a later confirmed run."""
    pactl = SafePactlRunner()
    journal = RecoveryJournal(Path.home() / ".local/share/opendubstream/routing.json")
    router = PipeWirePulseRouter(pactl, journal, PipeWireStreamDiscovery(pactl))

    def work() -> int | None:
        lease = select_active_chrome_route(pactl, router)
        physical_target = SinkRef(lease.physical_sink, True, sink_serial(pactl, lease.physical_sink))
        reference = router.open_reference_sink(lease, physical_target)
        capture = PipeWireMonitorCapture(OwnedReferenceMonitorDiscovery(pactl), journal=journal)
        capture.start_reference(reference)
        try:
            observation_kwargs = build_transport_offset_measurement_call(
                router=router, capture=capture, player=PwPlayPcmPlayer(), pactl=pactl,
                physical_target=SinkRef(reference.reference_sink, False), lease=lease,
                reference=reference,
            )
            return measure_transport_offset_samples(**observation_kwargs)
        finally:
            capture.stop()

    result = run_protected(router, work)
    router.recover()
    return result


def run_confirmed(
    *, voice: str = "ef_dora", transport_offset_samples: int | None = None, candidate_revision: str = "pending"
) -> dict[str, object]:
    """The real E2E path. Never called by tests; exercised only via `main()` with `--confirm`."""
    require_transport_offset_samples(transport_offset_samples)
    pactl = SafePactlRunner()
    root = DEFAULT_ROOT
    journal = RecoveryJournal(Path.home() / ".local/share/opendubstream/routing.json")
    router = PipeWirePulseRouter(pactl, journal, PipeWireStreamDiscovery(pactl))
    default_sink_before, default_source_before = default_endpoints(pactl)

    started = time.monotonic()

    def work() -> dict[str, object]:
        lease = select_active_chrome_route(pactl, router)
        capture = PipeWireMonitorCapture(OwnedMonitorDiscovery(pactl), journal=journal)
        monitor = capture.start(lease)
        time.sleep(0.3)  # let the freshly created virtual sink/monitor settle before parec attaches
        try:
            pcm = normalize_pcm16(read_duration(capture.read_phrase, 8.0, deadline=5.0))
        finally:
            capture.stop()

        if not has_voice(pcm, root):
            raise RuntimeError("no speech detected in the captured monitor audio; no ASR/translation/synthesis was run")

        import numpy as np  # local import: only touched on a confirmed live run
        os.environ.setdefault("ONNX_PROVIDER", "CUDAExecutionProvider")
        from faster_whisper import WhisperModel
        from kokoro_onnx import Kokoro
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        samples = np.frombuffer(pcm, dtype="<i2").astype("float32") / 32768.0
        asr_start = time.monotonic()
        asr_model = WhisperModel(str(root / "faster-whisper-distil-large-v3"), device="cuda", compute_type="float16", local_files_only=True)
        segments, _ = asr_model.transcribe(samples, language="en", task="transcribe", vad_filter=False)
        transcript = "".join(s.text for s in segments).strip()
        asr_seconds = time.monotonic() - asr_start

        tr_start = time.monotonic()
        tokenizer = AutoTokenizer.from_pretrained(str(root / "opus-mt-en-es"), local_files_only=True)
        mt_model = AutoModelForSeq2SeqLM.from_pretrained(str(root / "opus-mt-en-es"), local_files_only=True).to("cuda")
        translation = tokenizer.decode(mt_model.generate(**tokenizer(transcript, return_tensors="pt").to("cuda"))[0], skip_special_tokens=True).strip()
        translation_seconds = time.monotonic() - tr_start

        tts_start = time.monotonic()
        engine = Kokoro(str(root / "kokoro-onnx-v1/kokoro-v1.0.onnx"), str(root / "kokoro-onnx-v1/voices-v1.0.bin"))
        audio, kokoro_rate = engine.create(translation, voice=voice, lang="es")
        synthesized = resample_to_pcm16(audio, kokoro_rate, 16_000)
        tts_seconds = time.monotonic() - tts_start

        physical_target = SinkRef(lease.physical_sink, True, sink_serial(pactl, lease.physical_sink))
        # The selected isolation observation is intentionally physical -> selected virtual
        # monitor. A reference sink is valid only for transport-offset measurement: using it
        # here would make a positive result tautological rather than test feedback into ASR.
        capture2 = PipeWireMonitorCapture(OwnedMonitorDiscovery(pactl), journal=journal)
        selected_monitor = capture2.start(lease)
        selected_plan = SelectedCapturePlan.create(
            owned_monitor_name=selected_monitor.name, owned_monitor_serial=selected_monitor.serial,
            physical_target_name=physical_target.name, physical_target_serial=physical_target.serial,
            virtual_sink_name=lease.virtual_sink, virtual_sink_serial=sink_serial(pactl, lease.virtual_sink),
            virtual_sink_module_id=lease.virtual_module_id, candidate_revision=candidate_revision,
        )
        recorder = SelectedTopologyAuditRecorder(
            selected_plan, lambda: capture_selected_topology_snapshot(selected_plan),
        )
        try:
            # This is before the physical pw-play process is started. The recorder's
            # second callback is invoked by PwPlayPcmPlayer only after it has spawned
            # that process, so both graph reads bound the actual playback interval.
            recorder.capture_before_playback()
        except Exception:
            recorder = None
        player = PwPlayPcmPlayer(
            on_started=recorder.capture_during_playback if recorder is not None else None,
        )
        try:
            observation_kwargs = build_calibration_observation_call(
                router=router, capture=capture2, player=player, pactl=pactl,
                synthesized_pcm=synthesized, physical_target=physical_target, lease=lease,
                candidate_revision=candidate_revision, selected_plan=selected_plan,
            )
            raw_decision = run_calibrated_observation(**observation_kwargs)
            try:
                topology_audit = recorder.finish() if recorder is not None else None
            except Exception:
                topology_audit = None
            decision = admit_selected_decision(raw_decision, topology_audit)
        finally:
            capture2.stop()

        return {
            "monitor": monitor, "lease": lease, "resolved_sink": physical_target, "transcript": transcript,
            "translation": translation, "synthesized": synthesized, "decision": decision,
            "selected_plan": selected_plan, "topology_audit": topology_audit,
            "asr_seconds": asr_seconds, "translation_seconds": translation_seconds, "tts_seconds": tts_seconds,
        }

    result = run_protected(router, work)
    print(f"EN: {result['transcript']}", file=sys.stderr)
    print(f"ES: {result['translation']}", file=sys.stderr)
    recovery = router.recover()
    default_sink_after, default_source_after = default_endpoints(pactl)
    decision = result["decision"]

    return build_hardware_e2e_evidence(
        revision=candidate_revision,
        default_sink_before=default_sink_before, default_sink_after=default_sink_after,
        default_source_before=default_source_before, default_source_after=default_source_after,
        stream_identifier=result["lease"].stream.identifier, virtual_sink=result["lease"].virtual_sink,
        virtual_module_id=result["lease"].virtual_module_id,
        monitor_name=result["monitor"].name, monitor_serial=result["monitor"].serial,
        physical_sink_name=result["resolved_sink"].name, physical_sink_serial=result["resolved_sink"].serial,
        asr_seconds=result["asr_seconds"], translation_seconds=result["translation_seconds"], tts_seconds=result["tts_seconds"],
        total_seconds=time.monotonic() - started,
        calibration_decision=decision.value, null_p_values=decision.null_p_values,
        positive_p_value=decision.positive_p_value, selected_p_value=decision.selected_p_value,
        baseline_digest=decision.baseline_digest, routing_recovered=recovery.restored,
        selected_capture_plan=selected_capture_plan_trace(result["selected_plan"]),
        selected_topology_audit=selected_topology_audit_trace(result["topology_audit"]),
        transcript_hash=bounded_output_hash(result["transcript"].encode("utf-8"), max_bytes=4096),
        translation_hash=bounded_output_hash(result["translation"].encode("utf-8"), max_bytes=4096),
        synthesized_audio_hash=bounded_output_hash(result["synthesized"], max_bytes=50_000_000),
    )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Confirmed real-hardware E2E audio loop; no default mutation.")
    parser.add_argument("--confirm", action="store_true", help="required: run against the live desktop session")
    parser.add_argument("--voice", default="ef_dora")
    parser.add_argument(
        "--transport-offset-samples", type=int, default=None,
        help="required with --confirm: predeclared reader-clock offset for the byte-clock crop",
    )
    parser.add_argument(
        "--measure-transport-offset", action="store_true",
        help=(
            "real hardware only: run ONLY the one-time transport-offset measurement "
            "(plays one tag, searches for it, prints transport_offset_samples) and exit "
            "without running the ASR/MT/TTS/confirm pipeline; still requires --confirm"
        ),
    )
    args = parser.parse_args(argv)
    if args.measure_transport_offset:
        try:
            require_confirmation(args.confirm, reason="this plays one tag and routes a live Chrome stream; pass --confirm")
        except ConfirmationRequired as error:
            print(f"ERROR: {error}", file=sys.stderr)
            return 2
        offset = measure_transport_offset()
        if offset is None:
            print("ERROR: transport-offset measurement was inconclusive", file=sys.stderr)
            return 3
        print(offset)
        return 0
    try:
        require_confirmation(args.confirm, reason="this plays synthesized Spanish audio and routes a live Chrome stream; pass --confirm")
    except ConfirmationRequired as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    evidence = run_confirmed(voice=args.voice, transport_offset_samples=args.transport_offset_samples)
    # Flat human-readable dump, written regardless of admission outcome (for debugging only;
    # admission below gates the canonical no-feedback claim, not this raw dump).
    EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE_PATH.write_text(json.dumps(evidence, indent=2, sort_keys=True), encoding="utf-8")
    admission = persist_and_admit_hardware_rerun_evidence(evidence)
    if not admission.admitted:
        print(f"ERROR: hardware-rerun evidence was not admitted: {admission.reason}", file=sys.stderr)
        if evidence["calibration_decision"] == "no-feedback":
            print(
                "ERROR: refusing to publish a no-feedback verdict without admitted canonical evidence",
                file=sys.stderr,
            )
        print(json.dumps(evidence, indent=2, sort_keys=True))
        return 4
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
