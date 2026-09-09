"""PipeWire-Pulse routing adapter with journal-first, recoverable mutation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
import json
from typing import Protocol

from opendubstream.domain.contracts import (
    AudioRouter,
    OriginalMixLease,
    PhysicalSinkRequired,
    RecoveryResult,
    ReferenceLease,
    RouteLease,
    SinkRef,
    StreamRef,
    StreamIdentityChanged,
    StreamSelector,
    select_unique_stream,
)
from opendubstream.infrastructure.audio.journal import RecoveryJournal, RouteSnapshot
from opendubstream.infrastructure.audio.playback import PhysicalPlaybackTarget


class RoutingSafetyError(RuntimeError):
    pass


class Pactl(Protocol):
    def run(self, argv: tuple[str, ...], **kwargs: object) -> str: ...


class PipeWirePulseRouter(AudioRouter):
    def __init__(self, pactl: Pactl, journal: RecoveryJournal, streams: Callable[[], list[StreamRef]]) -> None:
        self._pactl = pactl
        self._journal = journal
        self._streams = streams
        self._playback = PhysicalPlaybackTarget()

    def select_unique(self, selector: StreamSelector) -> StreamRef:
        return select_unique_stream(selector, self._streams())

    def begin(self, stream: StreamRef, physical: SinkRef) -> RouteLease:
        try:
            physical = self._playback.validate_target(physical)
        except PhysicalSinkRequired as error:
            raise RoutingSafetyError("routing requires a physical TTS playback target") from error
        current = [candidate for candidate in self._streams() if candidate.identifier == stream.identifier]
        if len(current) != 1:
            raise RoutingSafetyError("selected browser stream identity changed before routing")
        try:
            stream.require_same_identity(current[0])
        except StreamIdentityChanged as error:
            raise RoutingSafetyError("selected browser stream identity changed before routing") from error
        stream = current[0]
        virtual_sink = f"opendubstream.{stream.identifier}"
        snapshot = RouteSnapshot(stream=stream, original_sink=stream.sink_name, virtual_sink=virtual_sink)
        self._journal.persist(snapshot)
        module_id = self._pactl.run(("pactl", "load-module", "module-null-sink", f"sink_name={virtual_sink}")).strip()
        if not module_id.isdigit():
            raise RoutingSafetyError("PipeWire-Pulse returned an invalid null-sink module identity")
        # Persist the module identity before moving the stream so recovery can undo either crash point.
        self._journal.persist(replace(snapshot, virtual_module_id=module_id))
        self._pactl.run(("pactl", "move-sink-input", stream.identifier, virtual_sink))
        return RouteLease(stream, virtual_sink, module_id, stream.sink_name, physical.name)

    def open_reference_sink(self, lease: RouteLease, physical: SinkRef) -> ReferenceLease:
        """Create a tool-owned reference null-sink, bridge it to `physical` with a
        `module-loopback` so it stays audible, and journal each step before its mutation
        -- same journal-before-mutation discipline as `begin()`."""
        snapshot = self._journal.load()
        if snapshot is None or snapshot.virtual_sink != lease.virtual_sink or snapshot.virtual_module_id != lease.virtual_module_id:
            raise RoutingSafetyError("journal does not match the active route lease")
        reference_sink = f"opendubstream-ref.{lease.stream.identifier}"
        snapshot = replace(snapshot, reference_sink=reference_sink)
        self._journal.persist(snapshot)
        reference_module_id = self._pactl.run(
            ("pactl", "load-module", "module-null-sink", f"sink_name={reference_sink}")
        ).strip()
        if not reference_module_id.isdigit():
            raise RoutingSafetyError("PipeWire-Pulse returned an invalid reference null-sink module identity")
        # Persist the reference module identity before loading the loopback so recovery can undo either crash point.
        snapshot = replace(snapshot, reference_module_id=reference_module_id)
        self._journal.persist(snapshot)
        loopback_module_id = self._pactl.run(
            ("pactl", "load-module", "module-loopback", f"source={reference_sink}.monitor", f"sink={physical.name}")
        ).strip()
        if not loopback_module_id.isdigit():
            raise RoutingSafetyError("PipeWire-Pulse returned an invalid loopback module identity")
        snapshot = replace(snapshot, loopback_module_id=loopback_module_id)
        self._journal.persist(snapshot)
        return ReferenceLease(reference_sink, reference_module_id, loopback_module_id, physical.name)

    def start_original_mix(self, lease: RouteLease, physical: SinkRef) -> OriginalMixLease:
        """Make only Chrome's owned virtual-monitor audible at a physical sink.

        The loopback source is intentionally the existing capture sink's monitor.  TTS
        never enters that sink: Spanish PCM continues to use its independent physical
        player.  Every identity required for a later volume mutation is journaled before
        it can be mutated or recovered.
        """
        self._validate_original_mix_target(physical)
        snapshot = self._matching_snapshot(lease)
        module_id = self._pactl.run(
            ("pactl", "load-module", "module-loopback", f"source={lease.virtual_sink}.monitor", f"sink={physical.name}")
        ).strip()
        if not module_id.isdigit():
            raise RoutingSafetyError("PipeWire-Pulse returned an invalid original loopback module identity")
        snapshot = replace(snapshot, original_loopback_module_id=module_id, original_loopback_physical_sink=physical.name)
        self._journal.persist(snapshot)
        try:
            sink_input_id = self._resolve_owned_loopback_sink_input(module_id, physical.name)
        except Exception:
            # This module is ours because its ID was returned by our exact load command.
            # Remove it immediately; retain the journal only if that cleanup itself fails.
            self._pactl.run(("pactl", "unload-module", module_id))
            self._journal.persist(
                replace(snapshot, original_loopback_module_id=None, original_loopback_physical_sink=None)
            )
            raise
        snapshot = replace(snapshot, original_loopback_sink_input_id=sink_input_id)
        self._journal.persist(snapshot)
        return OriginalMixLease(module_id, sink_input_id, physical.name)

    def set_original_mix(self, lease: OriginalMixLease, *, muted: bool, volume_percent: int) -> None:
        if not isinstance(volume_percent, int) or not 0 <= volume_percent <= 100:
            raise RoutingSafetyError("original mix volume must be an integer percentage from 0 through 100")
        snapshot = self._journal.load()
        if (
            snapshot is None
            or snapshot.original_loopback_module_id != lease.loopback_module_id
            or snapshot.original_loopback_sink_input_id != lease.sink_input_id
            or snapshot.original_loopback_physical_sink != lease.physical_sink
        ):
            raise RoutingSafetyError("journal does not match the active original mix lease")
        self._pactl.run(("pactl", "set-sink-input-mute", lease.sink_input_id, "1" if muted else "0"))
        self._pactl.run(("pactl", "set-sink-input-volume", lease.sink_input_id, f"{volume_percent}%"))

    def recover(self) -> RecoveryResult:
        snapshot = self._journal.load()
        if snapshot is None:
            return RecoveryResult(restored=False, detail="no recovery journal")
        # The original mix depends on the virtual monitor, so remove it before moving
        # Chrome back and before unloading the owned virtual sink.
        if snapshot.original_loopback_module_id is not None:
            self._unload_owned_module(snapshot.original_loopback_module_id)
        # Loopback depends on the reference sink existing, so it must be unloaded first;
        # both are a no-op when this journal never called `open_reference_sink` (old-shape
        # journals, or a lease that never measured transport offset).
        if snapshot.loopback_module_id is not None:
            self._pactl.run(("pactl", "unload-module", snapshot.loopback_module_id))
        if snapshot.reference_module_id is not None:
            self._pactl.run(("pactl", "unload-module", snapshot.reference_module_id))
        current_identifier = self._resolve_stream_identifier_on(snapshot.virtual_sink, snapshot.stream.identifier)
        self._pactl.run(("pactl", "move-sink-input", current_identifier, snapshot.original_sink))
        if snapshot.virtual_module_id is None:
            raise RoutingSafetyError("journal lacks virtual module identity; manual reconciliation is required")
        self._pactl.run(("pactl", "unload-module", snapshot.virtual_module_id))
        self._journal.clear()
        return RecoveryResult(restored=True, detail="restored journaled stream target")

    def _matching_snapshot(self, lease: RouteLease) -> RouteSnapshot:
        snapshot = self._journal.load()
        if snapshot is None or snapshot.virtual_sink != lease.virtual_sink or snapshot.virtual_module_id != lease.virtual_module_id:
            raise RoutingSafetyError("journal does not match the active route lease")
        if snapshot.original_loopback_module_id is not None:
            raise RoutingSafetyError("an original mix loopback is already active for this route")
        return snapshot

    def _unload_owned_module(self, module_id: str) -> None:
        """Treat only a proven already-absent owned module as idempotent cleanup.

        Recovery must still restore Chrome when another recovery/process has already
        removed this journaled module. Any other Pactl failure is actionable and leaves
        the journal intact for a later reconciliation attempt.
        """
        try:
            self._pactl.run(("pactl", "unload-module", module_id))
        except Exception as error:
            detail = str(error).lower()
            if any(marker in detail for marker in ("no such entity", "not found", "does not exist", "unknown module")):
                return
            raise

    def _validate_original_mix_target(self, physical: SinkRef) -> None:
        try:
            self._playback.validate_target(physical)
        except PhysicalSinkRequired as error:
            raise RoutingSafetyError("original mix requires a physical playback target") from error
        if (
            not physical.name
            or physical.name.endswith(".monitor")
            or physical.name.startswith("opendubstream.")
            or any(character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.:-" for character in physical.name)
        ):
            raise RoutingSafetyError("original mix requires a literal physical playback target")

    def _resolve_owned_loopback_sink_input(self, module_id: str, physical_sink: str) -> str:
        """Resolve exactly one sink-input owned by this module and physical sink.

        PipeWire-Pulse's JSON fields are locale independent.  Do not match an arbitrary
        stream by application label; `owner_module` plus the freshly resolved sink is the
        ownership boundary that prevents muting Chrome or another desktop application.
        """
        try:
            inputs = json.loads(self._pactl.run(("pactl", "-f", "json", "list", "sink-inputs")))
            sinks = json.loads(self._pactl.run(("pactl", "-f", "json", "list", "sinks")))
        except json.JSONDecodeError as error:
            raise RoutingSafetyError("malformed PipeWire-Pulse loopback inventory") from error
        if not isinstance(inputs, list) or not isinstance(sinks, list):
            raise RoutingSafetyError("malformed PipeWire-Pulse loopback inventory")
        physical_entries = [entry for entry in sinks if isinstance(entry, dict) and entry.get("name") == physical_sink]
        if len(physical_entries) != 1:
            raise RoutingSafetyError("physical loopback target is missing or ambiguous")
        physical_index = physical_entries[0].get("index")
        matches = [
            entry
            for entry in inputs
            if isinstance(entry, dict)
            and str(entry.get("owner_module")) == module_id
            and entry.get("sink") == physical_index
            and isinstance(entry.get("index"), int)
        ]
        if len(matches) != 1:
            raise RoutingSafetyError("owned loopback sink-input is missing or ambiguous")
        return str(matches[0]["index"])

    def _resolve_stream_identifier_on(self, sink_name: str, journaled_identifier: str) -> str:
        """The journaled stream identifier can go stale if PipeWire renumbers the stream
        during a long-running lease (confirmed live, 2026-09-03: identifier drifted across
        one ~20s measurement, leaving Chrome stuck on the virtual sink after `move-sink-input`
        failed against the stale identifier). `sink_name` is a private sink only this tool's
        lease could have moved a stream onto, so whichever single stream is discoverably
        there right now is unambiguously the one to restore -- the same "resolve fresh,
        never trust a stale reference" pattern `OwnedMonitorDiscovery` already uses. Falls
        back to the journaled identifier when discovery finds none (preserves prior
        behavior exactly); fails closed rather than guessing when it finds more than one."""
        matches = [candidate for candidate in self._streams() if candidate.sink_name == sink_name]
        if len(matches) > 1:
            raise RoutingSafetyError("more than one stream is on the owned virtual sink; refusing to guess which to restore")
        if len(matches) == 1:
            return matches[0].identifier
        return journaled_identifier
