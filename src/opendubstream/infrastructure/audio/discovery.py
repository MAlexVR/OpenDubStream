"""Real stream discovery via pactl's JSON output — locale-independent, read-only.

Human-readable `pactl list ...` output is translated by system locale (labels like
"Sink Input #" become "Entrada del destino #" under es_CO), which makes it an unsafe
parsing target for production code. `pactl -f json ...` returns stable, English
machine keys regardless of locale, so discovery is built on that instead.
"""

from __future__ import annotations

import json
from typing import Protocol

from opendubstream.domain.contracts import MonitorCaptureError, MonitorRef, ReferenceLease, RouteLease, StreamRef


class DiscoveryError(RuntimeError):
    """Real stream discovery could not produce a trustworthy stream list."""


class Pactl(Protocol):
    def run(self, argv: tuple[str, ...], **kwargs: object) -> str: ...


def parse_streams(sink_inputs_json: str, sinks_json: str) -> list[StreamRef]:
    try:
        sink_inputs = json.loads(sink_inputs_json)
    except json.JSONDecodeError as error:
        raise DiscoveryError(f"malformed sink-input listing: {error}") from error
    try:
        sinks = json.loads(sinks_json)
    except json.JSONDecodeError as error:
        raise DiscoveryError(f"malformed sink listing: {error}") from error

    sink_names = {str(sink["index"]): sink["name"] for sink in sinks}

    streams: list[StreamRef] = []
    for entry in sink_inputs:
        properties = entry.get("properties", {})
        application_name = properties.get("application.name")
        serial = properties.get("object.serial")
        if not application_name or not serial:
            continue
        sink_name = sink_names.get(str(entry.get("sink")))
        if not sink_name:
            continue
        streams.append(
            StreamRef(
                identifier=str(entry["index"]),
                application_name=application_name,
                media_name=properties.get("media.name", ""),
                serial=serial,
                sink_name=sink_name,
            )
        )
    return streams


class PipeWireStreamDiscovery:
    """A `streams: Callable[[], list[StreamRef]]` adapter for `AudioRouter`, backed by real pactl JSON output."""

    def __init__(self, pactl: Pactl) -> None:
        self._pactl = pactl

    def __call__(self) -> list[StreamRef]:
        sink_inputs_json = self._pactl.run(("pactl", "-f", "json", "list", "sink-inputs"))
        sinks_json = self._pactl.run(("pactl", "-f", "json", "list", "sinks"))
        return parse_streams(sink_inputs_json, sinks_json)


def _parse_inventory(payload: str, label: str) -> list[dict[str, object]]:
    try:
        decoded = json.loads(payload)
    except json.JSONDecodeError as error:
        raise MonitorCaptureError(f"malformed {label} JSON: {error}") from error
    if not isinstance(decoded, list) or not all(isinstance(entry, dict) for entry in decoded):
        raise MonitorCaptureError(f"malformed {label} inventory")
    return decoded


def _only(entries: list[dict[str, object]], label: str) -> dict[str, object]:
    if len(entries) != 1:
        raise MonitorCaptureError(f"{label} identity matched {len(entries)} entries; exactly one is required")
    return entries[0]


def resolve_owned_monitor(sources_json: str, sinks_json: str, modules_json: str, lease: RouteLease) -> MonitorRef:
    """Resolve one monitor whose source, sink, and null-sink module bind to a lease."""
    sources = _parse_inventory(sources_json, "source")
    sinks = _parse_inventory(sinks_json, "sink")
    modules = _parse_inventory(modules_json, "module")

    sink = _only([entry for entry in sinks if entry.get("name") == lease.virtual_sink], "virtual sink")
    # Real PipeWire-Pulse (confirmed live) leaves a source's monitor_of_sink unpopulated;
    # the sink's own monitor_source name is the reliable, authoritative link instead.
    monitor_source_name = sink.get("monitor_source")
    if not isinstance(monitor_source_name, str) or not monitor_source_name:
        raise MonitorCaptureError("virtual sink has no declared monitor source")

    monitor = _only(
        [entry for entry in sources if entry.get("name") == monitor_source_name],
        "monitor source",
    )
    name = monitor.get("name")
    properties = monitor.get("properties")
    serial = properties.get("object.serial") if isinstance(properties, dict) else None
    if not isinstance(name, str) or not name.endswith(".monitor") or not isinstance(serial, str) or not serial:
        raise MonitorCaptureError("monitor source has no stable monitor identity")

    # Real PipeWire-Pulse (confirmed live) module JSON entries have no "index" field at
    # all; name + the exact owned sink_name argument is the only available, and already
    # unique, identity check (our virtual_sink name is derived from the stream identifier).
    _only(
        [
            entry
            for entry in modules
            if entry.get("name") == "module-null-sink"
            and isinstance(entry.get("argument"), str)
            and f"sink_name={lease.virtual_sink}" in entry["argument"].split()
        ],
        "owned null-sink module",
    )
    return MonitorRef(name, serial, lease.virtual_sink, lease.virtual_module_id)


class OwnedMonitorDiscovery:
    """Refreshes PipeWire-Pulse JSON inventory before every monitor capture lease."""

    def __init__(self, pactl: Pactl) -> None:
        self._pactl = pactl

    def resolve(self, lease: RouteLease) -> MonitorRef:
        return resolve_owned_monitor(
            self._pactl.run(("pactl", "-f", "json", "list", "sources")),
            self._pactl.run(("pactl", "-f", "json", "list", "sinks")),
            self._pactl.run(("pactl", "-f", "json", "list", "modules")),
            lease,
        )


    def resolve_stream_monitor(self, stream: StreamRef) -> str:
        """Resolve the selected stream's original monitor, never a default source."""
        inputs = self._pactl.run(("pactl", "-f", "json", "list", "sink-inputs"))
        sinks_json = self._pactl.run(("pactl", "-f", "json", "list", "sinks"))
        current = [candidate for candidate in parse_streams(inputs, sinks_json)
                   if candidate.identifier == stream.identifier]
        if len(current) != 1:
            raise MonitorCaptureError("selected Chrome stream disappeared; stop and restart with the video playing")
        stream.require_same_identity(current[0])
        if current[0].sink_name != stream.sink_name or stream.sink_name.startswith("opendubstream"):
            raise MonitorCaptureError("selected stream output changed; stop and restart after restoring browser audio")
        sink = _only([entry for entry in _parse_inventory(sinks_json, "sink")
                      if entry.get("name") == stream.sink_name], "selected stream sink")
        name = sink.get("monitor_source")
        if not isinstance(name, str) or not name.endswith(".monitor"):
            raise MonitorCaptureError("selected stream sink has no explicit monitor source")
        return name


def resolve_reference_monitor(sources_json: str, sinks_json: str, modules_json: str, reference: ReferenceLease) -> MonitorRef:
    """Resolve one monitor whose source, sink, and null-sink module bind to a tool-owned
    `ReferenceLease` -- structurally identical to `resolve_owned_monitor`, but validating
    against the reference sink/module instead of a routed Chrome stream's virtual sink."""
    sources = _parse_inventory(sources_json, "source")
    sinks = _parse_inventory(sinks_json, "sink")
    modules = _parse_inventory(modules_json, "module")

    sink = _only([entry for entry in sinks if entry.get("name") == reference.reference_sink], "reference sink")
    monitor_source_name = sink.get("monitor_source")
    if not isinstance(monitor_source_name, str) or not monitor_source_name:
        raise MonitorCaptureError("reference sink has no declared monitor source")

    monitor = _only(
        [entry for entry in sources if entry.get("name") == monitor_source_name],
        "monitor source",
    )
    name = monitor.get("name")
    properties = monitor.get("properties")
    serial = properties.get("object.serial") if isinstance(properties, dict) else None
    if not isinstance(name, str) or not name.endswith(".monitor") or not isinstance(serial, str) or not serial:
        raise MonitorCaptureError("monitor source has no stable monitor identity")

    _only(
        [
            entry
            for entry in modules
            if entry.get("name") == "module-null-sink"
            and isinstance(entry.get("argument"), str)
            and f"sink_name={reference.reference_sink}" in entry["argument"].split()
        ],
        "owned reference null-sink module",
    )
    return MonitorRef(name, serial, reference.reference_sink, reference.reference_module_id)


class OwnedReferenceMonitorDiscovery:
    """Refreshes PipeWire-Pulse JSON inventory before every reference-sink monitor
    capture lease -- same refresh-before-every-lease pattern as `OwnedMonitorDiscovery`."""

    def __init__(self, pactl: Pactl) -> None:
        self._pactl = pactl

    def resolve(self, reference: ReferenceLease) -> MonitorRef:
        return resolve_reference_monitor(
            self._pactl.run(("pactl", "-f", "json", "list", "sources")),
            self._pactl.run(("pactl", "-f", "json", "list", "sinks")),
            self._pactl.run(("pactl", "-f", "json", "list", "modules")),
            reference,
        )
