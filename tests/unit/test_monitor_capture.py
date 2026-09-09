from __future__ import annotations

import json

import pytest

from opendubstream.domain.contracts import MonitorCaptureError, MonitorRef, ReferenceLease, RouteLease, StreamRef
from opendubstream.infrastructure.audio.capture import (
    CaptureReadError,
    ParecMonitorReader,
    PipeWireMonitorCapture,
    build_parec_argv,
)
from opendubstream.infrastructure.audio.discovery import (
    OwnedMonitorDiscovery,
    OwnedReferenceMonitorDiscovery,
    resolve_owned_monitor,
    resolve_reference_monitor,
)
from opendubstream.infrastructure.audio.journal import RouteSnapshot


def lease() -> RouteLease:
    return RouteLease(
        stream=StreamRef("41", "Google Chrome", "Video", "chrome-a", "alsa_output.speakers"),
        virtual_sink="opendubstream.41",
        virtual_module_id="77",
        original_sink="alsa_output.speakers",
        physical_sink="alsa_output.speakers",
    )


def inventory(
    *,
    source: dict[str, object] | None = None,
    sink: dict[str, object] | None = None,
    modules: list[dict[str, object]] | None = None,
) -> tuple[str, str, str]:
    # Real PipeWire-Pulse (confirmed live, 2026-09-03) never populates a source's
    # monitor_of_sink field; the sink's own monitor_source name is the reliable link.
    source = source or {
        "name": "opendubstream.41.monitor",
        "monitor_of_sink": None,
        "properties": {"object.serial": "source-serial"},
    }
    sink = sink or {"index": 12, "name": "opendubstream.41", "monitor_source": "opendubstream.41.monitor"}
    # Real PipeWire-Pulse (confirmed live, 2026-09-03) module entries have no "index" key.
    modules = modules if modules is not None else [
        {"name": "module-null-sink", "argument": "sink_name=opendubstream.41"}
    ]
    return (json.dumps([source]), json.dumps([sink]), json.dumps(modules))


def reference() -> ReferenceLease:
    return ReferenceLease(
        reference_sink="opendubstream-ref.41",
        reference_module_id="99",
        loopback_module_id="100",
        physical_sink="alsa_output.speakers",
    )


def reference_inventory(
    *,
    source: dict[str, object] | None = None,
    sink: dict[str, object] | None = None,
    modules: list[dict[str, object]] | None = None,
) -> tuple[str, str, str]:
    source = source or {
        "name": "opendubstream-ref.41.monitor",
        "monitor_of_sink": None,
        "properties": {"object.serial": "ref-source-serial"},
    }
    sink = sink or {"index": 20, "name": "opendubstream-ref.41", "monitor_source": "opendubstream-ref.41.monitor"}
    modules = modules if modules is not None else [
        {"name": "module-null-sink", "argument": "sink_name=opendubstream-ref.41"}
    ]
    return (json.dumps([source]), json.dumps([sink]), json.dumps(modules))


def test_resolves_only_monitor_bound_to_the_owned_reference_lease() -> None:
    sources, sinks, modules = reference_inventory()

    assert resolve_reference_monitor(sources, sinks, modules, reference()) == MonitorRef(
        name="opendubstream-ref.41.monitor",
        serial="ref-source-serial",
        monitored_sink="opendubstream-ref.41",
        owned_module_id="99",
    )


@pytest.mark.parametrize(
    ("sources", "sinks", "modules", "message"),
    [
        ("[]", reference_inventory()[1], reference_inventory()[2], "monitor"),
        (
            reference_inventory()[0],
            json.dumps([{"index": 20, "name": "alsa_output.speakers"}]),
            reference_inventory()[2],
            "reference sink",
        ),
        (
            reference_inventory()[0],
            reference_inventory()[1],
            json.dumps([]),
            "module",
        ),
        (
            reference_inventory()[0],
            reference_inventory()[1],
            json.dumps([{"name": "module-null-sink", "argument": "sink_name=opendubstream-ref.other-stream"}]),
            "module",
        ),
    ],
)
def test_resolve_reference_monitor_rejects_missing_stale_or_non_owned_inventory(
    sources: str, sinks: str, modules: str, message: str
) -> None:
    with pytest.raises(MonitorCaptureError, match=message):
        resolve_reference_monitor(sources, sinks, modules, reference())


def test_resolve_reference_monitor_rejects_malformed_json() -> None:
    with pytest.raises(MonitorCaptureError, match="malformed"):
        resolve_reference_monitor("not json", reference_inventory()[1], reference_inventory()[2], reference())


class FakeReferencePactl:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def run(self, argv: tuple[str, ...], **_: object) -> str:
        self.calls.append(argv)
        sources, sinks, modules = reference_inventory()
        if argv[-1] == "sources":
            return sources
        if argv[-1] == "sinks":
            return sinks
        return modules


def test_owned_reference_monitor_discovery_refreshes_inventory_before_every_lease() -> None:
    pactl = FakeReferencePactl()
    discovery = OwnedReferenceMonitorDiscovery(pactl)

    assert discovery.resolve(reference()) == MonitorRef(
        name="opendubstream-ref.41.monitor",
        serial="ref-source-serial",
        monitored_sink="opendubstream-ref.41",
        owned_module_id="99",
    )
    assert pactl.calls == [
        ("pactl", "-f", "json", "list", "sources"),
        ("pactl", "-f", "json", "list", "sinks"),
        ("pactl", "-f", "json", "list", "modules"),
    ]


def test_resolves_only_monitor_bound_to_the_owned_lease() -> None:
    sources, sinks, modules = inventory()

    assert resolve_owned_monitor(sources, sinks, modules, lease()) == MonitorRef(
        name="opendubstream.41.monitor",
        serial="source-serial",
        monitored_sink="opendubstream.41",
        owned_module_id="77",
    )


@pytest.mark.parametrize(
    ("sources", "sinks", "modules", "message"),
    [
        ("[]", inventory()[1], inventory()[2], "monitor"),
        (
            json.dumps([
                {"name": "opendubstream.41.monitor", "monitor_of_sink": None, "properties": {"object.serial": "one"}},
                {"name": "opendubstream.41.monitor", "monitor_of_sink": None, "properties": {"object.serial": "two"}},
            ]),
            inventory()[1],
            inventory()[2],
            "monitor",
        ),
        (inventory()[0], json.dumps([{"index": 12, "name": "alsa_output.speakers"}]), inventory()[2], "virtual sink"),
        (
            json.dumps([{"name": "alsa_input.mic", "properties": {"object.serial": "mic"}}]),
            inventory()[1],
            inventory()[2],
            "monitor",
        ),
        (
            inventory()[0],
            json.dumps([{"index": 12, "name": "opendubstream.41"}]),  # no monitor_source field
            inventory()[2],
            "monitor",
        ),
        (inventory()[0], inventory()[1], json.dumps([]), "module"),
        (
            inventory()[0],
            inventory()[1],
            json.dumps([{"name": "module-null-sink", "argument": "sink_name=opendubstream.other-stream"}]),
            "module",
        ),
    ],
)
def test_rejects_missing_ambiguous_stale_physical_non_monitor_or_non_owned_inventory(
    sources: str, sinks: str, modules: str, message: str
) -> None:
    with pytest.raises(MonitorCaptureError, match=message):
        resolve_owned_monitor(sources, sinks, modules, lease())


def test_rejects_malformed_json_before_reader_launch() -> None:
    with pytest.raises(MonitorCaptureError, match="malformed"):
        resolve_owned_monitor("not json", inventory()[1], inventory()[2], lease())


def test_parec_argv_is_literal_and_rejects_injected_monitor_name() -> None:
    """Confirmed live (2026-09-03, parec(1)/PulseAudio LatencyControl docs): a client that
    doesn't request an explicit latency gets the server's default, "usually relatively high
    for power saving reasons" -- observed as several seconds of first-connection delay on a
    freshly-created null-sink. `--latency-msec` pins a low, predictable connection latency."""
    assert build_parec_argv("opendubstream.41.monitor") == (
        "parec", "--device=opendubstream.41.monitor", "--format=s16le", "--rate=16000", "--channels=1",
        "--latency-msec=100",
    )
    with pytest.raises(MonitorCaptureError, match="literal"):
        build_parec_argv("opendubstream.41.monitor;id")


class FakeDiscovery:
    def __init__(self, monitor: MonitorRef) -> None:
        self.monitor = monitor
        self.leases: list[RouteLease] = []

    def resolve(self, route_lease: RouteLease) -> MonitorRef:
        self.leases.append(route_lease)
        return self.monitor


class FakeReader:
    def __init__(self, result: bytes | Exception) -> None:
        self.result = result
        self.deadlines: list[float] = []
        self.stop_calls = 0

    def read(self, deadline: float) -> bytes:
        self.deadlines.append(deadline)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result

    def stop(self) -> None:
        self.stop_calls += 1


class FakeJournal:
    def __init__(self, snapshot: RouteSnapshot | None) -> None:
        self.snapshot = snapshot

    def load(self) -> RouteSnapshot | None:
        return self.snapshot


def test_capture_reads_only_resolved_monitor_and_stops_child() -> None:
    monitor = MonitorRef("opendubstream.41.monitor", "source-serial", "opendubstream.41", "77")
    discovery = FakeDiscovery(monitor)
    reader = FakeReader(b"pcm")
    capture = PipeWireMonitorCapture(
        discovery,
        lambda reference: reader,
        journal=FakeJournal(
            RouteSnapshot(lease().stream, lease().original_sink, lease().virtual_sink, lease().virtual_module_id)
        ),
    )

    assert capture.start(lease()) == monitor
    assert capture.read_phrase(0.25) == b"pcm"
    capture.stop()

    assert discovery.leases == [lease()]
    assert reader.deadlines == [0.25]
    assert reader.stop_calls == 1


def test_capture_idle_timeout_preserves_child_and_validates_monitor() -> None:
    monitor = MonitorRef("opendubstream.41.monitor", "source-serial", "opendubstream.41", "77")
    reader = FakeReader(TimeoutError())
    capture = PipeWireMonitorCapture(
        FakeDiscovery(monitor),
        lambda reference: reader,
        journal=FakeJournal(
            RouteSnapshot(lease().stream, lease().original_sink, lease().virtual_sink, lease().virtual_module_id)
        ),
    )
    capture.start(lease())

    with pytest.raises(TimeoutError):
        capture.read_phrase(0.25)

    assert reader.stop_calls == 0


@pytest.mark.parametrize("module_id", [None, "stale-module"])
def test_capture_rejects_missing_or_stale_persisted_module_before_reader_launch(module_id: str | None) -> None:
    monitor = MonitorRef("opendubstream.41.monitor", "source-serial", "opendubstream.41", "77")
    reader = FakeReader(b"pcm")
    snapshot = None if module_id is None else RouteSnapshot(
        stream=lease().stream,
        original_sink=lease().original_sink,
        virtual_sink=lease().virtual_sink,
        virtual_module_id=module_id,
    )
    capture = PipeWireMonitorCapture(FakeDiscovery(monitor), lambda reference: reader, journal=FakeJournal(snapshot))

    with pytest.raises(MonitorCaptureError, match="journaled module"):
        capture.start(lease())

    assert reader.stop_calls == 0


def test_capture_start_reference_reads_only_the_resolved_reference_monitor_and_stops_child() -> None:
    monitor = MonitorRef("opendubstream-ref.41.monitor", "ref-source-serial", "opendubstream-ref.41", "99")
    discovery = FakeDiscovery(monitor)
    reader = FakeReader(b"pcm")
    capture = PipeWireMonitorCapture(
        discovery,
        lambda reference_ref: reader,
        journal=FakeJournal(
            RouteSnapshot(
                lease().stream,
                lease().original_sink,
                lease().virtual_sink,
                lease().virtual_module_id,
                reference_sink=reference().reference_sink,
                reference_module_id=reference().reference_module_id,
            )
        ),
    )

    assert capture.start_reference(reference()) == monitor
    assert capture.read_phrase(0.25) == b"pcm"
    capture.stop()

    assert discovery.leases == [reference()]
    assert reader.deadlines == [0.25]
    assert reader.stop_calls == 1


@pytest.mark.parametrize("reference_module_id", [None, "stale-module"])
def test_capture_start_reference_rejects_missing_or_stale_persisted_reference_module(
    reference_module_id: str | None,
) -> None:
    monitor = MonitorRef("opendubstream-ref.41.monitor", "ref-source-serial", "opendubstream-ref.41", "99")
    reader = FakeReader(b"pcm")
    snapshot = (
        None
        if reference_module_id is None
        else RouteSnapshot(
            stream=lease().stream,
            original_sink=lease().original_sink,
            virtual_sink=lease().virtual_sink,
            virtual_module_id=lease().virtual_module_id,
            reference_sink=reference().reference_sink,
            reference_module_id=reference_module_id,
        )
    )
    capture = PipeWireMonitorCapture(FakeDiscovery(monitor), lambda reference_ref: reader, journal=FakeJournal(snapshot))

    with pytest.raises(MonitorCaptureError, match="journaled reference module"):
        capture.start_reference(reference())

    assert reader.stop_calls == 0


def test_capture_start_reference_rejects_a_resolved_monitor_that_does_not_match_the_reference_lease() -> None:
    wrong_monitor = MonitorRef("opendubstream-ref.other.monitor", "other-serial", "opendubstream-ref.other", "99")
    reader = FakeReader(b"pcm")
    capture = PipeWireMonitorCapture(
        FakeDiscovery(wrong_monitor),
        lambda reference_ref: reader,
        journal=FakeJournal(
            RouteSnapshot(
                lease().stream,
                lease().original_sink,
                lease().virtual_sink,
                lease().virtual_module_id,
                reference_sink=reference().reference_sink,
                reference_module_id=reference().reference_module_id,
            )
        ),
    )

    with pytest.raises(MonitorCaptureError, match="resolved monitor"):
        capture.start_reference(reference())


def test_capture_start_ignores_reference_fields_already_present_in_the_journal() -> None:
    """A journal that has already run `open_reference_sink` still lets the Chrome-stream
    `start()` path work exactly as before -- the extra reference/loopback fields are
    simply irrelevant to `start()`'s own validation."""
    monitor = MonitorRef("opendubstream.41.monitor", "source-serial", "opendubstream.41", "77")
    discovery = FakeDiscovery(monitor)
    reader = FakeReader(b"pcm")
    capture = PipeWireMonitorCapture(
        discovery,
        lambda reference_ref: reader,
        journal=FakeJournal(
            RouteSnapshot(
                lease().stream,
                lease().original_sink,
                lease().virtual_sink,
                lease().virtual_module_id,
                reference_sink=reference().reference_sink,
                reference_module_id=reference().reference_module_id,
                loopback_module_id=reference().loopback_module_id,
            )
        ),
    )

    assert capture.start(lease()) == monitor


def test_parec_reader_keeps_live_child_when_select_times_out(monkeypatch: pytest.MonkeyPatch) -> None:
    class Child:
        stdout = object()

        def __init__(self) -> None:
            self.terminated = 0
            self.waited = 0

        def terminate(self) -> None:
            self.terminated += 1

        def wait(self, timeout: float) -> None:
            self.waited += 1

    child = Child()
    monkeypatch.setattr("opendubstream.infrastructure.audio.capture.select.select", lambda *_: ([], [], []))
    reader = ParecMonitorReader(
        MonitorRef("opendubstream.41.monitor", "source-serial", "opendubstream.41", "77"),
        launch=lambda argv: child,
    )

    with pytest.raises(TimeoutError):
        reader.read(0.25)

    assert child.terminated == 0
    assert child.waited == 0


def test_parec_reader_terminates_injected_child_when_stdout_read_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    class Stdout:
        def read(self, count: int) -> bytes:
            raise OSError("reader lost")

    class Child:
        stdout = Stdout()

        def __init__(self) -> None:
            self.terminated = 0
            self.waited = 0

        def terminate(self) -> None:
            self.terminated += 1

        def wait(self, timeout: float) -> None:
            self.waited += 1

    child = Child()
    monkeypatch.setattr("opendubstream.infrastructure.audio.capture.select.select", lambda *_: ([child.stdout], [], []))
    reader = ParecMonitorReader(
        MonitorRef("opendubstream.41.monitor", "source-serial", "opendubstream.41", "77"),
        launch=lambda argv: child,
    )

    with pytest.raises(OSError, match="reader lost"):
        reader.read(0.25)

    assert child.terminated == 1
    assert child.waited == 1


def test_stopped_monitor_reader_cannot_restart_a_child():
    launches = []
    monitor = MonitorRef('opendubstream.41.monitor', '1', 'opendubstream.41', '77')
    reader = ParecMonitorReader(monitor, launch=lambda argv: launches.append(argv))
    reader.stop()
    with pytest.raises(CaptureReadError, match='stopped'):
        reader.read(1)
    assert launches == []


def test_parec_reader_reassembles_partial_samples(monkeypatch):
    class Stdout:
        chunks = [b'\x01', b'\x02\x03\x04']
        def read1(self, size): return self.chunks.pop(0)
        read = read1
    class Child:
        stdout = Stdout()
    monitor = MonitorRef('opendubstream.41.monitor', '1', 'opendubstream.41', '77')
    monkeypatch.setattr('opendubstream.infrastructure.audio.capture.select.select', lambda *args: ([Child.stdout], [], []))
    reader = ParecMonitorReader(monitor, lambda argv: Child())
    assert reader.read(1) == b''
    assert reader.read(1) == b'\x01\x02\x03\x04'


def test_stream_monitor_argv_is_always_explicitly_per_stream():
    from opendubstream.infrastructure.audio.capture import build_stream_parec_argv
    assert '--monitor-stream=41' in build_stream_parec_argv('alsa_output.speakers.monitor', '41')
    for bad in ('', '-1', 'default', '4294967295', '41;echo'):
        with pytest.raises(MonitorCaptureError):
            build_stream_parec_argv('alsa_output.speakers.monitor', bad)


class StreamPactl:
    def __init__(self):
        self.calls = []
        self.inputs = [{'index': 41, 'sink': 2, 'properties': {
            'application.name': 'Google Chrome', 'object.serial': 'chrome-a', 'media.name': 'Video'}}]
        self.sinks = [{'index': 2, 'name': 'alsa_output.speakers',
                       'monitor_source': 'alsa_output.speakers.monitor'}]
    def run(self, argv):
        self.calls.append(argv)
        return json.dumps(self.inputs if argv[-1] == 'sink-inputs' else self.sinks)


def test_passive_cc_reader_binds_original_sink_and_only_selected_stream(monkeypatch):
    pactl = StreamPactl()
    capture = PipeWireMonitorCapture(OwnedMonitorDiscovery(pactl), journal=FakeJournal(None))
    launched = []
    class Stdout:
        def read(self, count): return b'\x01\x02'
    class Child:
        stdout = Stdout()
        def terminate(self): pass
        def wait(self, timeout): pass
    monkeypatch.setattr(ParecMonitorReader, '_subprocess_launch', staticmethod(lambda argv: launched.append(argv) or Child()))
    monkeypatch.setattr('opendubstream.infrastructure.audio.capture.select.select', lambda *_: ([Child.stdout], [], []))
    capture.start(lease().stream)
    assert capture.read_phrase(1) == b'\x01\x02'
    assert '--device=alsa_output.speakers.monitor' in launched[0]
    assert '--monitor-stream=41' in launched[0]
    assert len(pactl.calls) == 4  # Validation at Start and again at child launch.
    assert all(call[:4] == ('pactl', '-f', 'json', 'list') for call in pactl.calls)
    capture.stop()


@pytest.mark.parametrize('change', ['disappeared', 'reused', 'moved', 'monitor_missing', 'owned'])
def test_passive_cc_never_falls_back_on_missing_or_stale_stream(change, monkeypatch):
    from opendubstream.domain.contracts import RoutingContractError
    pactl = StreamPactl()
    capture = PipeWireMonitorCapture(OwnedMonitorDiscovery(pactl), journal=FakeJournal(None))
    capture.start(lease().stream)
    if change == 'disappeared': pactl.inputs.clear()
    if change == 'reused': pactl.inputs[0]['properties']['object.serial'] = 'replacement'
    if change == 'moved': pactl.sinks[0]['name'] = 'other'
    if change == 'monitor_missing': del pactl.sinks[0]['monitor_source']
    if change == 'owned': pactl.sinks[0]['name'] = 'opendubstream.41'
    launched = []
    monkeypatch.setattr(ParecMonitorReader, '_subprocess_launch', staticmethod(lambda argv: launched.append(argv)))
    with pytest.raises(RoutingContractError): capture.read_phrase(1)
    assert launched == []


def test_passive_cc_rejects_unrestored_owned_routing_journal():
    snapshot = RouteSnapshot(lease().stream, lease().original_sink, lease().virtual_sink, '77')
    pactl = StreamPactl()
    capture = PipeWireMonitorCapture(OwnedMonitorDiscovery(pactl), journal=FakeJournal(snapshot))
    with pytest.raises(MonitorCaptureError, match='recovery is pending'):
        capture.start(lease().stream)
    assert pactl.calls == []


def test_phrase_deadline_with_live_pipe_is_idle_then_resumes_without_restart():
    import os
    from opendubstream.infrastructure.audio.phrases import PhraseCapture
    read_fd, write_fd = os.pipe()
    class Child:
        stdout = os.fdopen(read_fd, 'rb', buffering=0)
        terminated = False
        def poll(self): return None
        def terminate(self): self.terminated = True
        def wait(self, timeout): pass
    child = Child()
    monitor = MonitorRef('opendubstream.41.monitor', 'source-serial', 'opendubstream.41', '77')
    raw = PipeWireMonitorCapture(FakeDiscovery(monitor),
        lambda _: ParecMonitorReader(monitor, lambda _: child),
        journal=FakeJournal(RouteSnapshot(lease().stream, lease().original_sink, lease().virtual_sink, '77')))
    capture = PhraseCapture(raw)
    capture.start(lease())
    try:
        for _ in range(2):
            assert capture.read_phrase(.02) == b''
            assert not child.terminated
        os.write(write_fd, b'\x00\x10' * 320)
        assert capture.read_phrase(.02) == b''
        assert capture.read_phrase(.33) == b'\x00\x10' * 320
        assert not child.terminated
    finally:
        capture.stop()
        child.stdout.close()
        os.close(write_fd)


def test_live_reader_stop_interrupts_long_select_wait():
    import os
    import threading
    import time
    read_fd, write_fd = os.pipe()
    entered = threading.Event()
    class Child:
        stdout = os.fdopen(read_fd, 'rb', buffering=0)
        def poll(self): return None
        def terminate(self): pass
        def wait(self, timeout): pass
    child = Child()
    reader = ParecMonitorReader(None, lambda _: entered.set() or child, argv_factory=lambda: ('fake',))
    errors = []
    def read():
        try: reader.read(5)
        except CaptureReadError as error: errors.append(error)
    worker = threading.Thread(target=read)
    worker.start()
    assert entered.wait(1)
    reader.stop()
    worker.join(.5)
    try:
        assert not worker.is_alive()
        assert errors
    finally:
        os.close(write_fd)
        worker.join(1)
        child.stdout.close()


@pytest.mark.parametrize('change', ['disappeared', 'reused', 'moved'])
def test_passive_cc_revalidates_stream_on_idle_timeout(change, monkeypatch):
    from opendubstream.domain.contracts import RoutingContractError
    pactl = StreamPactl()
    capture = PipeWireMonitorCapture(OwnedMonitorDiscovery(pactl), journal=FakeJournal(None))
    capture.start(lease().stream)
    reader = FakeReader(TimeoutError())
    capture._reader = reader
    if change == 'disappeared': pactl.inputs.clear()
    if change == 'reused': pactl.inputs[0]['properties']['object.serial'] = 'replacement'
    if change == 'moved': pactl.sinks[0]['name'] = 'other'
    with pytest.raises(RoutingContractError): capture.read_phrase(.01)
    assert reader.stop_calls == 1


def test_reader_checks_child_exit_after_empty_select(monkeypatch):
    class Child:
        stdout = object()
        exited = False
        stopped = False
        def poll(self): return 1 if self.exited else None
        def terminate(self): self.stopped = True
        def wait(self, timeout): pass
    child = Child()
    def select(*args):
        child.exited = True
        return [], [], []
    monkeypatch.setattr('opendubstream.infrastructure.audio.capture.select.select', select)
    reader = ParecMonitorReader(None, lambda _: child, argv_factory=lambda: ('fake',))
    with pytest.raises(CaptureReadError, match='exited'): reader.read(.01)
    assert child.stopped


def test_pipe_eof_is_fatal_not_idle():
    import os
    read_fd, write_fd = os.pipe()
    class Child:
        stdout = os.fdopen(read_fd, 'rb', buffering=0)
        stopped = False
        def poll(self): return None
        def terminate(self): self.stopped = True
        def wait(self, timeout): pass
    child = Child()
    os.close(write_fd)
    reader = ParecMonitorReader(None, lambda _: child, argv_factory=lambda: ('fake',))
    try:
        with pytest.raises(CaptureReadError, match='no PCM'): reader.read(.01)
        assert child.stopped
    finally:
        child.stdout.close()


def test_partial_sample_survives_idle_timeout_without_byte_loss():
    import os
    read_fd, write_fd = os.pipe()
    class Child:
        stdout = os.fdopen(read_fd, 'rb', buffering=0)
        def poll(self): return None
        def terminate(self): pass
        def wait(self, timeout): pass
    child = Child()
    reader = ParecMonitorReader(None, lambda _: child, argv_factory=lambda: ('fake',))
    try:
        os.write(write_fd, b'\x01')
        assert reader.read(.01) == b''
        with pytest.raises(TimeoutError): reader.read(.01)
        os.write(write_fd, b'\x02\x03\x04')
        assert reader.read(.01) == b'\x01\x02\x03\x04'
    finally:
        reader.stop()
        child.stdout.close()
        os.close(write_fd)
