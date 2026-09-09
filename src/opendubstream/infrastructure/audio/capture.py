"""Fail-closed, selected-monitor PCM capture boundary."""

from __future__ import annotations

import re
import select
import subprocess
import threading
import time
from collections.abc import Callable
from typing import Protocol

from opendubstream.domain.contracts import MonitorCaptureError, MonitorRef, ReferenceLease, RouteLease, StreamRef
from opendubstream.infrastructure.audio.discovery import OwnedMonitorDiscovery
from opendubstream.infrastructure.audio.journal import RecoveryJournal


class CaptureReadError(MonitorCaptureError):
    """A monitor reader could not return bounded PCM safely."""


class MonitorReader(Protocol):
    def read(self, deadline: float) -> bytes: ...

    def stop(self) -> None: ...


_SAFE_MONITOR_NAME = re.compile(r"^[A-Za-z0-9_.:-]+\.monitor$")


def build_parec_argv(monitor_name: str) -> tuple[str, ...]:
    """Build the only permitted monitor reader argv; never compose a shell string."""
    if not _SAFE_MONITOR_NAME.fullmatch(monitor_name):
        raise MonitorCaptureError("monitor name is not a literal safe capture endpoint")
    return (
        "parec", f"--device={monitor_name}", "--format=s16le", "--rate=16000", "--channels=1",
        "--latency-msec=100",
    )


def build_stream_parec_argv(monitor_name: str, stream_index: str) -> tuple[str, ...]:
    """A physical monitor is permitted only with an explicit, valid sink-input filter."""
    if not re.fullmatch(r"[0-9]+", stream_index) or not 0 <= int(stream_index) < 4294967295:
        raise MonitorCaptureError("selected stream index is invalid")
    return (*build_parec_argv(monitor_name), f"--monitor-stream={stream_index}")


class ParecMonitorReader:
    """Single child PCM reader with a bounded read and deterministic termination."""

    def __init__(self, monitor: MonitorRef | None, launch: Callable[[tuple[str, ...]], object] | None = None, *, argv_factory: Callable[[], tuple[str, ...]] | None = None) -> None:
        self._argv_factory = argv_factory
        self._monitor = monitor
        self._launch = launch or self._subprocess_launch
        self._child: object | None = None
        self._lock = threading.Lock()
        self._stopped = False
        self._partial_sample = b""

    def read(self, deadline: float) -> bytes:
        if deadline <= 0:
            raise TimeoutError()
        with self._lock:
            if self._stopped:
                raise CaptureReadError("monitor reader is stopped")
            if self._child is None:
                self._child = self._launch(self._argv_factory() if self._argv_factory else build_parec_argv(self._monitor.name))
            stdout = getattr(self._child, "stdout", None)
        if stdout is None:
            self.stop()
            raise CaptureReadError("parec child has no PCM stdout")
        until = time.monotonic() + deadline
        while True:
            with self._lock:
                if self._stopped:
                    raise CaptureReadError("monitor reader is stopped")
                child = self._child
            poll = getattr(child, "poll", None)
            if callable(poll) and poll() is not None:
                self.stop()
                raise CaptureReadError("parec reader exited")
            remaining = until - time.monotonic()
            if remaining <= 0:
                # A live producer may be paused. Preserve it and any partial sample.
                raise TimeoutError()
            ready, _, _ = select.select([stdout], [], [], min(remaining, 0.1))
            if ready:
                break
        try:
            pcm = getattr(stdout, "read1", stdout.read)(3200)
        except Exception:
            self.stop()
            raise
        if not isinstance(pcm, bytes) or not pcm:
            self.stop()
            raise CaptureReadError("parec reader returned no PCM")
        pcm = self._partial_sample + pcm
        complete = len(pcm) - len(pcm) % 2
        self._partial_sample = pcm[complete:]
        return pcm[:complete]

    def stop(self) -> None:
        with self._lock:
            self._stopped = True
            child, self._child = self._child, None
        if child is None:
            return
        terminate = getattr(child, "terminate", None)
        wait = getattr(child, "wait", None)
        kill = getattr(child, "kill", None)
        if callable(terminate):
            terminate()
        try:
            if callable(wait):
                wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            if callable(kill):
                kill()
            if callable(wait):
                wait(timeout=1.0)

    @staticmethod
    def _subprocess_launch(argv: tuple[str, ...]) -> subprocess.Popen[bytes]:
        return subprocess.Popen(argv, shell=False, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)


class PipeWireMonitorCapture:
    """Capture lifecycle that binds every reader to a freshly resolved owned monitor."""

    def __init__(
        self,
        discovery: OwnedMonitorDiscovery,
        reader_factory: Callable[[MonitorRef], MonitorReader] = ParecMonitorReader,
        *,
        journal: RecoveryJournal,
    ) -> None:
        self._discovery = discovery
        self._reader_factory = reader_factory
        self._journal = journal
        self._reader: MonitorReader | None = None
        self._monitor: MonitorRef | None = None
        self._lease: RouteLease | StreamRef | ReferenceLease | None = None

    def start(self, lease: RouteLease | StreamRef) -> MonitorRef | StreamRef:
        self.stop()
        self._lease = lease
        if isinstance(lease, StreamRef):
            if self._journal.load() is not None:
                raise MonitorCaptureError("audio recovery is pending; stop the existing dubbing session and restore browser audio before starting CC")
            # Resolve again at lazy process launch, not just when Start was clicked.
            def argv():
                return build_stream_parec_argv(self._discovery.resolve_stream_monitor(lease), lease.identifier)
            argv()  # Fail before worker launch on stale identity or missing monitor.
            self._reader = ParecMonitorReader(None, argv_factory=argv)
            return lease
        snapshot = self._journal.load()
        if (
            snapshot is None
            or snapshot.virtual_module_id is None
            or snapshot.virtual_module_id != lease.virtual_module_id
            or snapshot.virtual_sink != lease.virtual_sink
        ):
            raise MonitorCaptureError("journaled module identity does not match the active route lease")
        monitor = self._discovery.resolve(lease)
        if (
            monitor.owned_module_id != lease.virtual_module_id
            or monitor.owned_module_id != snapshot.virtual_module_id
            or monitor.monitored_sink != lease.virtual_sink
        ):
            raise MonitorCaptureError("resolved monitor does not match the active route lease")
        self._monitor = monitor
        self._reader = self._reader_factory(monitor)
        return monitor

    def start_reference(self, reference: ReferenceLease) -> MonitorRef:
        """Same validation and lifecycle as `start()`, mirrored exactly against a tool-owned
        `ReferenceLease` instead of a routed Chrome stream's `RouteLease`; does not weaken or
        alter `start()`'s existing validation for the Chrome-stream capture path."""
        self.stop()
        self._lease = reference
        snapshot = self._journal.load()
        if (
            snapshot is None
            or snapshot.reference_module_id is None
            or snapshot.reference_module_id != reference.reference_module_id
            or snapshot.reference_sink != reference.reference_sink
        ):
            raise MonitorCaptureError("journaled reference module identity does not match the active reference lease")
        monitor = self._discovery.resolve(reference)
        if (
            monitor.owned_module_id != reference.reference_module_id
            or monitor.owned_module_id != snapshot.reference_module_id
            or monitor.monitored_sink != reference.reference_sink
        ):
            raise MonitorCaptureError("resolved monitor does not match the active reference lease")
        self._monitor = monitor
        self._reader = self._reader_factory(monitor)
        return monitor

    def read_phrase(self, deadline: float) -> bytes:
        if self._reader is None:
            raise CaptureReadError("monitor capture has not started")
        try:
            return self._reader.read(deadline)
        except TimeoutError:
            # No PCM is not proof of failure. Recheck identity before allowing idle;
            # never fall back to an unfiltered monitor or silently follow a new stream.
            try:
                if isinstance(self._lease, StreamRef):
                    self._discovery.resolve_stream_monitor(self._lease)
                elif self._lease is not None:
                    if self._discovery.resolve(self._lease) != self._monitor:
                        raise CaptureReadError("capture monitor identity changed")
            except Exception:
                self.stop()
                raise
            raise
        except Exception:
            self.stop()
            raise

    def stop(self) -> None:
        reader, self._reader = self._reader, None
        self._monitor = None
        self._lease = None
        if reader is not None:
            reader.stop()
