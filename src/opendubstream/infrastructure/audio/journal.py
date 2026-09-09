"""Durable routing recovery state written before PipeWire-Pulse mutations."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from opendubstream.domain.contracts import StreamRef


@dataclass(frozen=True)
class RouteSnapshot:
    stream: StreamRef
    original_sink: str
    virtual_sink: str
    virtual_module_id: str | None = None
    reference_sink: str | None = None
    reference_module_id: str | None = None
    loopback_module_id: str | None = None
    original_loopback_module_id: str | None = None
    original_loopback_sink_input_id: str | None = None
    original_loopback_physical_sink: str | None = None


class RecoveryJournal:
    """An fsynced JSON journal that remains until recovery is verified."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def persist(self, snapshot: RouteSnapshot) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(asdict(snapshot), sort_keys=True, separators=(",", ":")).encode("utf-8")
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{self._path.name}.", dir=self._path.parent)
        try:
            with os.fdopen(descriptor, "wb") as temporary:
                temporary.write(payload)
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_name, self._path)
            self._fsync_directory()
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)

    def load(self) -> RouteSnapshot | None:
        if not self._path.exists():
            return None
        raw = json.loads(self._path.read_text(encoding="utf-8"))
        stream = StreamRef(**raw["stream"])
        return RouteSnapshot(
            stream=stream,
            original_sink=raw["original_sink"],
            virtual_sink=raw["virtual_sink"],
            virtual_module_id=raw.get("virtual_module_id"),
            reference_sink=raw.get("reference_sink"),
            reference_module_id=raw.get("reference_module_id"),
            loopback_module_id=raw.get("loopback_module_id"),
            original_loopback_module_id=raw.get("original_loopback_module_id"),
            original_loopback_sink_input_id=raw.get("original_loopback_sink_input_id"),
            original_loopback_physical_sink=raw.get("original_loopback_physical_sink"),
        )

    def clear(self) -> None:
        if self._path.exists():
            self._path.unlink()
            self._fsync_directory()

    def _fsync_directory(self) -> None:
        descriptor = os.open(self._path.parent, os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
