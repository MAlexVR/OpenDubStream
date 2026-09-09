"""Fail-closed contracts for isolated browser audio routing."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, Sequence


class RoutingContractError(RuntimeError):
    """Base error for a routing contract that cannot safely continue."""


class StreamSelectionError(RoutingContractError):
    """A selector did not identify exactly one stream."""


class StreamIdentityChanged(RoutingContractError):
    """A selected stream was replaced before the route could be mutated."""


class PhysicalSinkRequired(RoutingContractError):
    """Generated audio was not directed to a verified physical sink."""


class MonitorCaptureError(RoutingContractError):
    """A selected virtual-sink monitor cannot be proven safe to capture."""


@dataclass(frozen=True)
class StreamSelector:
    application_name: str
    serial: str | None = None
    media_name: str | None = None

    def matches(self, stream: "StreamRef") -> bool:
        return (
            stream.application_name == self.application_name
            and (self.serial is None or stream.serial == self.serial)
            and (self.media_name is None or stream.media_name == self.media_name)
        )


@dataclass(frozen=True)
class StreamRef:
    identifier: str
    application_name: str
    media_name: str
    serial: str
    sink_name: str

    def require_same_identity(self, current: "StreamRef") -> None:
        if self.identifier != current.identifier or self.serial != current.serial:
            raise StreamIdentityChanged("selected browser stream identity changed before routing")


@dataclass(frozen=True)
class SinkRef:
    name: str
    is_physical: bool
    serial: str = ""


class AudioMode(str, Enum):
    """Local presentation modes; translation_only remains a legacy mix value."""

    CAPTIONS = "captions"
    BOTH = "both"
    DUB = "dub"
    TRANSLATION_ONLY = "translation_only"


@dataclass(frozen=True)
class AudioMixSettings:
    """Editable per-session gain policy; it never represents a global PipeWire default."""

    mode: AudioMode = AudioMode.DUB
    original_volume: int = 20
    dub_volume: int = 100

    def __post_init__(self) -> None:
        for value in (self.original_volume, self.dub_volume):
            if not isinstance(value, int) or not 0 <= value <= 100:
                raise ValueError("mix volumes must be integer percentages from 0 through 100")


@dataclass(frozen=True)
class OriginalMixLease:
    """Only the sink-input created by this session may be locally muted or amplified."""

    loopback_module_id: str
    sink_input_id: str
    physical_sink: str


@dataclass(frozen=True)
class RouteLease:
    stream: StreamRef
    virtual_sink: str
    virtual_module_id: str
    original_sink: str
    physical_sink: str


@dataclass(frozen=True)
class ReferenceLease:
    """Identity of a tool-owned reference null-sink bridged to a physical sink via a
    `module-loopback`, used to measure transport offset without capturing a monitor
    this tool did not create (see `PipeWirePulseRouter.open_reference_sink`)."""

    reference_sink: str
    reference_module_id: str
    loopback_module_id: str
    physical_sink: str


@dataclass(frozen=True)
class MonitorRef:
    """Stable identity of the sole application-owned monitor capture source."""

    name: str
    serial: str
    monitored_sink: str
    owned_module_id: str


@dataclass(frozen=True)
class RecoveryResult:
    restored: bool
    detail: str


@dataclass(frozen=True)
class EndpointEvidence:
    """Proof that the real capture->playback->recovery loop actually ran, at one revision."""

    revision: str
    monitor_captured: bool
    physical_played: bool
    feedback_absent: bool
    routing_recovered: bool
    tdd_receipts_complete: bool

    def complete(self) -> bool:
        return self.monitor_captured and self.physical_played and self.feedback_absent and self.routing_recovered and self.tdd_receipts_complete


class AudioRouter(Protocol):
    def select_unique(self, selector: StreamSelector) -> StreamRef: ...

    def begin(self, stream: StreamRef, physical: SinkRef) -> RouteLease: ...

    def recover(self) -> RecoveryResult: ...


def select_unique_stream(selector: StreamSelector, streams: Sequence[StreamRef]) -> StreamRef:
    matches = [stream for stream in streams if selector.matches(stream)]
    if len(matches) != 1:
        raise StreamSelectionError(f"stream selection found {len(matches)} matches; exactly one is required")
    return matches[0]
