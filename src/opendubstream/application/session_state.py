"""Pure session-lifecycle state machine.

Per design.md: "the application layer emits message keys plus parameters, never prose, and
a pure state machine converts observer events into `SessionStatus`." `SessionEvent` is a
single reported transition (possibly partial -- e.g. a `PROCESSING` event carries no
transcript); `SessionStatus` is the accumulated, renderable state a UI observes. Fields a
new event omits (`transcript`/`translation`/`diagnostics`) carry forward from the previous
status rather than being cleared.

`RECOVERY_FAILED` is a sticky terminal state (per the "Recovery failure" design decision):
once entered, every event except a fresh Start (a `ROUTING` event) is ignored, so the
journal stays deliberately unreconciled until the next session genuinely restarts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Mapping, Protocol

from opendubstream.ui.resources import MessageKey


class SessionStage(StrEnum):
    IDLE = "idle"
    ARMED = "armed"
    ROUTING = "routing"
    CAPTURING = "capturing"
    PROCESSING = "processing"
    PLAYING = "playing"
    STOPPING = "stopping"
    RECOVERING = "recovering"
    FAILED = "failed"
    RECOVERY_FAILED = "recovery-failed"


@dataclass(frozen=True)
class Diagnostics:
    timings_ms: Mapping[str, float] = field(default_factory=dict)
    active_operation: str | None = None
    execution_mode: str | None = None
    backlog: int = 0
    overload_dropped: int = 0
    stale_dropped: int = 0
    last_routing_error: str | None = None
    recovery_restored: bool | None = None
    recovery_detail: str | None = None


@dataclass(frozen=True)
class SessionEvent:
    stage: SessionStage
    message: MessageKey | None = None
    params: Mapping[str, str] = field(default_factory=dict)
    transcript: str | None = None
    translation: str | None = None
    diagnostics: Diagnostics | None = None


class SessionObserver(Protocol):
    """Callback boundary a UI (or test double) implements to receive `SessionEvent`s as a
    session runs. Added in Phase 2 (`application/dubbing_session.py`'s `DubbingSessionRunner`
    is its sole producer so far) alongside the `SessionEvent` it wraps -- design.md's
    Interfaces/Contracts section specifies this protocol but did not attribute it to a file;
    it lives here, beside `SessionEvent`, rather than being redefined in `dubbing_session.py`.
    No existing Phase 1 symbol in this module changes."""

    def on_event(self, event: SessionEvent) -> None: ...


@dataclass(frozen=True)
class SessionStatus:
    stage: SessionStage
    message: MessageKey | None = None
    params: Mapping[str, str] = field(default_factory=dict)
    transcript: str | None = None
    translation: str | None = None
    diagnostics: Diagnostics | None = None


class SessionStateMachine:
    """Not thread-safe by itself -- callers on a `QThread` boundary marshal `apply()`
    calls onto the main thread via Qt signals, per design.md's data flow; this class has
    no Qt dependency and no I/O."""

    def __init__(self) -> None:
        self._status = SessionStatus(stage=SessionStage.IDLE)

    @property
    def status(self) -> SessionStatus:
        return self._status

    def apply(self, event: SessionEvent) -> SessionStatus:
        if self._status.stage is SessionStage.RECOVERY_FAILED and event.stage is not SessionStage.ROUTING:
            return self._status
        self._status = SessionStatus(
            stage=event.stage,
            message=event.message,
            params=event.params,
            transcript=event.transcript if event.transcript is not None else self._status.transcript,
            translation=event.translation if event.translation is not None else self._status.translation,
            diagnostics=event.diagnostics if event.diagnostics is not None else self._status.diagnostics,
        )
        return self._status
