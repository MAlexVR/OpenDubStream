"""RED-first pure coverage for the session state machine (task 1.7): stage transitions
including `RECOVERY_FAILED` -> Start -> `ROUTING`, and `Error` vs `Recovering`."""

from __future__ import annotations

from opendubstream.application.session_state import (
    Diagnostics,
    SessionEvent,
    SessionStage,
    SessionStateMachine,
)
from opendubstream.ui.resources import MessageKey


def test_state_machine_starts_idle() -> None:
    machine = SessionStateMachine()

    assert machine.status.stage is SessionStage.IDLE


def test_state_machine_transitions_through_a_normal_session() -> None:
    machine = SessionStateMachine()

    machine.apply(SessionEvent(stage=SessionStage.ROUTING))
    machine.apply(SessionEvent(stage=SessionStage.CAPTURING))
    machine.apply(SessionEvent(stage=SessionStage.PROCESSING))
    status = machine.apply(SessionEvent(stage=SessionStage.PLAYING))

    assert status.stage is SessionStage.PLAYING


def test_state_machine_distinguishes_error_from_recovering() -> None:
    failed_machine = SessionStateMachine()
    recovering_machine = SessionStateMachine()

    failed_status = failed_machine.apply(SessionEvent(stage=SessionStage.FAILED))
    recovering_status = recovering_machine.apply(SessionEvent(stage=SessionStage.RECOVERING))

    assert failed_status.stage is SessionStage.FAILED
    assert recovering_status.stage is SessionStage.RECOVERING
    assert failed_status.stage is not recovering_status.stage


def test_recovery_failed_is_sticky_until_a_fresh_start() -> None:
    machine = SessionStateMachine()
    machine.apply(SessionEvent(stage=SessionStage.RECOVERY_FAILED, message=MessageKey.INELIGIBLE_GPU_UNAVAILABLE))

    ignored = machine.apply(SessionEvent(stage=SessionStage.CAPTURING))

    assert ignored.stage is SessionStage.RECOVERY_FAILED
    assert ignored.message == MessageKey.INELIGIBLE_GPU_UNAVAILABLE  # unchanged: the event was ignored


def test_recovery_failed_exits_only_on_a_fresh_start_routing_event() -> None:
    machine = SessionStateMachine()
    machine.apply(SessionEvent(stage=SessionStage.RECOVERY_FAILED))

    resumed = machine.apply(SessionEvent(stage=SessionStage.ROUTING))

    assert resumed.stage is SessionStage.ROUTING


def test_state_machine_carries_forward_the_last_transcript_and_translation() -> None:
    machine = SessionStateMachine()
    machine.apply(SessionEvent(stage=SessionStage.PROCESSING, transcript="hello", translation="hola"))

    status = machine.apply(SessionEvent(stage=SessionStage.PLAYING))

    assert status.transcript == "hello"
    assert status.translation == "hola"


def test_state_machine_carries_forward_diagnostics_when_a_new_event_omits_them() -> None:
    machine = SessionStateMachine()
    diagnostics = Diagnostics(execution_mode="cuda", backlog=2, overload_dropped=1, stale_dropped=0)
    machine.apply(SessionEvent(stage=SessionStage.CAPTURING, diagnostics=diagnostics))

    status = machine.apply(SessionEvent(stage=SessionStage.PROCESSING))

    assert status.diagnostics == diagnostics


def test_state_machine_overwrites_diagnostics_when_a_new_event_supplies_them() -> None:
    machine = SessionStateMachine()
    first = Diagnostics(execution_mode="cuda", backlog=1, overload_dropped=0, stale_dropped=0)
    second = Diagnostics(execution_mode="cuda", backlog=3, overload_dropped=2, stale_dropped=1)
    machine.apply(SessionEvent(stage=SessionStage.CAPTURING, diagnostics=first))

    status = machine.apply(SessionEvent(stage=SessionStage.PROCESSING, diagnostics=second))

    assert status.diagnostics == second
