"""Qt boundary regression coverage for persisted presentation-language selection.

These tests intentionally use the inference environment: the lightweight test environment
does not install PySide6 by design.  They create the real ``MainWindow`` but replace only
the eligibility query, so no PipeWire, model, or audio process is invoked.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from opendubstream.application.eligibility import Eligibility
from opendubstream.application.session_state import Diagnostics, SessionEvent, SessionStage
from opendubstream.ui import app as ui_app
from opendubstream.ui.resources import Language, MessageKey, PreferencesStore, resolve


@pytest.fixture(scope="module")
def qt_application() -> QApplication:
    return QApplication.instance() or QApplication([])


@pytest.fixture
def main_window(monkeypatch: pytest.MonkeyPatch, tmp_path, qt_application: QApplication):
    """Build the real window with a deterministic eligible desktop seam."""
    monkeypatch.setattr(ui_app, "DEFAULT_PREFERENCES_PATH", tmp_path / "ui.json")
    monkeypatch.setattr(ui_app, "DEFAULT_JOURNAL_PATH", tmp_path / "routing.json")
    monkeypatch.setattr(ui_app.MainWindow, "_current_eligibility", lambda _self: Eligibility(startable=True))
    window = ui_app.MainWindow()
    yield window
    window._thread.quit()
    assert window._thread.wait(2_000)
    window.close()


def test_selecting_spanish_persists_an_enum_and_preserves_start_eligibility(main_window) -> None:
    """Changing the real combo box must not turn its Qt string payload into UI state."""
    main_window._language_combo.setCurrentIndex(1)

    persisted = PreferencesStore(ui_app.DEFAULT_PREFERENCES_PATH).load()

    assert main_window._language is Language.SPANISH
    assert persisted.language is Language.SPANISH
    assert main_window._start_button.isEnabled()


def test_selecting_english_after_spanish_round_trips_the_real_combo_payload(main_window) -> None:
    main_window._language_combo.setCurrentIndex(1)
    main_window._language_combo.setCurrentIndex(0)

    persisted = PreferencesStore(ui_app.DEFAULT_PREFERENCES_PATH).load()

    assert main_window._language is Language.ENGLISH
    assert persisted.language is Language.ENGLISH
    assert main_window._start_button.isEnabled()


@pytest.mark.parametrize(
    ("eligibility", "reason"),
    [
        (Eligibility(startable=False, reason=MessageKey.INELIGIBLE_NO_CHROME_STREAM), MessageKey.INELIGIBLE_NO_CHROME_STREAM),
        (Eligibility(startable=False, reason=MessageKey.INELIGIBLE_GPU_UNAVAILABLE), MessageKey.INELIGIBLE_GPU_UNAVAILABLE),
    ],
)
def test_preflight_keeps_start_actionable_and_shows_refusal_reason(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    qt_application: QApplication,
    eligibility: Eligibility,
    reason: MessageKey,
) -> None:
    monkeypatch.setattr(ui_app, "DEFAULT_PREFERENCES_PATH", tmp_path / "ui.json")
    monkeypatch.setattr(ui_app, "DEFAULT_JOURNAL_PATH", tmp_path / "routing.json")
    monkeypatch.setattr(ui_app.MainWindow, "_current_eligibility", lambda _self: eligibility)
    window = ui_app.MainWindow()
    try:
        assert window._start_button.isEnabled()
        assert window._state_machine.status.stage is SessionStage.IDLE
        assert window._reason_label.text() == resolve(Language.ENGLISH, reason)

        if reason is MessageKey.INELIGIBLE_GPU_UNAVAILABLE:
            # An unmet precondition is normal waiting state, never the alarming `FAILED`
            # banner -- pressing Start while ineligible must stay on the same neutral
            # `IDLE`-with-reason display the periodic preflight refresh already shows.
            window._on_start_clicked()
            assert window._start_button.isEnabled()
            assert not window._stop_button.isEnabled()
            assert window._state_machine.status.stage is SessionStage.IDLE
            assert window._reason_label.text() == resolve(Language.ENGLISH, reason)
    finally:
        window._thread.quit()
        assert window._thread.wait(2_000)
        window.close()


def test_preflight_exception_is_visible_without_disabling_start(
    monkeypatch: pytest.MonkeyPatch, tmp_path, qt_application: QApplication,
) -> None:
    monkeypatch.setattr(ui_app, "DEFAULT_PREFERENCES_PATH", tmp_path / "ui.json")
    monkeypatch.setattr(ui_app, "DEFAULT_JOURNAL_PATH", tmp_path / "routing.json")

    def unavailable(_self):
        raise RuntimeError("pactl unavailable")

    monkeypatch.setattr(ui_app.MainWindow, "_current_eligibility", unavailable)
    window = ui_app.MainWindow()
    try:
        assert window._start_button.isEnabled()
        assert window._state_machine.status.stage is SessionStage.FAILED
        assert "pactl unavailable" in window._reason_label.text()
    finally:
        window._thread.quit()
        assert window._thread.wait(2_000)
        window.close()


def test_preflight_refresh_never_reenables_start_during_an_active_session(main_window) -> None:
    main_window._session_active = True
    main_window._start_button.setEnabled(False)
    main_window._preflight_timer.timeout.emit()

    assert not main_window._start_button.isEnabled()

    main_window._session_active = False
    main_window._preflight_timer.timeout.emit()

    assert main_window._start_button.isEnabled()


def test_finished_session_returns_to_idle_actionable_preflight(main_window) -> None:
    main_window._session_active = True
    main_window._start_button.setEnabled(False)
    main_window._stop_button.setEnabled(True)
    main_window._on_session_event(SessionEvent(stage=SessionStage.CAPTURING))

    main_window._on_session_finished()

    assert main_window._state_machine.status.stage is SessionStage.IDLE
    assert main_window._start_button.isEnabled()
    assert not main_window._stop_button.isEnabled()


def test_preflight_refresh_preserves_failed_session_diagnostics(main_window) -> None:
    """A later advisory preflight must not erase the failure the user needs to inspect."""
    main_window._on_session_event(
        SessionEvent(
            stage=SessionStage.FAILED,
            diagnostics=Diagnostics(last_routing_error="vad rejected all windows", recovery_restored=True),
        )
    )

    main_window._preflight_timer.timeout.emit()

    status = main_window._state_machine.status
    assert status.stage is SessionStage.FAILED
    assert status.diagnostics is not None
    assert status.diagnostics.last_routing_error == "vad rejected all windows"
    assert main_window._start_button.isEnabled()


def test_stop_sets_the_worker_cancel_event_without_waiting_for_worker_queue(main_window) -> None:
    """Stop must be observable while the worker thread is blocked in its session loop."""
    main_window._worker._cancel.clear()
    main_window._stop_button.setEnabled(True)

    main_window._on_stop_clicked()

    assert main_window._worker._cancel.is_set()
    assert not main_window._stop_button.isEnabled()
