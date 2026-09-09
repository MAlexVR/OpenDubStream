"""Offscreen RED coverage for the Phase 3 desktop workspace."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from opendubstream.application.eligibility import Eligibility
from opendubstream.application.session_state import SessionEvent, SessionStage
from opendubstream.domain.contracts import AudioMode, SinkRef, StreamRef
from opendubstream.ui import app as ui_app
from opendubstream.ui.resources import Language, MessageKey, Preferences, PreferencesStore


@pytest.fixture(scope="module")
def qt_application() -> QApplication:
    return QApplication.instance() or QApplication([])


@pytest.fixture
def workspace(monkeypatch: pytest.MonkeyPatch, tmp_path, qt_application: QApplication):
    monkeypatch.setattr(ui_app, "DEFAULT_PREFERENCES_PATH", tmp_path / "ui.json")
    monkeypatch.setattr(ui_app, "DEFAULT_JOURNAL_PATH", tmp_path / "routing.json")
    monkeypatch.setattr(ui_app.MainWindow, "_current_eligibility", lambda _self: Eligibility(startable=True))
    window = ui_app.MainWindow()
    yield window
    window._thread.quit()
    assert window._thread.wait(2_000)
    window.close()


def test_workspace_merges_audio_and_settings_tabs(workspace) -> None:
    assert workspace._workspace_tabs.count() == 3
    assert [workspace._workspace_tabs.tabText(index) for index in range(3)] == [
        "Dubbing", "Settings", "General",
    ]
    assert workspace._start_button.accessibleName() == "Start dubbing"
    assert workspace._stop_button.accessibleName() == "Stop dubbing"


def test_general_tab_shows_app_identity_and_moves_the_language_selector(workspace) -> None:
    """The interface-language selector lives in the General tab (identity/about info),
    not the Dubbing tab's action row, so that row stays focused on Start/Stop alone."""
    assert workspace._general_title_label.text() == "OpenDubStream"
    assert workspace._app_description_label.text() != ""
    assert workspace._language_combo.parentWidget() is workspace._general_tab
    assert workspace._language_combo.parentWidget() is not workspace._dubbing_tab


def test_audio_and_settings_controls_keep_only_editable_preferences(workspace) -> None:
    workspace._audio_mode_combo.setCurrentIndex(1)
    workspace._original_volume_spin.setValue(37)
    workspace._dub_volume_spin.setValue(81)
    workspace._voice_combo.setCurrentText("ef_dora")
    workspace._speed_spin.setValue(1.15)

    stored = PreferencesStore(ui_app.DEFAULT_PREFERENCES_PATH).load()

    assert stored.audio_mode is AudioMode.DUB
    assert stored.original_volume == 37
    assert stored.dub_volume == 81
    assert stored.tts_voice == "ef_dora"
    assert stored.speech_speed == 1.15
    assert workspace._chrome_stream_combo.currentData() is None


def test_start_arms_without_chrome_when_cuda_and_physical_output_are_available(
    workspace, monkeypatch: pytest.MonkeyPatch,
) -> None:
    physical = SinkRef("alsa_output.speakers", True, "speaker-1")
    monkeypatch.setattr(workspace, "_available_physical_sinks", lambda: [physical])
    monkeypatch.setattr(workspace, "_available_chrome_streams", lambda: [])
    workspace._refresh_audio_choices()
    started: list[object] = []
    monkeypatch.setattr(workspace, "_launch_runner", lambda runner: started.append(runner))

    workspace._on_start_clicked()

    assert len(started) == 1
    assert workspace._session_active
    assert workspace._stop_button.isEnabled()


def test_start_arms_even_when_real_eligibility_reports_no_chrome_stream(
    workspace, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing Chrome stream must never refuse Start: `DubbingSessionRunner` itself arms
    and waits for one (`SessionStage.ARMED`) -- only a missing GPU is a real blocker. This
    exercises the real `Eligibility(startable=False, reason=INELIGIBLE_NO_CHROME_STREAM)`
    shape `evaluate_eligibility` actually returns when no stream is playing yet, overriding
    the `workspace` fixture's always-`startable=True` stub for this one call."""
    monkeypatch.setattr(
        ui_app.MainWindow, "_current_eligibility",
        lambda _self: Eligibility(startable=False, reason=MessageKey.INELIGIBLE_NO_CHROME_STREAM),
    )
    physical = SinkRef("alsa_output.speakers", True, "speaker-1")
    monkeypatch.setattr(workspace, "_available_physical_sinks", lambda: [physical])
    monkeypatch.setattr(workspace, "_available_chrome_streams", lambda: [])
    workspace._refresh_audio_choices()
    started: list[object] = []
    monkeypatch.setattr(workspace, "_launch_runner", lambda runner: started.append(runner))

    workspace._on_start_clicked()

    assert len(started) == 1
    assert workspace._session_active


def test_start_still_refuses_without_a_gpu(workspace, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ui_app.MainWindow, "_current_eligibility",
        lambda _self: Eligibility(startable=False, reason=MessageKey.INELIGIBLE_GPU_UNAVAILABLE),
    )
    started: list[object] = []
    monkeypatch.setattr(workspace, "_launch_runner", lambda runner: started.append(runner))

    workspace._on_start_clicked()

    assert len(started) == 0
    assert not workspace._session_active
    assert workspace._state_machine.status.stage is SessionStage.IDLE


def test_selected_stream_and_output_must_be_current_before_start(workspace, monkeypatch: pytest.MonkeyPatch) -> None:
    current = StreamRef("42", "Google Chrome", "Lesson", "chrome-42", "alsa_output.speakers")
    physical = SinkRef("alsa_output.speakers", True, "speaker-1")
    monkeypatch.setattr(workspace, "_available_physical_sinks", lambda: [physical])
    monkeypatch.setattr(workspace, "_available_chrome_streams", lambda: [current])
    workspace._refresh_audio_choices()
    workspace._chrome_stream_combo.setCurrentIndex(1)
    workspace._output_combo.setCurrentIndex(1)
    monkeypatch.setattr(workspace, "_available_chrome_streams", lambda: [])

    workspace._on_start_clicked()

    assert not workspace._session_active
    assert "stale" in workspace._reason_label.text().lower() or "available" in workspace._reason_label.text().lower()


def test_selected_physical_output_must_not_be_used_after_it_disappears(workspace, monkeypatch: pytest.MonkeyPatch) -> None:
    physical = SinkRef("alsa_output.speakers", True, "speaker-1")
    monkeypatch.setattr(workspace, "_available_physical_sinks", lambda: [physical])
    monkeypatch.setattr(workspace, "_available_chrome_streams", lambda: [])
    workspace._refresh_audio_choices()
    workspace._output_combo.setCurrentIndex(1)
    monkeypatch.setattr(workspace, "_available_physical_sinks", lambda: [])

    workspace._on_start_clicked()

    assert not workspace._session_active
    assert "output" in workspace._reason_label.text().lower() or "salida" in workspace._reason_label.text().lower()


def test_output_combo_defaults_to_the_hosts_actual_default_sink_not_the_first_candidate(
    workspace, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real host can list its sinks in an order where the first one is not where the
    rest of the desktop (e.g. headphones) is actually playing -- auto-selecting "index 1"
    unconditionally silently dubs into the wrong device. The host's own default sink is a
    much better guess than list order when there is no prior explicit selection."""
    first_listed = SinkRef("alsa_output.usb-dock", True, "dock-1")
    actual_default = SinkRef("alsa_output.analog-stereo", True, "analog-1")
    monkeypatch.setattr(workspace, "_available_physical_sinks", lambda: [first_listed, actual_default])
    monkeypatch.setattr(workspace, "_available_chrome_streams", lambda: [])
    monkeypatch.setattr(workspace, "_default_sink_name", lambda: actual_default.name)

    workspace._refresh_audio_choices()

    assert workspace._output_combo.currentData() == actual_default


def test_preferences_round_trip_only_editable_values_and_malformed_content_fails_closed(tmp_path) -> None:
    path = tmp_path / "ui.json"
    store = PreferencesStore(path)
    expected = Preferences(
        language=Language.SPANISH,
        audio_mode=AudioMode.TRANSLATION_ONLY,
        original_volume=0,
        dub_volume=72,
        tts_voice="ef_dora",
        speech_speed=1.15,
    )
    store.save(expected)

    assert store.load() == expected
    assert set(json.loads(path.read_text(encoding="utf-8"))) == {
        "audio_mode", "dub_volume", "language", "original_volume", "speech_speed", "tts_voice",
    }

    path.write_text('{"language":"unexpected","pcm":"private"}', encoding="utf-8")
    assert store.load() == Preferences()


def test_original_icon_and_provenance_are_tracked() -> None:
    root = Path(__file__).resolve().parents[2]
    icon = root / "assets" / "opendubstream.svg"
    provenance = root / "assets" / "OPENDUBSTREAM-ICON-PROVENANCE.md"
    license_file = root / "assets" / "LICENSES" / "CC0-1.0.txt"

    assert icon.exists()
    assert "OpenDubStream" in provenance.read_text(encoding="utf-8")
    assert "SPDX-License-Identifier: `CC0-1.0`" in provenance.read_text(encoding="utf-8")
    assert license_file.exists()


def test_caption_overlay_starts_hidden_and_toggles_with_the_dubbing_tab_button(workspace) -> None:
    assert not workspace._caption_overlay.isVisible()
    assert not workspace._caption_toggle_button.isChecked()

    workspace._set_caption_overlay_visible(True)

    assert workspace._caption_overlay.isVisible()
    assert workspace._caption_toggle_button.isChecked()

    workspace._set_caption_overlay_visible(False)

    assert not workspace._caption_overlay.isVisible()
    assert not workspace._caption_toggle_button.isChecked()


def test_caption_overlay_mirrors_start_stop_state_and_receives_the_live_translation(workspace) -> None:
    assert workspace._caption_overlay._start_button.isEnabled()
    assert not workspace._caption_overlay._stop_button.isEnabled()

    workspace._on_session_event(
        SessionEvent(stage=SessionStage.PROCESSING, translation="hola mundo"),
    )

    assert workspace._caption_overlay._caption_label.text() == "hola mundo"


def test_tray_icon_is_gracefully_absent_under_offscreen_platform(workspace) -> None:
    """`QSystemTrayIcon.isSystemTrayAvailable()` is false under `QT_QPA_PLATFORM=offscreen`
    (every automated test runs this way) -- the tray must be omitted, not crash."""
    assert workspace._tray_icon is None
    assert workspace._tray_caption_action is None


def test_compact_workspace_hides_technical_details_until_requested(workspace):
    assert workspace.minimumHeight() <= 620
    assert workspace._diagnostics_group.isHidden()
    workspace._diagnostics_toggle.click()
    assert not workspace._diagnostics_group.isHidden()


def test_overlay_close_updates_the_shared_visibility_control(workspace):
    workspace._set_caption_overlay_visible(True)
    workspace._caption_overlay._close_button.click()
    assert not workspace._caption_toggle_button.isChecked()
    assert not workspace._caption_overlay.isVisible()


def test_live_audio_controls_and_overlay_share_one_settings_snapshot(workspace):
    workspace._dub_volume_spin.setValue(63)
    workspace._original_volume_spin.setValue(34)
    workspace._speed_spin.setValue(1.25)
    assert workspace._live_controls.snapshot().mix.dub_volume == 63
    assert workspace._live_controls.snapshot().mix.original_volume == 34
    assert workspace._live_controls.snapshot().speed == 1.25
    workspace._caption_overlay._voice_button.click()
    assert not workspace._live_controls.snapshot().voice_enabled
    assert not workspace._voice_toggle.isChecked()
    workspace._caption_overlay._font_size.setValue(28)
    assert '28px' in workspace._caption_overlay._caption_label.styleSheet()


def test_default_window_keeps_main_actions_and_details_in_view(workspace, qt_application):
    from PySide6.QtCore import QPoint
    workspace.resize(680, 740)
    workspace.show()
    qt_application.processEvents()
    for widget in (workspace._start_button, workspace._diagnostics_toggle):
        bottom = widget.mapTo(workspace, QPoint(0, widget.height())).y()
        assert bottom < workspace.height()
    assert ui_app._dark_palette().color(__import__('PySide6.QtGui', fromlist=['QPalette']).QPalette.ColorRole.Window).name() == '#0a0f0a'


def test_voice_name_is_friendly_without_changing_persisted_voice_id(workspace):
    assert workspace._voice_combo.currentText() == 'Dora'
    assert workspace._voice_combo.currentData() == 'ef_dora'


def test_null_device_descriptions_use_device_and_active_port(workspace, monkeypatch):
    inventory = [{"name": "alsa_output.headphones", "description": "(null)",
                  "active_port": "analog-output-headphones",
                  "ports": [{"name": "analog-output-headphones", "description": "Headphones"}],
                  "properties": {"device.class": "sound", "device.bus": "pci",
                                 "object.serial": "59", "device.description": "Built-in Audio"}}]
    monkeypatch.setattr(workspace._pactl, "run", lambda argv: json.dumps(inventory))
    sinks = workspace._available_physical_sinks()
    assert len(sinks) == 1
    assert workspace._sink_labels[sinks[0].name] == "Built-in Audio — Headphones"


def test_caption_window_is_independent_and_controls_have_visible_labels(workspace):
    assert workspace._caption_overlay.parentWidget() is None
    assert workspace._caption_overlay._font_size.prefix() != "A  "
    assert workspace._caption_overlay._size_label.text()
    assert workspace._caption_overlay._volume_label.text()


def test_user_modes_control_speech_and_captions(workspace):
    assert workspace._audio_mode_combo.count() == 3
    workspace._audio_mode_combo.setCurrentIndex(0)
    assert workspace._live_controls.snapshot().mix.mode is AudioMode.CAPTIONS
    assert workspace._caption_overlay.isVisible()
    workspace._audio_mode_combo.setCurrentIndex(1)
    assert workspace._live_controls.snapshot().mix.mode is AudioMode.DUB
    assert not workspace._caption_overlay.isVisible()
    workspace._audio_mode_combo.setCurrentIndex(2)
    assert workspace._live_controls.snapshot().mix.mode is AudioMode.BOTH
    assert workspace._caption_overlay.isVisible()


def test_main_language_pair_and_equal_text_scale(workspace):
    assert "English" in workspace._transcript_label.text()
    assert "Spanish" in workspace._translation_label.text()
    assert workspace._title_bar._title_label.alignment() & ui_app.Qt.AlignmentFlag.AlignHCenter


def test_tray_quick_actions_are_localized_and_share_state(monkeypatch, tmp_path, qt_application):
    monkeypatch.setattr(ui_app, 'DEFAULT_PREFERENCES_PATH', tmp_path / 'ui.json')
    monkeypatch.setattr(ui_app, 'DEFAULT_JOURNAL_PATH', tmp_path / 'routing.json')
    monkeypatch.setattr(ui_app.MainWindow, '_current_eligibility', lambda _: Eligibility(startable=True))
    monkeypatch.setattr(ui_app.QSystemTrayIcon, 'isSystemTrayAvailable', lambda: True)
    window = ui_app.MainWindow()
    try:
        assert window._tray_start_action.text() == 'Start'
        assert not window._tray_stop_action.isEnabled()
        window._tray_mode_actions[0].trigger()
        assert window._audio_mode_combo.currentData() == AudioMode.CAPTIONS
        assert window._tray_mode_actions[0].isChecked()
        assert not window._tray_voice_action.isEnabled()
        window._tray_mode_actions[2].trigger()
        assert window._tray_voice_action.isEnabled()
        window._tray_voice_action.trigger()
        assert not window._voice_toggle.isChecked()
    finally:
        window._thread.quit()
        window._thread.wait(2000)
        window.close()


def test_return_to_video_hides_settings_not_captions(workspace, qt_application):
    workspace.show()
    workspace._audio_mode_combo.setCurrentIndex(2)
    workspace._back_to_video()
    qt_application.processEvents()
    assert not workspace.isVisible()
    assert workspace._caption_overlay.isVisible()
    workspace._caption_overlay._settings_button.click()
    assert workspace.isVisible()


def test_tray_current_mode_cannot_be_unchecked(monkeypatch, tmp_path, qt_application):
    monkeypatch.setattr(ui_app, 'DEFAULT_PREFERENCES_PATH', tmp_path / 'ui.json')
    monkeypatch.setattr(ui_app, 'DEFAULT_JOURNAL_PATH', tmp_path / 'routing.json')
    monkeypatch.setattr(ui_app.MainWindow, '_current_eligibility', lambda _: Eligibility(startable=True))
    monkeypatch.setattr(ui_app.QSystemTrayIcon, 'isSystemTrayAvailable', lambda: True)
    window = ui_app.MainWindow()
    try:
        selected = window._tray_mode_actions[window._audio_mode_combo.currentIndex()]
        selected.trigger()
        assert selected.isChecked()
    finally:
        window._thread.quit()
        window._thread.wait(2000)
        window.close()


def test_output_menu_is_not_rebuilt_while_user_is_choosing(workspace, monkeypatch, qt_application):
    workspace._workspace_tabs.setCurrentIndex(1)
    workspace._output_combo.addItem('Headphones', SinkRef('phones', True, '1'))
    workspace.show()
    workspace._output_combo.showPopup()
    qt_application.processEvents()
    called = []
    monkeypatch.setattr(workspace, '_available_chrome_streams', lambda: called.append(True) or [])
    workspace._refresh_audio_choices()
    assert not called
    workspace._output_combo.hidePopup()


def test_discovery_includes_suspended_physical_outputs_but_not_virtuals(workspace, monkeypatch):
    def sink(name, state, device_class='sound'):
        return {'name': name, 'state': state, 'description': name,
                'properties': {'device.class': device_class, 'device.bus': 'usb', 'object.serial': name}}
    entries = [sink('usb_speakers', 'RUNNING'), sink('usb_headphones', 'SUSPENDED'),
               sink('opendubstream.owned', 'RUNNING'), sink('input.monitor', 'IDLE'),
               sink('virtual_mix', 'IDLE', 'abstract')]
    monkeypatch.setattr(workspace._pactl, 'run', lambda _: json.dumps(entries))
    assert [s.name for s in workspace._available_physical_sinks()] == ['usb_speakers', 'usb_headphones']


def test_active_session_mode_requires_stop_before_switching(workspace):
    workspace._session_active = True
    workspace._sync_tray()
    assert not workspace._audio_mode_combo.isEnabled()
    assert 'Stop' in workspace._audio_mode_combo.toolTip()
    workspace._on_session_finished()
    assert workspace._audio_mode_combo.isEnabled()


def test_overlay_pages_two_lines_retains_full_text_and_does_not_restart_same_caption(workspace, qt_application):
    overlay = workspace._caption_overlay
    overlay.show()
    qt_application.processEvents()
    full = ' '.join(['Una frase completa que conserva todas sus palabras.'] * 8)
    overlay.set_caption(full)
    assert len(overlay._caption_label.text().splitlines()) <= 2
    assert overlay._caption_label.accessibleDescription() == full
    assert overlay._caption_label.toolTip() == full
    assert len(overlay._caption_pages) > 1
    assert ' '.join(' '.join(overlay._caption_pages).split()) == full
    overlay._advance_caption_page()
    page = overlay._caption_label.text()
    overlay.set_caption(full)
    assert overlay._caption_label.text() == page
    assert overlay._caption_page_label.text().startswith('2 / ')


def test_status_exposes_operation_and_latest_stage_timings(workspace):
    from opendubstream.application.session_state import Diagnostics
    workspace._on_session_event(SessionEvent(stage=SessionStage.PROCESSING,
        diagnostics=Diagnostics(active_operation='voice', timings_ms={'voice': 12000, 'caption_age': 3000})))
    assert workspace._stage_label.text() == 'Generating Spanish voice…'
    rendered = workspace._diagnostics_rows[MessageKey.DIAGNOSTICS_TIMINGS_LABEL][1].text()
    assert '12000' in rendered and '3000' in rendered


@pytest.mark.parametrize('size', [16, 22, 36])
def test_caption_two_line_pages_fit_actual_font_and_window(workspace, qt_application, size):
    from PySide6.QtGui import QFontMetrics
    overlay = workspace._caption_overlay
    overlay.resize(560, 240)
    overlay.show()
    overlay._font_size.setValue(size)
    qt_application.processEvents()
    overlay.set_caption('Una explicación más larga necesita mantener cada palabra visible y conservar la idea completa para el usuario.')
    metrics = QFontMetrics(overlay._caption_label.font())
    for page in overlay._caption_pages:
        assert len(page.splitlines()) <= 2
        for line in page.splitlines():
            assert metrics.horizontalAdvance(line) <= overlay._caption_label.width()
    assert overlay._caption_label.height() >= metrics.lineSpacing() * 2


def test_titlebar_matches_reference_without_duplicate_icon(workspace):
    from PySide6.QtWidgets import QLabel
    bar = workspace._title_bar
    workspace.show()
    QApplication.processEvents()
    assert all(label.pixmap().isNull() for label in bar.findChildren(QLabel))
    assert abs(bar._title_label.geometry().center().x() - bar.rect().center().x()) <= 1
    assert bar._close_button.width() == bar._close_button.height() == 30
    assert bar._title_label.font().bold()
