from __future__ import annotations

import json
from pathlib import Path
from threading import Event

import pytest

from opendubstream.domain.contracts import (
    AudioMixSettings,
    AudioMode,
    PhysicalSinkRequired,
    SinkRef,
    StreamIdentityChanged,
    StreamRef,
    StreamSelectionError,
    StreamSelector,
    select_unique_stream,
)
from opendubstream.infrastructure.audio.journal import RecoveryJournal, RouteSnapshot
from opendubstream.infrastructure.audio.playback import PhysicalPlaybackTarget
from opendubstream.infrastructure.audio.process import (
    ProcessCancelled,
    ProcessExecutionError,
    ProcessOutputError,
    ProcessTimeout,
    SafePactlRunner,
)


def stream(identifier: str = "41", serial: str = "chrome-a") -> StreamRef:
    return StreamRef(
        identifier=identifier,
        application_name="Google Chrome",
        media_name="Video",
        serial=serial,
        sink_name="alsa_output.speakers",
    )


def test_unique_selector_returns_only_exact_stable_identity() -> None:
    selector = StreamSelector(application_name="Google Chrome", serial="chrome-a")

    assert select_unique_stream(selector, [stream(), stream("42", "other")]) == stream()


def test_selector_rejects_missing_or_ambiguous_identity() -> None:
    selector = StreamSelector(application_name="Google Chrome")

    with pytest.raises(StreamSelectionError, match="found 0"):
        select_unique_stream(selector, [])
    with pytest.raises(StreamSelectionError, match="found 2"):
        select_unique_stream(selector, [stream(), stream("42", "chrome-b")])


def test_selected_stream_requires_same_identity_after_refresh() -> None:
    selected = stream()

    with pytest.raises(StreamIdentityChanged):
        selected.require_same_identity(stream(serial="replaced"))


def test_recovery_journal_durably_round_trips_and_clears(tmp_path: Path) -> None:
    journal = RecoveryJournal(tmp_path / "routing.json")
    snapshot = RouteSnapshot(stream=stream(), original_sink="alsa_output.speakers", virtual_sink="ods.41")

    journal.persist(snapshot)

    assert journal.load() == snapshot
    journal.clear()
    assert journal.load() is None


def test_recovery_journal_loads_an_old_shape_file_without_reference_fields(tmp_path: Path) -> None:
    """A journal written before `open_reference_sink` existed has no `reference_sink`,
    `reference_module_id`, or `loopback_module_id` keys at all. `load()` must still parse
    it, defaulting the new fields to `None` -- exactly like `virtual_module_id` already
    degrades gracefully for pre-`begin()`-mutation journals."""
    path = tmp_path / "routing.json"
    old_shape_payload = {
        "stream": {
            "identifier": "41",
            "application_name": "Google Chrome",
            "media_name": "Video",
            "serial": "chrome-a",
            "sink_name": "alsa_output.speakers",
        },
        "original_sink": "alsa_output.speakers",
        "virtual_sink": "opendubstream.41",
        "virtual_module_id": "7",
    }
    path.write_text(json.dumps(old_shape_payload), encoding="utf-8")
    journal = RecoveryJournal(path)

    loaded = journal.load()

    assert loaded == RouteSnapshot(
        stream=stream(), original_sink="alsa_output.speakers", virtual_sink="opendubstream.41", virtual_module_id="7"
    )
    assert loaded.reference_sink is None
    assert loaded.reference_module_id is None
    assert loaded.loopback_module_id is None


def test_physical_playback_refuses_virtual_or_nonphysical_target() -> None:
    playback = PhysicalPlaybackTarget()
    physical = SinkRef(name="alsa_output.speakers", is_physical=True)

    assert playback.validate_target(physical) == physical
    with pytest.raises(PhysicalSinkRequired):
        playback.validate_target(SinkRef(name="ods.41", is_physical=False))


def test_audio_mix_settings_allow_independent_percentages_and_reject_out_of_range_values() -> None:
    settings = AudioMixSettings(mode=AudioMode.TRANSLATION_ONLY, original_volume=0, dub_volume=73)

    assert settings.mode is AudioMode.TRANSLATION_ONLY
    assert settings.original_volume == 0
    assert settings.dub_volume == 73
    with pytest.raises(ValueError, match="0 through 100"):
        AudioMixSettings(original_volume=101)
    with pytest.raises(ValueError, match="integer"):
        AudioMixSettings(dub_volume=1.5)  # type: ignore[arg-type]


def test_runner_rejects_shell_metacharacters_and_unknown_options_without_launch() -> None:
    launches: list[tuple[str, ...]] = []
    runner = SafePactlRunner(launch=lambda argv, timeout: launches.append(argv))

    with pytest.raises(ProcessExecutionError):
        runner.run(("pactl", "move-sink-input", "41;id", "alsa_output.speakers"))
    with pytest.raises(ProcessExecutionError):
        runner.run(("pactl", "--server=attacker", "list", "sinks", "short"))

    assert launches == []


def test_runner_maps_timeout_cancellation_malformed_nonzero_and_child_death_to_fail_closed() -> None:
    cancelled = Event()
    cancelled.set()
    runner = SafePactlRunner(launch=lambda argv, timeout: "irrelevant")

    with pytest.raises(ProcessCancelled):
        runner.run(("pactl", "list", "sinks", "short"), cancel=cancelled)

    with pytest.raises(ProcessTimeout):
        SafePactlRunner(launch=lambda argv, timeout: (_ for _ in ()).throw(TimeoutError())).run(
            ("pactl", "list", "sinks", "short")
        )
    with pytest.raises(ProcessOutputError):
        SafePactlRunner(launch=lambda argv, timeout: object()).run(("pactl", "list", "sinks", "short"))
    with pytest.raises(ProcessExecutionError, match="exited 1"):
        SafePactlRunner(launch=lambda argv, timeout: (1, "failure")).run(("pactl", "list", "sinks", "short"))
    with pytest.raises(ProcessExecutionError, match="child exited"):
        SafePactlRunner(launch=lambda argv, timeout: None).run(("pactl", "list", "sinks", "short"))


def test_runner_allows_read_only_json_discovery_listing_forms() -> None:
    launches: list[tuple[str, ...]] = []
    runner = SafePactlRunner(launch=lambda argv, timeout: launches.append(argv) or "[]")

    runner.run(("pactl", "-f", "json", "list", "sink-inputs"))
    runner.run(("pactl", "-f", "json", "list", "sinks"))

    assert launches == [
        ("pactl", "-f", "json", "list", "sink-inputs"),
        ("pactl", "-f", "json", "list", "sinks"),
    ]


def test_runner_allows_the_argument_less_default_sink_query() -> None:
    launches: list[tuple[str, ...]] = []
    runner = SafePactlRunner(launch=lambda argv, timeout: launches.append(argv) or "alsa_output.speakers\n")

    result = runner.run(("pactl", "get-default-sink"))

    assert launches == [("pactl", "get-default-sink")]
    assert result == "alsa_output.speakers\n"


def test_runner_allows_only_literal_routing_argv_forms() -> None:
    launches: list[tuple[str, ...]] = []
    runner = SafePactlRunner(launch=lambda argv, timeout: launches.append(argv) or "7")

    runner.run(("pactl", "load-module", "module-null-sink", "sink_name=opendubstream.41"))
    runner.run(("pactl", "move-sink-input", "41", "opendubstream.41"))
    runner.run(("pactl", "unload-module", "7"))

    assert launches == [
        ("pactl", "load-module", "module-null-sink", "sink_name=opendubstream.41"),
        ("pactl", "move-sink-input", "41", "opendubstream.41"),
        ("pactl", "unload-module", "7"),
    ]


def test_runner_allows_only_bounded_owned_sink_input_mix_argv() -> None:
    launches: list[tuple[str, ...]] = []
    runner = SafePactlRunner(launch=lambda argv, timeout: launches.append(argv) or "")

    runner.run(("pactl", "set-sink-input-mute", "77", "1"))
    runner.run(("pactl", "set-sink-input-volume", "77", "20%"))

    assert launches == [
        ("pactl", "set-sink-input-mute", "77", "1"),
        ("pactl", "set-sink-input-volume", "77", "20%"),
    ]
    with pytest.raises(ProcessExecutionError):
        runner.run(("pactl", "set-sink-input-volume", "77", "101%"))
    with pytest.raises(ProcessExecutionError):
        runner.run(("pactl", "set-sink-input-mute", "77", "yes"))


def test_runner_allows_the_exact_literal_module_loopback_form_used_by_open_reference_sink() -> None:
    """Confirmed live (2026-09-03): `router.open_reference_sink` calls this exact argv shape
    and the allowlist had never been extended for it, so the real call crashed with
    `ProcessExecutionError` even though every fake-backed unit/integration test passed."""
    launches: list[tuple[str, ...]] = []
    runner = SafePactlRunner(launch=lambda argv, timeout: launches.append(argv) or "9")

    runner.run(("pactl", "load-module", "module-loopback", "source=opendubstream-ref.41.monitor", "sink=alsa_output.speakers"))

    assert launches == [
        ("pactl", "load-module", "module-loopback", "source=opendubstream-ref.41.monitor", "sink=alsa_output.speakers"),
    ]


def test_runner_rejects_a_malformed_module_loopback_form() -> None:
    runner = SafePactlRunner(launch=lambda argv, timeout: "9")

    with pytest.raises(ProcessExecutionError):
        runner.run(("pactl", "load-module", "module-loopback", "source=opendubstream-ref.41.monitor;rm -rf", "sink=alsa_output.speakers"))
    with pytest.raises(ProcessExecutionError):
        runner.run(("pactl", "load-module", "module-loopback", "source=opendubstream-ref.41", "sink=alsa_output.speakers"))
    with pytest.raises(ProcessExecutionError):
        runner.run(("pactl", "load-module", "module-loopback", "source=opendubstream-ref.41.monitor"))
