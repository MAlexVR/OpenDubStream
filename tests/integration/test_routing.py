from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from opendubstream.application.session import run_protected
from opendubstream.domain.contracts import SinkRef, StreamRef, StreamSelector, StreamSelectionError
from opendubstream.infrastructure.audio.journal import RecoveryJournal
from opendubstream.infrastructure.audio.router import PipeWirePulseRouter, RoutingSafetyError


def stream(identifier: str = "41", serial: str = "chrome-a") -> StreamRef:
    return StreamRef(identifier, "Google Chrome", "Video", serial, "alsa_output.speakers")


PHYSICAL_SINK = SinkRef("alsa_output.speakers", True)


class FakePactl:
    def __init__(self) -> None:
        self.commands: list[tuple[str, ...]] = []

    def run(self, argv: tuple[str, ...], **_: object) -> str:
        self.commands.append(argv)
        return "1"


class LoopbackInventoryPactl(FakePactl):
    """Fake JSON inventory which identifies exactly one module-owned loopback input."""

    def __init__(self, sink_inputs: list[dict[str, object]] | None = None) -> None:
        super().__init__()
        self._sink_inputs = sink_inputs if sink_inputs is not None else [
            {"index": 77, "sink": 5, "owner_module": 1, "properties": {"application.name": "OpenDubStream original mix"}}
        ]

    def run(self, argv: tuple[str, ...], **kwargs: object) -> str:
        if argv == ("pactl", "-f", "json", "list", "sink-inputs"):
            self.commands.append(argv)
            return json.dumps(self._sink_inputs)
        if argv == ("pactl", "-f", "json", "list", "sinks"):
            self.commands.append(argv)
            return json.dumps([{"index": 5, "name": "alsa_output.speakers"}])
        return super().run(argv, **kwargs)


def test_router_journals_before_mutation_keeps_default_and_routes_only_selected_stream(tmp_path: Path) -> None:
    fake = FakePactl()
    journal = RecoveryJournal(tmp_path / "routing.json")
    router = PipeWirePulseRouter(fake, journal, streams=lambda: [stream()])

    selected = router.select_unique(StreamSelector(application_name="Google Chrome", serial="chrome-a"))
    lease = router.begin(selected, SinkRef("alsa_output.speakers", True))

    assert journal.load() is not None
    assert fake.commands == [
        ("pactl", "load-module", "module-null-sink", "sink_name=opendubstream.41"),
        ("pactl", "move-sink-input", "41", "opendubstream.41"),
    ]
    assert lease.original_sink == "alsa_output.speakers"
    assert lease.physical_sink == "alsa_output.speakers"
    assert all("set-default-sink" not in command for command in fake.commands)


def test_router_rejects_changed_or_ambiguous_stream_before_mutation(tmp_path: Path) -> None:
    fake = FakePactl()
    journal = RecoveryJournal(tmp_path / "routing.json")
    router = PipeWirePulseRouter(fake, journal, streams=lambda: [stream(), stream("42", "chrome-b")])

    with pytest.raises(StreamSelectionError):
        router.select_unique(StreamSelector(application_name="Google Chrome"))

    assert fake.commands == []
    assert journal.load() is None


def test_recovery_is_idempotent_after_crash_or_stale_journal(tmp_path: Path) -> None:
    fake = FakePactl()
    journal = RecoveryJournal(tmp_path / "routing.json")
    router = PipeWirePulseRouter(fake, journal, streams=lambda: [stream()])
    router.begin(router.select_unique(StreamSelector(application_name="Google Chrome", serial="chrome-a")), SinkRef("alsa_output.speakers", True))

    assert router.recover().restored is True
    assert router.recover().restored is False
    assert journal.load() is None
    assert fake.commands[-2:] == [
        ("pactl", "move-sink-input", "41", "alsa_output.speakers"),
        ("pactl", "unload-module", "1"),
    ]


def test_recovery_uses_the_current_stream_identifier_when_it_has_drifted_from_the_journal(tmp_path: Path) -> None:
    """Confirmed live (2026-09-03): PipeWire renumbered Chrome's sink-input identifier
    during a ~20s measurement, so the journaled identifier ("41") no longer existed by
    recovery time and `move-sink-input 41 ...` failed, leaving Chrome silently stuck on
    the virtual sink until manual intervention. `virtual_sink` is a private sink only this
    tool's lease could have moved a stream onto, so whichever stream is discoverably there
    right now is unambiguously the one to restore -- not a guess, the same "resolve fresh,
    never trust a stale reference" pattern `OwnedMonitorDiscovery` already uses."""
    fake = FakePactl()
    journal = RecoveryJournal(tmp_path / "routing.json")
    current_streams = [stream()]
    router = PipeWirePulseRouter(fake, journal, streams=lambda: current_streams)
    lease = router.begin(
        router.select_unique(StreamSelector(application_name="Google Chrome", serial="chrome-a")),
        SinkRef("alsa_output.speakers", True),
    )
    # Simulate PipeWire renumbering the stream while it sits on the owned virtual sink.
    current_streams[:] = [replace(stream(identifier="99"), sink_name=lease.virtual_sink)]
    fake.commands.clear()

    result = router.recover()

    assert result.restored is True
    assert ("pactl", "move-sink-input", "99", "alsa_output.speakers") in fake.commands
    assert ("pactl", "move-sink-input", "41", "alsa_output.speakers") not in fake.commands


def test_recovery_rejects_more_than_one_stream_on_the_owned_virtual_sink(tmp_path: Path) -> None:
    fake = FakePactl()
    journal = RecoveryJournal(tmp_path / "routing.json")
    current_streams = [stream()]
    router = PipeWirePulseRouter(fake, journal, streams=lambda: current_streams)
    lease = router.begin(
        router.select_unique(StreamSelector(application_name="Google Chrome", serial="chrome-a")),
        SinkRef("alsa_output.speakers", True),
    )
    current_streams[:] = [
        replace(stream(identifier="41"), sink_name=lease.virtual_sink),
        replace(stream(identifier="99", serial="chrome-b"), sink_name=lease.virtual_sink),
    ]
    fake.commands.clear()

    with pytest.raises(RoutingSafetyError, match="more than one stream"):
        router.recover()


def test_router_refuses_virtual_tts_target_before_audio_mutation(tmp_path: Path) -> None:
    fake = FakePactl()
    router = PipeWirePulseRouter(fake, RecoveryJournal(tmp_path / "routing.json"), streams=lambda: [stream()])

    with pytest.raises(RoutingSafetyError, match="physical"):
        router.begin(router.select_unique(StreamSelector(application_name="Google Chrome")), SinkRef("opendubstream.41", False))

    assert fake.commands == []


def test_router_persists_module_identity_before_the_next_mutation(tmp_path: Path) -> None:
    fake = FakePactl()
    journal = RecoveryJournal(tmp_path / "routing.json")
    router = PipeWirePulseRouter(fake, journal, streams=lambda: [stream()])

    router.begin(router.select_unique(StreamSelector(application_name="Google Chrome", serial="chrome-a")), SinkRef("alsa_output.speakers", True))

    assert journal.load().virtual_module_id == "1"


@pytest.mark.parametrize(
    "target",
    [
        SinkRef("opendubstream.41", False),
        SinkRef("opendubstream.41.monitor", False),
        SinkRef("", True),
    ],
)
def test_original_mix_rejects_unsafe_target_before_loopback_mutation(tmp_path: Path, target: SinkRef) -> None:
    fake = LoopbackInventoryPactl()
    journal = RecoveryJournal(tmp_path / "routing.json")
    router = PipeWirePulseRouter(fake, journal, streams=lambda: [stream()])
    lease = router.begin(
        router.select_unique(StreamSelector(application_name="Google Chrome", serial="chrome-a")),
        SinkRef("alsa_output.speakers", True),
    )
    fake.commands.clear()

    with pytest.raises(RoutingSafetyError, match="physical"):
        router.start_original_mix(lease, target)

    assert fake.commands == []
    assert journal.load().original_loopback_module_id is None


def test_original_mix_creates_owned_virtual_monitor_loopback_and_journals_resolved_input(tmp_path: Path) -> None:
    fake = LoopbackInventoryPactl()
    journal = RecoveryJournal(tmp_path / "routing.json")
    router = PipeWirePulseRouter(fake, journal, streams=lambda: [stream()])
    lease = router.begin(
        router.select_unique(StreamSelector(application_name="Google Chrome", serial="chrome-a")),
        SinkRef("alsa_output.speakers", True),
    )
    fake.commands.clear()

    mix = router.start_original_mix(lease, SinkRef("alsa_output.speakers", True))

    assert mix.loopback_module_id == "1"
    assert mix.sink_input_id == "77"
    assert fake.commands == [
        ("pactl", "load-module", "module-loopback", "source=opendubstream.41.monitor", "sink=alsa_output.speakers"),
        ("pactl", "-f", "json", "list", "sink-inputs"),
        ("pactl", "-f", "json", "list", "sinks"),
    ]
    snapshot = journal.load()
    assert snapshot.original_loopback_module_id == "1"
    assert snapshot.original_loopback_sink_input_id == "77"
    assert snapshot.original_loopback_physical_sink == "alsa_output.speakers"


def test_original_mix_updates_only_owned_input_and_cleanup_precedes_chrome_restore(tmp_path: Path) -> None:
    fake = LoopbackInventoryPactl()
    journal = RecoveryJournal(tmp_path / "routing.json")
    router = PipeWirePulseRouter(fake, journal, streams=lambda: [stream()])
    lease = router.begin(
        router.select_unique(StreamSelector(application_name="Google Chrome", serial="chrome-a")),
        SinkRef("alsa_output.speakers", True),
    )
    mix = router.start_original_mix(lease, SinkRef("alsa_output.speakers", True))
    fake.commands.clear()

    router.set_original_mix(mix, muted=True, volume_percent=0)
    router.set_original_mix(mix, muted=False, volume_percent=20)
    result = router.recover()

    assert result.restored is True
    assert fake.commands == [
        ("pactl", "set-sink-input-mute", "77", "1"),
        ("pactl", "set-sink-input-volume", "77", "0%"),
        ("pactl", "set-sink-input-mute", "77", "0"),
        ("pactl", "set-sink-input-volume", "77", "20%"),
        ("pactl", "unload-module", "1"),
        ("pactl", "move-sink-input", "41", "alsa_output.speakers"),
        ("pactl", "unload-module", "1"),
    ]
    assert journal.load() is None
    assert all("set-default" not in command for command in fake.commands)


def test_recovery_tolerates_an_already_absent_owned_original_loopback_and_restores_chrome(tmp_path: Path) -> None:
    class AbsentOriginalLoopbackPactl(LoopbackInventoryPactl):
        def __init__(self) -> None:
            super().__init__()
            self._unload_calls = 0

        def run(self, argv: tuple[str, ...], **kwargs: object) -> str:
            if argv == ("pactl", "unload-module", "1"):
                self.commands.append(argv)
                self._unload_calls += 1
                if self._unload_calls == 1:
                    raise RuntimeError("Failure: No such entity")
                return ""
            return super().run(argv, **kwargs)

    fake = AbsentOriginalLoopbackPactl()
    journal = RecoveryJournal(tmp_path / "routing.json")
    router = PipeWirePulseRouter(fake, journal, streams=lambda: [stream()])
    lease = router.begin(router.select_unique(StreamSelector(application_name="Google Chrome", serial="chrome-a")), PHYSICAL_SINK)
    router.start_original_mix(lease, PHYSICAL_SINK)
    fake.commands.clear()

    result = router.recover()

    assert result.restored is True
    assert fake.commands == [
        ("pactl", "unload-module", "1"),
        ("pactl", "move-sink-input", "41", "alsa_output.speakers"),
        ("pactl", "unload-module", "1"),
    ]
    assert journal.load() is None


def test_recovery_preserves_unexpected_original_loopback_cleanup_failure(tmp_path: Path) -> None:
    class BrokenOriginalLoopbackPactl(LoopbackInventoryPactl):
        def run(self, argv: tuple[str, ...], **kwargs: object) -> str:
            if argv == ("pactl", "unload-module", "1"):
                self.commands.append(argv)
                raise RuntimeError("PipeWire disconnected")
            return super().run(argv, **kwargs)

    fake = BrokenOriginalLoopbackPactl()
    journal = RecoveryJournal(tmp_path / "routing.json")
    router = PipeWirePulseRouter(fake, journal, streams=lambda: [stream()])
    lease = router.begin(router.select_unique(StreamSelector(application_name="Google Chrome", serial="chrome-a")), PHYSICAL_SINK)
    router.start_original_mix(lease, PHYSICAL_SINK)
    fake.commands.clear()

    with pytest.raises(RuntimeError, match="disconnected"):
        router.recover()

    assert fake.commands == [("pactl", "unload-module", "1")]
    assert journal.load() is not None


@pytest.mark.parametrize(
    "inputs",
    [[], [{"index": 77, "sink": 5, "owner_module": 1}, {"index": 78, "sink": 5, "owner_module": 1}]],
)
def test_original_mix_removes_created_module_when_owned_input_is_missing_or_ambiguous(
    tmp_path: Path, inputs: list[dict[str, object]]
) -> None:
    fake = LoopbackInventoryPactl(inputs)
    journal = RecoveryJournal(tmp_path / "routing.json")
    router = PipeWirePulseRouter(fake, journal, streams=lambda: [stream()])
    lease = router.begin(
        router.select_unique(StreamSelector(application_name="Google Chrome", serial="chrome-a")),
        SinkRef("alsa_output.speakers", True),
    )
    fake.commands.clear()

    with pytest.raises(RoutingSafetyError, match="loopback sink-input"):
        router.start_original_mix(lease, SinkRef("alsa_output.speakers", True))

    assert fake.commands[-1] == ("pactl", "unload-module", "1")
    assert journal.load().original_loopback_module_id is None


def test_reader_or_player_failure_triggers_exactly_one_recovery_and_reraises(tmp_path: Path) -> None:
    fake = FakePactl()
    journal = RecoveryJournal(tmp_path / "routing.json")
    router = PipeWirePulseRouter(fake, journal, streams=lambda: [stream()])
    router.begin(router.select_unique(StreamSelector(application_name="Google Chrome", serial="chrome-a")), SinkRef("alsa_output.speakers", True))
    fake.commands.clear()

    def failing_reader_or_player() -> None:
        raise RuntimeError("reader or player failed")

    with pytest.raises(RuntimeError, match="reader or player failed"):
        run_protected(router, failing_reader_or_player)

    assert fake.commands == [
        ("pactl", "move-sink-input", "41", "alsa_output.speakers"),
        ("pactl", "unload-module", "1"),
    ]
    assert journal.load() is None
    assert all("set-default-sink" not in command for command in fake.commands)


class FailsWhenMovingBackToPhysicalSinkPactl(FakePactl):
    """Fails only the recovery move (back to the original physical sink), not the routing move."""

    def run(self, argv: tuple[str, ...], **_: object) -> str:
        if argv[:2] == ("pactl", "move-sink-input") and argv[3] == "alsa_output.speakers":
            self.commands.append(argv)
            raise RuntimeError("pactl unreachable during recovery")
        return super().run(argv)


def test_recovery_failure_during_protected_work_retains_the_journal(tmp_path: Path) -> None:
    fake = FailsWhenMovingBackToPhysicalSinkPactl()
    journal = RecoveryJournal(tmp_path / "routing.json")
    router = PipeWirePulseRouter(fake, journal, streams=lambda: [stream()])
    router.begin(router.select_unique(StreamSelector(application_name="Google Chrome", serial="chrome-a")), SinkRef("alsa_output.speakers", True))

    def failing_work() -> None:
        raise RuntimeError("reader failed")

    with pytest.raises(RuntimeError, match="pactl unreachable during recovery"):
        run_protected(router, failing_work)

    assert journal.load() is not None


def test_open_reference_sink_creates_reference_and_loopback_modules_journaling_before_each_mutation(tmp_path: Path) -> None:
    fake = FakePactl()
    journal = RecoveryJournal(tmp_path / "routing.json")
    router = PipeWirePulseRouter(fake, journal, streams=lambda: [stream()])
    lease = router.begin(
        router.select_unique(StreamSelector(application_name="Google Chrome", serial="chrome-a")),
        SinkRef("alsa_output.speakers", True),
    )
    fake.commands.clear()

    reference = router.open_reference_sink(lease, SinkRef("alsa_output.speakers", True))

    assert fake.commands == [
        ("pactl", "load-module", "module-null-sink", "sink_name=opendubstream-ref.41"),
        ("pactl", "load-module", "module-loopback", "source=opendubstream-ref.41.monitor", "sink=alsa_output.speakers"),
    ]
    assert reference.reference_sink == "opendubstream-ref.41"
    assert reference.reference_module_id == "1"
    assert reference.loopback_module_id == "1"
    assert reference.physical_sink == "alsa_output.speakers"
    snapshot = journal.load()
    assert snapshot.reference_sink == "opendubstream-ref.41"
    assert snapshot.reference_module_id == "1"
    assert snapshot.loopback_module_id == "1"
    # begin()'s Chrome-stream journal fields are untouched by this call.
    assert snapshot.virtual_sink == lease.virtual_sink
    assert snapshot.virtual_module_id == lease.virtual_module_id


def test_open_reference_sink_persists_the_reference_sink_name_before_loading_the_null_sink_module(tmp_path: Path) -> None:
    journal = RecoveryJournal(tmp_path / "routing.json")
    persisted_before_first_mutation: list[object] = []

    class PersistProbePactl(FakePactl):
        def run(self, argv: tuple[str, ...], **_: object) -> str:
            if argv[:3] == ("pactl", "load-module", "module-null-sink") and "sink_name=opendubstream-ref." in argv[3]:
                persisted_before_first_mutation.append(journal.load())
            return super().run(argv)

    fake = PersistProbePactl()
    router = PipeWirePulseRouter(fake, journal, streams=lambda: [stream()])
    lease = router.begin(
        router.select_unique(StreamSelector(application_name="Google Chrome", serial="chrome-a")),
        SinkRef("alsa_output.speakers", True),
    )

    router.open_reference_sink(lease, SinkRef("alsa_output.speakers", True))

    assert persisted_before_first_mutation[0].reference_sink == "opendubstream-ref.41"
    assert persisted_before_first_mutation[0].reference_module_id is None


def test_open_reference_sink_rejects_a_lease_that_does_not_match_the_journal(tmp_path: Path) -> None:
    fake = FakePactl()
    journal = RecoveryJournal(tmp_path / "routing.json")
    router = PipeWirePulseRouter(fake, journal, streams=lambda: [stream()])
    lease = router.begin(
        router.select_unique(StreamSelector(application_name="Google Chrome", serial="chrome-a")),
        SinkRef("alsa_output.speakers", True),
    )
    stale_lease = replace(lease, virtual_module_id="stale-module")
    fake.commands.clear()

    with pytest.raises(RoutingSafetyError, match="journal"):
        router.open_reference_sink(stale_lease, SinkRef("alsa_output.speakers", True))

    assert fake.commands == []


def test_open_reference_sink_rejects_when_no_journal_exists(tmp_path: Path) -> None:
    fake = FakePactl()
    router = PipeWirePulseRouter(fake, RecoveryJournal(tmp_path / "routing.json"), streams=lambda: [stream()])
    lease = router.begin(
        router.select_unique(StreamSelector(application_name="Google Chrome", serial="chrome-a")),
        SinkRef("alsa_output.speakers", True),
    )
    # A second router instance pointed at a fresh, journal-less path simulates a lease
    # whose journal was already cleared (e.g. after an unrelated recover()).
    orphan_router = PipeWirePulseRouter(fake, RecoveryJournal(tmp_path / "orphan-routing.json"), streams=lambda: [stream()])

    with pytest.raises(RoutingSafetyError, match="journal"):
        orphan_router.open_reference_sink(lease, SinkRef("alsa_output.speakers", True))


def test_recovery_unloads_loopback_then_reference_sink_before_the_chrome_stream_restoration(tmp_path: Path) -> None:
    fake = FakePactl()
    journal = RecoveryJournal(tmp_path / "routing.json")
    router = PipeWirePulseRouter(fake, journal, streams=lambda: [stream()])
    lease = router.begin(
        router.select_unique(StreamSelector(application_name="Google Chrome", serial="chrome-a")),
        SinkRef("alsa_output.speakers", True),
    )
    router.open_reference_sink(lease, SinkRef("alsa_output.speakers", True))
    fake.commands.clear()

    result = router.recover()

    assert result.restored is True
    assert fake.commands == [
        ("pactl", "unload-module", "1"),  # loopback, unloaded first (it depends on the reference sink)
        ("pactl", "unload-module", "1"),  # reference null-sink
        ("pactl", "move-sink-input", "41", "alsa_output.speakers"),
        ("pactl", "unload-module", "1"),  # virtual (Chrome) sink
    ]
    assert journal.load() is None


def test_recovery_is_a_no_op_for_reference_fields_on_an_old_shape_journal(tmp_path: Path) -> None:
    """A journal persisted before `open_reference_sink` ever ran (or by a version of this
    tool that predates it) has no reference/loopback module keys. `recover()` must restore
    the Chrome stream exactly as before -- issuing the same two commands, in the same
    order, with no extra unload-module calls for absent fields."""
    fake = FakePactl()
    journal_path = tmp_path / "routing.json"
    journal_path.write_text(
        json.dumps(
            {
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
        ),
        encoding="utf-8",
    )
    journal = RecoveryJournal(journal_path)
    router = PipeWirePulseRouter(fake, journal, streams=lambda: [stream()])

    result = router.recover()

    assert result.restored is True
    assert fake.commands == [
        ("pactl", "move-sink-input", "41", "alsa_output.speakers"),
        ("pactl", "unload-module", "7"),
    ]
    assert journal.load() is None


def test_router_rejects_stream_replaced_after_user_selection_before_mutation(tmp_path: Path) -> None:
    fake = FakePactl()
    selected = stream()
    router = PipeWirePulseRouter(
        fake,
        RecoveryJournal(tmp_path / "routing.json"),
        streams=lambda: [stream(serial="replaced")],
    )

    with pytest.raises(RoutingSafetyError, match="identity changed"):
        router.begin(selected, SinkRef("alsa_output.speakers", True))

    assert fake.commands == []
