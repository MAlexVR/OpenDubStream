from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "verify-real-audio-loop.py"


def load_module():
    spec = importlib.util.spec_from_file_location("verify_real_audio_loop", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_require_confirmation_blocks_without_explicit_confirmation() -> None:
    module = load_module()

    with pytest.raises(module.ConfirmationRequired, match="reference hardware"):
        module.require_confirmation(False, reason="reference hardware run")

    module.require_confirmation(True, reason="reference hardware run")  # does not raise


def test_bounded_output_hash_rejects_output_over_the_size_bound() -> None:
    module = load_module()

    with pytest.raises(module.BoundedOutputError, match="exceeds"):
        module.bounded_output_hash(b"x" * 10, max_bytes=5)

    assert module.bounded_output_hash(b"hello", max_bytes=5) == hashlib.sha256(b"hello").hexdigest()


def evidence_fields(**overrides: object) -> dict[str, object]:
    fields: dict[str, object] = dict(
        revision="rev-1",
        default_sink_before="alsa_output.speakers", default_sink_after="alsa_output.speakers",
        default_source_before="alsa_input.mic", default_source_after="alsa_input.mic",
        stream_identifier="41", virtual_sink="opendubstream.41", virtual_module_id="7",
        monitor_name="opendubstream.41.monitor", monitor_serial="mon-1",
        physical_sink_name="alsa_output.speakers", physical_sink_serial="sink-1",
        asr_seconds=0.3, translation_seconds=0.2, tts_seconds=0.3, total_seconds=0.8,
        calibration_decision="no-feedback", null_p_values=(0.5, 0.6, 0.7),
        positive_p_value=0.005, selected_p_value=0.5,
        selected_capture_plan={"schema": "opendubstream.selected-capture-plan/v1", "analysis_samples": 64_000},
        selected_topology_audit={
            "schema": "opendubstream.selected-topology-audit/v1",
            "pre_snapshot_sha256": "d" * 64,
            "during_snapshot_sha256": "e" * 64,
            "valid": True,
            "no_physical_to_selected_path": True,
            "reason": "",
        },
        baseline_digest="db82ba31a54ab1a70dc0de1e1a3256c2c9993217ea05eeba4c6a35ae1bd6bc72",
        routing_recovered=True,
        transcript_hash="a" * 64, translation_hash="b" * 64, synthesized_audio_hash="c" * 64,
    )
    fields.update(overrides)
    return fields


def test_build_hardware_e2e_evidence_rejects_missing_fields() -> None:
    module = load_module()
    fields = evidence_fields()
    del fields["monitor_serial"]

    with pytest.raises(module.EvidenceSchemaError, match="monitor_serial"):
        module.build_hardware_e2e_evidence(**fields)


def test_build_hardware_e2e_evidence_rejects_a_changed_default_device() -> None:
    module = load_module()

    with pytest.raises(module.EvidenceSchemaError, match="default sink changed"):
        module.build_hardware_e2e_evidence(**evidence_fields(default_sink_after="alsa_output.other"))
    with pytest.raises(module.EvidenceSchemaError, match="default source changed"):
        module.build_hardware_e2e_evidence(**evidence_fields(default_source_after="alsa_input.other"))


def test_build_hardware_e2e_evidence_produces_the_canonical_schema() -> None:
    module = load_module()

    evidence = module.build_hardware_e2e_evidence(**evidence_fields())

    assert evidence["schema"] == "opendubstream.hardware-e2e/v1"
    assert evidence["revision"] == "rev-1"
    assert evidence["feedback_absent"] is True
    assert evidence["selected_capture_plan"]["schema"] == "opendubstream.selected-capture-plan/v1"
    assert evidence["selected_topology_audit"]["valid"] is True


def test_selected_topology_recorder_snapshots_before_and_during_the_physical_player() -> None:
    module = load_module()
    plan = _selected_plan(module)
    events: list[str] = []
    snapshots = iter((_selected_graph_without_player(), _selected_graph()))

    def snapshot() -> dict[str, object]:
        events.append("snapshot")
        return next(snapshots)

    recorder = module.SelectedTopologyAuditRecorder(plan, snapshot)
    recorder.capture_before_playback()

    class FakePhysicalPlayer:
        def play(self) -> None:
            events.append("physical-player-started")
            recorder.capture_during_playback()
            events.append("physical-player-finished")

    FakePhysicalPlayer().play()
    audit = recorder.finish()

    assert events == ["snapshot", "physical-player-started", "snapshot", "physical-player-finished"]
    assert audit is not None
    assert audit.valid is True
    assert audit.no_physical_to_selected_path is True


def test_selected_topology_recorder_retries_the_during_snapshot_until_the_physical_player_registers() -> None:
    # Regression: `on_started` fires immediately after `Popen()`, before the spawned
    # `pw-play` process has registered its PipeWire node -- the "during" snapshot must
    # tolerate that registration race with a bounded, injectable-sleep retry loop.
    module = load_module()
    plan = _selected_plan(module)
    snapshots = iter((
        _selected_graph_without_player(),  # consumed by capture_before_playback
        _selected_graph_without_player(),  # during attempt 1: player not registered yet
        _selected_graph_without_player(),  # during attempt 2: still not registered
        _selected_graph(),  # during attempt 3: player has now registered
    ))
    sleeps: list[float] = []

    recorder = module.SelectedTopologyAuditRecorder(plan, lambda: next(snapshots), sleep=sleeps.append)
    recorder.capture_before_playback()
    recorder.capture_during_playback()
    audit = recorder.finish()

    assert len(sleeps) == 2
    assert sleeps[0] == sleeps[1]
    assert 0 < sleeps[0] < 1.0  # short, bounded retry interval -- not a multi-second stall
    assert audit is not None
    assert audit.valid is True
    assert audit.no_physical_to_selected_path is True


def test_selected_topology_recorder_during_snapshot_retry_is_bounded_and_fails_closed() -> None:
    # The player must never appear within budget in this test -- the retry must give up,
    # fall back to the last snapshot taken, and let `SelectedTopologyAudit` report
    # `inconclusive` rather than block indefinitely or fabricate validity.
    module = load_module()
    plan = _selected_plan(module)
    call_count = 0

    def snapshot() -> dict[str, object]:
        nonlocal call_count
        call_count += 1
        return _selected_graph_without_player()

    sleeps: list[float] = []
    recorder = module.SelectedTopologyAuditRecorder(plan, snapshot, sleep=sleeps.append)
    recorder.capture_before_playback()
    recorder.capture_during_playback()
    audit = recorder.finish()

    assert 0 < len(sleeps) < 10  # retried at least once, but bounded -- never indefinite
    assert 2 < call_count < 12  # bounded total snapshot calls (before + retried during)
    assert audit is not None
    assert audit.valid is False
    assert audit.no_physical_to_selected_path is False
    assert audit.reason == "inconclusive"


def test_build_hardware_e2e_evidence_rejects_missing_calibration_decision() -> None:
    module = load_module()
    fields = evidence_fields()
    del fields["calibration_decision"]

    with pytest.raises(module.EvidenceSchemaError, match="calibration_decision"):
        module.build_hardware_e2e_evidence(**fields)


def test_build_hardware_e2e_evidence_derives_feedback_absent_from_the_calibration_decision() -> None:
    module = load_module()

    no_feedback = module.build_hardware_e2e_evidence(**evidence_fields(calibration_decision="no-feedback"))
    inconclusive = module.build_hardware_e2e_evidence(**evidence_fields(calibration_decision="inconclusive"))
    feedback_present = module.build_hardware_e2e_evidence(**evidence_fields(calibration_decision="feedback-present"))

    assert no_feedback["feedback_absent"] is True
    assert inconclusive["feedback_absent"] is False
    assert feedback_present["feedback_absent"] is False
    assert no_feedback["null_p_values"] == (0.5, 0.6, 0.7)
    assert no_feedback["positive_p_value"] == 0.005
    assert no_feedback["selected_p_value"] == 0.5
    assert no_feedback["baseline_digest"] == "db82ba31a54ab1a70dc0de1e1a3256c2c9993217ea05eeba4c6a35ae1bd6bc72"


def test_main_never_runs_the_live_harness_without_confirm(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    module = load_module()

    exit_code = module.main([])

    assert exit_code == 2
    assert "--confirm" in capsys.readouterr().err


def test_run_confirmed_ties_every_stage_to_exactly_one_recovery_on_failure() -> None:
    module = load_module()
    calls: list[str] = []

    class FakeRouter:
        def recover(self) -> None:
            calls.append("recover")

    def failing_stage() -> None:
        calls.append("stage")
        raise RuntimeError("capture failed")

    with pytest.raises(RuntimeError, match="capture failed"):
        module.run_protected(FakeRouter(), failing_stage)

    assert calls == ["stage", "recover"]


def test_main_accepts_an_optional_transport_offset_samples_argument() -> None:
    module = load_module()

    exit_code = module.main(["--transport-offset-samples", "-17"])

    assert exit_code == 2  # still blocked without --confirm; the offset alone never triggers a live run


def test_require_unchanged_default_devices_rejects_a_changed_sink_or_source() -> None:
    module = load_module()

    module.require_unchanged_default_devices(
        sink_before="alsa_output.speakers", sink_after="alsa_output.speakers",
        source_before="alsa_input.mic", source_after="alsa_input.mic",
    )  # does not raise

    with pytest.raises(module.EvidenceSchemaError, match="default sink changed"):
        module.require_unchanged_default_devices(
            sink_before="alsa_output.speakers", sink_after="alsa_output.other",
            source_before="alsa_input.mic", source_after="alsa_input.mic",
        )
    with pytest.raises(module.EvidenceSchemaError, match="default source changed"):
        module.require_unchanged_default_devices(
            sink_before="alsa_output.speakers", sink_after="alsa_output.speakers",
            source_before="alsa_input.mic", source_after="alsa_input.other",
        )


def test_settle_and_collect_null_controls_defaults_to_a_six_second_settle_period() -> None:
    module = load_module()

    assert module._SETTLE_SECONDS == 6.0


def test_sample_counting_reader_tracks_cumulative_samples_and_the_full_stream() -> None:
    module = load_module()
    chunks = iter([b"\x01\x00" * 2, b"\x02\x00" * 3])
    reader = module.SampleCountingReader(lambda deadline: next(chunks, b""))

    first = reader(1.0)
    second = reader(1.0)

    assert first == b"\x01\x00" * 2
    assert second == b"\x02\x00" * 3
    assert reader.sample_count == 5
    assert reader.captured_bytes() == b"\x01\x00" * 2 + b"\x02\x00" * 3


def test_advance_reader_to_stops_once_the_target_is_reached() -> None:
    module = load_module()
    calls: list[float] = []
    chunks = iter([b"\x00\x00" * 4, b"\x00\x00" * 4])

    def fake_read(deadline: float) -> bytes:
        calls.append(deadline)
        return next(chunks, b"")

    reader = module.SampleCountingReader(fake_read)

    assert module.advance_reader_to(reader, 4, deadline=2.0) is True
    assert reader.sample_count == 4
    assert calls == [2.0]  # stops after exactly one read; the second chunk is never consumed


def test_advance_reader_to_fails_closed_when_the_reader_stalls() -> None:
    module = load_module()
    reader = module.SampleCountingReader(lambda deadline: b"")

    assert module.advance_reader_to(reader, 4, deadline=0.1, max_reads=3) is False


def test_settle_and_collect_null_controls_waits_the_predeclared_period_before_any_read() -> None:
    module = load_module()
    import opendubstream.infrastructure.audio.playback as playback

    config = playback.ProtocolConfig(analysis_samples=4)
    events: list[str] = []

    def fake_sleep(seconds: float) -> None:
        events.append(f"sleep:{seconds}")

    def fake_read(deadline: float) -> bytes:
        events.append("read")
        return b"\x01\x00" * 4

    reader = module.SampleCountingReader(fake_read)

    nulls = module.settle_and_collect_null_controls(reader, config, settle_seconds=6.0, sleep=fake_sleep, deadline=1.0)

    assert events[0] == "sleep:6.0"
    assert events.count("read") == 3
    assert nulls == (b"\x01\x00" * 4, b"\x01\x00" * 4, b"\x01\x00" * 4)
    assert all(len(null) == config.analysis_samples * 2 for null in nulls)


def test_settle_and_collect_null_controls_fails_closed_on_a_stalled_reader() -> None:
    module = load_module()
    import opendubstream.infrastructure.audio.playback as playback

    config = playback.ProtocolConfig(analysis_samples=4)
    stalled_reader = module.SampleCountingReader(lambda deadline: b"")

    result = module.settle_and_collect_null_controls(
        stalled_reader, config, settle_seconds=0.0, sleep=lambda seconds: None, deadline=0.1
    )

    assert result is None


def test_default_protocol_config_uses_three_64000_sample_null_windows() -> None:
    module = load_module()
    import opendubstream.infrastructure.audio.playback as playback

    config = playback.ProtocolConfig()
    reader = module.SampleCountingReader(lambda deadline: b"\x00\x00" * config.analysis_samples)

    nulls = module.settle_and_collect_null_controls(reader, config, settle_seconds=0.0, sleep=lambda seconds: None, deadline=1.0)

    assert config.analysis_samples == 64_000
    assert nulls is not None
    assert all(len(null) == 64_000 * 2 for null in nulls)


def test_build_positive_control_overwrites_exactly_the_declared_tag_window() -> None:
    module = load_module()
    import opendubstream.infrastructure.audio.playback as playback

    config = playback.ProtocolConfig(tag_samples=2, tag_insertion_samples=1, analysis_samples=4)
    null_zero = b"\x11\x00" * 4
    tag = b"\x22\x00\x33\x00"

    positive = module.build_positive_control(null_zero, tag, config)

    assert positive == b"\x11\x00" + tag + b"\x11\x00"
    assert len(positive) == len(null_zero)


def test_build_positive_control_rejects_a_null_or_tag_of_the_wrong_length() -> None:
    module = load_module()
    import opendubstream.infrastructure.audio.playback as playback

    config = playback.ProtocolConfig(tag_samples=2, tag_insertion_samples=1, analysis_samples=4)

    with pytest.raises(module.EvidenceSchemaError, match="one full null dataset"):
        module.build_positive_control(b"\x11\x00" * 3, b"\x22\x00\x33\x00", config)
    with pytest.raises(module.EvidenceSchemaError, match="one full null dataset"):
        module.build_positive_control(b"\x11\x00" * 4, b"\x22\x00", config)


def test_require_transport_offset_samples_rejects_missing_or_non_integer_values() -> None:
    module = load_module()

    with pytest.raises(module.TransportOffsetRequired, match="predeclared"):
        module.require_transport_offset_samples(None)
    with pytest.raises(module.TransportOffsetRequired, match="predeclared"):
        module.require_transport_offset_samples("17")

    assert module.require_transport_offset_samples(-17) == -17
    assert module.require_transport_offset_samples(0) == 0


# --- Task 1.2: `select_active_chrome_route` delegates to the extracted pure selection ---


def test_select_active_chrome_route_delegates_to_the_extracted_pure_selection_and_stays_byte_identical() -> None:
    """Regression for the `chrome_route.py` extraction (task 1.2): the CLI wiring --
    exact pactl calls issued, exact `StreamSelector`/`SinkRef` passed to the router, exact
    resulting `RouteLease` -- must stay identical to the pre-extraction inline
    implementation."""
    module = load_module()
    from opendubstream.domain.contracts import RouteLease, SinkRef, StreamRef, StreamSelector

    sink_inputs_json = json.dumps([
        {
            "index": 1, "corked": False, "sink": 5,
            "properties": {"application.name": "Google Chrome", "media.name": "Video", "object.serial": "serial-1"},
        },
    ])
    sinks_json = json.dumps([{"index": 5, "name": "alsa_output.speakers"}])

    class FakePactl:
        def __init__(self) -> None:
            self.calls: list[tuple[str, ...]] = []

        def run(self, argv: tuple[str, ...], **kwargs: object) -> str:
            self.calls.append(argv)
            return sink_inputs_json if argv[-1] == "sink-inputs" else sinks_json

    resolved_stream = StreamRef(
        identifier="1", application_name="Google Chrome", media_name="Video",
        serial="serial-1", sink_name="alsa_output.speakers",
    )

    class FakeRouter:
        def __init__(self) -> None:
            self.select_unique_calls: list[StreamSelector] = []
            self.begin_calls: list[tuple[StreamRef, SinkRef]] = []

        def select_unique(self, selector: StreamSelector) -> StreamRef:
            self.select_unique_calls.append(selector)
            return resolved_stream

        def begin(self, stream: StreamRef, physical: SinkRef) -> RouteLease:
            self.begin_calls.append((stream, physical))
            return RouteLease(
                stream=stream, virtual_sink="opendubstream.1", virtual_module_id="7",
                original_sink=stream.sink_name, physical_sink=stream.sink_name,
            )

    pactl = FakePactl()
    router = FakeRouter()

    lease = module.select_active_chrome_route(pactl, router)

    assert [call[-1] for call in pactl.calls] == ["sink-inputs", "sinks", "sink-inputs"]  # exact pre-extraction call order
    assert router.select_unique_calls == [StreamSelector(application_name="Google Chrome", serial="serial-1")]
    assert router.begin_calls == [(resolved_stream, SinkRef("alsa_output.speakers", True))]
    assert lease.virtual_sink == "opendubstream.1"
    assert lease.stream.serial == "serial-1"


def test_select_active_chrome_route_raises_the_exact_pre_extraction_error_when_no_stream_is_active() -> None:
    module = load_module()

    class FakePactl:
        def run(self, argv: tuple[str, ...], **kwargs: object) -> str:
            return "[]"

    class FakeRouter:
        pass

    with pytest.raises(RuntimeError, match="no actively playing \\(non-corked\\) Chrome stream found"):
        module.select_active_chrome_route(FakePactl(), FakeRouter())


def _lease() -> object:
    from opendubstream.domain.contracts import RouteLease, StreamRef

    return RouteLease(
        stream=StreamRef("41", "Google Chrome", "Video", "chrome-a", "alsa_output.speakers"),
        virtual_sink="opendubstream.41",
        virtual_module_id="77",
        original_sink="alsa_output.speakers",
        physical_sink="alsa_output.speakers",
    )


def _physical_sink_inventory() -> str:
    return json.dumps([{"name": "alsa_output.speakers", "properties": {"object.serial": "sink-serial", "device.bus": "pci"}}])


class _FakePlayer:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[tuple[object, bytes, float]] = []
        self.stop_calls = 0

    def play(self, target: object, pcm: bytes, deadline: float) -> None:
        self.calls.append((target, pcm, deadline))
        if self.error is not None:
            raise self.error

    def stop(self) -> None:
        self.stop_calls += 1


def test_play_predeclared_probe_emits_exactly_the_probe_pcm_with_nothing_appended() -> None:
    from opendubstream.domain.contracts import SinkRef

    module = load_module()
    target = SinkRef("alsa_output.speakers", True, "sink-serial")
    player = _FakePlayer()
    probe = b"\x00\x00" * 4 + b"\x01\x00" * 2

    resolved = module.play_predeclared_probe(player, _physical_sink_inventory, probe, target, _lease(), deadline=5.0)

    assert resolved == target
    assert player.calls == [(target, probe, 5.0)]  # exactly the predeclared bytes; no appended tag
    assert player.stop_calls == 0


def test_play_predeclared_probe_reports_the_exact_resolved_playback_target(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Diagnostic visibility: a low correlation score alone can't distinguish "tag reached
    the wrong device" from "tag reached the right device but wasn't found" -- the operator
    needs to see exactly which sink name/serial the probe was actually sent to."""
    from opendubstream.domain.contracts import SinkRef

    module = load_module()
    target = SinkRef("alsa_output.speakers", True, "sink-serial")
    player = _FakePlayer()
    probe = b"\x00\x00" * 4

    module.play_predeclared_probe(player, _physical_sink_inventory, probe, target, _lease(), deadline=5.0)

    diagnostic = capsys.readouterr().err
    assert "alsa_output.speakers" in diagnostic
    assert "sink-serial" in diagnostic


def test_play_predeclared_probe_stops_the_player_and_reraises_on_timeout() -> None:
    from opendubstream.domain.contracts import SinkRef

    module = load_module()
    target = SinkRef("alsa_output.speakers", True, "sink-serial")
    player = _FakePlayer(TimeoutError())

    with pytest.raises(module.PcmPlaybackError, match="timed out"):
        module.play_predeclared_probe(player, _physical_sink_inventory, b"\x00\x00", target, _lease(), deadline=1.0)

    assert player.stop_calls == 1


def _reference() -> object:
    from opendubstream.domain.contracts import ReferenceLease

    return ReferenceLease(
        reference_sink="opendubstream-ref.41",
        reference_module_id="99",
        loopback_module_id="100",
        physical_sink="alsa_output.speakers",
    )


def _reference_sink_inventory() -> str:
    return json.dumps([{"name": "opendubstream-ref.41"}])


def test_resolve_reference_target_accepts_the_owned_reference_sink() -> None:
    from opendubstream.domain.contracts import SinkRef

    module = load_module()
    target = SinkRef("opendubstream-ref.41", False)

    resolved = module.resolve_reference_target(_reference_sink_inventory(), target, _reference())

    assert resolved == target


def test_resolve_reference_target_rejects_a_physical_target() -> None:
    from opendubstream.domain.contracts import SinkRef

    module = load_module()
    target = SinkRef("opendubstream-ref.41", True)

    with pytest.raises(module.ReferenceTargetError, match="non-physical"):
        module.resolve_reference_target(_reference_sink_inventory(), target, _reference())


def test_resolve_reference_target_rejects_a_name_that_does_not_match_the_lease() -> None:
    from opendubstream.domain.contracts import SinkRef

    module = load_module()
    target = SinkRef("opendubstream-ref.other", False)

    with pytest.raises(module.ReferenceTargetError, match="owned reference sink"):
        module.resolve_reference_target(_reference_sink_inventory(), target, _reference())


def test_resolve_reference_target_rejects_missing_or_ambiguous_inventory() -> None:
    from opendubstream.domain.contracts import SinkRef

    module = load_module()
    target = SinkRef("opendubstream-ref.41", False)

    with pytest.raises(module.ReferenceTargetError, match="0"):
        module.resolve_reference_target("[]", target, _reference())
    with pytest.raises(module.ReferenceTargetError, match="2"):
        two = json.dumps([{"name": "opendubstream-ref.41"}, {"name": "opendubstream-ref.41"}])
        module.resolve_reference_target(two, target, _reference())


def test_resolve_reference_target_rejects_malformed_json() -> None:
    from opendubstream.domain.contracts import SinkRef

    module = load_module()
    target = SinkRef("opendubstream-ref.41", False)

    with pytest.raises(module.ReferenceTargetError, match="malformed"):
        module.resolve_reference_target("not json", target, _reference())


def test_play_predeclared_probe_to_reference_emits_exactly_the_probe_pcm_with_nothing_appended() -> None:
    from opendubstream.domain.contracts import SinkRef

    module = load_module()
    target = SinkRef("opendubstream-ref.41", False)
    player = _FakePlayer()
    probe = b"\x00\x00" * 4 + b"\x01\x00" * 2

    resolved = module.play_predeclared_probe_to_reference(
        player, _reference_sink_inventory, probe, target, _reference(), deadline=5.0
    )

    assert resolved == target
    assert player.calls == [(target, probe, 5.0)]
    assert player.stop_calls == 0


def test_play_predeclared_probe_to_reference_reports_the_resolved_target(capsys: pytest.CaptureFixture[str]) -> None:
    from opendubstream.domain.contracts import SinkRef

    module = load_module()
    target = SinkRef("opendubstream-ref.41", False)
    player = _FakePlayer()

    module.play_predeclared_probe_to_reference(
        player, _reference_sink_inventory, b"\x00\x00", target, _reference(), deadline=5.0
    )

    diagnostic = capsys.readouterr().err
    assert "opendubstream-ref.41" in diagnostic


def test_play_predeclared_probe_to_reference_stops_the_player_and_reraises_on_timeout() -> None:
    from opendubstream.domain.contracts import SinkRef

    module = load_module()
    target = SinkRef("opendubstream-ref.41", False)
    player = _FakePlayer(TimeoutError())

    with pytest.raises(module.PcmPlaybackError, match="timed out"):
        module.play_predeclared_probe_to_reference(
            player, _reference_sink_inventory, b"\x00\x00", target, _reference(), deadline=1.0
        )

    assert player.stop_calls == 1


def test_run_calibrated_observation_crops_the_selected_window_at_the_predeclared_byte_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_module()
    import opendubstream.infrastructure.audio.playback as playback

    config = playback.ProtocolConfig(
        code_count=1, tag_samples=2, tag_insertion_samples=2, analysis_samples=6, max_lag=1, permutation_count=100
    )
    null_marker = b"\x01\x00" * config.analysis_samples
    selected_marker = b"\x03\x00" * config.analysis_samples
    chunks = iter([b"\x00\x00", null_marker, null_marker, null_marker, selected_marker, b"\x00\x00" * 2])

    def fake_read_phrase(deadline: float) -> bytes:
        return next(chunks, b"")

    monkeypatch.setattr(module, "play_predeclared_probe", lambda *args, **kwargs: None)

    seen: dict[str, bytes] = {}

    def fake_score(observed_pcm: bytes, candidate_revision: str, dataset_label: str, cfg: object) -> "playback.CalibrationScore":
        seen[dataset_label] = observed_pcm
        if dataset_label == "positive":
            return playback.CalibrationScore(active_max=0.95, permutation_global_max=tuple([0.3] * config.permutation_count))
        return playback.CalibrationScore(active_max=0.05, permutation_global_max=tuple([0.6] * config.permutation_count))

    monkeypatch.setattr(module, "score_pcm_dataset", fake_score)

    class FakeRouter:
        def recover(self) -> None:
            return None

    decision = module.run_calibrated_observation(
        router=FakeRouter(),
        read_phrase=fake_read_phrase,
        player=object(),
        inventory=lambda: "[]",
        synthesized_pcm=b"\x02\x00" * 4,
        target=None,
        lease=None,
        candidate_revision="candidate-a",
        config=config,
        settle_seconds=0.0,
        sleep=lambda seconds: None,
    )

    assert seen["null-0"] == null_marker
    assert seen["null-1"] == null_marker
    assert seen["null-2"] == null_marker
    assert seen["selected"] == selected_marker
    assert seen["positive"] == module.build_positive_control(null_marker, playback.deterministic_tag(config, 0), config)
    assert decision.value == "no-feedback"


def test_run_calibrated_observation_warms_up_the_reader_before_collecting_null_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same class of bug already fixed in `measure_transport_offset_samples` (4.1c-fix3):
    `settle_and_collect_null_controls` sleeps before its own first read, so without a
    warm-up the reader (and therefore `parec`) may not have started flowing until after
    the settle countdown has already been spent."""
    module = load_module()
    import opendubstream.infrastructure.audio.playback as playback

    config = playback.ProtocolConfig(code_count=1, tag_samples=2, tag_insertion_samples=2, analysis_samples=6, max_lag=1, permutation_count=100)
    events: list[str] = []

    def fake_read_phrase(deadline: float) -> bytes:
        events.append("read")
        return b"\x00\x00" * 20

    class FakeRouter:
        def recover(self) -> None:
            return None

    monkeypatch.setattr(module, "play_predeclared_probe", lambda *args, **kwargs: events.append("play"))

    module.run_calibrated_observation(
        router=FakeRouter(),
        read_phrase=fake_read_phrase,
        player=object(),
        inventory=lambda: "[]",
        synthesized_pcm=b"\x00\x00",
        target=None,
        lease=None,
        candidate_revision="candidate-a",
        config=config,
        settle_seconds=0.0,
        sleep=lambda seconds: None,
    )

    assert events[0] == "read"
    assert "play" in events
    assert events.index("read") < events.index("play")


def test_selected_observation_rejects_reference_route_arguments() -> None:
    """The selected branch may not be converted into a tautological reference probe."""
    module = load_module()

    with pytest.raises(TypeError, match="reference"):
        module.run_calibrated_observation(  # type: ignore[call-arg]
            router=object(), read_phrase=lambda _: b"", player=object(), inventory=lambda: "[]",
            synthesized_pcm=b"", target=None, lease=None, candidate_revision="rev",
            reference=_reference(),
        )

def test_run_calibrated_observation_recovers_the_route_once_and_writes_inconclusive_on_a_reader_failure() -> None:
    module = load_module()
    import opendubstream.infrastructure.audio.playback as playback

    config = playback.ProtocolConfig(analysis_samples=4)
    recover_calls: list[str] = []

    class FakeRouter:
        def recover(self) -> None:
            recover_calls.append("recover")

    def failing_read(deadline: float) -> bytes:
        raise RuntimeError("reader crashed")

    decision = module.run_calibrated_observation(
        router=FakeRouter(),
        read_phrase=failing_read,
        player=object(),
        inventory=lambda: "[]",
        synthesized_pcm=b"\x00\x00",
        target=None,
        lease=None,
        candidate_revision="candidate-a",
        config=config,
        settle_seconds=0.0,
        sleep=lambda seconds: None,
    )

    assert recover_calls == ["recover"]
    assert decision.value == "inconclusive"
    assert decision.null_p_values == ()


def test_run_calibrated_observation_recovers_the_route_once_and_writes_inconclusive_on_a_player_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_module()
    import opendubstream.infrastructure.audio.playback as playback

    config = playback.ProtocolConfig(analysis_samples=2)
    recover_calls: list[str] = []

    class FakeRouter:
        def recover(self) -> None:
            recover_calls.append("recover")

    filler = iter([b"\x00\x00" * 2] * 10)

    def fake_read_phrase(deadline: float) -> bytes:
        return next(filler, b"")

    def failing_play(*args: object, **kwargs: object) -> None:
        raise module.PcmPlaybackError("PCM player timed out")

    monkeypatch.setattr(module, "play_predeclared_probe", failing_play)

    decision = module.run_calibrated_observation(
        router=FakeRouter(),
        read_phrase=fake_read_phrase,
        player=object(),
        inventory=lambda: "[]",
        synthesized_pcm=b"\x00\x00",
        target=None,
        lease=None,
        candidate_revision="candidate-a",
        config=config,
        settle_seconds=0.0,
        sleep=lambda seconds: None,
    )

    assert recover_calls == ["recover"]
    assert decision.value == "inconclusive"


def test_build_calibration_observation_call_wires_the_live_objects_into_run_calibrated_observation_kwargs() -> None:
    from opendubstream.domain.contracts import SinkRef

    module = load_module()

    class FakeCapture:
        def read_phrase(self, deadline: float) -> bytes:
            return b""

    class FakePactl:
        def __init__(self) -> None:
            self.calls: list[tuple[str, ...]] = []

        def run(self, argv: tuple[str, ...], **kwargs: object) -> str:
            self.calls.append(argv)
            return "[]"

    class FakeRouter:
        pass

    capture = FakeCapture()
    pactl = FakePactl()
    player = _FakePlayer()
    router = FakeRouter()
    physical_target = SinkRef("alsa_output.speakers", True, "sink-1")
    lease = _lease()
    selected_plan = _selected_plan(module)

    kwargs = module.build_calibration_observation_call(
        router=router,
        capture=capture,
        player=player,
        pactl=pactl,
        synthesized_pcm=b"\x01\x00",
        physical_target=physical_target,
        lease=lease,
        candidate_revision="rev-x",
        selected_plan=selected_plan,
    )

    assert kwargs["router"] is router
    assert kwargs["read_phrase"] == capture.read_phrase
    assert kwargs["player"] is player
    assert kwargs["synthesized_pcm"] == b"\x01\x00"
    assert kwargs["target"] is physical_target
    assert kwargs["lease"] is lease
    assert kwargs["candidate_revision"] == "rev-x"
    assert kwargs["selected_plan"] is selected_plan
    assert kwargs["inventory"]() == "[]"
    assert pactl.calls == [("pactl", "-f", "json", "list", "sinks")]


def test_selected_observation_builder_requires_the_live_selected_plan() -> None:
    from opendubstream.domain.contracts import SinkRef

    module = load_module()

    class FakeCapture:
        def read_phrase(self, deadline: float) -> bytes:
            return b""

    class FakePactl:
        def run(self, argv: tuple[str, ...], **kwargs: object) -> str:
            return "[]"

    with pytest.raises(TypeError, match="selected_plan"):
        module.build_calibration_observation_call(
            router=object(), capture=FakeCapture(), player=_FakePlayer(), pactl=FakePactl(),
            synthesized_pcm=b"", physical_target=SinkRef("alsa_output.speakers", True, "sink-1"),
            lease=_lease(), candidate_revision="rev",
        )


def test_selected_observation_call_rejects_reference_route_arguments() -> None:
    from opendubstream.domain.contracts import SinkRef

    module = load_module()

    class FakeCapture:
        def read_phrase(self, deadline: float) -> bytes:
            return b""

    class FakePactl:
        def run(self, argv: tuple[str, ...], **kwargs: object) -> str:
            return "[]"

    with pytest.raises(TypeError, match="reference"):
        module.build_calibration_observation_call(  # type: ignore[call-arg]
            router=object(), capture=FakeCapture(), player=_FakePlayer(), pactl=FakePactl(),
            synthesized_pcm=b"", physical_target=SinkRef("alsa_output.speakers", True, "sink-1"),
            lease=_lease(), candidate_revision="rev", reference=_reference(),
        )

def test_measure_transport_offset_samples_derives_the_offset_from_the_best_scoring_position(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_module()
    import opendubstream.infrastructure.audio.playback as playback

    config = playback.ProtocolConfig(tag_samples=4, tag_insertion_samples=2, analysis_samples=4)
    tag = playback.deterministic_tag(config, 0)
    true_offset = 5
    lead_silence_samples = config.tag_insertion_samples + true_offset
    captured = b"\x00\x00" * lead_silence_samples + tag + b"\x00\x00" * 100

    chunks = iter([b"\x00\x00", captured])  # first chunk is the warm-up read, consumed before settle/play

    def fake_read_phrase(deadline: float) -> bytes:
        return next(chunks, b"")

    class FakeRouter:
        def recover(self) -> None:
            return None

    monkeypatch.setattr(module, "play_predeclared_probe", lambda *args, **kwargs: None)

    offset = module.measure_transport_offset_samples(
        router=FakeRouter(),
        read_phrase=fake_read_phrase,
        player=object(),
        inventory=lambda: "[]",
        target=None,
        lease=None,
        config=config,
        settle_seconds=0.0,
        sleep=lambda seconds: None,
        search_range_samples=60,
        confidence_floor=0.5,
    )

    assert offset == true_offset


def test_measure_transport_offset_samples_warms_up_the_reader_before_playing_the_tag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Confirmed live (2026-09-03, parec(1)/PulseAudio LatencyControl docs): a freshly
    launched `parec` subprocess negotiates the server's default (often multi-second)
    latency before it delivers its first byte. Playing the tag before the reader has
    started actually flowing means the tag can fully finish before any of it is ever
    captured. The reader MUST be warmed up (its first real read issued) before the
    settle sleep and the tag playback -- not after."""
    module = load_module()
    import opendubstream.infrastructure.audio.playback as playback

    config = playback.ProtocolConfig(tag_samples=4, tag_insertion_samples=2, analysis_samples=4)
    events: list[str] = []

    def fake_read_phrase(deadline: float) -> bytes:
        events.append("read")
        return b"\x00\x00" * 200

    class FakeRouter:
        def recover(self) -> None:
            return None

    monkeypatch.setattr(module, "play_predeclared_probe", lambda *args, **kwargs: events.append("play"))

    module.measure_transport_offset_samples(
        router=FakeRouter(),
        read_phrase=fake_read_phrase,
        player=object(),
        inventory=lambda: "[]",
        target=None,
        lease=None,
        config=config,
        settle_seconds=0.0,
        sleep=lambda seconds: None,
        search_range_samples=10,
        confidence_floor=0.5,
    )

    assert events[0] == "read"
    assert "play" in events
    assert events.index("read") < events.index("play")


def test_measure_transport_offset_samples_fails_closed_when_the_capture_stalls(monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_module()
    import opendubstream.infrastructure.audio.playback as playback

    config = playback.ProtocolConfig(analysis_samples=4)

    class FakeRouter:
        def recover(self) -> None:
            return None

    monkeypatch.setattr(module, "play_predeclared_probe", lambda *args, **kwargs: None)

    offset = module.measure_transport_offset_samples(
        router=FakeRouter(),
        read_phrase=lambda deadline: b"",
        player=object(),
        inventory=lambda: "[]",
        target=None,
        lease=None,
        config=config,
        settle_seconds=0.0,
        sleep=lambda seconds: None,
        search_range_samples=10,
        confidence_floor=0.5,
    )

    assert offset is None


def test_measure_transport_offset_samples_reports_captured_vs_needed_when_the_capture_stalls(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A silent `None` return doesn't tell an operator whether nothing was captured or a
    real match just missed the confidence floor -- these are different problems to fix."""
    module = load_module()
    import opendubstream.infrastructure.audio.playback as playback

    config = playback.ProtocolConfig(analysis_samples=4)

    class FakeRouter:
        def recover(self) -> None:
            return None

    monkeypatch.setattr(module, "play_predeclared_probe", lambda *args, **kwargs: None)

    module.measure_transport_offset_samples(
        router=FakeRouter(),
        read_phrase=lambda deadline: b"",
        player=object(),
        inventory=lambda: "[]",
        target=None,
        lease=None,
        config=config,
        settle_seconds=0.0,
        sleep=lambda seconds: None,
        search_range_samples=10,
        confidence_floor=0.5,
    )

    diagnostic = capsys.readouterr().err
    assert "captured=0" in diagnostic
    assert "needed=" in diagnostic


def test_measure_transport_offset_samples_reports_the_best_score_found_below_the_floor(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = load_module()
    import opendubstream.infrastructure.audio.playback as playback

    config = playback.ProtocolConfig(tag_samples=4, tag_insertion_samples=2, analysis_samples=4)
    uncorrelated = b"\x01\x00" * 100  # non-empty capture, but no real tag match anywhere in it

    class FakeRouter:
        def recover(self) -> None:
            return None

    monkeypatch.setattr(module, "play_predeclared_probe", lambda *args, **kwargs: None)

    offset = module.measure_transport_offset_samples(
        router=FakeRouter(),
        read_phrase=lambda deadline: uncorrelated,
        player=object(),
        inventory=lambda: "[]",
        target=None,
        lease=None,
        config=config,
        settle_seconds=0.0,
        sleep=lambda seconds: None,
        search_range_samples=60,
        confidence_floor=0.99,
    )

    assert offset is None
    diagnostic = capsys.readouterr().err
    assert "best_correlation=" in diagnostic
    assert "confidence_floor=0.99" in diagnostic


def test_measure_transport_offset_samples_recovers_the_route_once_on_a_playback_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_module()
    import opendubstream.infrastructure.audio.playback as playback

    config = playback.ProtocolConfig(analysis_samples=4)
    recover_calls: list[str] = []

    class FakeRouter:
        def recover(self) -> None:
            recover_calls.append("recover")

    def failing_play(*args: object, **kwargs: object) -> None:
        raise module.PcmPlaybackError("PCM player timed out")

    monkeypatch.setattr(module, "play_predeclared_probe", failing_play)

    offset = module.measure_transport_offset_samples(
        router=FakeRouter(),
        read_phrase=lambda deadline: b"\x00\x00" * 4,
        player=object(),
        inventory=lambda: "[]",
        target=None,
        lease=None,
        config=config,
        settle_seconds=0.0,
        sleep=lambda seconds: None,
        search_range_samples=10,
        confidence_floor=0.5,
    )

    assert offset is None
    assert recover_calls == ["recover"]


def test_measure_transport_offset_samples_plays_to_the_reference_sink_when_a_reference_lease_is_supplied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When `reference` is supplied, playback must route through
    `play_predeclared_probe_to_reference` (the owned reference sink + loopback path), not
    the physical-target `play_predeclared_probe` -- proving the wiring described in
    design.md's owned-reference-sink amendment without touching real hardware."""
    from opendubstream.domain.contracts import SinkRef

    module = load_module()
    import opendubstream.infrastructure.audio.playback as playback

    config = playback.ProtocolConfig(tag_samples=4, tag_insertion_samples=2, analysis_samples=4)
    tag = playback.deterministic_tag(config, 0)
    true_offset = 3
    lead_silence_samples = config.tag_insertion_samples + true_offset
    captured = b"\x00\x00" * lead_silence_samples + tag + b"\x00\x00" * 100

    chunks = iter([b"\x00\x00", captured])  # first chunk is the warm-up read, consumed before settle/play

    def fake_read_phrase(deadline: float) -> bytes:
        return next(chunks, b"")

    class FakeRouter:
        def recover(self) -> None:
            return None

    physical_calls: list[object] = []
    reference_calls: list[object] = []
    monkeypatch.setattr(module, "play_predeclared_probe", lambda *args, **kwargs: physical_calls.append(args))
    monkeypatch.setattr(
        module, "play_predeclared_probe_to_reference", lambda *args, **kwargs: reference_calls.append(args)
    )

    offset = module.measure_transport_offset_samples(
        router=FakeRouter(),
        read_phrase=fake_read_phrase,
        player=object(),
        inventory=lambda: "[]",
        target=SinkRef("opendubstream-ref.41", False),
        lease=_lease(),
        reference=_reference(),
        config=config,
        settle_seconds=0.0,
        sleep=lambda seconds: None,
        search_range_samples=60,
        confidence_floor=0.5,
    )

    assert offset == true_offset
    assert len(reference_calls) == 1
    assert physical_calls == []


def test_build_transport_offset_measurement_call_wires_the_live_objects_into_measurement_kwargs() -> None:
    from opendubstream.domain.contracts import SinkRef

    module = load_module()

    class FakeCapture:
        def read_phrase(self, deadline: float) -> bytes:
            return b""

    class FakePactl:
        def __init__(self) -> None:
            self.calls: list[tuple[str, ...]] = []

        def run(self, argv: tuple[str, ...], **kwargs: object) -> str:
            self.calls.append(argv)
            return "[]"

    class FakeRouter:
        pass

    capture = FakeCapture()
    pactl = FakePactl()
    player = _FakePlayer()
    router = FakeRouter()
    physical_target = SinkRef("alsa_output.speakers", True, "sink-1")
    lease = _lease()

    kwargs = module.build_transport_offset_measurement_call(
        router=router,
        capture=capture,
        player=player,
        pactl=pactl,
        physical_target=physical_target,
        lease=lease,
    )

    assert kwargs["router"] is router
    assert kwargs["read_phrase"] == capture.read_phrase
    assert kwargs["player"] is player
    assert kwargs["target"] is physical_target
    assert kwargs["lease"] is lease
    assert kwargs["inventory"]() == "[]"
    assert pactl.calls == [("pactl", "-f", "json", "list", "sinks")]
    assert kwargs["reference"] is None


def test_build_transport_offset_measurement_call_passes_through_a_supplied_reference_lease() -> None:
    from opendubstream.domain.contracts import SinkRef

    module = load_module()

    class FakeCapture:
        def read_phrase(self, deadline: float) -> bytes:
            return b""

    class FakePactl:
        def run(self, argv: tuple[str, ...], **kwargs: object) -> str:
            return "[]"

    class FakeRouter:
        pass

    reference = _reference()
    kwargs = module.build_transport_offset_measurement_call(
        router=FakeRouter(),
        capture=FakeCapture(),
        player=_FakePlayer(),
        pactl=FakePactl(),
        physical_target=SinkRef("opendubstream-ref.41", False),
        lease=_lease(),
        reference=reference,
    )

    assert kwargs["reference"] is reference


def test_main_measure_transport_offset_still_requires_confirm(capsys: pytest.CaptureFixture[str]) -> None:
    module = load_module()

    exit_code = module.main(["--measure-transport-offset"])

    assert exit_code == 2
    assert "--confirm" in capsys.readouterr().err


def _selected_graph(*, include_path: bool = False, player_serial: str = "player-1") -> dict[str, object]:
    # Shaped like real `pw-dump` output: a sink's monitor is never a distinct
    # `PipeWire:Interface:Node` -- it only exists as a pair of monitor Ports
    # (`port.monitor: true`) attached to the *same* Node as the sink itself.
    # There is deliberately no node named `f"{virtual_sink_name}.monitor"` here,
    # matching the real graph shape confirmed by both a throwaway pactl/pw-dump
    # diagnostic and both real failed 4.1d-c hardware runs' own recorded evidence.
    nodes: list[dict[str, object]] = [
        {"id": "player", "serial": player_serial, "name": "pw-play", "role": "physical-player"},
        {"id": "physical", "serial": "sink-1", "name": "alsa_output.speakers", "role": "physical-target"},
        {"id": "virtual", "serial": "virtual-1", "name": "opendubstream.41", "module_id": "7", "role": "virtual-sink"},
    ]
    links: list[dict[str, str]] = []
    if include_path:
        links.extend(({"from": "physical", "to": "virtual"}, {"from": "player", "to": "physical"}))
    return {"nodes": nodes, "links": links}


def _selected_graph_without_player() -> dict[str, object]:
    graph = _selected_graph()
    graph["nodes"] = [node for node in graph["nodes"] if node["role"] != "physical-player"]
    return graph


def _selected_plan(module: object):
    # `owned_monitor_serial` equals `virtual_sink_serial` by construction: both real
    # failed 4.1d-c hardware runs recorded exactly this ("1348"=="1348", "1434"=="1434"),
    # because PipeWire's monitor Ports live on the sink's own Node, not a separate one.
    return module.SelectedCapturePlan.create(
        owned_monitor_name="opendubstream.41.monitor", owned_monitor_serial="virtual-1",
        physical_target_name="alsa_output.speakers", physical_target_serial="sink-1",
        virtual_sink_name="opendubstream.41", virtual_sink_serial="virtual-1",
        virtual_sink_module_id="7", candidate_revision="candidate-a",
    )


def test_selected_capture_plan_freezes_the_exact_unshifted_window_and_coverage() -> None:
    module = load_module()
    plan = _selected_plan(module)
    before = 40_000
    completed = before + 96_000
    pcm = b"\x00\x00" * (completed + 1)

    window = plan.extract_selected_window(
        captured_pcm=pcm, reader_count_before_player=before,
        reader_count_after_completion=completed,
        capture_started_monotonic_ns=10, player_invoked_monotonic_ns=20,
        capture_completed_monotonic_ns=30,
    )

    assert plan.schema == "opendubstream.selected-capture-plan/v1"
    assert plan.pre_roll_samples == 32_000
    assert plan.analysis_samples == 64_000
    assert plan.post_roll_samples == 32_000
    assert len(window) == 64_000 * 2
    assert plan.completion_target(before) == before + 96_000


@pytest.mark.parametrize(
    ("frame_bytes", "sample_count", "completed", "started", "player", "finished"),
    [
        (2, 200_000, 135_999, 10, 20, 30),
        (1, 200_000, 136_000, 10, 20, 30),
        (2, 200_000, 136_000, 20, 10, 30),
        (2, 100, 136_000, 10, 20, 30),
    ],
)
def test_selected_capture_plan_rejects_incomplete_noncontiguous_or_nonmonotonic_coverage(
    frame_bytes: int, sample_count: int, completed: int, started: int, player: int, finished: int,
) -> None:
    pcm = b"\x00" * frame_bytes * sample_count
    module = load_module()
    plan = _selected_plan(module)

    assert plan.extract_selected_window(
        captured_pcm=pcm, reader_count_before_player=40_000,
        reader_count_after_completion=completed,
        capture_started_monotonic_ns=started, player_invoked_monotonic_ns=player,
        capture_completed_monotonic_ns=finished,
    ) is None


def test_selected_topology_audit_requires_identical_complete_no_path_snapshots() -> None:
    module = load_module()
    plan = _selected_plan(module)

    audit = module.SelectedTopologyAudit.from_snapshots(plan, _selected_graph(), _selected_graph())

    assert audit.valid is True
    assert audit.no_physical_to_selected_path is True
    assert audit.pre_snapshot_sha256 == audit.during_snapshot_sha256


@pytest.mark.parametrize(
    ("pre", "during"),
    [
        (_selected_graph(include_path=True), _selected_graph(include_path=True)),
        (_selected_graph(), _selected_graph(player_serial="player-2")),
        ({"nodes": [], "links": []}, _selected_graph()),
        (_selected_graph(), {"nodes": "malformed", "links": []}),
    ],
)
def test_selected_topology_audit_fails_closed_for_path_drift_or_invalid_graph(
    pre: dict[str, object], during: dict[str, object],
) -> None:
    module = load_module()

    audit = module.SelectedTopologyAudit.from_snapshots(_selected_plan(module), pre, during)

    assert audit.valid is False
    assert audit.no_physical_to_selected_path is False


def test_selected_decision_admission_fails_closed_without_a_valid_topology_audit() -> None:
    module = load_module()
    import opendubstream.infrastructure.audio.playback as playback

    no_feedback = playback.CalibrationDecision("no-feedback", (0.5,), 0.001, 0.5, "baseline")
    feedback_present = playback.CalibrationDecision("feedback-present", (0.5,), 0.001, 0.001, "baseline")
    valid_audit = module.SelectedTopologyAudit.from_snapshots(_selected_plan(module), _selected_graph(), _selected_graph())

    assert module.admit_selected_decision(no_feedback, None).value == "inconclusive"
    assert module.admit_selected_decision(feedback_present, None).value == "inconclusive"
    assert module.admit_selected_decision(no_feedback, valid_audit).value == "no-feedback"


def test_selected_topology_audit_does_not_false_negative_when_pipewire_has_no_separate_monitor_node() -> None:
    """Regression for the false negative that hit BOTH real 4.1d-c hardware runs (2026-09-04).

    Both live runs completed cleanly end-to-end with clean statistics
    (`positive_p_value=0.001`, `selected_p_value=0.993` then `1.0`) but still came back
    `selected_topology_audit.valid=false, reason="inconclusive"`. Root cause: real PipeWire
    never exposes a sink's monitor as a distinct `PipeWire:Interface:Node` -- it is only a
    pair of monitor Ports on the *same* Node as the sink -- so a node named
    `f"{virtual_sink_name}.monitor"` never exists, and the old `role="selected-monitor"`
    lookup unconditionally found zero matches. This fixture matches that real shape
    (confirmed both by a throwaway pactl/pw-dump diagnostic and by both real failed runs'
    own recorded `monitor_serial`==`virtual_sink_serial` evidence: run 1 "1348"=="1348",
    run 2 "1434"=="1434"). With no physical-to-virtual path present, the audit must be
    valid -- this is a false negative in the pre-fix code, not a genuine feedback path.
    """
    module = load_module()

    audit = module.SelectedTopologyAudit.from_snapshots(_selected_plan(module), _selected_graph(), _selected_graph())

    assert audit.valid is True
    assert audit.no_physical_to_selected_path is True
    assert audit.reason == ""


def test_selected_topology_audit_fails_closed_when_monitor_serial_invariant_is_violated() -> None:
    """The monitor-is-an-alias-of-the-virtual-sink invariant (`owned_monitor_serial ==
    virtual_sink_serial`) holds by construction on every PipeWire system observed so far,
    but the audit must still fail closed -- never crash uncaught, never fabricate
    validity -- if a differently-configured PipeWire version ever violates it."""
    module = load_module()
    mismatched_plan = module.SelectedCapturePlan.create(
        owned_monitor_name="opendubstream.41.monitor", owned_monitor_serial="mismatched-serial",
        physical_target_name="alsa_output.speakers", physical_target_serial="sink-1",
        virtual_sink_name="opendubstream.41", virtual_sink_serial="virtual-1",
        virtual_sink_module_id="7", candidate_revision="candidate-a",
    )

    audit = module.SelectedTopologyAudit.from_snapshots(mismatched_plan, _selected_graph(), _selected_graph())

    assert audit.valid is False
    assert audit.no_physical_to_selected_path is False
    assert audit.reason == "inconclusive"


# --- Task 4.2: canonical `evidence/hardware-rerun.json` admission -------------------------


def test_evidence_path_targets_the_active_change_not_the_historical_one() -> None:
    module = load_module()

    path_text = str(module.EVIDENCE_PATH)

    assert "calibrate-feedback-isolation" in path_text
    assert "complete-real-audio-loop" not in path_text
    assert module.EVIDENCE_PATH.name == "hardware-rerun.json"


def test_build_hardware_rerun_raw_payload_maps_evidence_into_the_canonical_raw_shape() -> None:
    module = load_module()
    evidence = module.build_hardware_e2e_evidence(**evidence_fields())

    payload = module.build_hardware_rerun_raw_payload(evidence, now=lambda: "2026-09-04T00:00:00Z")

    assert payload["schema"] == "opendubstream.hardware-rerun/v1"
    assert payload["object_id"] == "hardware-rerun/rev-1/hardware-rerun"
    assert payload["candidate_revision"] == "rev-1"
    assert payload["recorded_at"] == "2026-09-04T00:00:00Z"
    assert payload["baseline_reference"] == {"sha256": evidence["baseline_digest"]}
    protocol = payload["protocol"]
    assert protocol["permutation_version"] == "circular-shift-guarded-v1"
    assert protocol["alpha"] == 0.01
    assert protocol["code_count"] == 16
    assert protocol["tag_samples"] == 800
    assert protocol["analysis_samples"] == 64_000
    assert protocol["permutation_count"] == 999
    assert protocol["max_lag"] == 3_200
    assert payload["datasets"] == [{"kind": "hardware-rerun-observation", **evidence}]
    assert payload["decision"] == {
        "calibration_decision": evidence["calibration_decision"],
        "feedback_absent": evidence["feedback_absent"],
    }
    assert "canonical_payload_sha256" not in payload


def test_build_hardware_rerun_raw_payload_defaults_to_a_real_utc_clock_injected_as_a_seam() -> None:
    import datetime as _datetime

    module = load_module()
    evidence = module.build_hardware_e2e_evidence(**evidence_fields())

    payload = module.build_hardware_rerun_raw_payload(evidence)

    assert payload["recorded_at"].endswith("Z")
    _datetime.datetime.strptime(payload["recorded_at"], "%Y-%m-%dT%H:%M:%SZ")


def test_build_hardware_rerun_raw_payload_satisfies_receipts_required_raw_fields() -> None:
    module = load_module()
    from opendubstream.application import receipts

    evidence = module.build_hardware_e2e_evidence(**evidence_fields())
    payload = module.build_hardware_rerun_raw_payload(evidence, now=lambda: "2026-09-04T00:00:00Z")

    raw_json = receipts.canonical_raw_json(payload)
    parsed = json.loads(raw_json)

    assert set(receipts.REQUIRED_RAW_FIELDS) <= parsed.keys()
    assert parsed["canonical_payload_sha256"]


def test_persist_and_admit_hardware_rerun_evidence_admits_the_written_canonical_object(tmp_path: Path) -> None:
    module = load_module()
    evidence = module.build_hardware_e2e_evidence(**evidence_fields())

    admission = module.persist_and_admit_hardware_rerun_evidence(evidence, openspec_root=tmp_path)

    assert admission.admitted is True
    assert admission.object_id == "hardware-rerun/rev-1/hardware-rerun"
    assert admission.candidate_revision == "rev-1"
    written = tmp_path / "evidence/hardware-rerun/hardware-rerun/rev-1/hardware-rerun.json"
    assert written.exists()
    assert json.loads(written.read_text(encoding="utf-8"))["schema"] == "opendubstream.hardware-rerun/v1"


def test_admission_rejects_the_historical_flat_hardware_e2e_shape_at_the_hardware_rerun_path(tmp_path: Path) -> None:
    """Neither the historical `hardware-e2e/v1` shape nor its bare fields can satisfy
    canonical `hardware-rerun` admission, even placed at the exact deterministic path."""
    module = load_module()
    from opendubstream.application import receipts

    old_flat_evidence = module.build_hardware_e2e_evidence(**evidence_fields())
    object_id = "hardware-rerun/rev-1/hardware-rerun"
    target = tmp_path / "evidence/hardware-rerun" / f"{object_id}.json"
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps(old_flat_evidence, sort_keys=True, separators=(",", ":")), encoding="utf-8")

    admission = receipts.admit_canonical_openspec_evidence(
        openspec_root=tmp_path, kind="hardware-rerun", object_id=object_id, candidate_revision="rev-1",
    )

    assert admission.admitted is False
    assert "required evidence fields are missing" in admission.reason


def test_admission_does_not_find_hardware_rerun_evidence_written_under_the_historical_changes_root(
    tmp_path: Path,
) -> None:
    """The historical `complete-real-audio-loop` evidence store and the active
    `calibrate-feedback-isolation` evidence store are not interchangeable."""
    module = load_module()
    from opendubstream.application import receipts

    historical_root = tmp_path / "complete-real-audio-loop"
    active_root = tmp_path / "calibrate-feedback-isolation"
    evidence = module.build_hardware_e2e_evidence(**evidence_fields())

    admission_under_historical = module.persist_and_admit_hardware_rerun_evidence(evidence, openspec_root=historical_root)
    assert admission_under_historical.admitted is True  # sanity: valid evidence in its own store

    admission_under_active = receipts.admit_canonical_openspec_evidence(
        openspec_root=active_root, kind="hardware-rerun",
        object_id=admission_under_historical.object_id,
        candidate_revision=admission_under_historical.candidate_revision,
    )

    assert admission_under_active.admitted is False
    # OSError text is locale-dependent (e.g. Spanish "No existe el fichero o el directorio"
    # instead of "No such file or directory"); the missing object's own id is not.
    assert admission_under_historical.object_id in admission_under_active.reason


def test_admission_rejects_evidence_relabelled_with_a_mismatched_candidate_revision(tmp_path: Path) -> None:
    """Reusing/relabelling one revision's admitted evidence for another revision must fail."""
    module = load_module()
    from opendubstream.application import receipts

    evidence = module.build_hardware_e2e_evidence(**evidence_fields(revision="rev-1"))
    admission = module.persist_and_admit_hardware_rerun_evidence(evidence, openspec_root=tmp_path)
    assert admission.admitted is True

    reused = receipts.admit_canonical_openspec_evidence(
        openspec_root=tmp_path, kind="hardware-rerun",
        object_id=admission.object_id, candidate_revision="rev-2",
    )

    assert reused.admitted is False
    assert "candidate revision" in reused.reason


def test_main_confirmed_run_never_claims_no_feedback_without_admitted_canonical_evidence(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path,
) -> None:
    module = load_module()
    from opendubstream.application import receipts

    evidence = module.build_hardware_e2e_evidence(**evidence_fields())  # calibration_decision="no-feedback"
    monkeypatch.setattr(module, "run_confirmed", lambda **kwargs: evidence)
    monkeypatch.setattr(module, "EVIDENCE_PATH", tmp_path / "hardware-rerun.json")
    recorded: dict[str, object] = {}

    def fake_persist(passed_evidence: dict[str, object]) -> receipts.CanonicalOpenSpecEvidenceAdmission:
        recorded["evidence"] = passed_evidence
        return receipts.CanonicalOpenSpecEvidenceAdmission(False, None, None, None, None, "canonical evidence write failed")

    monkeypatch.setattr(module, "persist_and_admit_hardware_rerun_evidence", fake_persist)

    exit_code = module.main(["--confirm", "--transport-offset-samples", "94124"])

    assert recorded["evidence"] is evidence
    assert exit_code != 0
    assert "canonical evidence write failed" in capsys.readouterr().err
    assert (tmp_path / "hardware-rerun.json").exists()  # flat dump is still written for human debugging


def test_main_confirmed_run_exits_cleanly_when_hardware_rerun_evidence_is_admitted(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path,
) -> None:
    module = load_module()
    from opendubstream.application import receipts

    evidence = module.build_hardware_e2e_evidence(**evidence_fields())
    monkeypatch.setattr(module, "run_confirmed", lambda **kwargs: evidence)
    monkeypatch.setattr(module, "EVIDENCE_PATH", tmp_path / "hardware-rerun.json")
    monkeypatch.setattr(
        module, "persist_and_admit_hardware_rerun_evidence",
        lambda passed_evidence: receipts.CanonicalOpenSpecEvidenceAdmission(
            True, "hardware-rerun/rev-1/hardware-rerun", "rev-1",
            "evidence/hardware-rerun/hardware-rerun/rev-1/hardware-rerun.json", "d" * 64, None,
        ),
    )

    exit_code = module.main(["--confirm", "--transport-offset-samples", "94124"])

    assert exit_code == 0
    assert "ERROR" not in capsys.readouterr().err
