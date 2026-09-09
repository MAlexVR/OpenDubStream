"""RED-first fakes-only coverage for `DubbingSessionRunner` (Phase 2, tasks 2.1-2.6).

No real PipeWire, Chrome, GPU, or model call anywhere: the router, capture, player, and
CUDA probe are all fakes; `LocalDubbingPipeline` and `FixedCapacityScheduler` are the real
(already-tested) pure/fakes-testable objects `run_confirmed` itself uses, matching this
project's established "reuse the real object, fake only the hardware/process boundary"
seam (see `tests/integration/test_routing.py`, `tests/integration/test_pipeline.py`).
"""

from __future__ import annotations

import threading
from typing import Callable

from opendubstream.application.dubbing_session import DubbingSessionRunner
from opendubstream.application.session_state import SessionEvent, SessionStage
from opendubstream.domain.contracts import AudioMixSettings, AudioMode, OriginalMixLease, RecoveryResult, RouteLease, SinkRef, StreamRef, StreamSelector
from opendubstream.domain.pipeline import Utterance
from opendubstream.pipeline.local_pipeline import LocalDubbingPipeline
from opendubstream.pipeline.scheduler import FixedCapacityScheduler

PHYSICAL_SINK = SinkRef("alsa_output.speakers", True, "phys-serial")


def chrome_stream(identifier: str = "41", serial: str = "chrome-a") -> StreamRef:
    return StreamRef(identifier, "Google Chrome", "Video", serial, "alsa_output.speakers")


def one_active_chrome_sink_inputs(serial: str = "chrome-a") -> list[dict[str, object]]:
    return [{"corked": False, "properties": {"object.serial": serial}}]


class FakeVad:
    def accepts(self, audio: bytes) -> bool:
        return True


class FailingVad:
    def accepts(self, audio: bytes) -> bool:
        raise RuntimeError("onnxruntime CUDA provider initialization failed")


class RejectThenAcceptVad:
    """Rejects the first VAD window and accepts the next voiced phrase."""

    def __init__(self) -> None:
        self.calls: list[bytes] = []

    def accepts(self, audio: bytes) -> bool:
        self.calls.append(audio)
        return audio == b"voiced"


class FakeAsr:
    def transcribe(self, audio: bytes) -> str:
        return "hello"


class FailingAsr:
    def transcribe(self, audio: bytes) -> str:
        raise RuntimeError("asr exploded")


class FakeTranslator:
    def translate(self, text: str) -> str:
        return "hola"


class CancelingTranslator:
    """Sets the cancel token mid-`pipeline.process()`, so the checkpoint-before-playback
    test can prove the cancellation is observed before `player.play()` -- not mid-action."""

    def __init__(self, cancel: threading.Event) -> None:
        self._cancel = cancel

    def translate(self, text: str) -> str:
        self._cancel.set()
        return "hola"


class FakeTts:
    def synthesize(self, text: str) -> bytes:
        return b"spanish-audio"


class PcmTts:
    def synthesize(self, text: str) -> bytes:
        return (20_000).to_bytes(2, "little", signed=True)


def default_pipeline() -> LocalDubbingPipeline:
    return LocalDubbingPipeline(FakeVad(), FakeAsr(), FakeTranslator(), FakeTts(), clock=lambda: 0.0)


def pcm_pipeline() -> LocalDubbingPipeline:
    return LocalDubbingPipeline(FakeVad(), FakeAsr(), FakeTranslator(), PcmTts(), clock=lambda: 0.0)


def default_scheduler() -> FixedCapacityScheduler:
    return FixedCapacityScheduler(capacity=4, max_age_seconds=30.0)


class FakeRouter:
    """Mirrors `PipeWirePulseRouter`'s externally observable behavior: `select_unique`
    resolves fresh against whatever `streams` currently reports (never a cached identifier),
    `begin` returns a `RouteLease`, and `recover` is configurable to simulate a restored or
    failed (or raising) recovery."""

    def __init__(
        self,
        streams: Callable[[], list[StreamRef]],
        *,
        recover_result: RecoveryResult | None = None,
        recover_raises: Exception | None = None,
    ) -> None:
        self._streams = streams
        self.select_unique_calls: list[StreamSelector] = []
        self.begin_calls: list[tuple[StreamRef, SinkRef]] = []
        self.recover_calls = 0
        self._recover_result = recover_result or RecoveryResult(restored=True, detail="restored")
        self._recover_raises = recover_raises

    def select_unique(self, selector: StreamSelector) -> StreamRef:
        self.select_unique_calls.append(selector)
        matches = [candidate for candidate in self._streams() if selector.matches(candidate)]
        assert len(matches) == 1
        return matches[0]

    def begin(self, stream: StreamRef, physical: SinkRef) -> RouteLease:
        self.begin_calls.append((stream, physical))
        return RouteLease(stream, f"opendubstream.{stream.identifier}", "1", stream.sink_name, physical.name)

    def recover(self) -> RecoveryResult:
        self.recover_calls += 1
        if self._recover_raises is not None:
            raise self._recover_raises
        return self._recover_result


class FakeCapture:
    def __init__(self, phrases: list[bytes], *, cancel_on_read: threading.Event | None = None) -> None:
        self._phrases = list(phrases)
        self._cancel_on_read = cancel_on_read
        self.start_calls: list[RouteLease] = []
        self.stop_calls = 0
        self.read_calls = 0
        self._stopped = threading.Event()

    def start(self, lease: RouteLease) -> object:
        self.start_calls.append(lease)
        return object()

    def read_phrase(self, deadline: float) -> bytes:
        if not self._phrases:
            self._stopped.wait(deadline)
            return b""
        self.read_calls += 1
        phrase = self._phrases.pop(0)
        if self._cancel_on_read is not None:
            self._cancel_on_read.set()
        return phrase

    def stop(self) -> None:
        self._stopped.set()
        self.stop_calls += 1


class FakePlayer:
    def __init__(self) -> None:
        self.play_calls: list[tuple[SinkRef, bytes, float]] = []

    def play(self, target: SinkRef, pcm: bytes, deadline: float) -> None:
        self.play_calls.append((target, pcm, deadline))

    def stop(self) -> None:
        pass


class FakeOriginalMix:
    def __init__(self) -> None:
        self.started: list[tuple[RouteLease, SinkRef]] = []
        self.updated: list[tuple[OriginalMixLease, bool, int]] = []

    def start_original_mix(self, lease: RouteLease, physical: SinkRef) -> OriginalMixLease:
        self.started.append((lease, physical))
        return OriginalMixLease("50", "77", physical.name)

    def set_original_mix(self, lease: OriginalMixLease, *, muted: bool, volume_percent: int) -> None:
        self.updated.append((lease, muted, volume_percent))


class CancelingAfterPlayPlayer(FakePlayer):
    """Sets the cancel token right after a successful play -- simulates the user pressing
    Stop immediately after one utterance finished, so the *next* loop iteration's
    checkpoint-before-capture halts before another capture window opens."""

    def __init__(self, cancel: threading.Event) -> None:
        super().__init__()
        self._cancel = cancel

    def play(self, target: SinkRef, pcm: bytes, deadline: float) -> None:
        super().play(target, pcm, deadline)
        self._cancel.set()


class RecordingObserver:
    def __init__(self) -> None:
        self.events: list[SessionEvent] = []

    def on_event(self, event: SessionEvent) -> None:
        self.events.append(event)


def build_runner(
    *,
    router,
    capture,
    player,
    observer,
    chrome_streams: Callable[[], list[StreamRef]] | None = None,
    chrome_sink_inputs: Callable[[], list[dict[str, object]]] | None = None,
    pipeline: LocalDubbingPipeline | None = None,
    scheduler: FixedCapacityScheduler | None = None,
    cuda_probe: Callable[[], bool] = lambda: True,
    original_mix_controller=None,
    mix_settings: AudioMixSettings = AudioMixSettings(),
) -> DubbingSessionRunner:
    return DubbingSessionRunner(
        router=router,
        chrome_streams=chrome_streams or (lambda: [chrome_stream()]),
        chrome_sink_inputs=chrome_sink_inputs or (lambda: one_active_chrome_sink_inputs()),
        physical_sink=PHYSICAL_SINK,
        capture=capture,
        pipeline=pipeline or default_pipeline(),
        scheduler=scheduler or default_scheduler(),
        player=player,
        cuda_probe=cuda_probe,
        observer=observer,
        clock=lambda: 0.0,
        original_mix_controller=original_mix_controller,
        mix_settings=mix_settings,
    )


# --- 2.1: observer event order ------------------------------------------------------


def test_event_order_for_a_normal_start_one_utterance_then_stop() -> None:
    cancel = threading.Event()
    streams = lambda: [chrome_stream()]
    router = FakeRouter(streams)
    capture = FakeCapture([b"phrase"])
    player = CancelingAfterPlayPlayer(cancel)
    observer = RecordingObserver()
    runner = build_runner(router=router, capture=capture, player=player, observer=observer, chrome_streams=streams)

    runner.run(cancel)

    stages = [event.stage for event in observer.events]
    # Detailed inference timing emits additional PROCESSING events, without changing
    # the safety lifecycle or withholding the two text results until voice playback.
    assert stages[:2] == [SessionStage.ROUTING, SessionStage.CAPTURING]
    assert stages[-4:] == [SessionStage.PLAYING, SessionStage.STOPPING, SessionStage.RECOVERING, SessionStage.IDLE]
    text_events = [event for event in observer.events if event.transcript is not None]
    assert text_events[0].transcript == "hello"
    assert text_events[1].translation == "hola"
    assert capture.start_calls and capture.stop_calls == 1
    assert player.play_calls == [(PHYSICAL_SINK, b"spanish-audio", 15.0)]


def test_session_starts_only_owned_original_mix_and_applies_translation_only_local_mute() -> None:
    cancel = threading.Event()
    streams = lambda: [chrome_stream()]
    router = FakeRouter(streams)
    capture = FakeCapture([b"phrase"])
    player = CancelingAfterPlayPlayer(cancel)
    observer = RecordingObserver()
    mix = FakeOriginalMix()
    runner = build_runner(
        router=router,
        capture=capture,
        player=player,
        observer=observer,
        chrome_streams=streams,
        original_mix_controller=mix,
        mix_settings=AudioMixSettings(mode=AudioMode.TRANSLATION_ONLY, original_volume=20, dub_volume=50),
        pipeline=pcm_pipeline(),
    )

    runner.run(cancel)

    assert len(mix.started) == 1
    assert [update[1:] for update in mix.updated] == [(False, 100), (True, 20), (False, 100)]
    # Independent dubbed gain applies to physical PCM; the original loopback is the only mute target.
    assert player.play_calls == [(PHYSICAL_SINK, (10_000).to_bytes(2, "little", signed=True), 15.0)]


def test_ineligible_start_emits_only_a_failed_event_with_the_refusal_reason() -> None:
    cancel = threading.Event()
    streams = lambda: [chrome_stream()]
    router = FakeRouter(streams)
    capture = FakeCapture([])
    player = FakePlayer()
    observer = RecordingObserver()
    runner = build_runner(
        router=router, capture=capture, player=player, observer=observer, chrome_streams=streams,
        cuda_probe=lambda: False,
    )

    runner.run(cancel)

    assert len(observer.events) == 1
    assert observer.events[0].stage == SessionStage.FAILED
    assert capture.start_calls == []
    assert router.begin_calls == []


# --- 2.2: exactly-one router.recover() on normal stop and mid-session failure --------


def test_exactly_one_recover_call_on_normal_stop() -> None:
    cancel = threading.Event()
    streams = lambda: [chrome_stream()]
    router = FakeRouter(streams)
    capture = FakeCapture([b"phrase"])
    player = CancelingAfterPlayPlayer(cancel)
    observer = RecordingObserver()
    runner = build_runner(router=router, capture=capture, player=player, observer=observer, chrome_streams=streams)

    runner.run(cancel)

    assert router.recover_calls == 1


def test_exactly_one_recover_call_on_mid_session_failure() -> None:
    cancel = threading.Event()
    streams = lambda: [chrome_stream()]
    router = FakeRouter(streams)
    capture = FakeCapture([b"phrase"])
    player = FakePlayer()
    observer = RecordingObserver()
    pipeline = LocalDubbingPipeline(FakeVad(), FailingAsr(), FakeTranslator(), FakeTts(), clock=lambda: 0.0)
    runner = build_runner(
        router=router, capture=capture, player=player, observer=observer, chrome_streams=streams, pipeline=pipeline,
    )

    runner.run(cancel)

    assert router.recover_calls == 1
    assert observer.events[-1].stage == SessionStage.FAILED
    assert player.play_calls == []


def test_model_initialization_failure_after_routing_stops_capture_and_recovers_owned_route() -> None:
    """A lazy ASR backend failure occurs only after `begin`; it must not strand routing."""
    cancel = threading.Event()
    streams = lambda: [chrome_stream()]
    router = FakeRouter(streams)
    capture = FakeCapture([b"phrase"])
    player = FakePlayer()
    observer = RecordingObserver()
    pipeline = LocalDubbingPipeline(FakeVad(), FailingAsr(), FakeTranslator(), FakeTts(), clock=lambda: 0.0)
    runner = build_runner(
        router=router, capture=capture, player=player, observer=observer, chrome_streams=streams, pipeline=pipeline,
    )

    runner.run(cancel)

    assert router.begin_calls == [(chrome_stream(), PHYSICAL_SINK)]
    assert capture.stop_calls == 1
    assert router.recover_calls == 1
    assert observer.events[-1].stage is SessionStage.FAILED


def test_vad_initialization_failure_after_routing_stops_capture_and_recovers_owned_route() -> None:
    """A lazy VAD backend failure after routing must not strand Chrome on the capture sink."""
    cancel = threading.Event()
    streams = lambda: [chrome_stream()]
    router = FakeRouter(streams)
    capture = FakeCapture([b"phrase"])
    player = FakePlayer()
    observer = RecordingObserver()
    pipeline = LocalDubbingPipeline(FailingVad(), FakeAsr(), FakeTranslator(), FakeTts(), clock=lambda: 0.0)
    runner = build_runner(
        router=router, capture=capture, player=player, observer=observer, chrome_streams=streams, pipeline=pipeline,
    )

    runner.run(cancel)

    assert router.begin_calls == [(chrome_stream(), PHYSICAL_SINK)]
    assert capture.stop_calls == 1
    assert router.recover_calls == 1
    assert observer.events[-1].stage is SessionStage.FAILED
    assert player.play_calls == []


def test_vad_rejection_is_skipped_and_a_later_voiced_phrase_reaches_playing() -> None:
    """Silence is ordinary input, not a session failure requiring route recovery."""
    cancel = threading.Event()
    streams = lambda: [chrome_stream()]
    router = FakeRouter(streams)
    capture = FakeCapture([b"silence", b"voiced"])
    player = CancelingAfterPlayPlayer(cancel)
    observer = RecordingObserver()
    vad = RejectThenAcceptVad()
    pipeline = LocalDubbingPipeline(vad, FakeAsr(), FakeTranslator(), FakeTts(), clock=lambda: 0.0)
    runner = build_runner(
        router=router,
        capture=capture,
        player=player,
        observer=observer,
        chrome_streams=streams,
        pipeline=pipeline,
    )

    runner.run(cancel)

    stages = [event.stage for event in observer.events]
    assert vad.calls == [b"silence", b"voiced"]
    assert SessionStage.FAILED not in stages
    assert stages.count(SessionStage.PLAYING) == 1
    assert player.play_calls == [(PHYSICAL_SINK, b"spanish-audio", 15.0)]
    assert router.recover_calls == 1


# --- 2.3: cancel-at-checkpoint halts before the next real action, not mid-action -----


def test_cancel_before_run_halts_before_any_capture_window() -> None:
    cancel = threading.Event()
    cancel.set()
    streams = lambda: [chrome_stream()]
    router = FakeRouter(streams)
    capture = FakeCapture([b"phrase"])
    player = FakePlayer()
    observer = RecordingObserver()
    runner = build_runner(router=router, capture=capture, player=player, observer=observer, chrome_streams=streams)

    runner.run(cancel)

    assert capture.start_calls  # the monitor is opened before the first checkpoint
    assert capture.read_calls == 0  # but no capture window is ever opened
    assert player.play_calls == []
    assert capture.stop_calls == 1
    assert SessionStage.STOPPING in [event.stage for event in observer.events]


def test_cancel_set_during_the_capture_window_halts_before_enqueue() -> None:
    cancel = threading.Event()
    streams = lambda: [chrome_stream()]
    router = FakeRouter(streams)
    capture = FakeCapture([b"phrase"], cancel_on_read=cancel)
    player = FakePlayer()
    observer = RecordingObserver()
    scheduler = default_scheduler()
    runner = build_runner(
        router=router, capture=capture, player=player, observer=observer, chrome_streams=streams, scheduler=scheduler,
    )

    runner.run(cancel)

    assert capture.read_calls == 1
    assert scheduler.backlog == 0  # enqueue() was never reached for the just-captured phrase
    assert player.play_calls == []
    assert SessionStage.PROCESSING not in [event.stage for event in observer.events]


def test_cancel_set_during_processing_halts_before_playback() -> None:
    cancel = threading.Event()
    streams = lambda: [chrome_stream()]
    router = FakeRouter(streams)
    capture = FakeCapture([b"phrase"])
    player = FakePlayer()
    observer = RecordingObserver()
    pipeline = LocalDubbingPipeline(FakeVad(), FakeAsr(), CancelingTranslator(cancel), FakeTts(), clock=lambda: 0.0)
    runner = build_runner(
        router=router, capture=capture, player=player, observer=observer, chrome_streams=streams, pipeline=pipeline,
    )

    runner.run(cancel)

    assert player.play_calls == []  # translation (and tts) already ran, but playback never did
    stages = [event.stage for event in observer.events]
    assert SessionStage.PROCESSING in stages
    assert SessionStage.PLAYING not in stages
    assert stages[-1] == SessionStage.IDLE


# --- 2.4: restored=False or a recover() raise -> RECOVERY_FAILED, no auto-retry ------


def test_recovery_returning_restored_false_produces_recovery_failed_on_clean_stop() -> None:
    cancel = threading.Event()
    streams = lambda: [chrome_stream()]
    router = FakeRouter(streams, recover_result=RecoveryResult(restored=False, detail="stale journal"))
    capture = FakeCapture([b"phrase"])
    player = CancelingAfterPlayPlayer(cancel)
    observer = RecordingObserver()
    runner = build_runner(router=router, capture=capture, player=player, observer=observer, chrome_streams=streams)

    runner.run(cancel)

    last = observer.events[-1]
    assert last.stage == SessionStage.RECOVERY_FAILED
    assert last.diagnostics is not None
    assert last.diagnostics.recovery_restored is False
    assert last.diagnostics.recovery_detail == "stale journal"
    assert router.recover_calls == 1


def test_recover_raising_during_mid_session_failure_produces_recovery_failed_not_generic_failed() -> None:
    cancel = threading.Event()
    streams = lambda: [chrome_stream()]
    router = FakeRouter(streams, recover_raises=RuntimeError("pactl unreachable during recovery"))
    capture = FakeCapture([b"phrase"])
    player = FakePlayer()
    observer = RecordingObserver()
    pipeline = LocalDubbingPipeline(FakeVad(), FailingAsr(), FakeTranslator(), FakeTts(), clock=lambda: 0.0)
    runner = build_runner(
        router=router, capture=capture, player=player, observer=observer, chrome_streams=streams, pipeline=pipeline,
    )

    runner.run(cancel)

    last = observer.events[-1]
    assert last.stage == SessionStage.RECOVERY_FAILED
    assert last.diagnostics is not None
    assert last.diagnostics.recovery_restored is False
    assert last.diagnostics.recovery_detail == "pactl unreachable during recovery"
    assert router.recover_calls == 1  # no auto-retry


# --- 2.5: scheduler overload_dropped/stale_dropped surfaced via Diagnostics ----------


def test_scheduler_overload_dropped_count_is_surfaced_via_diagnostics() -> None:
    cancel = threading.Event()
    streams = lambda: [chrome_stream()]
    router = FakeRouter(streams)
    capture = FakeCapture([b"phrase"])
    player = CancelingAfterPlayPlayer(cancel)
    observer = RecordingObserver()
    scheduler = FixedCapacityScheduler(capacity=1, max_age_seconds=100.0)
    scheduler.enqueue(Utterance("pre-1", b"x", captured_at=0.0, phrase_end_at=0.0), now=0.0)
    scheduler.enqueue(Utterance("pre-2", b"y", captured_at=0.0, phrase_end_at=0.0), now=0.0)  # evicts pre-1
    scheduler.start_next(now=0.0)  # drains pre-2, backlog back to 0
    assert scheduler.metrics.overload_dropped == 1
    runner = build_runner(
        router=router, capture=capture, player=player, observer=observer, chrome_streams=streams, scheduler=scheduler,
    )

    runner.run(cancel)

    playing_events = [event for event in observer.events if event.stage == SessionStage.PLAYING]
    assert playing_events and playing_events[0].diagnostics is not None
    assert playing_events[0].diagnostics.overload_dropped == 1


def test_scheduler_stale_dropped_count_is_surfaced_via_diagnostics() -> None:
    cancel = threading.Event()
    streams = lambda: [chrome_stream()]
    router = FakeRouter(streams)
    capture = FakeCapture([b"phrase"])
    player = CancelingAfterPlayPlayer(cancel)
    observer = RecordingObserver()
    scheduler = FixedCapacityScheduler(capacity=4, max_age_seconds=1.0)
    scheduler.enqueue(Utterance("stale-1", b"x", captured_at=0.0, phrase_end_at=0.0), now=0.0)
    scheduler.start_next(now=200.0)  # discards the stale entry, no crash
    assert scheduler.metrics.stale_dropped == 1
    runner = build_runner(
        router=router, capture=capture, player=player, observer=observer, chrome_streams=streams, scheduler=scheduler,
    )

    runner.run(cancel)

    playing_events = [event for event in observer.events if event.stage == SessionStage.PLAYING]
    assert playing_events and playing_events[0].diagnostics is not None
    assert playing_events[0].diagnostics.stale_dropped == 1


# --- 2.6: fresh Chrome discovery after RECOVERY_FAILED; no stream identifier reuse ---


def test_fresh_chrome_discovery_after_recovery_failed_never_reuses_a_stale_stream_identifier() -> None:
    state = {"identifier": "41"}
    streams_seen: list[str] = []

    def chrome_streams() -> list[StreamRef]:
        current = chrome_stream(identifier=state["identifier"])
        streams_seen.append(current.identifier)
        return [current]

    sink_inputs = lambda: one_active_chrome_sink_inputs()
    router = FakeRouter(chrome_streams, recover_result=RecoveryResult(restored=False, detail="stale"))
    observer = RecordingObserver()

    cancel_1 = threading.Event()
    capture_1 = FakeCapture([b"phrase"])
    player_1 = CancelingAfterPlayPlayer(cancel_1)
    runner_1 = build_runner(
        router=router, capture=capture_1, player=player_1, observer=observer,
        chrome_streams=chrome_streams, chrome_sink_inputs=sink_inputs,
    )
    runner_1.run(cancel_1)
    assert observer.events[-1].stage == SessionStage.RECOVERY_FAILED
    # `chrome_streams` is queried multiple times per run() (eligibility count, selection,
    # and the router's own fresh `select_unique` resolution) -- every one of them saw "41".
    assert set(streams_seen) == {"41"}

    # Simulate PipeWire renumbering the Chrome stream identifier before the next Start.
    state["identifier"] = "99"
    streams_seen.clear()
    cancel_2 = threading.Event()
    capture_2 = FakeCapture([b"phrase"])
    player_2 = CancelingAfterPlayPlayer(cancel_2)
    runner_2 = build_runner(
        router=router, capture=capture_2, player=player_2, observer=observer,
        chrome_streams=chrome_streams, chrome_sink_inputs=sink_inputs,
    )

    runner_2.run(cancel_2)

    # The second, fresh `DubbingSessionRunner` invocation never reuses the stale "41"
    # identifier -- every discovery call after the identifier drifted saw "99" only.
    assert set(streams_seen) == {"99"}
    assert router.begin_calls[0][0].identifier == "41"
    assert router.begin_calls[-1][0].identifier == "99"
