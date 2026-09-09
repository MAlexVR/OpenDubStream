"""RED-first unit tests for the armable continuous session boundary.

These use only fakes.  They deliberately avoid PipeWire, Chrome, models and timers whose
wall-clock execution would make a cancellation test flaky.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

from opendubstream.application.dubbing_session import DubbingSessionRunner
from opendubstream.application.session_state import SessionEvent, SessionStage
from opendubstream.domain.contracts import RecoveryResult, RouteLease, SinkRef, StreamRef, StreamSelector
from opendubstream.pipeline.local_pipeline import LocalDubbingPipeline
from opendubstream.pipeline.scheduler import FixedCapacityScheduler


PHYSICAL = SinkRef("alsa_output.speakers", True, "physical")


def chrome_stream(identifier: str = "42") -> StreamRef:
    return StreamRef(identifier, "Google Chrome", "Video", f"serial-{identifier}", PHYSICAL.name)


def active_inputs(streams: list[StreamRef]) -> list[dict[str, object]]:
    return [{"corked": False, "properties": {"object.serial": stream.serial}} for stream in streams]


class Router:
    def __init__(self, streams: Callable[[], list[StreamRef]]) -> None:
        self.streams = streams
        self.begin_calls: list[StreamRef] = []
        self.recover_calls = 0

    def select_unique(self, selector: StreamSelector) -> StreamRef:
        matches = [stream for stream in self.streams() if selector.matches(stream)]
        assert len(matches) == 1
        return matches[0]

    def begin(self, stream: StreamRef, physical: SinkRef) -> RouteLease:
        self.begin_calls.append(stream)
        return RouteLease(stream, "opendubstream.test", "1", stream.sink_name, physical.name)

    def recover(self) -> RecoveryResult:
        self.recover_calls += 1
        return RecoveryResult(restored=True, detail="restored")


class Capture:
    def start(self, lease: RouteLease) -> object:
        return object()

    def read_phrase(self, deadline: float) -> bytes:
        raise AssertionError("capture must not start in armed-only tests")

    def stop(self) -> None:
        pass


class Player:
    def play(self, target: SinkRef, pcm: bytes, deadline: float) -> None:
        raise AssertionError("playback must not start in armed-only tests")

    def stop(self) -> None:
        pass


class SilentVad:
    def accepts(self, audio: bytes) -> bool:
        return False


class PassthroughAsr:
    def transcribe(self, audio: bytes) -> str:
        return ""


class PassthroughTranslator:
    def translate(self, text: str) -> str:
        return ""


class PassthroughTts:
    def synthesize(self, text: str) -> bytes:
        return b""


class Observer:
    def __init__(self, on_event: Callable[[SessionEvent], None] | None = None) -> None:
        self.events: list[SessionEvent] = []
        self._on_event = on_event

    def on_event(self, event: SessionEvent) -> None:
        self.events.append(event)
        if self._on_event is not None:
            self._on_event(event)


def runner_for(
    streams: list[StreamRef],
    *,
    observer: Observer,
) -> tuple[DubbingSessionRunner, Router]:
    router = Router(lambda: list(streams))
    runner = DubbingSessionRunner(
        router=router,
        chrome_streams=lambda: list(streams),
        chrome_sink_inputs=lambda: active_inputs(streams),
        physical_sink=PHYSICAL,
        capture=Capture(),
        pipeline=LocalDubbingPipeline(SilentVad(), PassthroughAsr(), PassthroughTranslator(), PassthroughTts(), clock=lambda: 0),
        scheduler=FixedCapacityScheduler(capacity=2, max_age_seconds=2),
        player=Player(),
        cuda_probe=lambda: True,
        observer=observer,
        clock=lambda: 0,
        arm_poll_interval=0,
    )
    return runner, router


def test_start_arms_then_activates_when_one_chrome_stream_appears() -> None:
    streams: list[StreamRef] = []
    cancel = threading.Event()

    def appear_then_cancel(event: SessionEvent) -> None:
        if event.stage is SessionStage.ARMED and not streams:
            streams.append(chrome_stream())
        elif event.stage is SessionStage.CAPTURING:
            cancel.set()

    observer = Observer(appear_then_cancel)
    runner, router = runner_for(streams, observer=observer)

    runner.run(cancel)

    stages = [event.stage for event in observer.events]
    assert stages[:3] == [SessionStage.ARMED, SessionStage.ROUTING, SessionStage.CAPTURING]
    assert router.begin_calls == [chrome_stream()]


def test_stop_while_armed_cancels_without_any_route_mutation() -> None:
    streams: list[StreamRef] = []
    cancel = threading.Event()
    observer = Observer(lambda event: cancel.set() if event.stage is SessionStage.ARMED else None)
    runner, router = runner_for(streams, observer=observer)

    runner.run(cancel)

    assert [event.stage for event in observer.events] == [SessionStage.ARMED, SessionStage.STOPPING, SessionStage.IDLE]
    assert router.begin_calls == []
    assert router.recover_calls == 0


def test_multiple_chrome_streams_remain_armed_until_stop_without_routing() -> None:
    streams = [chrome_stream("1"), chrome_stream("2")]
    cancel = threading.Event()
    observer = Observer(lambda event: cancel.set() if event.stage is SessionStage.ARMED else None)
    runner, router = runner_for(streams, observer=observer)

    runner.run(cancel)

    assert [event.stage for event in observer.events] == [SessionStage.ARMED, SessionStage.STOPPING, SessionStage.IDLE]
    assert router.begin_calls == []
    assert router.recover_calls == 0


def test_route_revalidates_stream_identity_before_begin() -> None:
    first = chrome_stream("first")
    replacement = chrome_stream("replacement")
    calls = 0
    cancel = threading.Event()

    def discovered_streams() -> list[StreamRef]:
        nonlocal calls
        calls += 1
        # Initial arm sees one candidate, while the identity revalidation sees a
        # different candidate.  The session must return to ARMED before `begin`.
        if calls == 1:
            return [first]
        if calls == 2:
            return [replacement]
        return []

    router = Router(discovered_streams)
    observer = Observer(lambda event: cancel.set() if event.stage is SessionStage.ARMED else None)
    runner = DubbingSessionRunner(
        router=router,
        chrome_streams=discovered_streams,
        chrome_sink_inputs=lambda: active_inputs([first, replacement]),
        physical_sink=PHYSICAL,
        capture=Capture(),
        pipeline=LocalDubbingPipeline(SilentVad(), PassthroughAsr(), PassthroughTranslator(), PassthroughTts(), clock=lambda: 0),
        scheduler=FixedCapacityScheduler(capacity=2, max_age_seconds=2),
        player=Player(),
        cuda_probe=lambda: True,
        observer=observer,
        clock=lambda: 0,
        arm_poll_interval=0,
    )

    runner.run(cancel)

    assert router.begin_calls == []
    assert SessionStage.ARMED in [event.stage for event in observer.events]
