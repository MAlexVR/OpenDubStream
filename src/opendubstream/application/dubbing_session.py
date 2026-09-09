"""Continuous capture, single-thread inference and independent bounded playback.

Only the owner thread recovers routing, after both I/O workers have stopped.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Callable, Protocol

from opendubstream.application.live_controls import LiveControls, LiveSettings
from opendubstream.application.audio_mix import apply_pcm_gain
from opendubstream.application.chrome_route import count_active_chrome_streams, select_active_chrome_stream
from opendubstream.application.eligibility import CudaProbe, evaluate_eligibility
from opendubstream.application.session import run_protected
from opendubstream.application.session_state import Diagnostics, SessionEvent, SessionObserver, SessionStage
from opendubstream.domain.contracts import (
    AudioMixSettings,
    AudioMode,
    AudioRouter,
    OriginalMixLease,
    RecoveryResult,
    RouteLease,
    SinkRef,
    StreamRef,
    StreamSelector,
)
from opendubstream.domain.pipeline import Utterance
from opendubstream.ui.resources import MessageKey
from opendubstream.pipeline.local_pipeline import LocalDubbingPipeline, RejectedUtterance, PipelineStageError
from opendubstream.pipeline.scheduler import FixedCapacityScheduler


class MonitorCapture(Protocol):
    """Structural match for `PipeWireMonitorCapture`; this application-layer module never
    imports it directly, so a fake can satisfy this Protocol without any infrastructure
    dependency in tests."""

    def start(self, lease: RouteLease | StreamRef) -> object: ...

    def read_phrase(self, deadline: float) -> bytes: ...

    def stop(self) -> None: ...


class PcmPlayer(Protocol):
    """Structural match for `PwPlayPcmPlayer` / `infrastructure.audio.playback.PcmPlayer`."""

    def play(self, target: SinkRef, pcm: bytes, deadline: float) -> None: ...

    def stop(self) -> None: ...


class OriginalMixController(Protocol):
    """Infrastructure seam for the owned original-audio loopback only."""

    def start_original_mix(self, lease: RouteLease, physical: SinkRef) -> OriginalMixLease: ...

    def set_original_mix(self, lease: OriginalMixLease, *, muted: bool, volume_percent: int) -> None: ...


class _RecoveryRecordingRouter:
    """Wraps the real `AudioRouter` so that `run_protected`'s internal recovery call (on a
    mid-session failure) and this runner's own trailing recovery call (on a clean/cancelled
    stop) both funnel through exactly one call to the underlying `router.recover()`, with its
    `RecoveryResult` (or a raised exception) captured for `RECOVERY_FAILED` classification.
    `select_unique`/`begin` are never invoked through this proxy -- routing happens once,
    before this proxy is constructed."""

    def __init__(self, router: AudioRouter) -> None:
        self._router = router
        self.recover_calls = 0
        self.recovery_result: RecoveryResult | None = None
        self.recovery_error: Exception | None = None

    def select_unique(self, selector: StreamSelector) -> StreamRef:
        return self._router.select_unique(selector)

    def begin(self, stream: StreamRef, physical: SinkRef) -> RouteLease:
        return self._router.begin(stream, physical)

    def recover(self) -> RecoveryResult:
        self.recover_calls += 1
        try:
            result = self._router.recover()
        except Exception as error:
            self.recovery_error = error
            raise
        self.recovery_result = result
        return result


class DubbingSessionRunner:
    def __init__(
        self,
        *,
        router: AudioRouter,
        chrome_streams: Callable[[], list[StreamRef]],
        chrome_sink_inputs: Callable[[], list[dict[str, object]]],
        physical_sink: SinkRef,
        capture: MonitorCapture,
        pipeline: LocalDubbingPipeline,
        scheduler: FixedCapacityScheduler,
        player: PcmPlayer,
        cuda_probe: CudaProbe,
        observer: SessionObserver,
        clock: Callable[[], float],
        capture_deadline: float = 5.0,
        play_deadline: float = 15.0,
        arm_poll_interval: float = 0.5,
        original_mix_controller: OriginalMixController | None = None,
        mix_settings: AudioMixSettings = AudioMixSettings(),
        controls: LiveControls | None = None,
    ) -> None:
        self._timings: dict[str, float] = {}
        self._timing_lock = threading.Lock()
        self._active_operation: str | None = None
        self._router = router
        self._chrome_streams = chrome_streams
        self._chrome_sink_inputs = chrome_sink_inputs
        self._physical_sink = physical_sink
        self._capture = capture
        self._pipeline = pipeline
        self._scheduler = scheduler
        self._player = player
        self._cuda_probe = cuda_probe
        self._observer = observer
        self._clock = clock
        self._capture_deadline = capture_deadline
        self._play_deadline = play_deadline
        if arm_poll_interval < 0:
            raise ValueError("arm poll interval must not be negative")
        self._arm_poll_interval = arm_poll_interval
        self._original_mix_controller = original_mix_controller
        self._mix_settings = mix_settings
        self._controls = controls or LiveControls(LiveSettings(mix=mix_settings))

    def run(self, cancel: threading.Event) -> None:
        """Run one full session: eligibility -> routing -> capture/process/playback loop
        -> stop/failure -> exactly-one recovery. Never raises: every failure, including a
        failed recovery, is reported to `observer` as a `SessionEvent` instead."""
        if not self._cuda_probe():
            eligibility = evaluate_eligibility(cuda_available=False, active_chrome_streams=0)
            self._emit(SessionStage.FAILED, message=eligibility.reason)
            return

        passive = self._controls.snapshot().mix.mode is AudioMode.CAPTIONS
        while True:
            stream = self._wait_for_unique_stream(cancel)
            if stream is None:
                return

            self._emit(SessionStage.ROUTING)
            # A stream can disappear or a second Chrome stream can appear in the gap
            # between a successful arm poll and routing. Revalidate the exact identity
            # before the first mutation. A changed identity returns to the polling loop
            # without using recursion or retaining any stale selection.
            fresh_stream = self._select_unique_active_stream()
            if fresh_stream is None or fresh_stream.serial != stream.serial:
                continue
            try:
                selected = self._router.select_unique(
                    StreamSelector(application_name=fresh_stream.application_name, serial=fresh_stream.serial)
                )
                fresh_stream.require_same_identity(selected)
                lease = selected if passive else self._router.begin(selected, self._physical_sink)
            except Exception as error:
                self._emit(SessionStage.FAILED, diagnostics=self._diagnostics(last_routing_error=str(error)))
                return
            break

        if passive:
            try:
                self._session_work(lease, cancel)
            except Exception as error:
                self._emit(SessionStage.FAILED, diagnostics=self._diagnostics(last_routing_error=str(error)))
            else:
                self._emit(SessionStage.IDLE)
            return

        proxy = _RecoveryRecordingRouter(self._router)
        work_error: Exception | None = None
        try:
            run_protected(proxy, lambda: self._session_work(lease, cancel))
        except Exception as error:
            work_error = error

        self._emit(
            SessionStage.RECOVERING,
            diagnostics=self._diagnostics(last_routing_error=str(work_error) if work_error is not None else None),
        )
        if work_error is None:
            # `run_protected` only calls `recover()` on a failure; on a clean/cancelled
            # stop this is the sole, trailing recovery call -- never a second one.
            try:
                proxy.recover()
            except Exception:
                pass  # captured on the proxy as `recovery_error`; handled below

        if proxy.recovery_error is not None or (proxy.recovery_result is not None and not proxy.recovery_result.restored):
            detail = str(proxy.recovery_error) if proxy.recovery_error is not None else proxy.recovery_result.detail
            self._emit(
                SessionStage.RECOVERY_FAILED,
                diagnostics=self._diagnostics(
                    last_routing_error=str(work_error) if work_error is not None else None,
                    recovery_restored=False, recovery_detail=detail,
                ),
            )
            return

        recovery_detail = proxy.recovery_result.detail if proxy.recovery_result is not None else None
        if work_error is not None:
            message = {
                "capture": MessageKey.ERROR_CAPTURE,
                "vad": MessageKey.ERROR_CAPTURE,
                "transcription": MessageKey.ERROR_TRANSCRIPTION,
                "translation": MessageKey.ERROR_TRANSLATION,
                "voice": MessageKey.ERROR_VOICE,
                "playback": MessageKey.ERROR_PLAYBACK,
            }.get(getattr(work_error, "stage", None))
            self._emit(
                SessionStage.FAILED, message=message,
                diagnostics=self._diagnostics(
                    last_routing_error=str(work_error), recovery_restored=True, recovery_detail=recovery_detail,
                ),
            )
            return

        self._emit(SessionStage.IDLE, diagnostics=self._diagnostics(recovery_restored=True, recovery_detail=recovery_detail))

    def _wait_for_unique_stream(self, cancel: threading.Event) -> StreamRef | None:
        """Wait without mutating PipeWire until exactly one eligible Chrome stream exists.

        `Event.wait()` makes the poll cancellable: Stop wakes a normal half-second wait
        immediately instead of waiting for another Chrome discovery cycle.
        """
        # Preserve the established active-session checkpoint contract: a cancellation
        # already set when a unique stream is immediately available is observed by
        # `_session_work` before its first capture window.  ARMED cancellation is the
        # new no-mutation path, and applies only after a poll finds no unique stream.
        initial = self._select_unique_active_stream()
        if initial is not None:
            return initial

        armed_reported = False
        while not cancel.is_set():
            stream = self._select_unique_active_stream()
            if stream is not None:
                return stream
            if not armed_reported:
                self._emit(SessionStage.ARMED, message=MessageKey.WAITING_FOR_CHROME, diagnostics=self._diagnostics())
                armed_reported = True
            cancel.wait(self._arm_poll_interval)

        self._emit(SessionStage.STOPPING, diagnostics=self._diagnostics())
        self._emit(SessionStage.IDLE, diagnostics=self._diagnostics(recovery_restored=True, recovery_detail="armed session cancelled"))
        return None

    def _select_unique_active_stream(self) -> StreamRef | None:
        streams = self._chrome_streams()
        sink_inputs = self._chrome_sink_inputs()
        if count_active_chrome_streams(streams, sink_inputs) != 1:
            return None
        return select_active_chrome_stream(streams, sink_inputs)

    def _session_work(self, lease: RouteLease | StreamRef, cancel: threading.Event) -> None:
        stop = threading.Event()
        condition = threading.Condition()
        playback_pending = deque()
        errors: list[Exception] = []
        passive = isinstance(lease, StreamRef)
        original_mix = None
        if not passive and self._original_mix_controller is not None:
            original_mix = self._original_mix_controller.start_original_mix(lease, self._physical_sink)
        last_mix = None

        def mix(playing: bool) -> None:
            nonlocal last_mix
            if original_mix is None:
                return
            settings = self._controls.snapshot()
            speaking = playing and settings.mix.mode is not AudioMode.CAPTIONS and settings.voice_enabled and settings.mix.dub_volume > 0
            # Keep the source audible while models load, between phrases, and on failure.
            desired = (
                speaking and settings.mix.mode is AudioMode.TRANSLATION_ONLY,
                settings.mix.original_volume if speaking else 100,
            )
            if desired != last_mix:
                self._original_mix_controller.set_original_mix(
                    original_mix, muted=desired[0], volume_percent=desired[1],
                )
                last_mix = desired

        def fail(error: Exception) -> None:
            with condition:
                errors.append(error)
                stop.set()
                condition.notify_all()

        def capture_loop() -> None:
            try:
                while not stop.is_set() and not cancel.is_set():
                    audio = self._capture.read_phrase(self._capture_deadline)
                    now = self._clock()
                    if stop.is_set() or cancel.is_set():
                        break
                    if not audio:
                        continue
                    utterance = Utterance(
                        identifier=f"utterance-{now}", audio=audio,
                        captured_at=now - len(audio) / 32000, phrase_end_at=now,
                    )
                    with condition:
                        self._scheduler.enqueue(utterance, now=now)
                        condition.notify_all()
            except Exception as error:
                if not stop.is_set() and not cancel.is_set():
                    fail(PipelineStageError("capture", error))

        def playback_loop() -> None:
            try:
                while not stop.is_set() and not cancel.is_set():
                    mix(False)
                    with condition:
                        if not playback_pending:
                            condition.wait(0.1)
                            continue
                        processed, queued_at = playback_pending.popleft()
                    settings = self._controls.snapshot()
                    if passive or settings.mix.mode is AudioMode.CAPTIONS or not settings.voice_enabled or not settings.mix.dub_volume:
                        continue
                    # Expire waiting output, not completed model work. Slow inference
                    # is reported as voice_age rather than silently starving playback.
                    if self._clock() - queued_at > self._scheduler.max_age_seconds:
                        with condition:
                            self._scheduler.metrics.stale_dropped += 1
                        continue
                    self._record_timings(playback_wait=(self._clock() - queued_at) * 1000,
                                         voice_age=(self._clock() - processed.source.phrase_end_at) * 1000)
                    pcm = processed.synthesized_audio
                    if settings.mix.dub_volume != 100:
                        pcm = apply_pcm_gain(pcm, settings.mix.dub_volume)
                    mix(True)
                    try:
                        self._emit(SessionStage.PLAYING, diagnostics=self._diagnostics())
                        self._player.play(self._physical_sink, pcm, self._play_deadline)
                    finally:
                        mix(False)
            except Exception as error:
                fail(PipelineStageError("playback", error))

        def on_stage(stage, duration) -> None:
            self._active_operation = stage
            if duration is not None:
                self._record_timings(**{stage: duration})
            if not cancel.is_set() and not stop.is_set():
                self._emit(SessionStage.PROCESSING, diagnostics=self._diagnostics())

        def on_text(transcript, translation) -> None:
            if cancel.is_set() or stop.is_set():
                return
            if translation is not None:
                self._record_timings(caption_age=(self._clock() - ready.phrase_end_at) * 1000)
            self._emit(SessionStage.PROCESSING, transcript=transcript, translation=translation,
                       diagnostics=self._diagnostics())

        cancel_check = getattr(self._player, "set_cancel_check", None)
        if cancel_check is not None:
            cancel_check(lambda: stop.is_set() or cancel.is_set() or not self._controls.snapshot().voice_enabled or self._controls.snapshot().mix.mode is AudioMode.CAPTIONS)
        mix(False)
        self._pipeline.reset()
        self._emit(SessionStage.CAPTURING, diagnostics=self._diagnostics())
        capture_thread = threading.Thread(target=capture_loop, name="dubbing-capture")
        playback_thread = threading.Thread(target=playback_loop, name="dubbing-playback")
        try:
            self._capture.start(lease)
            capture_thread.start()
            playback_thread.start()
            while not stop.is_set():
                if self._checkpoint_cancelled(cancel):
                    break
                with condition:
                    ready = self._scheduler.start_next(now=self._clock())
                    if ready is None:
                        condition.wait(0.05)
                        continue
                self._record_timings(capture=(ready.phrase_end_at - ready.captured_at) * 1000,
                                     inference_wait=(self._clock() - ready.phrase_end_at) * 1000)
                self._emit(SessionStage.PROCESSING, diagnostics=self._diagnostics())
                try:
                    processed = self._pipeline.process(
                        ready, on_text=on_text, on_stage=on_stage,
                        should_synthesize=lambda: not passive and not cancel.is_set() and not stop.is_set()
                        and self._controls.snapshot().voice_enabled and self._controls.snapshot().mix.dub_volume > 0
                        and self._controls.snapshot().mix.mode is not AudioMode.CAPTIONS,
                    )
                except RejectedUtterance:
                    self._active_operation = None
                    self._emit(SessionStage.CAPTURING, diagnostics=self._diagnostics())
                    continue
                if self._checkpoint_cancelled(cancel):
                    break
                self._active_operation = None
                self._emit(SessionStage.CAPTURING, diagnostics=self._diagnostics())
                if not processed.synthesized_audio:
                    continue
                with condition:
                    if len(playback_pending) >= 2:
                        playback_pending.popleft()
                        self._scheduler.metrics.overload_dropped += 1
                    playback_pending.append((processed, self._clock()))
                    condition.notify_all()
        finally:
            stop.set()
            with condition:
                condition.notify_all()
            # One failed cleanup must not skip another worker's shutdown/join.
            for adapter in (self._capture, self._player):
                try:
                    adapter.stop()
                except Exception as error:
                    errors.append(error)
            # Production adapters have bounded reads and cancellable playback. Do not
            # recover the lease while a worker can still access it.
            for thread in (capture_thread, playback_thread):
                if thread.ident is not None:
                    thread.join()
        if errors:
            raise errors[0]

    def _checkpoint_cancelled(self, cancel: threading.Event) -> bool:
        """Report cancellation on the owner thread; I/O workers also poll the token.

        Model inference is not forcibly interrupted; its next boundary observes Stop.
        """
        if cancel.is_set():
            self._emit(SessionStage.STOPPING, diagnostics=self._diagnostics())
            return True
        return False

    def _count_active_chrome_streams(self) -> int:
        """Mirrors `chrome_route.select_active_chrome_stream`'s own active-non-corked
        Chrome filter, but counts matches instead of selecting one -- `evaluate_eligibility`
        needs the count (0/1/>1), while `select_active_chrome_stream` needs the single
        selection and fails closed on zero or silently first-match-wins on >1. Always
        re-reads `chrome_streams`/`chrome_sink_inputs` fresh; nothing is cached."""
        return count_active_chrome_streams(self._chrome_streams(), self._chrome_sink_inputs())

    def _record_timings(self, **values: float) -> None:
        with self._timing_lock:
            self._timings.update(values)

    def _diagnostics(
        self, *, last_routing_error: str | None = None, recovery_restored: bool | None = None,
        recovery_detail: str | None = None,
    ) -> Diagnostics:
        with self._timing_lock:
            timings = dict(self._timings)
        return Diagnostics(
            timings_ms=timings, active_operation=self._active_operation,
            backlog=self._scheduler.backlog,
            overload_dropped=self._scheduler.metrics.overload_dropped,
            stale_dropped=self._scheduler.metrics.stale_dropped,
            last_routing_error=last_routing_error,
            recovery_restored=recovery_restored,
            recovery_detail=recovery_detail,
        )

    def _emit(
        self, stage: SessionStage, *, message=None, transcript: str | None = None,
        translation: str | None = None, diagnostics: Diagnostics | None = None,
    ) -> None:
        self._observer.on_event(
            SessionEvent(
                stage=stage, message=message, transcript=transcript, translation=translation, diagnostics=diagnostics,
            )
        )
