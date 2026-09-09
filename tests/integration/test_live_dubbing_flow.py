"""Concurrent fake devices prove progress without real audio or inference."""
import threading

from test_dubbing_session import (
    FakeCapture, FakeRouter, RecordingObserver, FakeOriginalMix, FakePlayer,
    FakeVad, FakeAsr, FakeTranslator, FakeTts, build_runner, chrome_stream,
)
from opendubstream.application.session_state import SessionStage
from opendubstream.domain.contracts import AudioMixSettings
from opendubstream.pipeline.local_pipeline import LocalDubbingPipeline


def test_capture_continues_while_tts_runs_and_text_is_already_visible():
    cancel = threading.Event()
    captured_again = threading.Event()
    observer = RecordingObserver()
    class Capture(FakeCapture):
        def read_phrase(self, deadline):
            if self.read_calls:
                captured_again.set()
                self.stopped.wait(deadline)
                return b''
            return super().read_phrase(deadline)
        def __init__(self):
            super().__init__([b'phrase'])
            self.stopped = threading.Event()
        def stop(self):
            self.stopped.set()
            super().stop()
    class Tts(FakeTts):
        def synthesize(self, text):
            assert captured_again.wait(1), 'capture stalled behind TTS'
            assert any(e.translation == 'hola' for e in observer.events), 'captions waited for TTS'
            cancel.set()
            return super().synthesize(text)
    router = FakeRouter(lambda: [chrome_stream()])
    runner = build_runner(router=router, capture=Capture(), player=FakePlayer(), observer=observer,
        pipeline=LocalDubbingPipeline(FakeVad(), FakeAsr(), FakeTranslator(), Tts(), clock=lambda: 0))
    runner.run(cancel)
    assert observer.events[-1].stage == SessionStage.IDLE
    assert router.recover_calls == 1


def test_original_is_audible_before_tts_and_ducking_restores_after_playback_error():
    cancel = threading.Event()
    mix = FakeOriginalMix()
    class Tts(FakeTts):
        def synthesize(self, text):
            assert mix.updated[-1][1:] == (False, 100)
            return super().synthesize(text)
    class Player(FakePlayer):
        def play(self, *args):
            assert mix.updated[-1][1:] == (False, 20)
            raise RuntimeError('speaker gone')
    router = FakeRouter(lambda: [chrome_stream()])
    observer = RecordingObserver()
    runner = build_runner(router=router, capture=FakeCapture([b'phrase']), player=Player(), observer=observer,
        original_mix_controller=mix,
        pipeline=LocalDubbingPipeline(FakeVad(), FakeAsr(), FakeTranslator(), Tts(), clock=lambda: 0))
    runner.run(cancel)
    assert any(e.translation == 'hola' for e in observer.events)
    assert mix.updated[-1][1:] == (False, 100)
    assert observer.events[-1].stage == SessionStage.FAILED
    assert observer.events[-1].diagnostics.last_routing_error == "speaker gone"
    assert router.recover_calls == 1


def test_empty_asr_does_not_invoke_translation_or_voice():
    class Asr:
        def transcribe(self, audio): return '  '
    class Forbidden:
        def translate(self, text): raise AssertionError('empty translation')
        def synthesize(self, text): raise AssertionError('empty voice')
    from opendubstream.pipeline.local_pipeline import RejectedUtterance
    from opendubstream.domain.pipeline import Utterance
    import pytest
    with pytest.raises(RejectedUtterance):
        LocalDubbingPipeline(FakeVad(), Asr(), Forbidden(), Forbidden(), clock=lambda: 0).process(Utterance('1', b'a', 0, 0))


def test_voice_off_skips_tts_but_keeps_translated_captions():
    from opendubstream.application.live_controls import LiveSettings
    cancel = threading.Event()
    observer = RecordingObserver()
    class Tts(FakeTts):
        def synthesize(self, text):
            raise AssertionError('disabled voice must not synthesize')
    class Observer(RecordingObserver):
        def on_event(self, event):
            super().on_event(event)
            if event.translation == 'hola': cancel.set()
    observer = Observer()
    runner = build_runner(router=FakeRouter(lambda: [chrome_stream()]), capture=FakeCapture([b'phrase']),
        player=FakePlayer(), observer=observer,
        pipeline=LocalDubbingPipeline(FakeVad(), FakeAsr(), FakeTranslator(), Tts(), clock=lambda: 0))
    runner._controls.update(LiveSettings(voice_enabled=False))
    runner.run(cancel)
    assert observer.events[-1].stage == SessionStage.IDLE
    assert any(e.translation == 'hola' for e in observer.events)


def test_capture_start_failure_still_stops_adapter_and_recovers():
    class Capture(FakeCapture):
        def start(self, lease):
            super().start(lease)
            raise RuntimeError('partial capture startup')
    capture = Capture([])
    router = FakeRouter(lambda: [chrome_stream()])
    observer = RecordingObserver()
    build_runner(router=router, capture=capture, player=FakePlayer(), observer=observer).run(threading.Event())
    assert capture.stop_calls == 1
    assert router.recover_calls == 1
    assert observer.events[-1].diagnostics.last_routing_error == 'partial capture startup'


def test_inference_advances_while_prior_audio_is_playing():
    cancel = threading.Event()
    playback_started = threading.Event()
    next_inferred = threading.Event()
    class Capture(FakeCapture):
        def read_phrase(self, deadline):
            if self.read_calls == 1:
                assert playback_started.wait(1)
            return super().read_phrase(deadline)
    class Asr(FakeAsr):
        def transcribe(self, audio):
            if audio == b'second': next_inferred.set()
            return super().transcribe(audio)
    class Player(FakePlayer):
        def play(self, *args):
            playback_started.set()
            assert next_inferred.wait(1), 'inference stalled behind playback'
            cancel.set()
    observer = RecordingObserver()
    build_runner(router=FakeRouter(lambda: [chrome_stream()]), capture=Capture([b'first', b'second']),
        player=Player(), observer=observer,
        pipeline=LocalDubbingPipeline(FakeVad(), Asr(), FakeTranslator(), FakeTts(), clock=lambda: 0)).run(cancel)
    assert next_inferred.is_set()
    assert observer.events[-1].stage == SessionStage.IDLE


def test_capture_overload_keeps_newest_audio_and_reports_actual_drops():
    from opendubstream.pipeline.scheduler import FixedCapacityScheduler
    cancel = threading.Event()
    processing_started = threading.Event()
    burst_complete = threading.Event()
    seen = []
    class Capture(FakeCapture):
        def read_phrase(self, deadline):
            if self.read_calls == 1:
                assert processing_started.wait(1)
            if self.read_calls == 7:
                burst_complete.set()
            return super().read_phrase(deadline)
    class Asr(FakeAsr):
        def transcribe(self, audio):
            seen.append(audio)
            if audio == b'first':
                processing_started.set()
                assert burst_complete.wait(1)
            if audio == b'6': cancel.set()
            return super().transcribe(audio)
    scheduler = FixedCapacityScheduler(capacity=2, max_age_seconds=30)
    observer = RecordingObserver()
    build_runner(router=FakeRouter(lambda: [chrome_stream()]),
        capture=Capture([b'first'] + [str(i).encode() for i in range(1, 7)]),
        player=FakePlayer(), observer=observer, scheduler=scheduler,
        pipeline=LocalDubbingPipeline(FakeVad(), Asr(), FakeTranslator(), FakeTts(), clock=lambda: 0)).run(cancel)
    assert seen == [b'first', b'5', b'6']
    assert scheduler.metrics.overload_dropped == 4
    assert observer.events[-1].diagnostics.overload_dropped == 4
    assert observer.events[-1].stage == SessionStage.IDLE


def test_playback_cancel_check_tracks_voice_toggle_and_stop():
    from opendubstream.application.live_controls import LiveSettings
    cancel = threading.Event()
    class Player(FakePlayer):
        def set_cancel_check(self, check): self.check = check
        def play(self, *args):
            assert not self.check()
            runner._controls.update(LiveSettings(voice_enabled=False))
            assert self.check()
            runner._controls.update(LiveSettings(voice_enabled=True))
            cancel.set()
            assert self.check()
    player = Player()
    observer = RecordingObserver()
    runner = build_runner(router=FakeRouter(lambda: [chrome_stream()]), capture=FakeCapture([b'first']),
        player=player, observer=observer)
    runner.run(cancel)
    assert observer.events[-1].stage == SessionStage.IDLE


def test_tts_failure_reports_voice_stage_without_blame_on_output():
    from opendubstream.ui.resources import MessageKey
    class Tts(FakeTts):
        def synthesize(self, text):
            raise RuntimeError('synthetic voice backend unavailable')
    observer = RecordingObserver()
    router = FakeRouter(lambda: [chrome_stream()])
    runner = build_runner(router=router, capture=FakeCapture([b'phrase']), player=FakePlayer(), observer=observer,
        pipeline=LocalDubbingPipeline(FakeVad(), FakeAsr(), FakeTranslator(), Tts(), clock=lambda: 0))
    runner.run(threading.Event())
    assert observer.events[-1].message is MessageKey.ERROR_VOICE
    assert observer.events[-1].diagnostics.last_routing_error == 'synthetic voice backend unavailable'
    assert router.recover_calls == 1


def test_caption_mode_never_synthesizes_or_ducks_even_with_voice_toggle_on():
    from opendubstream.application.live_controls import LiveSettings
    from opendubstream.domain.contracts import AudioMode
    cancel = threading.Event()
    mix = FakeOriginalMix()
    class Tts(FakeTts):
        def synthesize(self, text):
            raise AssertionError('caption mode must not synthesize')
    class Observer(RecordingObserver):
        def on_event(self, event):
            super().on_event(event)
            if event.translation:
                cancel.set()
    observer = Observer()
    player = FakePlayer()
    runner = build_runner(router=FakeRouter(lambda: [chrome_stream()]), capture=FakeCapture([b'phrase']),
        player=player, observer=observer, original_mix_controller=mix,
        pipeline=LocalDubbingPipeline(FakeVad(), FakeAsr(), FakeTranslator(), Tts(), clock=lambda: 0))
    runner._controls.update(LiveSettings(mix=AudioMixSettings(mode=AudioMode.CAPTIONS), voice_enabled=True))
    runner.run(cancel)
    assert observer.events[-1].stage == SessionStage.IDLE
    assert any(event.translation for event in observer.events)
    assert all(update[1:] == (False, 100) for update in mix.updated)


def test_cc_preserves_browser_route_and_never_recovers_or_mixes():
    from opendubstream.application.live_controls import LiveSettings
    from opendubstream.domain.contracts import AudioMode
    cancel = threading.Event()
    router = FakeRouter(lambda: [chrome_stream()])
    mix = FakeOriginalMix()
    capture = FakeCapture([b'phrase'])
    class Observer(RecordingObserver):
        def on_event(self, event):
            super().on_event(event)
            if event.translation:
                cancel.set()
    observer = Observer()
    runner = build_runner(router=router, capture=capture, player=FakePlayer(),
                          observer=observer, original_mix_controller=mix)
    runner._controls.update(LiveSettings(mix=AudioMixSettings(mode=AudioMode.CAPTIONS)))
    runner.run(cancel)
    assert observer.events[-1].stage == SessionStage.IDLE
    assert capture.start_calls == [chrome_stream()]
    assert router.begin_calls == []
    assert router.recover_calls == 0
    assert mix.started == []
    assert mix.updated == []


def test_cc_capture_failure_does_not_touch_browser_route():
    from opendubstream.application.live_controls import LiveSettings
    from opendubstream.domain.contracts import AudioMode
    class Capture(FakeCapture):
        def start(self, lease):
            raise RuntimeError('selected stream disappeared')
    router = FakeRouter(lambda: [chrome_stream()])
    observer = RecordingObserver()
    capture = Capture([])
    runner = build_runner(router=router, capture=capture, player=FakePlayer(), observer=observer)
    runner._controls.update(LiveSettings(mix=AudioMixSettings(mode=AudioMode.CAPTIONS)))
    runner.run(threading.Event())
    assert observer.events[-1].stage == SessionStage.FAILED
    assert observer.events[-1].diagnostics.last_routing_error == 'selected stream disappeared'
    assert capture.stop_calls == 1
    assert router.begin_calls == []
    assert router.recover_calls == 0


def test_passive_session_cannot_synthesize_after_live_mode_changes():
    from opendubstream.application.live_controls import LiveSettings
    from opendubstream.domain.contracts import AudioMode
    cancel = threading.Event()
    class Tts(FakeTts):
        def synthesize(self, text):
            raise AssertionError('passive stream must never synthesize')
    class Translator(FakeTranslator):
        def translate(self, text):
            runner._controls.update(LiveSettings(mix=AudioMixSettings(mode=AudioMode.DUB)))
            return super().translate(text)
    class Capture(FakeCapture):
        def read_phrase(self, deadline):
            if self.read_calls == 1:
                cancel.wait(0.1)
                cancel.set()
                return b''
            return super().read_phrase(deadline)
    observer = RecordingObserver()
    player = FakePlayer()
    runner = build_runner(router=FakeRouter(lambda: [chrome_stream()]), capture=Capture([b'phrase']),
        player=player, observer=observer,
        pipeline=LocalDubbingPipeline(FakeVad(), FakeAsr(), Translator(), Tts(), clock=lambda: 0))
    runner._controls.update(LiveSettings(mix=AudioMixSettings(mode=AudioMode.CAPTIONS)))
    runner.run(cancel)
    assert observer.events[-1].stage == SessionStage.IDLE
    assert any(e.translation == 'hola' for e in observer.events)


def test_fresh_voice_survives_slow_inference_without_source_age_starvation():
    now = [0.0]
    cancel = threading.Event()
    class Tts(FakeTts):
        def synthesize(self, text):
            now[0] = 12.0
            return super().synthesize(text)
    class Player(FakePlayer):
        def play(self, *args):
            super().play(*args)
            cancel.set()
    player = Player()
    observer = RecordingObserver()
    runner = build_runner(router=FakeRouter(lambda: [chrome_stream()]),
        capture=FakeCapture([b'phrase']), player=player, observer=observer,
        pipeline=LocalDubbingPipeline(FakeVad(), FakeAsr(), FakeTranslator(), Tts(), clock=lambda: now[0]))
    runner._clock = lambda: now[0]
    runner._scheduler.max_age_seconds = 10.0
    watchdog = threading.Timer(0.5, cancel.set)
    watchdog.start()
    try:
        runner.run(cancel)
    finally:
        watchdog.cancel()
    assert len(player.play_calls) == 1, 'fresh voice discarded solely for inference time'
    played = next(event for event in observer.events if event.stage is SessionStage.PLAYING)
    assert played.diagnostics.timings_ms['voice_age'] == 12000
    assert played.diagnostics.timings_ms['playback_wait'] == 0
    assert played.diagnostics.timings_ms['voice'] == 12000


def test_pipeline_reports_stage_durations_without_text_or_audio():
    from opendubstream.domain.pipeline import Utterance
    now = [0.0]
    class Asr(FakeAsr):
        def transcribe(self, audio):
            now[0] += 2
            return super().transcribe(audio)
    class Translator(FakeTranslator):
        def translate(self, text):
            now[0] += 3
            return super().translate(text)
    class Tts(FakeTts):
        def synthesize(self, text):
            now[0] += 4
            return super().synthesize(text)
    stages = []
    result = LocalDubbingPipeline(FakeVad(), Asr(), Translator(), Tts(), clock=lambda: now[0]).process(
        Utterance('x', b'private audio', 0, 0), on_stage=lambda name, duration: stages.append((name, duration)))
    assert ('transcription', 2000.0) in stages
    assert ('translation', 3000.0) in stages
    assert ('voice', 4000.0) in stages
    assert ('voice', None) in stages
    assert all(isinstance(name, str) and (duration is None or isinstance(duration, float)) for name, duration in stages)
    assert result.timestamps['tts'] == 9.0


def test_completed_voice_still_expires_after_waiting_in_playback_queue():
    now = [0.0]
    cancel = threading.Event()
    first_playing = threading.Event()
    second_queued = threading.Event()
    after_second_voice = [False]
    class Capture(FakeCapture):
        def read_phrase(self, deadline):
            if self.read_calls == 1:
                assert first_playing.wait(1)
            return super().read_phrase(deadline)
    class Tts(FakeTts):
        def __init__(self): self.calls = 0
        def synthesize(self, text):
            self.calls += 1
            if self.calls == 2: after_second_voice[0] = True
            return super().synthesize(text)
    def clock():
        value = now[0]
        # Runner calls its clock with the queue condition held when publishing voice.
        # The playback worker can only dequeue after that tuple is fully published.
        if after_second_voice[0] and threading.current_thread().name != 'dubbing-playback':
            second_queued.set()
        return value
    class Player(FakePlayer):
        def play(self, *args):
            super().play(*args)
            first_playing.set()
            assert second_queued.wait(1)
            now[0] = 11.0
    player = Player()
    runner = build_runner(router=FakeRouter(lambda: [chrome_stream()]),
        capture=Capture([b'first', b'second']), player=player, observer=RecordingObserver(),
        pipeline=LocalDubbingPipeline(FakeVad(), FakeAsr(), FakeTranslator(), Tts(), clock=lambda: 0))
    runner._clock = clock
    runner._scheduler.max_age_seconds = 10
    watchdog = threading.Timer(.5, cancel.set)
    watchdog.start()
    try:
        runner.run(cancel)
    finally:
        watchdog.cancel()
    assert len(player.play_calls) == 1
    assert runner._scheduler.metrics.stale_dropped == 1
