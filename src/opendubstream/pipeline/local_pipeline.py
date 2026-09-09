"""Synchronous, testable local VAD → ASR → OPUS-MT → TTS phrase path."""

from __future__ import annotations

from collections.abc import Callable

from opendubstream.domain.pipeline import Asr, ProcessedUtterance, Translator, Tts, Utterance, Vad


class RejectedUtterance(ValueError):
    """VAD rejected an audio segment before any inference begins."""


class PipelineStageError(RuntimeError):
    """Retain the failing boundary without storing transcript or translated text."""

    def __init__(self, stage: str, error: Exception) -> None:
        super().__init__(str(error))
        self.stage = stage


def _at_stage(stage, function, *args):
    try:
        return function(*args)
    except Exception as error:
        raise PipelineStageError(stage, error) from error


class LocalDubbingPipeline:
    def __init__(self, vad: Vad, asr: Asr, translator: Translator, tts: Tts, *, clock: Callable[[], float]) -> None:
        self._vad = vad
        self._asr = asr
        self._translator = translator
        self._tts = tts
        self._clock = clock

    def reset(self) -> None:
        reset = getattr(self._vad, "reset", None)
        if reset is not None:
            reset()

    def process(self, utterance: Utterance, *, on_text=None, should_synthesize=None, on_stage=None) -> ProcessedUtterance:
        def measured(stage, function, *args):
            if on_stage is None:
                return _at_stage(stage, function, *args)
            on_stage(stage, None)
            started = self._clock()
            result = _at_stage(stage, function, *args)
            on_stage(stage, (self._clock() - started) * 1000.0)
            return result

        if not measured("vad", self._vad.accepts, utterance.audio):
            raise RejectedUtterance(utterance.identifier)
        transcript = measured("transcription", self._asr.transcribe, utterance.audio)
        if not transcript.strip():
            raise RejectedUtterance(utterance.identifier)
        if on_text is not None:
            on_text(transcript, None)
        timestamps = {"asr": self._clock()}
        translation = measured("translation", self._translator.translate, transcript)
        if not translation.strip():
            raise RejectedUtterance(utterance.identifier)
        if on_text is not None:
            on_text(transcript, translation)
        timestamps["translation"] = self._clock()
        synthesized = measured("voice", self._tts.synthesize, translation) if should_synthesize is None or should_synthesize() else b""
        timestamps["tts"] = self._clock()
        timestamps["playback"] = self._clock()
        return ProcessedUtterance(utterance, transcript, translation, synthesized, timestamps)
