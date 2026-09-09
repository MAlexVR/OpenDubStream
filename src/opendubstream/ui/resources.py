"""Localization catalog and persisted language preference.

Per design.md's "Localization" decision: every user-visible string the UI ever renders is
a `MessageKey` resolved by this pure, Qt-free module -- the application layer (`eligibility`,
`session_state`, `dubbing_session`) emits message keys plus parameters, never prose, and
`resolve()` is the single place English/Spanish text is authored. `resolve()` is total: an
unmapped key raises (`KeyError`) rather than silently falling back to English, matching the
`bilingual-desktop-control` spec's "Presentation-language selection and persistence"
requirement that the selected language applies to *every* user-visible string, including
diagnostics and recovery-status text -- not just static button/menu chrome.

`MessageKey` covers, per that same requirement and the spec's "Session stage visibility" /
"Diagnostics visibility" / "Recovery-failure surfacing" requirements:
- Pre-Start eligibility refusal reasons (Phase 1: `INELIGIBLE_GPU_UNAVAILABLE`,
  `INELIGIBLE_NO_CHROME_STREAM` -- values unchanged from Phase 1's stub so existing
  `eligibility.py`/`session_state.py` references keep working unmodified).
- Session stage labels, one per `SessionStage` member.
- Diagnostics field labels (execution mode, backlog, drop counts, last routing error,
  recovery restored/detail).
- The distinct, non-auto-dismissing recovery-failure message text, and its clean-recovery
  counterpart.
- Static UI chrome the design's proposal names: transcript/translation labels, Start/Stop,
  the language selector and its two language names, the window title.
- A generic unexpected-error template taking a `detail` format parameter, for error text
  that is not one of the above named cases.

Language persistence (`PreferencesStore`) follows this project's existing simple
file-based persistence convention (`infrastructure/audio/journal.py`'s `RecoveryJournal`):
one small JSON file at an injected path, `tmp_path`-testable, no Qt/`QSettings`. Per
design.md, it lives beside the routing journal (`~/.local/share/opendubstream/ui.json`) --
the concrete default path is chosen by the caller (Phase 4's Qt shell), not by this module.
A missing file defaults to `Language.ENGLISH`, matching design.md's Migration/Rollout note.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Mapping

from opendubstream.domain.contracts import AudioMode


class Language(StrEnum):
    ENGLISH = "en"
    SPANISH = "es"


class MessageKey(StrEnum):
    """Every user-visible string key, referenced by application-layer events rather than
    prose."""

    # Pre-Start eligibility refusal reasons (Phase 1 values, unchanged).
    INELIGIBLE_GPU_UNAVAILABLE = "ineligible-gpu-unavailable"
    INELIGIBLE_NO_CHROME_STREAM = "ineligible-no-chrome-stream"

    # Session stage labels -- one per `session_state.SessionStage` member.
    STAGE_IDLE = "stage-idle"
    STAGE_ARMED = "stage-armed"
    STAGE_ROUTING = "stage-routing"
    STAGE_CAPTURING = "stage-capturing"
    STAGE_TRANSCRIBING = "stage-transcribing"
    STAGE_TRANSLATING = "stage-translating"
    STAGE_VOICE = "stage-voice"
    DIAGNOSTICS_TIMINGS_LABEL = "diagnostics-timings-label"
    STAGE_PROCESSING = "stage-processing"
    STAGE_PLAYING = "stage-playing"
    STAGE_STOPPING = "stage-stopping"
    STAGE_RECOVERING = "stage-recovering"
    STAGE_FAILED = "stage-failed"
    STAGE_RECOVERY_FAILED = "stage-recovery-failed"

    # Diagnostics field labels.
    DIAGNOSTICS_EXECUTION_MODE_LABEL = "diagnostics-execution-mode-label"
    DIAGNOSTICS_BACKLOG_LABEL = "diagnostics-backlog-label"
    DIAGNOSTICS_OVERLOAD_DROPPED_LABEL = "diagnostics-overload-dropped-label"
    DIAGNOSTICS_STALE_DROPPED_LABEL = "diagnostics-stale-dropped-label"
    DIAGNOSTICS_LAST_ROUTING_ERROR_LABEL = "diagnostics-last-routing-error-label"
    DIAGNOSTICS_RECOVERY_RESTORED_LABEL = "diagnostics-recovery-restored-label"
    DIAGNOSTICS_RECOVERY_DETAIL_LABEL = "diagnostics-recovery-detail-label"

    # Recovery outcome text.
    RECOVERY_FAILED_MESSAGE = "recovery-failed-message"
    RECOVERY_SUCCEEDED_MESSAGE = "recovery-succeeded-message"

    # Last transcript/translation display.
    TRANSCRIPT_LABEL = "transcript-label"
    TRANSLATION_LABEL = "translation-label"

    # Static UI chrome.
    START_BUTTON_LABEL = "start-button-label"
    STOP_BUTTON_LABEL = "stop-button-label"
    LANGUAGE_SELECTOR_LABEL = "language-selector-label"
    LANGUAGE_ENGLISH_LABEL = "language-english-label"
    LANGUAGE_SPANISH_LABEL = "language-spanish-label"
    WINDOW_TITLE = "window-title"

    # Phase 3 workspace tabs and editable controls.
    TAB_DUBBING = "tab-dubbing"
    TAB_AUDIO = "tab-audio"
    TAB_SETTINGS = "tab-settings"
    TAB_GENERAL = "tab-general"
    APP_DESCRIPTION = "app-description"
    AUDIO_SOURCE_LABEL = "audio-source-label"
    AUDIO_OUTPUT_LABEL = "audio-output-label"
    AUDIO_MODE_LABEL = "audio-mode-label"
    MODE_CAPTIONS = "mode-captions"
    MODE_BOTH = "mode-both"
    AUDIO_MODE_DUB = "audio-mode-dub"
    AUDIO_MODE_TRANSLATION_ONLY = "audio-mode-translation-only"
    ORIGINAL_VOLUME_LABEL = "original-volume-label"
    DUB_VOLUME_LABEL = "dub-volume-label"
    SETTINGS_ASR_LABEL = "settings-asr-label"
    SETTINGS_TRANSLATION_LABEL = "settings-translation-label"
    SETTINGS_TTS_LABEL = "settings-tts-label"
    SETTINGS_VOICE_LABEL = "settings-voice-label"
    SETTINGS_SPEED_LABEL = "settings-speed-label"
    WAITING_FOR_CHROME = "waiting-for-chrome"
    SELECTED_WHEN_SESSION_STARTS = "selected-when-session-starts"
    AUTOMATIC_STREAM_SELECTION = "automatic-stream-selection"
    SELECT_PHYSICAL_OUTPUT = "select-physical-output"
    ERROR_CAPTURE = "error-capture"
    ERROR_TRANSCRIPTION = "error-transcription"
    ERROR_TRANSLATION = "error-translation"
    ERROR_VOICE = "error-voice"
    ERROR_PLAYBACK = "error-playback"
    ERROR_STALE_SELECTION = "error-stale-selection"

    # Floating caption overlay and system tray.
    CAPTION_OVERLAY_TOGGLE_LABEL = "caption-overlay-toggle-label"
    CAPTION_OVERLAY_LANGUAGE_PAIR = "caption-overlay-language-pair"
    TRAY_SHOW_WINDOW_LABEL = "tray-show-window-label"
    TRAY_QUIT_LABEL = "tray-quit-label"

    MIX_HELP = "mix-help"
    BACK_TO_VIDEO = "back-to-video"
    WORKSPACE_SUBTITLE = "workspace-subtitle"
    VOICE_TOGGLE = "voice-toggle"
    TECHNICAL_DETAILS = "technical-details"
    CAPTION_TEXT_SIZE = "caption-text-size"
    CLOSE_CAPTIONS = "close-captions"
    CAPTION_PLACEHOLDER = "caption-placeholder"
    SOURCE_PLACEHOLDER = "source-placeholder"
    SESSION_ERROR_HELP = "session-error-help"
    START_HELP = "start-help"
    CATCHING_UP = "catching-up"
    VOICE_HELP = "voice-help"
    SPEECH_SPEED_HELP = "speech-speed-help"
    NEXT_PHRASE_HELP = "next-phrase-help"

    # Generic error text.
    ERROR_UNEXPECTED = "error-unexpected"


_EDITABLE_PREFERENCE_KEYS = frozenset({
    "language", "audio_mode", "original_volume", "dub_volume", "tts_voice", "speech_speed",
})


_CATALOG: dict[Language, dict[MessageKey, str]] = {
    Language.ENGLISH: {
        MessageKey.MIX_HELP: "The original returns to full volume between Spanish phrases. These controls do not change your system volume.",
        MessageKey.BACK_TO_VIDEO: "Back to video",
        MessageKey.ERROR_CAPTURE: 'Could not read the video audio. Check that Chrome is playing, then open Troubleshooting details.',
        MessageKey.ERROR_TRANSCRIPTION: 'Could not transcribe the speech. Open Troubleshooting details and share the last error.',
        MessageKey.ERROR_TRANSLATION: 'Could not translate the text. Open Troubleshooting details and share the last error.',
        MessageKey.ERROR_VOICE: 'Could not generate the Spanish voice. Try Spanish captions (CC), or share the last error in Troubleshooting details.',
        MessageKey.ERROR_PLAYBACK: 'Could not play the Spanish voice. Check the selected output in Settings and open Troubleshooting details.',
        MessageKey.MODE_CAPTIONS: "Spanish captions (CC)",
        MessageKey.MODE_BOTH: "Dubbing + captions",
        MessageKey.SPEECH_SPEED_HELP: "Applies to newly generated speech. Already queued speech keeps its speed.",
        MessageKey.VOICE_HELP: "Turn off the translated voice immediately. Captions continue.",
        MessageKey.WORKSPACE_SUBTITLE: 'Your videos. Your language.',
        MessageKey.VOICE_TOGGLE: 'Voice',
        MessageKey.TECHNICAL_DETAILS: 'Troubleshooting details',
        MessageKey.CAPTION_TEXT_SIZE: 'Caption text size',
        MessageKey.CLOSE_CAPTIONS: 'Hide captions',
        MessageKey.CAPTION_PLACEHOLDER: 'Your Spanish subtitles will appear here.',
        MessageKey.SOURCE_PLACEHOLDER: 'Listening for the original speech…',
        MessageKey.SESSION_ERROR_HELP: 'We could not continue. Open Troubleshooting details and share the last error before trying again.',
        MessageKey.START_HELP: 'Press Start, then play a video in Chrome. You can keep listening while the first translation is prepared.',
        MessageKey.CATCHING_UP: 'Some audio was skipped to catch up. The original stays available; try a faster voice or a less demanding video.',
        MessageKey.NEXT_PHRASE_HELP: 'Applies to the next spoken phrase. Captions keep running when Voice is off.',

        MessageKey.INELIGIBLE_GPU_UNAVAILABLE: (
            "A compatible graphics processor is required for this setup. Check the setup guide before starting."
        ),
        MessageKey.INELIGIBLE_NO_CHROME_STREAM: (
            "No single actively playing Chrome audio stream was found."
        ),
        MessageKey.STAGE_IDLE: "Ready when you are",
        MessageKey.STAGE_ARMED: "Ready",
        MessageKey.STAGE_ROUTING: "Connecting your audio…",
        MessageKey.STAGE_CAPTURING: "Listening…",
        MessageKey.STAGE_TRANSCRIBING: "Recognizing English speech…",
        MessageKey.STAGE_TRANSLATING: "Translating into Spanish…",
        MessageKey.STAGE_VOICE: "Generating Spanish voice…",
        MessageKey.DIAGNOSTICS_TIMINGS_LABEL: "Latest timings (ms; includes first model load)",
        MessageKey.STAGE_PROCESSING: "Preparing your translation…",
        MessageKey.STAGE_PLAYING: "Speaking in Spanish",
        MessageKey.STAGE_STOPPING: "Stopping",
        MessageKey.STAGE_RECOVERING: "Restoring your audio…",
        MessageKey.STAGE_FAILED: "Error",
        MessageKey.STAGE_RECOVERY_FAILED: "Recovery Failed",
        MessageKey.DIAGNOSTICS_EXECUTION_MODE_LABEL: "Execution mode",
        MessageKey.DIAGNOSTICS_BACKLOG_LABEL: "Backlog",
        MessageKey.DIAGNOSTICS_OVERLOAD_DROPPED_LABEL: "Dropped (overload)",
        MessageKey.DIAGNOSTICS_STALE_DROPPED_LABEL: "Dropped (stale)",
        MessageKey.DIAGNOSTICS_LAST_ROUTING_ERROR_LABEL: "Last error",
        MessageKey.DIAGNOSTICS_RECOVERY_RESTORED_LABEL: "Recovery restored",
        MessageKey.DIAGNOSTICS_RECOVERY_DETAIL_LABEL: "Recovery detail",
        MessageKey.RECOVERY_FAILED_MESSAGE: (
            "Routing recovery failed. Manual PipeWire state may need attention."
        ),
        MessageKey.RECOVERY_SUCCEEDED_MESSAGE: "Routing recovery completed successfully.",
        MessageKey.TRANSCRIPT_LABEL: "Original · English",
        MessageKey.TRANSLATION_LABEL: "Translation · Spanish",
        MessageKey.START_BUTTON_LABEL: "Start",
        MessageKey.STOP_BUTTON_LABEL: "Stop",
        MessageKey.LANGUAGE_SELECTOR_LABEL: "Language",
        MessageKey.LANGUAGE_ENGLISH_LABEL: "English",
        MessageKey.LANGUAGE_SPANISH_LABEL: "Spanish",
        MessageKey.WINDOW_TITLE: "OpenDubStream",
        MessageKey.TAB_DUBBING: "Dubbing",
        MessageKey.TAB_AUDIO: "Audio",
        MessageKey.TAB_SETTINGS: "Settings",
        MessageKey.TAB_GENERAL: "General",
        MessageKey.APP_DESCRIPTION: (
            "Local, offline, real-time English-to-Spanish dubbing for any Chrome video."
        ),
        MessageKey.AUDIO_SOURCE_LABEL: "Chrome source",
        MessageKey.AUDIO_OUTPUT_LABEL: "Dubbing output",
        MessageKey.AUDIO_MODE_LABEL: "Mode",
        MessageKey.AUDIO_MODE_DUB: "Dubbing",
        MessageKey.AUDIO_MODE_TRANSLATION_ONLY: "Translation only",
        MessageKey.ORIGINAL_VOLUME_LABEL: "Background while speaking",
        MessageKey.DUB_VOLUME_LABEL: "Dubbed audio",
        MessageKey.SETTINGS_ASR_LABEL: "ASR: distil-large-v3 · CUDA / CPU fallback",
        MessageKey.SETTINGS_TRANSLATION_LABEL: "Translation: OPUS-MT English → Spanish (local)",
        MessageKey.SETTINGS_TTS_LABEL: "TTS: Kokoro (local)",
        MessageKey.SETTINGS_VOICE_LABEL: "Spanish voice",
        MessageKey.SETTINGS_SPEED_LABEL: "Speech speed",
        MessageKey.WAITING_FOR_CHROME: "Ready — play English audio in Chrome to start dubbing",
        MessageKey.SELECTED_WHEN_SESSION_STARTS: "Selected when the session starts",
        MessageKey.AUTOMATIC_STREAM_SELECTION: "Choose automatically",
        MessageKey.SELECT_PHYSICAL_OUTPUT: "Select a physical output",
        MessageKey.ERROR_STALE_SELECTION: "The selected Chrome stream or output is no longer available.",
        MessageKey.CAPTION_OVERLAY_TOGGLE_LABEL: "Floating captions",
        MessageKey.CAPTION_OVERLAY_LANGUAGE_PAIR: "English → Spanish",
        MessageKey.TRAY_SHOW_WINDOW_LABEL: "Show OpenDubStream",
        MessageKey.TRAY_QUIT_LABEL: "Quit",
        MessageKey.ERROR_UNEXPECTED: "An unexpected error occurred: {detail}",
    },
    Language.SPANISH: {
        MessageKey.MIX_HELP: "El original recupera su volumen entre frases en español. Estos controles no cambian el volumen del sistema.",
        MessageKey.BACK_TO_VIDEO: "Volver al video",
        MessageKey.ERROR_CAPTURE: 'No se pudo leer el audio del video. Comprueba que Chrome esté reproduciendo y abre los detalles para resolver problemas.',
        MessageKey.ERROR_TRANSCRIPTION: 'No se pudo transcribir la voz. Abre los detalles para resolver problemas y comparte el último error.',
        MessageKey.ERROR_TRANSLATION: 'No se pudo traducir el texto. Abre los detalles para resolver problemas y comparte el último error.',
        MessageKey.ERROR_VOICE: 'No se pudo generar la voz en español. Prueba el modo Subtítulos en español (CC) o comparte el último error de los detalles.',
        MessageKey.ERROR_PLAYBACK: 'No se pudo reproducir la voz en español. Revisa la salida elegida en Configuración y abre los detalles para resolver problemas.',
        MessageKey.MODE_CAPTIONS: "Subtítulos en español (CC)",
        MessageKey.MODE_BOTH: "Doblaje + subtítulos",
        MessageKey.SPEECH_SPEED_HELP: "Se aplica a la voz que se genere a continuación. La voz ya preparada conserva su velocidad.",
        MessageKey.VOICE_HELP: "Desactiva la voz traducida de inmediato. Los subtítulos continúan.",
        MessageKey.WORKSPACE_SUBTITLE: 'Tus videos. Tu idioma.',
        MessageKey.VOICE_TOGGLE: 'Voz',
        MessageKey.TECHNICAL_DETAILS: 'Detalles para resolver problemas',
        MessageKey.CAPTION_TEXT_SIZE: 'Tamaño de subtítulos',
        MessageKey.CLOSE_CAPTIONS: 'Ocultar subtítulos',
        MessageKey.CAPTION_PLACEHOLDER: 'Tus subtítulos en español aparecerán aquí.',
        MessageKey.SOURCE_PLACEHOLDER: 'Esperando la voz original…',
        MessageKey.SESSION_ERROR_HELP: 'No pudimos continuar. Abre los detalles para resolver problemas y comparte el último error antes de reintentar.',
        MessageKey.START_HELP: 'Pulsa Iniciar y reproduce un video en Chrome. Seguirás escuchando mientras se prepara la primera traducción.',
        MessageKey.CATCHING_UP: 'Se omitió parte del audio para reducir el retraso. El original sigue disponible. Abre los detalles para consultar los tiempos de procesamiento.',
        MessageKey.NEXT_PHRASE_HELP: 'Se aplica a la siguiente frase hablada. Los subtítulos continúan al desactivar Voz.',

        MessageKey.INELIGIBLE_GPU_UNAVAILABLE: (
            "No hay GPU disponible; la ejecución solo con CPU no cumple el requisito de "
            "latencia en tiempo real."
        ),
        MessageKey.INELIGIBLE_NO_CHROME_STREAM: (
            "No se encontró una única transmisión de audio de Chrome activa."
        ),
        MessageKey.STAGE_IDLE: "Todo listo para empezar",
        MessageKey.STAGE_ARMED: "Listo",
        MessageKey.STAGE_ROUTING: "Conectando el audio…",
        MessageKey.STAGE_CAPTURING: "Escuchando…",
        MessageKey.STAGE_TRANSCRIBING: "Reconociendo la voz en inglés…",
        MessageKey.STAGE_TRANSLATING: "Traduciendo al español…",
        MessageKey.STAGE_VOICE: "Generando la voz en español…",
        MessageKey.DIAGNOSTICS_TIMINGS_LABEL: "Últimos tiempos (ms; incluyen la primera carga del modelo)",
        MessageKey.STAGE_PROCESSING: "Preparando tu traducción…",
        MessageKey.STAGE_PLAYING: "Hablando en español",
        MessageKey.STAGE_STOPPING: "Deteniendo",
        MessageKey.STAGE_RECOVERING: "Restaurando el audio…",
        MessageKey.STAGE_FAILED: "Falla",
        MessageKey.STAGE_RECOVERY_FAILED: "Recuperación fallida",
        MessageKey.DIAGNOSTICS_EXECUTION_MODE_LABEL: "Modo de ejecución",
        MessageKey.DIAGNOSTICS_BACKLOG_LABEL: "Trabajo pendiente",
        MessageKey.DIAGNOSTICS_OVERLOAD_DROPPED_LABEL: "Descartes (sobrecarga)",
        MessageKey.DIAGNOSTICS_STALE_DROPPED_LABEL: "Descartes (obsoletos)",
        MessageKey.DIAGNOSTICS_LAST_ROUTING_ERROR_LABEL: "Último error",
        MessageKey.DIAGNOSTICS_RECOVERY_RESTORED_LABEL: "Recuperación restaurada",
        MessageKey.DIAGNOSTICS_RECOVERY_DETAIL_LABEL: "Detalle de recuperación",
        MessageKey.RECOVERY_FAILED_MESSAGE: (
            "La recuperación del enrutamiento falló. El estado de PipeWire puede requerir "
            "atención manual."
        ),
        MessageKey.RECOVERY_SUCCEEDED_MESSAGE: "La recuperación del enrutamiento se completó correctamente.",
        MessageKey.TRANSCRIPT_LABEL: "Original · Inglés",
        MessageKey.TRANSLATION_LABEL: "Traducción · Español",
        MessageKey.START_BUTTON_LABEL: "Iniciar",
        MessageKey.STOP_BUTTON_LABEL: "Detener",
        MessageKey.LANGUAGE_SELECTOR_LABEL: "Idioma",
        MessageKey.LANGUAGE_ENGLISH_LABEL: "Inglés",
        MessageKey.LANGUAGE_SPANISH_LABEL: "Español",
        MessageKey.WINDOW_TITLE: "OpenDubStream",
        MessageKey.TAB_DUBBING: "Doblaje",
        MessageKey.TAB_AUDIO: "Sonido",
        MessageKey.TAB_SETTINGS: "Configuración",
        MessageKey.TAB_GENERAL: "General",
        MessageKey.APP_DESCRIPTION: (
            "Doblaje local, sin conexión y en tiempo real, de inglés a español, para "
            "cualquier video de Chrome."
        ),
        MessageKey.AUDIO_SOURCE_LABEL: "Fuente de Chrome",
        MessageKey.AUDIO_OUTPUT_LABEL: "Salida del doblaje",
        MessageKey.AUDIO_MODE_LABEL: "Modo",
        MessageKey.AUDIO_MODE_DUB: "Doblaje",
        MessageKey.AUDIO_MODE_TRANSLATION_ONLY: "Solo traducción",
        MessageKey.ORIGINAL_VOLUME_LABEL: "Fondo durante el doblaje",
        MessageKey.DUB_VOLUME_LABEL: "Audio doblado",
        MessageKey.SETTINGS_ASR_LABEL: "ASR: distil-large-v3 · CUDA / CPU alternativo",
        MessageKey.SETTINGS_TRANSLATION_LABEL: "Traducción: OPUS-MT Inglés → Español (local)",
        MessageKey.SETTINGS_TTS_LABEL: "TTS: Kokoro (sin conexión)",
        MessageKey.SETTINGS_VOICE_LABEL: "Voz española",
        MessageKey.SETTINGS_SPEED_LABEL: "Velocidad de voz",
        MessageKey.WAITING_FOR_CHROME: "Listo — reproduce audio en inglés en Chrome para comenzar el doblaje",
        MessageKey.SELECTED_WHEN_SESSION_STARTS: "Se selecciona al iniciar la sesión",
        MessageKey.AUTOMATIC_STREAM_SELECTION: "Elegir automáticamente",
        MessageKey.SELECT_PHYSICAL_OUTPUT: "Selecciona una salida física",
        MessageKey.ERROR_STALE_SELECTION: "La transmisión de Chrome o la salida seleccionada ya no está disponible.",
        MessageKey.CAPTION_OVERLAY_TOGGLE_LABEL: "Subtítulos flotantes",
        MessageKey.CAPTION_OVERLAY_LANGUAGE_PAIR: "Inglés → Español",
        MessageKey.TRAY_SHOW_WINDOW_LABEL: "Mostrar OpenDubStream",
        MessageKey.TRAY_QUIT_LABEL: "Salir",
        MessageKey.ERROR_UNEXPECTED: "Ocurrió un error inesperado: {detail}",
    },
}


def resolve(language: Language, key: MessageKey, params: Mapping[str, str] | None = None) -> str:
    """Resolve `key` to localized text for `language`. Total per design.md: an unmapped
    key raises `KeyError` rather than silently falling back to English -- there is no
    `.get(..., default)` anywhere in this lookup. `params` is optional and defaults to an
    empty mapping; a template with no placeholders resolves fine with none supplied, and
    unused keys in `params` are silently ignored by `str.format`, matching normal Python
    formatting semantics."""
    template = _CATALOG[language][key]
    return template.format(**(params or {}))


@dataclass(frozen=True)
class Preferences:
    language: Language = Language.ENGLISH
    audio_mode: AudioMode = AudioMode.DUB
    original_volume: int = 20
    dub_volume: int = 100
    tts_voice: str = "ef_dora"
    speech_speed: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.audio_mode, AudioMode):
            raise ValueError("audio mode must be an AudioMode")
        if any(not isinstance(value, int) or not 0 <= value <= 100 for value in (self.original_volume, self.dub_volume)):
            raise ValueError("volume percentages must be integers from 0 through 100")
        if self.tts_voice != "ef_dora":
            raise ValueError("voice must be an installed Spanish voice")
        if not isinstance(self.speech_speed, (int, float)) or isinstance(self.speech_speed, bool) or not 0.5 <= self.speech_speed <= 2.0:
            raise ValueError("speech speed must be from 0.5 through 2.0")


class PreferencesStore:
    """Durable single-value language preference, matching `RecoveryJournal`'s file-based
    persistence convention (`infrastructure/audio/journal.py`) but without its PipeWire
    crash-recovery fsync/directory-fsync guarantees -- a UI preference is not safety
    critical to that degree; a simple atomic write (temp file + `os.replace`) is enough to
    avoid a partially written `ui.json`. A missing file defaults to `Language.ENGLISH`,
    matching design.md's Migration/Rollout note ("A missing `ui.json` defaults to English
    and is written on first change")."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def load(self) -> Preferences:
        if not self._path.exists():
            return Preferences()
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict) or set(raw) - _EDITABLE_PREFERENCE_KEYS:
                return Preferences()
            return Preferences(
                language=Language(raw.get("language", Language.ENGLISH.value)),
                audio_mode=AudioMode(raw.get("audio_mode", AudioMode.DUB.value)),
                original_volume=raw.get("original_volume", 20),
                dub_volume=raw.get("dub_volume", 100),
                tts_voice=raw.get("tts_voice", "ef_dora"),
                speech_speed=raw.get("speech_speed", 1.0),
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return Preferences()

    def save(self, preferences: Preferences) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({
            "audio_mode": preferences.audio_mode.value,
            "dub_volume": preferences.dub_volume,
            "language": preferences.language.value,
            "original_volume": preferences.original_volume,
            "speech_speed": preferences.speech_speed,
            "tts_voice": preferences.tts_voice,
        }, sort_keys=True).encode("utf-8")
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{self._path.name}.", dir=self._path.parent)
        try:
            with os.fdopen(descriptor, "wb") as temporary:
                temporary.write(payload)
            os.replace(temporary_name, self._path)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)
