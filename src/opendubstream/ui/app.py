"""Localized desktop controller and independent captions, covered by offscreen tests.

Model construction remains lazy; session work runs off the Qt thread.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QObject, Qt, QSignalBlocker, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QActionGroup, QColor, QIcon, QMouseEvent, QPainter, QPalette, QPixmap, QTextLayout, QFontMetrics
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QPushButton,
    QSpinBox,
    QSlider,
    QScrollArea,
    QSystemTrayIcon,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from opendubstream.application.chrome_route import count_active_chrome_streams, select_active_chrome_stream
from opendubstream.application.dubbing_session import DubbingSessionRunner
from opendubstream.application.live_controls import LiveControls, LiveSettings
from opendubstream.application.eligibility import evaluate_eligibility
from opendubstream.application.session_state import SessionEvent, SessionStage, SessionStateMachine, SessionStatus
from opendubstream.domain.contracts import AudioMixSettings, AudioMode, SinkRef, StreamRef
from opendubstream.domain.pipeline import Utterance  # noqa: F401  (documents the Vad/Asr/Translator/Tts data shape below)
from opendubstream.infrastructure.audio.capture import PipeWireMonitorCapture
from opendubstream.infrastructure.audio.phrases import PhraseCapture
from opendubstream.infrastructure.audio.discovery import OwnedMonitorDiscovery, PipeWireStreamDiscovery
from opendubstream.infrastructure.audio.journal import RecoveryJournal
from opendubstream.infrastructure.audio.playback import PwPlayPcmPlayer
from opendubstream.infrastructure.audio.process import SafePactlRunner
from opendubstream.infrastructure.audio.router import PipeWirePulseRouter
from opendubstream.infrastructure.models.asr import FasterWhisperTranscriber, create_faster_whisper_model
from opendubstream.infrastructure.models.assets import AssetManifest, AssetPin, LocalAssets
from opendubstream.infrastructure.models.translation import OpusMtEnglishSpanishTranslator
from opendubstream.infrastructure.models.tts import KokoroSpanishSynthesizer
from opendubstream.infrastructure.models.vad import SileroVadDetector, create_silero_vad_session
from opendubstream.pipeline.local_pipeline import LocalDubbingPipeline
from opendubstream.pipeline.scheduler import FixedCapacityScheduler
from opendubstream.ui.resources import Language, MessageKey, Preferences, PreferencesStore, resolve

DEFAULT_MODEL_ROOT = Path(os.environ.get("OPENDUBSTREAM_MODEL_ROOT", Path.home() / ".local/share/opendubstream/models"))
DEFAULT_JOURNAL_PATH = Path.home() / ".local/share/opendubstream/routing.json"
DEFAULT_PREFERENCES_PATH = Path.home() / ".local/share/opendubstream/ui.json"
DEFAULT_VOICE = "ef_dora"
_ICON_PATH = Path(__file__).resolve().parents[3] / "assets" / "opendubstream.svg"
_SCHEDULER_CAPACITY = 4
_SCHEDULER_MAX_AGE_SECONDS = 10.0
_PREFLIGHT_REFRESH_MILLISECONDS = 2_000

# BluCast reference palette with an original OpenDubStream layout and icon.
_STYLESHEET = """
QMainWindow, QWidget#centralSurface { background: #0a0f0a; }
QWidget { color: #e2e8f0; font-family: 'Ubuntu', 'Inter', sans-serif; font-size: 13px; }
QLabel { color: #94a3b8; background: transparent; border: none; }
QPushButton { background: #1a1f1a; color: #e2e8f0; border: 1px solid #2d3d2d; border-radius: 9px;
    padding: 8px 14px; min-height: 22px; font-weight: 500; }
QPushButton:hover { background: #1f2a1f; border-color: #3b82f6; }
QPushButton:pressed { background: #2d3d2d; }
QPushButton:checked { background: #1f2a1f; color: #60a5fa; border-color: #3b82f6; }
QPushButton:disabled { background: #14170f; border-color: #1f261f; color: #64748b; }
QPushButton:default:enabled { background: #3b82f6; color: #ffffff; border-color: #3b82f6; font-weight: 700; }
QPushButton:default:hover { background: #2563eb; }
QPushButton:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus { border: 2px solid #3b82f6; }
QTabWidget::pane { background: #111611; border: 1px solid #1f2a1f; border-radius: 12px; top: -1px; }
QTabBar::tab { background: transparent; color: #94a3b8; padding: 12px 21px; border-bottom: 3px solid transparent; }
QTabBar::tab:selected { color: #60a5fa; border-bottom: 3px solid #3b82f6; }
QTabBar::tab:hover { color: #e2e8f0; }
QGroupBox { border: 1px solid #1f2a1f; border-radius: 10px; background: #111611; padding: 12px; margin-top: 4px; }
QTextEdit { border: none; border-radius: 8px; background: #0d120d; color: #e2e8f0; padding: 10px; selection-background-color: #3b82f6; }
QTextEdit#translationText { font-size: 15px; font-weight: 500; }
QTextEdit#sourceText { color: #94a3b8; font-size: 15px; }
QComboBox, QSpinBox, QDoubleSpinBox { background: #1a1f1a; border: 1px solid #2d3d2d; border-radius: 8px; padding: 7px 11px; min-height: 22px; }
QComboBox::drop-down { border: none; width: 24px; }
QComboBox QAbstractItemView { background: #1a1f1a; color: #e2e8f0; selection-background-color: #3b82f6; padding: 8px; }
QSlider::groove:horizontal { height: 5px; background: #2d3d2d; border-radius: 2px; }
QSlider::sub-page:horizontal { background: #3b82f6; border-radius: 2px; }
QSlider::handle:horizontal { background: #60a5fa; width: 14px; margin: -5px 0; border-radius: 7px; }
QMenu { background: #111611; color: #e2e8f0; border: 1px solid #2d3d2d; padding: 6px; }
QMenu::item { padding: 7px 18px; }
QMenu::item:selected { background: #1f2a1f; }
QMenu::item:disabled { color: #64748b; }
QScrollArea { border: none; background: transparent; }
QScrollBar:vertical { background: #111611; width: 8px; }
QScrollBar::handle:vertical { background: #3d4d3d; border-radius: 4px; min-height: 20px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QWidget#titleBar { background: #1b1b1b; border-top-left-radius: 18px; border-top-right-radius: 18px; }
QLabel#titleBarLabel { color: #dedede; background: transparent; font-size: 14px; font-weight: 700; }
QPushButton#titleBarMinimize, QPushButton#titleBarMaximize, QPushButton#titleBarClose { border: none; background: #303030; color: #dedede; border-radius: 15px; padding: 0; min-height: 0; font-size: 16px; }
QPushButton#titleBarClose:hover, QPushButton#captionOverlayClose:hover { background: #913943; color: white; }
QLabel#brandTitle, QLabel#generalAppTitle { color: #e2e8f0; font-size: 21px; font-weight: 700; }
QLabel#captionOverlayLanguages { color: #3b82f6; font-size: 12px; font-weight: 600; }
QWidget#captionOverlay { background: #0a0f0a; border: 1px solid #2d3d2d; border-radius: 16px; }
QLabel#captionOverlayText { color: #ffffff; font-size: 22px; font-weight: 600; }
QPushButton#captionOverlayClose { border: none; background: transparent; padding: 0; min-height: 0; }
"""


def _dark_palette() -> QPalette:
    """Verbatim port of Blucast's own `main()` palette (`app/control_panel.py`). QSS only
    paints what it explicitly targets; a top-level popup (e.g. a `QComboBox` dropdown)
    still fills its own window rect with `QPalette.Window`/`Base` before drawing the
    QSS-styled rounded content over it, so without this dark palette the pixels outside a
    rounded QSS corner render the platform's default white instead of matching the theme."""
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor("#0a0f0a"))
    palette.setColor(QPalette.ColorRole.WindowText, QColor("#e2e8f0"))
    palette.setColor(QPalette.ColorRole.Base, QColor("#111611"))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#1a1f1a"))
    palette.setColor(QPalette.ColorRole.Text, QColor("#e2e8f0"))
    palette.setColor(QPalette.ColorRole.Button, QColor("#1a1f1a"))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor("#94a3b8"))
    palette.setColor(QPalette.ColorRole.Highlight, QColor("#3b82f6"))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    return palette

_REASON_LABEL_DEFAULT_STYLE = "color: #94a3b8; background: transparent;"
_REASON_LABEL_RECOVERY_FAILED_STYLE = "font-weight: bold; color: white; background-color: #b00020; padding: 4px; border-radius: 6px;"
_STAGE_LABEL_DEFAULT_STYLE = "color: #e2e8f0; background: transparent; font-size: 14px; font-weight: 600;"
_STAGE_LABEL_RECOVERY_FAILED_STYLE = "font-weight: bold; color: white; background-color: #b00020; padding: 4px; border-radius: 6px;"

_STAGE_MESSAGE_KEYS: dict[SessionStage, MessageKey] = {
    SessionStage.IDLE: MessageKey.STAGE_IDLE,
    SessionStage.ARMED: MessageKey.STAGE_ARMED,
    SessionStage.ROUTING: MessageKey.STAGE_ROUTING,
    SessionStage.CAPTURING: MessageKey.STAGE_CAPTURING,
    SessionStage.PROCESSING: MessageKey.STAGE_PROCESSING,
    SessionStage.PLAYING: MessageKey.STAGE_PLAYING,
    SessionStage.STOPPING: MessageKey.STAGE_STOPPING,
    SessionStage.RECOVERING: MessageKey.STAGE_RECOVERING,
    SessionStage.FAILED: MessageKey.STAGE_FAILED,
    SessionStage.RECOVERY_FAILED: MessageKey.STAGE_RECOVERY_FAILED,
}


def cuda_available() -> bool:
    """Lazy CUDA probe: `torch` is imported only when actually checking, matching
    `run_confirmed`'s own lazy-ML-import discipline so UI startup never touches CUDA."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _resolve_physical_sink(pactl: SafePactlRunner, sink_name: str) -> SinkRef:
    """Real-pactl JSON lookup mirroring `scripts/verify-real-audio-loop.py`'s own
    `sink_serial()` helper verbatim -- one loop, no branching decision, matching the
    already-untested-by-design real-hardware wiring boundary this codebase already uses
    for `OwnedMonitorDiscovery`/`resolve_physical_target`. `physical_sink` is the sink
    Chrome is already outputting to before routing moves its stream onto a fresh virtual
    sink -- the same real target `run_confirmed` resolves, but computed once per Start
    press instead of `run_confirmed`'s live per-write re-resolution (see apply-progress.md's
    Phase 4 Deviations: `DubbingSessionRunner.__init__` takes one constructor-injected
    `physical_sink`, not a live per-write re-resolution callback)."""
    sinks = json.loads(pactl.run(("pactl", "-f", "json", "list", "sinks")))
    serial = ""
    for entry in sinks:
        if entry.get("name") == sink_name:
            candidate = entry.get("properties", {}).get("object.serial")
            if isinstance(candidate, str) and candidate:
                serial = candidate
            break
    return SinkRef(sink_name, True, serial)


def _local_assets(model_root: Path) -> LocalAssets:
    """Build the `LocalAssets` handle the tested adapter classes require. See the module
    docstring: `.require()` only confirms a pinned filename is declared, it never reads or
    hashes the file, so the `sha256` field below is a documented placeholder, not an active
    verification value -- hash verification already happened during provisioning."""
    return LocalAssets(
        root=model_root,
        manifest=AssetManifest(
            pins=(
                AssetPin(name="silero-vad", filename="silero-vad/silero_vad.onnx", sha256="unverified-at-ui-runtime"),
                AssetPin(
                    name="faster-whisper-distil-large-v3", filename="faster-whisper-distil-large-v3",
                    sha256="unverified-at-ui-runtime",
                ),
                AssetPin(name="opus-mt-en-es", filename="opus-mt-en-es", sha256="unverified-at-ui-runtime"),
                AssetPin(name="kokoro-onnx-v1", filename="kokoro-onnx-v1", sha256="unverified-at-ui-runtime"),
            ),
        ),
    )


def _make_vad_infer(model_root: Path) -> Callable[[list[float]], float]:
    """Lazily constructs the real Silero VAD session on first use, mirroring
    `run_confirmed()`'s inline `has_voice()` session construction exactly. Returns a
    single-window scoring callable; `SileroVadDetector` owns the windowing/threshold logic."""
    state: dict[str, object] = {}

    def infer(window: list[float]) -> float:
        if "session" not in state:
            import onnxruntime as ort

            state["session"] = create_silero_vad_session(str(model_root / "silero-vad/silero_vad.onnx"), ort.InferenceSession)
            state["rnn_state"] = None
        import numpy as np

        session = state["session"]
        rnn_state = state["rnn_state"]
        if rnn_state is None:
            rnn_state = np.zeros((2, 1, 128), dtype="float32")
        if len(window) != 512:
            raise ValueError("Silero requires 512 samples at 16 kHz")
        context = state.get("context", np.zeros((1, 64), dtype="float32"))
        chunk = np.concatenate((context, np.asarray(window, dtype="float32")[None, :]), axis=1)
        score, rnn_state = session.run(None, {"input": chunk, "state": rnn_state, "sr": np.array(16000, dtype="int64")})
        state["rnn_state"] = rnn_state
        state["context"] = chunk[:, -64:].copy()
        return float(score[0][0])

    def reset() -> None:
        state.pop("context", None)
        state["rnn_state"] = None

    infer.reset = reset
    return infer


def _make_asr_infer(model_root: Path):
    """Lazily constructs the real Faster-Whisper model, mirroring `run_confirmed()`'s
    inline `WhisperModel` construction exactly. Returns the real segment iterator;
    `FasterWhisperTranscriber` owns joining/stripping the transcript text."""
    state: dict[str, object] = {}

    def infer(samples: list[float]):
        if "model" not in state:
            from faster_whisper import WhisperModel

            state["model"] = create_faster_whisper_model(str(model_root / "faster-whisper-distil-large-v3"), WhisperModel)
        import numpy as np

        segments, _ = state["model"].transcribe(np.asarray(samples, dtype="float32"), language="en", task="transcribe", vad_filter=False)
        return segments

    return infer


def _make_translation_infer(model_root: Path) -> Callable[[str], str]:
    """Lazily constructs the real OPUS-MT tokenizer/model, mirroring `run_confirmed()`'s
    inline construction exactly."""
    state: dict[str, object] = {}

    def infer(text: str) -> str:
        if "model" not in state:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            state["tokenizer"] = AutoTokenizer.from_pretrained(str(model_root / "opus-mt-en-es"), local_files_only=True)
            state["model"] = AutoModelForSeq2SeqLM.from_pretrained(str(model_root / "opus-mt-en-es"), local_files_only=True).to("cuda")
        tokenizer, model = state["tokenizer"], state["model"]
        encoded = tokenizer(text, return_tensors="pt").to("cuda")
        generated = model.generate(**encoded)
        return tokenizer.decode(generated[0], skip_special_tokens=True).strip()

    return infer


def _make_tts_infer(model_root: Path, voice: str, controls: LiveControls | None = None):
    """Lazily constructs the real Kokoro engine, mirroring `run_confirmed()`'s inline
    construction exactly. Returns the real `(audio, source_rate)` pair;
    `KokoroSpanishSynthesizer` owns resampling/PCM16 packing."""
    state: dict[str, object] = {}

    def infer(text: str):
        if "engine" not in state:
            from kokoro_onnx import Kokoro

            state["engine"] = Kokoro(
                str(model_root / "kokoro-onnx-v1/kokoro-v1.0.onnx"), str(model_root / "kokoro-onnx-v1/voices-v1.0.bin"),
            )
        return state["engine"].create(text, voice=voice, lang="es", speed=controls.snapshot().speed if controls else 1.0)

    return infer


class SessionWorker(QObject):
    """Lives on the `QThread` (via `moveToThread`, never subclassed). Its slots run only
    when invoked through a queued cross-thread signal connection.  Its cancellation token is
    intentionally an exception: `threading.Event` is safe to set from the GUI thread, which
    lets Stop reach a worker blocked in its synchronous capture loop instead of waiting behind
    that loop in the worker's queued Qt event stream."""

    event_occurred = Signal(object)  # SessionEvent
    finished = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._cancel = threading.Event()

    def begin_session(self) -> None:
        """Prepare the token before queuing work, avoiding a Start/Stop race."""
        self._cancel.clear()

    @Slot(object)
    def run_session(self, runner: DubbingSessionRunner) -> None:
        try:
            runner.run(self._cancel)
        finally:
            self.finished.emit()

    def request_stop(self) -> None:
        """Thread-safe immediate cancellation; deliberately not a queued Qt slot."""
        self._cancel.set()


class _SignalObserver:
    """`SessionObserver` implementation that relays every `SessionEvent` across the
    thread boundary via `SessionWorker.event_occurred` -- Qt delivers this as a queued
    connection automatically because the emitting `QObject` (the worker) lives on a
    different thread than `MainWindow`'s connected slot."""

    def __init__(self, worker: SessionWorker) -> None:
        self._worker = worker

    def on_event(self, event: SessionEvent) -> None:
        self._worker.event_occurred.emit(event)


class _TitleBar(QWidget):
    """Custom-drawn replacement for the OS window decoration, which paints from the host
    desktop's own theme and ignores this app's QSS -- the only way to keep the title bar
    on the same Blucast-matched dark palette as the rest of the window (see design.md's
    "Custom title bar" amendment). Dragging and maximize/restore delegate to
    `QWindow.startSystemMove`/`isMaximized`/`showMaximized`/`showNormal` rather than
    manual position math, so behavior stays correct under both X11 and Wayland."""

    def __init__(self, window: "MainWindow") -> None:
        super().__init__(window)
        self._window = window
        self.setObjectName("titleBar")
        self.setFixedHeight(48)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 0, 10, 0)
        layout.setSpacing(8)

        self._title_label = QLabel()
        self._title_label.setObjectName("titleBarLabel")
        self._title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addSpacing(114)
        layout.addWidget(self._title_label, 1)

        self._minimize_button = QPushButton("–")
        self._maximize_button = QPushButton("□")
        self._close_button = QPushButton("✕")
        for button, object_name in (
            (self._minimize_button, "titleBarMinimize"),
            (self._maximize_button, "titleBarMaximize"),
            (self._close_button, "titleBarClose"),
        ):
            button.setObjectName(object_name)
            button.setFixedSize(30, 30)
            button.setFocusPolicy(Qt.NoFocus)
            layout.addWidget(button)

        self._minimize_button.clicked.connect(window.showMinimized)
        self._maximize_button.clicked.connect(self._toggle_maximize)
        self._close_button.clicked.connect(window.close)

    def set_title(self, text: str) -> None:
        self._title_label.setText(text)

    def _toggle_maximize(self) -> None:
        if self._window.isMaximized():
            self._window.showNormal()
        else:
            self._window.showMaximized()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 (Qt override signature)
        if event.button() == Qt.MouseButton.LeftButton:
            handle = self._window.windowHandle()
            if handle is not None:
                handle.startSystemMove()
                return
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._toggle_maximize()
            return
        super().mouseDoubleClickEvent(event)


class _CaptionOverlay(QWidget):
    """Floating, draggable, always-on-top Spanish caption display with its own Start/Stop
    controls. Deliberately delegates every action to `MainWindow`'s own handlers
    (`_on_start_clicked`/`_on_stop_clicked`) instead of duplicating session logic -- there is
    only ever one `DubbingSessionRunner`/`SessionStateMachine`, this is a second *view* onto
    it, never a second controller. A parentless top-level window avoids the main window's
    transient stacking/minimize relationship; dragging uses
    `QWindow.startSystemMove()`, the same technique `_TitleBar` already uses, for the same
    X11/Wayland-correctness reason.

    Pause is deliberately not offered here: `DubbingSessionRunner` has no mid-session pause
    capability today (only fresh-discovery Start and full-teardown-with-recovery Stop), and
    faking one at the UI layer (e.g. silently muting output while pretending to "pause")
    would misrepresent what actually happens. Start/Stop are real; a real pause would need a
    small addition to the pipeline layer, not just this widget.
    """

    def __init__(self, window: "MainWindow") -> None:
        super().__init__(
            None, Qt.WindowType.Window | Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint,
        )
        self.setObjectName("captionOverlay")
        self.resize(720, 240)
        self.setWindowTitle("OpenDubStream CC")
        self.setMinimumWidth(560)
        self.setStyleSheet(_STYLESHEET)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 10, 12, 14)
        layout.setSpacing(8)

        header = QHBoxLayout()
        self._language_pair_label = QLabel()
        self._language_pair_label.setObjectName("captionOverlayLanguages")
        header.addWidget(self._language_pair_label)
        header.addStretch(1)
        self._caption_page_label = QLabel()
        header.addWidget(self._caption_page_label)
        close_button = self._close_button = QPushButton("✕")
        close_button.setObjectName("captionOverlayClose")
        close_button.setFixedSize(22, 22)
        close_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        close_button.clicked.connect(lambda: window._set_caption_overlay_visible(False))
        header.addWidget(close_button)
        layout.addLayout(header)

        self._caption_label = QLabel()
        self._caption_label.setObjectName("captionOverlayText")
        self._caption_label.setWordWrap(True)
        self._caption_label.setTextFormat(Qt.TextFormat.PlainText)
        self._caption_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._caption_label.setMinimumHeight(64)
        self._caption_text = ""
        self._caption_pages = []
        self._caption_page = 0
        self._caption_timer = QTimer(self)
        self._caption_timer.timeout.connect(self._advance_caption_page)
        layout.addWidget(self._caption_label, 1)

        controls = QHBoxLayout()
        self._start_button = QPushButton()
        self._start_button.clicked.connect(window._on_start_clicked)
        self._stop_button = QPushButton()
        self._stop_button.clicked.connect(window._on_stop_clicked)
        self._stop_button.setEnabled(False)
        controls.addWidget(self._start_button)
        controls.addWidget(self._stop_button)
        self._voice_button = QPushButton()
        self._voice_button.setCheckable(True)
        self._voice_button.setChecked(True)
        self._voice_button.clicked.connect(window._set_voice_enabled)
        controls.addWidget(self._voice_button)
        self._settings_button = QPushButton()
        self._settings_button.clicked.connect(window._show_settings)
        controls.addStretch(1)
        controls.addWidget(self._settings_button)
        layout.addLayout(controls)
        controls = QHBoxLayout()
        self._volume_label = QLabel()
        controls.addWidget(self._volume_label)
        self._volume = QSlider(Qt.Orientation.Horizontal)
        self._volume.setRange(0, 100)
        self._volume.setValue(window._preferences.dub_volume)
        self._volume.setMinimumWidth(80)
        self._volume.valueChanged.connect(lambda value: window._dub_volume_spin.setValue(value))
        self._volume.valueChanged.connect(lambda value: self._volume_label.setText(
            f"{resolve(window._language, MessageKey.DUB_VOLUME_LABEL)} {value}%"))
        controls.addWidget(self._volume, 1)
        self._font_size = QSpinBox()
        self._font_size.setRange(16, 36)
        self._font_size.setValue(22)
        self._font_size.setSuffix(" px")
        self._size_label = QLabel()
        controls.addWidget(self._size_label)
        self._font_size.valueChanged.connect(lambda size: self._caption_label.setStyleSheet(f"font-size: {size}px;"))
        self._font_size.valueChanged.connect(lambda _: self._paginate_caption())
        controls.addWidget(self._font_size)
        layout.addLayout(controls)

        self.retranslate(window._language)

    def retranslate(self, language: Language) -> None:
        self._language_pair_label.setText(resolve(language, MessageKey.CAPTION_OVERLAY_LANGUAGE_PAIR))
        self._voice_button.setText(resolve(language, MessageKey.VOICE_TOGGLE))
        self._settings_button.setText(resolve(language, MessageKey.TAB_SETTINGS))
        self._voice_button.setAccessibleName(resolve(language, MessageKey.VOICE_TOGGLE))
        self._volume.setAccessibleName(resolve(language, MessageKey.DUB_VOLUME_LABEL))
        self._volume_label.setText(f"{resolve(language, MessageKey.DUB_VOLUME_LABEL)} {self._volume.value()}%")
        self._size_label.setText(resolve(language, MessageKey.CAPTION_TEXT_SIZE))
        self._volume.setToolTip(resolve(language, MessageKey.NEXT_PHRASE_HELP))
        self._voice_button.setToolTip(resolve(language, MessageKey.VOICE_HELP))
        self._font_size.setAccessibleName(resolve(language, MessageKey.CAPTION_TEXT_SIZE))
        self._close_button.setAccessibleName(resolve(language, MessageKey.CLOSE_CAPTIONS))
        self._caption_label.setText(resolve(language, MessageKey.CAPTION_PLACEHOLDER))
        self._start_button.setText(resolve(language, MessageKey.START_BUTTON_LABEL))
        self._stop_button.setText(resolve(language, MessageKey.STOP_BUTTON_LABEL))

    def set_caption(self, text: str) -> None:
        # Status/timing events repeat the last translation: never restart its reading time.
        if text == self._caption_text:
            return
        self._caption_text = text
        self._caption_label.setToolTip(text)
        self._caption_label.setAccessibleDescription(text)
        self._paginate_caption()

    def _paginate_caption(self) -> None:
        self._caption_timer.stop()
        self._caption_label.ensurePolished()
        font = self._caption_label.font()
        font.setPixelSize(self._font_size.value())
        self._caption_label.setMinimumHeight(QFontMetrics(font).lineSpacing() * 2 + 8)
        if not self._caption_text:
            return
        text = " ".join(self._caption_text.split())
        typesetter = QTextLayout(text, font)
        typesetter.beginLayout()
        lines = []
        while True:
            line = typesetter.createLine()
            if not line.isValid():
                break
            line.setLineWidth(max(80, self._caption_label.width() - 16))
            lines.append(text[line.textStart():line.textStart() + line.textLength()].strip())
        typesetter.endLayout()
        self._caption_pages = ["\n".join(lines[i:i + 2]) for i in range(0, len(lines), 2)] or [""]
        self._caption_page = 0
        self._show_caption_page()

    def _show_caption_page(self) -> None:
        page = self._caption_pages[self._caption_page]
        self._caption_label.setText(page)
        count = len(self._caption_pages)
        self._caption_page_label.setText(f"{self._caption_page + 1} / {count}" if count > 1 else "")
        if count > 1:
            self._caption_timer.start(max(2500, int(len(page) / 15 * 1000)))

    def _advance_caption_page(self) -> None:
        self._caption_page = (self._caption_page + 1) % len(self._caption_pages)
        self._show_caption_page()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._paginate_caption()

    def set_session_active(self, active: bool) -> None:
        self._start_button.setEnabled(not active)
        self._stop_button.setEnabled(active)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 (Qt override signature)
        if event.button() == Qt.MouseButton.LeftButton:
            handle = self.windowHandle()
            if handle is not None:
                handle.startSystemMove()
                return
        super().mousePressEvent(event)


class MainWindow(QMainWindow):
    _start_worker = Signal(object)  # DubbingSessionRunner

    def __init__(self) -> None:
        super().__init__()
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        # Compact default size; the content scrolls when details are expanded.
        self.setMinimumSize(600, 600)
        self.resize(680, 740)
        self._preferences_store = PreferencesStore(DEFAULT_PREFERENCES_PATH)
        self._preferences: Preferences = self._preferences_store.load()
        if self._preferences.audio_mode is AudioMode.TRANSLATION_ONLY:
            from dataclasses import replace
            self._preferences = replace(self._preferences, audio_mode=AudioMode.DUB)
        self._language: Language = self._preferences.language
        self._state_machine = SessionStateMachine()
        self._session_active = False
        self._sink_labels: dict[str, str] = {}
        self._live_controls = LiveControls(LiveSettings(
            mix=AudioMixSettings(self._preferences.audio_mode, self._preferences.original_volume, self._preferences.dub_volume),
            speed=self._preferences.speech_speed,
        ))

        self._pactl = SafePactlRunner()
        self._journal = RecoveryJournal(DEFAULT_JOURNAL_PATH)
        self._router = PipeWirePulseRouter(self._pactl, self._journal, PipeWireStreamDiscovery(self._pactl))

        # Adapters are held once and reused across sessions -- each wraps a lazily-loaded
        # real model (first use only, cached inside the injected `infer` closure) so a
        # second Start in the same process never reloads the model stack. The adapters
        # themselves (windowing/joining/resampling) are the tested classes from
        # `infrastructure/models/`; only real model construction happens here.
        assets = _local_assets(DEFAULT_MODEL_ROOT)
        self._vad = SileroVadDetector(assets, _make_vad_infer(DEFAULT_MODEL_ROOT))
        self._asr = FasterWhisperTranscriber(assets, _make_asr_infer(DEFAULT_MODEL_ROOT))
        self._translator = OpusMtEnglishSpanishTranslator(assets, _make_translation_infer(DEFAULT_MODEL_ROOT))
        self._tts = KokoroSpanishSynthesizer(assets, _make_tts_infer(DEFAULT_MODEL_ROOT, self._preferences.tts_voice, self._live_controls))

        self._thread = QThread()
        self._worker = SessionWorker()
        self._worker.moveToThread(self._thread)
        self._start_worker.connect(self._worker.run_session)
        self._worker.event_occurred.connect(self._on_session_event)
        self._worker.finished.connect(self._on_session_finished)
        self._thread.start()

        self._translatable_labels: list[tuple[QLabel, MessageKey]] = []
        self._translatable_buttons: list[tuple[QPushButton, MessageKey]] = []
        self._caption_overlay = _CaptionOverlay(self)
        self._build_ui()
        self.setWindowIcon(QIcon(str(_ICON_PATH)))
        self._build_tray_icon()
        self._retranslate_ui()
        self._preflight_timer = QTimer(self)
        self._preflight_timer.setInterval(_PREFLIGHT_REFRESH_MILLISECONDS)
        self._preflight_timer.timeout.connect(self._refresh_preflight_status)
        self._refresh_preflight_status()
        self._preflight_timer.start()

    # ---- UI construction -------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        central.setObjectName("centralSurface")
        layout = QVBoxLayout(central)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        brand = QHBoxLayout()
        logo = QLabel()
        logo.setPixmap(QIcon(str(_ICON_PATH)).pixmap(48, 48))
        brand.addWidget(logo)
        identity = QVBoxLayout()
        title = QLabel("OpenDubStream")
        title.setObjectName("brandTitle")
        identity.addWidget(title)
        self._brand_subtitle = QLabel()
        self._translatable_labels.append((self._brand_subtitle, MessageKey.WORKSPACE_SUBTITLE))
        identity.addWidget(self._brand_subtitle)
        brand.addLayout(identity)
        brand.addStretch()
        layout.addLayout(brand)
        self._workspace_tabs = QTabWidget()
        self._dubbing_tab = QWidget()
        self._audio_tab = QWidget()
        self._general_tab = QWidget()
        self._workspace_tabs.addTab(self._dubbing_tab, "")
        self._workspace_tabs.addTab(self._audio_tab, "")
        self._workspace_tabs.addTab(self._general_tab, "")
        layout.addWidget(self._workspace_tabs)

        self._build_dubbing_tab()
        self._build_audio_tab()
        self._build_settings_tab()
        self._build_general_tab()
        self._start_button.setDefault(True)  # `:enabled:default` in _STYLESHEET is the blue accent
        self.setStyleSheet(_STYLESHEET)

        # The frameless window has no OS title bar, so a custom `_TitleBar` sits above the
        # existing tabbed `central` content in a plain wrapper. No edge-drag resize is
        # implemented on purpose: the user only resizes this window via the title bar's
        # maximize button (or its double-click), never by dragging an edge/corner.
        frame = QWidget(self)
        frame_layout = QVBoxLayout(frame)
        frame_layout.setContentsMargins(0, 0, 0, 0)
        frame_layout.setSpacing(0)
        self._title_bar = _TitleBar(self)
        frame_layout.addWidget(self._title_bar)
        frame_layout.addWidget(central)
        self.setCentralWidget(frame)

    def _build_dubbing_tab(self) -> None:
        outer = QVBoxLayout(self._dubbing_tab)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)
        scroll.setWidget(content)
        outer.addWidget(scroll)

        controls = QHBoxLayout()
        self._start_button = QPushButton()
        self._start_button.clicked.connect(self._on_start_clicked)
        self._stop_button = QPushButton()
        self._stop_button.clicked.connect(self._on_stop_clicked)
        self._stop_button.setEnabled(False)
        self._translatable_buttons.append((self._start_button, MessageKey.START_BUTTON_LABEL))
        self._translatable_buttons.append((self._stop_button, MessageKey.STOP_BUTTON_LABEL))

        self._caption_toggle_button = QPushButton()
        self._caption_toggle_button.setCheckable(True)
        self._caption_toggle_button.clicked.connect(self._set_caption_overlay_visible)
        self._translatable_buttons.append((self._caption_toggle_button, MessageKey.CAPTION_OVERLAY_TOGGLE_LABEL))

        controls.addWidget(self._start_button)
        controls.addWidget(self._stop_button)
        self._voice_toggle = QPushButton()
        self._voice_toggle.setCheckable(True)
        self._voice_toggle.setChecked(True)
        self._voice_toggle.clicked.connect(self._set_voice_enabled)
        self._translatable_buttons.append((self._voice_toggle, MessageKey.VOICE_TOGGLE))
        controls.addStretch(1)
        controls.addWidget(self._voice_toggle)
        controls.addWidget(self._caption_toggle_button)
        layout.addLayout(controls)
        self._mode_row = QHBoxLayout()
        layout.addLayout(self._mode_row)

        self._stage_label = QLabel()
        self._stage_label.setFixedHeight(26)
        layout.addWidget(self._stage_label)

        self._reason_label = QLabel()
        self._reason_label.setWordWrap(True)
        layout.addWidget(self._reason_label)

        transcript_group = QGroupBox()
        transcript_layout = QVBoxLayout(transcript_group)
        self._transcript_label = QLabel()
        self._translatable_labels.append((self._transcript_label, MessageKey.TRANSCRIPT_LABEL))
        self._transcript_view = QTextEdit()
        self._transcript_view.setReadOnly(True)
        self._transcript_view.setObjectName("sourceText")
        self._transcript_view.setFixedHeight(64)
        self._translation_label = QLabel()
        self._translatable_labels.append((self._translation_label, MessageKey.TRANSLATION_LABEL))
        self._translation_view = QTextEdit()
        self._translation_view.setReadOnly(True)
        self._translation_view.setObjectName("translationText")
        self._translation_view.setFixedHeight(106)
        transcript_layout.addWidget(self._transcript_label)
        transcript_layout.addWidget(self._transcript_view)
        transcript_layout.addWidget(self._translation_label)
        transcript_layout.addWidget(self._translation_view)
        layout.addWidget(transcript_group)

        self._diagnostics_toggle = QPushButton()
        self._diagnostics_toggle.setCheckable(True)
        self._translatable_buttons.append((self._diagnostics_toggle, MessageKey.TECHNICAL_DETAILS))
        layout.addWidget(self._diagnostics_toggle)
        self._diagnostics_group = QGroupBox()
        self._diagnostics_group.hide()
        self._diagnostics_toggle.toggled.connect(self._diagnostics_group.setVisible)
        diagnostics_layout = QVBoxLayout(self._diagnostics_group)
        self._diagnostics_rows: dict[MessageKey, tuple[QLabel, QLabel]] = {}
        for key in (
            MessageKey.DIAGNOSTICS_TIMINGS_LABEL,
            MessageKey.DIAGNOSTICS_EXECUTION_MODE_LABEL,
            MessageKey.DIAGNOSTICS_BACKLOG_LABEL,
            MessageKey.DIAGNOSTICS_OVERLOAD_DROPPED_LABEL,
            MessageKey.DIAGNOSTICS_STALE_DROPPED_LABEL,
            MessageKey.DIAGNOSTICS_LAST_ROUTING_ERROR_LABEL,
            MessageKey.DIAGNOSTICS_RECOVERY_RESTORED_LABEL,
            MessageKey.DIAGNOSTICS_RECOVERY_DETAIL_LABEL,
        ):
            row = QHBoxLayout()
            name_label = QLabel()
            self._translatable_labels.append((name_label, key))
            value_label = QLabel("-")
            value_label.setWordWrap(True)
            self._diagnostics_rows[key] = (name_label, value_label)
            row.addWidget(name_label)
            row.addWidget(value_label)
            row.addStretch(1)
            diagnostics_layout.addLayout(row)
        layout.addWidget(self._diagnostics_group)
        layout.addStretch(1)

    def _build_audio_tab(self) -> None:
        layout = QVBoxLayout(self._audio_tab)
        source_group = QGroupBox()
        source_layout = QVBoxLayout(source_group)
        self._chrome_stream_label = QLabel()
        self._chrome_stream_combo = QComboBox()
        source_layout.addWidget(self._chrome_stream_label)
        source_layout.addWidget(self._chrome_stream_combo)
        self._output_label = QLabel()
        self._output_combo = QComboBox()
        source_layout.addWidget(self._output_label)
        source_layout.addWidget(self._output_combo)
        layout.addWidget(source_group)

        mix_group = QGroupBox()
        mix_layout = QVBoxLayout(mix_group)
        self._audio_mode_label = QLabel()
        self._audio_mode_combo = QComboBox()
        for mode in (AudioMode.CAPTIONS, AudioMode.DUB, AudioMode.BOTH):
            self._audio_mode_combo.addItem("", mode)
        self._audio_mode_combo.setCurrentIndex(max(0, self._audio_mode_combo.findData(self._preferences.audio_mode)))
        self._audio_mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self._original_volume_label = QLabel()
        self._original_volume_spin = QSpinBox()
        self._original_volume_spin.setRange(0, 100)
        self._original_volume_spin.setSuffix(" %")
        self._original_volume_spin.setValue(self._preferences.original_volume)
        self._original_volume_spin.valueChanged.connect(self._save_editable_preferences)
        self._dub_volume_label = QLabel()
        self._dub_volume_spin = QSpinBox()
        self._dub_volume_spin.setRange(0, 100)
        self._dub_volume_spin.setSuffix(" %")
        self._dub_volume_spin.setValue(self._preferences.dub_volume)
        self._dub_volume_spin.valueChanged.connect(self._save_editable_preferences)
        self._mode_row.addWidget(self._audio_mode_label)
        self._mode_row.addWidget(self._audio_mode_combo, 1)
        self._video_button = QPushButton()
        self._translatable_buttons.append((self._video_button, MessageKey.BACK_TO_VIDEO))
        self._video_button.clicked.connect(self._back_to_video)
        self._mode_row.addWidget(self._video_button)
        for label, control in ((self._original_volume_label, self._original_volume_spin), (self._dub_volume_label, self._dub_volume_spin)):
            row = QHBoxLayout()
            row.addWidget(label)
            row.addStretch(1)
            row.addWidget(control)
            mix_layout.addLayout(row)
        help_label = QLabel()
        help_label.setWordWrap(True)
        self._translatable_labels.append((help_label, MessageKey.MIX_HELP))
        mix_layout.addWidget(help_label)
        layout.addWidget(mix_group)

    def _build_settings_tab(self) -> None:
        layout = self._audio_tab.layout()
        engines_group = QGroupBox()
        engines_layout = QVBoxLayout(engines_group)
        self._asr_value = QLabel()
        self._translation_engine_value = QLabel()
        self._tts_value = QLabel()
        engines_layout.addWidget(self._asr_value)
        engines_layout.addWidget(self._translation_engine_value)
        engines_layout.addWidget(self._tts_value)
        self._diagnostics_group.layout().addWidget(engines_group)
        voice_group = QGroupBox()
        voice_layout = QVBoxLayout(voice_group)
        self._voice_label = QLabel()
        self._voice_combo = QComboBox()
        self._voice_combo.addItem("Dora", "ef_dora")
        self._voice_combo.currentTextChanged.connect(self._save_editable_preferences)
        self._speed_label = QLabel()
        self._speed_spin = QDoubleSpinBox()
        self._speed_spin.setRange(0.5, 2.0)
        self._speed_spin.setSingleStep(0.05)
        self._speed_spin.setDecimals(2)
        self._speed_spin.setSuffix(" ×")
        self._speed_spin.setValue(self._preferences.speech_speed)
        self._speed_spin.valueChanged.connect(self._save_editable_preferences)
        for label, control in ((self._voice_label, self._voice_combo), (self._speed_label, self._speed_spin)):
            row = QHBoxLayout()
            row.addWidget(label)
            row.addStretch(1)
            row.addWidget(control)
            voice_layout.addLayout(row)
        layout.addWidget(voice_group)
        layout.addStretch(1)

    @staticmethod
    def _separator() -> QFrame:
        """Matches the reference UX/UI's own `_separator()` helper verbatim."""
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("background: #1f2a1f; max-height: 1px; border: none;")
        return line

    def _build_general_tab(self) -> None:
        """App identity (icon/name/description) plus the interface-language selector,
        moved here out of the Dubbing tab's action row to keep that row focused on the one
        thing it does -- matching the reference UX/UI's own General tab layout and its
        "shown first since it identifies what the app is" ordering within the tab."""
        layout = QVBoxLayout(self._general_tab)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(14)

        icon_label = QLabel()
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if _ICON_PATH.exists():
            pixmap = QPixmap(64, 64)
            pixmap.fill(Qt.GlobalColor.transparent)
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            QSvgRenderer(str(_ICON_PATH)).render(painter)
            painter.end()
            icon_label.setPixmap(pixmap)
        layout.addWidget(icon_label)

        title_label = QLabel(resolve(self._language, MessageKey.WINDOW_TITLE))
        title_label.setObjectName("generalAppTitle")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._general_title_label = title_label
        layout.addWidget(title_label)

        layout.addWidget(self._separator())

        self._app_description_label = QLabel()
        self._app_description_label.setWordWrap(True)
        self._app_description_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._app_description_label.setObjectName("generalAppDescription")
        layout.addWidget(self._app_description_label)

        layout.addWidget(self._separator())

        self._language_label = QLabel()
        self._translatable_labels.append((self._language_label, MessageKey.LANGUAGE_SELECTOR_LABEL))
        layout.addWidget(self._language_label)
        self._language_combo = QComboBox()
        self._language_combo.addItem("", Language.ENGLISH)
        self._language_combo.addItem("", Language.SPANISH)
        self._language_combo.setCurrentIndex(0 if self._language is Language.ENGLISH else 1)
        self._language_combo.currentIndexChanged.connect(self._on_language_changed)
        layout.addWidget(self._language_combo)

        layout.addStretch(1)

    # ---- Floating caption overlay / system tray --------------------------------------

    def _build_tray_icon(self) -> None:
        """`QSystemTrayIcon.isSystemTrayAvailable()` is false under `QT_QPA_PLATFORM=offscreen`
        (every automated test) and on any desktop with no tray host running -- both are real,
        expected conditions, not errors, so the tray is simply omitted rather than raising."""
        self._tray_icon: QSystemTrayIcon | None = None
        self._tray_caption_action = None
        self._tray_show_action = None
        self._tray_quit_action = None
        self._tray_start_action = None
        self._tray_stop_action = None
        self._tray_voice_action = None
        self._tray_mode_actions = []
        if not QSystemTrayIcon.isSystemTrayAvailable():
            return

        self._tray_icon = QSystemTrayIcon(QIcon(str(_ICON_PATH)), self)
        menu = QMenu()
        self._tray_menu = menu
        menu.setStyleSheet(_STYLESHEET)
        self._tray_start_action = menu.addAction("")
        self._tray_start_action.triggered.connect(self._on_start_clicked)
        self._tray_stop_action = menu.addAction("")
        self._tray_stop_action.triggered.connect(self._on_stop_clicked)
        menu.addSeparator()
        mode_group = QActionGroup(menu)
        mode_group.setExclusive(True)
        for index in range(3):
            action = menu.addAction("")
            action.setCheckable(True)
            mode_group.addAction(action)
            action.triggered.connect(lambda _checked, i=index: self._audio_mode_combo.setCurrentIndex(i))
            self._tray_mode_actions.append(action)
        menu.addSeparator()
        self._tray_voice_action = menu.addAction("")
        self._tray_voice_action.setCheckable(True)
        self._tray_voice_action.toggled.connect(self._set_voice_enabled)
        self._tray_caption_action = menu.addAction("")
        self._tray_caption_action.setCheckable(True)
        self._tray_caption_action.toggled.connect(self._set_caption_overlay_visible)
        menu.addSeparator()
        self._tray_show_action = menu.addAction("")
        self._tray_show_action.triggered.connect(self._show_settings)
        self._tray_quit_action = menu.addAction("")
        self._tray_quit_action.triggered.connect(QApplication.instance().quit)
        self._tray_icon.setContextMenu(menu)
        self._tray_icon.activated.connect(self._on_tray_activated)
        self._tray_icon.setVisible(True)

    def _sync_tray(self) -> None:
        self._audio_mode_combo.setEnabled(not self._session_active)
        self._audio_mode_combo.setToolTip(
            ("Detén la sesión para cambiar de modo." if self._language is Language.SPANISH else
             "Stop the session to change mode.") if self._session_active else ""
        )
        voice_available = self._preferences.audio_mode is not AudioMode.CAPTIONS
        self._voice_toggle.setEnabled(voice_available)
        self._caption_overlay._voice_button.setEnabled(voice_available)
        self._caption_overlay._volume.setEnabled(voice_available)
        if self._tray_start_action is None:
            return
        self._tray_voice_action.setEnabled(voice_available)
        for action, key in ((self._tray_start_action, MessageKey.START_BUTTON_LABEL),
                            (self._tray_stop_action, MessageKey.STOP_BUTTON_LABEL),
                            (self._tray_voice_action, MessageKey.VOICE_TOGGLE),
                            (self._tray_show_action, MessageKey.TAB_SETTINGS)):
            action.setText(resolve(self._language, key))
        self._tray_start_action.setEnabled(not self._session_active)
        self._tray_stop_action.setEnabled(self._session_active)
        with QSignalBlocker(self._tray_voice_action):
            self._tray_voice_action.setChecked(self._live_controls.snapshot().voice_enabled)
        for index, action in enumerate(self._tray_mode_actions):
            action.setEnabled(not self._session_active)
            action.setText(self._audio_mode_combo.itemText(index))
            action.setChecked(index == self._audio_mode_combo.currentIndex())

    def _on_tray_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            self._show_main_window()

    def _show_main_window(self) -> None:
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def _set_caption_overlay_visible(self, visible: bool) -> None:
        """Single toggle shared by the Dubbing tab button and the tray menu action -- each
        stays in sync with the other via `QSignalBlocker` so toggling one never re-triggers
        this method through the other's own signal."""
        self._caption_overlay.setVisible(visible)
        if visible:
            self._caption_overlay.show()
            self._caption_overlay.raise_()
        with QSignalBlocker(self._caption_toggle_button):
            self._caption_toggle_button.setChecked(visible)
        if self._tray_caption_action is not None:
            with QSignalBlocker(self._tray_caption_action):
                self._tray_caption_action.setChecked(visible)

    # ---- Localization -------------------------------------------------------------

    def _retranslate_ui(self) -> None:
        self.setWindowTitle(resolve(self._language, MessageKey.WINDOW_TITLE))
        self._title_bar.set_title(resolve(self._language, MessageKey.WINDOW_TITLE))
        for label, key in self._translatable_labels:
            label.setText(resolve(self._language, key))
        for button, key in self._translatable_buttons:
            button.setText(resolve(self._language, key))
        self._language_combo.setItemText(0, resolve(self._language, MessageKey.LANGUAGE_ENGLISH_LABEL))
        self._language_combo.setItemText(1, resolve(self._language, MessageKey.LANGUAGE_SPANISH_LABEL))
        self._workspace_tabs.setTabText(0, resolve(self._language, MessageKey.TAB_DUBBING))
        self._workspace_tabs.setTabText(1, resolve(self._language, MessageKey.TAB_SETTINGS))
        self._workspace_tabs.setTabText(2, resolve(self._language, MessageKey.TAB_GENERAL))
        self._general_title_label.setText(resolve(self._language, MessageKey.WINDOW_TITLE))
        self._app_description_label.setText(resolve(self._language, MessageKey.APP_DESCRIPTION))
        self._start_button.setAccessibleName(f"{resolve(self._language, MessageKey.START_BUTTON_LABEL)} {resolve(self._language, MessageKey.TAB_DUBBING).lower()}")
        self._stop_button.setAccessibleName(f"{resolve(self._language, MessageKey.STOP_BUTTON_LABEL)} {resolve(self._language, MessageKey.TAB_DUBBING).lower()}")
        self._chrome_stream_label.setText(resolve(self._language, MessageKey.AUDIO_SOURCE_LABEL))
        self._output_label.setText(resolve(self._language, MessageKey.AUDIO_OUTPUT_LABEL))
        self._audio_mode_label.setText(resolve(self._language, MessageKey.AUDIO_MODE_LABEL))
        for index, key in enumerate((MessageKey.MODE_CAPTIONS, MessageKey.AUDIO_MODE_DUB, MessageKey.MODE_BOTH)):
            self._audio_mode_combo.setItemText(index, resolve(self._language, key))
        self._original_volume_label.setText(resolve(self._language, MessageKey.ORIGINAL_VOLUME_LABEL))
        self._dub_volume_label.setText(resolve(self._language, MessageKey.DUB_VOLUME_LABEL))
        self._asr_value.setText(resolve(self._language, MessageKey.SETTINGS_ASR_LABEL))
        self._translation_engine_value.setText(resolve(self._language, MessageKey.SETTINGS_TRANSLATION_LABEL))
        self._tts_value.setText(resolve(self._language, MessageKey.SETTINGS_TTS_LABEL))
        self._voice_label.setText(resolve(self._language, MessageKey.SETTINGS_VOICE_LABEL))
        self._speed_label.setText(resolve(self._language, MessageKey.SETTINGS_SPEED_LABEL))
        self._transcript_view.setPlaceholderText(resolve(self._language, MessageKey.SOURCE_PLACEHOLDER))
        self._translation_view.setPlaceholderText(resolve(self._language, MessageKey.CAPTION_PLACEHOLDER))
        self._speed_spin.setToolTip(resolve(self._language, MessageKey.SPEECH_SPEED_HELP))
        self._dub_volume_spin.setToolTip(resolve(self._language, MessageKey.NEXT_PHRASE_HELP))
        self._original_volume_spin.setToolTip(resolve(self._language, MessageKey.NEXT_PHRASE_HELP))
        self._voice_toggle.setToolTip(resolve(self._language, MessageKey.VOICE_HELP))
        self._caption_overlay.retranslate(self._language)
        if self._tray_caption_action is not None:
            self._tray_caption_action.setText(resolve(self._language, MessageKey.CAPTION_OVERLAY_TOGGLE_LABEL))
            self._tray_show_action.setText(resolve(self._language, MessageKey.TRAY_SHOW_WINDOW_LABEL))
            self._tray_quit_action.setText(resolve(self._language, MessageKey.TRAY_QUIT_LABEL))
        self._sync_tray()
        self._refresh_audio_choices()
        self._render_status(self._state_machine.status)

    def _on_language_changed(self, index: int) -> None:
        # Qt transports ``StrEnum`` item data as its underlying ``str``. Rebuild the
        # domain enum at this boundary so both localization and persistence retain
        # their explicit Language contract.
        self._language = Language(self._language_combo.itemData(index))
        self._save_editable_preferences()
        self._retranslate_ui()

    def _save_editable_preferences(self, *_ignored: object) -> None:
        """Persist the only values the user may edit; session data never crosses this boundary."""
        if not hasattr(self, "_audio_mode_combo"):
            return
        self._preferences = Preferences(
            language=self._language,
            audio_mode=AudioMode(self._audio_mode_combo.currentData()),
            original_volume=self._original_volume_spin.value(),
            dub_volume=self._dub_volume_spin.value(),
            tts_voice=self._voice_combo.currentData(),
            speech_speed=self._speed_spin.value(),
        )
        self._preferences_store.save(self._preferences)
        self._live_controls.update(LiveSettings(
            mix=AudioMixSettings(self._preferences.audio_mode, self._preferences.original_volume, self._preferences.dub_volume),
            speed=self._preferences.speech_speed,
            voice_enabled=self._live_controls.snapshot().voice_enabled,
        ))
        with QSignalBlocker(self._caption_overlay._volume):
            self._caption_overlay._volume.setValue(self._preferences.dub_volume)
        self._caption_overlay._volume_label.setText(f"{resolve(self._language, MessageKey.DUB_VOLUME_LABEL)} {self._preferences.dub_volume}%")

    def _back_to_video(self) -> None:
        if self._caption_overlay.isVisible():
            self.hide()
        else:
            self.showMinimized()

    def _show_settings(self) -> None:
        self._workspace_tabs.setCurrentIndex(1)
        self._show_main_window()

    def _on_mode_changed(self, *_ignored) -> None:
        self._save_editable_preferences()
        mode = self._preferences.audio_mode
        self._set_caption_overlay_visible(mode in (AudioMode.CAPTIONS, AudioMode.BOTH))
        self._sync_tray()

    def _set_voice_enabled(self, enabled: bool) -> None:
        settings = self._live_controls.snapshot()
        self._live_controls.update(LiveSettings(settings.mix, settings.speed, enabled))
        for button in (self._voice_toggle, self._caption_overlay._voice_button):
            with QSignalBlocker(button):
                button.setChecked(enabled)
        self._sync_tray()

    # ---- Eligibility gate -----------------------------------------------------------

    def _current_eligibility(self):
        streams = PipeWireStreamDiscovery(self._pactl)()
        raw_sink_inputs = json.loads(self._pactl.run(("pactl", "-f", "json", "list", "sink-inputs")))
        return evaluate_eligibility(
            cuda_available=cuda_available(),
            active_chrome_streams=count_active_chrome_streams(streams, raw_sink_inputs),
        )

    def _available_chrome_streams(self) -> list[StreamRef]:
        """Return current Chrome candidates only; no selection is cached as routing authority."""
        return [
            stream for stream in PipeWireStreamDiscovery(self._pactl)()
            if "chrome" in stream.application_name.lower()
        ]

    def _available_physical_sinks(self) -> list[SinkRef]:
        """Discover literal hardware outputs from pactl JSON without mutating any route."""
        raw_sinks = json.loads(self._pactl.run(("pactl", "-f", "json", "list", "sinks")))
        sinks: list[SinkRef] = []
        self._sink_labels = {}
        for entry in raw_sinks:
            name = entry.get("name")
            properties = entry.get("properties", {})
            if not isinstance(name, str) or not isinstance(properties, dict):
                continue
            device_class = properties.get("device.class")
            if device_class != "sound" or name.endswith(".monitor") or name.startswith("opendubstream."):
                continue
            serial = properties.get("object.serial")
            if isinstance(serial, (str, int)):
                sinks.append(SinkRef(name, True, str(serial)))
                def usable(value):
                    return isinstance(value, str) and value.strip() and value.strip().lower() not in {"(null)", "null", "none"}
                label = next((value.strip() for value in (
                    entry.get("description"), properties.get("device.description"), properties.get("node.nick"), name,
                ) if usable(value)), name)
                ports = entry.get("ports", [])
                active = entry.get("active_port")
                port = next((p.get("description") for p in ports if isinstance(p, dict) and p.get("name") == active), None)
                if usable(port) and port.strip() not in label:
                    label += " — " + port.strip()
                self._sink_labels[name] = label
        return sinks

    def _default_sink_name(self) -> str | None:
        """The name PipeWire-Pulse itself would route ordinary audio to right now -- used
        only to pick a sensible *default* output selection, never to authorize routing."""
        try:
            return self._pactl.run(("pactl", "get-default-sink")).strip() or None
        except Exception:
            return None

    @staticmethod
    def _same_stream(left: StreamRef, right: StreamRef) -> bool:
        return left.identifier == right.identifier and left.serial == right.serial

    @staticmethod
    def _same_sink(left: SinkRef, right: SinkRef) -> bool:
        return left.name == right.name and left.serial == right.serial and left.is_physical and right.is_physical

    def _refresh_audio_choices(self) -> None:
        """Refresh display choices only. Running sessions retain their already validated lease."""
        if self._session_active or self._output_combo.view().isVisible() or self._chrome_stream_combo.view().isVisible():
            return
        try:
            streams = self._available_chrome_streams()
            sinks = self._available_physical_sinks()
        except Exception:
            return

        previous_stream = self._chrome_stream_combo.currentData()
        previous_sink = self._output_combo.currentData()
        with QSignalBlocker(self._chrome_stream_combo), QSignalBlocker(self._output_combo):
            self._chrome_stream_combo.clear()
            self._chrome_stream_combo.addItem(resolve(self._language, MessageKey.AUTOMATIC_STREAM_SELECTION), None)
            for stream in streams:
                self._chrome_stream_combo.addItem(f"{stream.application_name} — {stream.media_name}", stream)
            if isinstance(previous_stream, StreamRef):
                for index in range(1, self._chrome_stream_combo.count()):
                    candidate = self._chrome_stream_combo.itemData(index)
                    if isinstance(candidate, StreamRef) and self._same_stream(candidate, previous_stream):
                        self._chrome_stream_combo.setCurrentIndex(index)
                        break

            self._output_combo.clear()
            self._output_combo.addItem(resolve(self._language, MessageKey.SELECT_PHYSICAL_OUTPUT), None)
            for sink in sinks:
                self._output_combo.addItem(self._sink_labels.get(sink.name, sink.name), sink)
            restored = False
            if isinstance(previous_sink, SinkRef):
                for index in range(1, self._output_combo.count()):
                    candidate = self._output_combo.itemData(index)
                    if isinstance(candidate, SinkRef) and self._same_sink(candidate, previous_sink):
                        self._output_combo.setCurrentIndex(index)
                        restored = True
                        break
            if not restored and sinks:
                # No prior selection, or the prior selection is no longer present (e.g. the
                # host's real default sink changed, or a test replaces the candidate list) --
                # prefer the host's actual default sink (wherever the rest of the desktop,
                # e.g. headphones, is already playing) over an arbitrary list order; only
                # fall back to the first candidate if the default sink can't be determined or
                # isn't among the discovered physical outputs.
                default_name = self._default_sink_name()
                default_index = next(
                    (index for index, sink in enumerate(sinks, start=1) if sink.name == default_name), None,
                )
                self._output_combo.setCurrentIndex(default_index if default_index is not None else 1)

    def _refresh_preflight_status(self) -> None:
        """Keep the Start action available while exposing the *current* preflight state.

        Preflight is advisory UI state, not authorization: `_on_start_clicked` repeats the
        exact query and refuses before it creates routing or model work.  This means a user
        can click Start to see a fresh failure reason even when Chrome has not yet emitted a
        stream, while an active session still owns Start until its worker finishes.
        """
        if self._session_active:
            self._start_button.setEnabled(False)
            return
        self._refresh_audio_choices()
        if self._state_machine.status.stage is SessionStage.FAILED:
            # Preflight is advisory.  A periodic success must not turn a real session
            # failure into a misleading Idle state or hide its recovery diagnostics.
            self._start_button.setEnabled(True)
            return
        try:
            eligibility = self._current_eligibility()
        except Exception as error:
            status = self._state_machine.apply(
                SessionEvent(
                    stage=SessionStage.FAILED,
                    message=MessageKey.ERROR_UNEXPECTED,
                    params={"detail": str(error)},
                ),
            )
        else:
            if eligibility.startable:
                status = self._state_machine.apply(SessionEvent(stage=SessionStage.IDLE))
            else:
                status = self._state_machine.apply(SessionEvent(stage=SessionStage.IDLE, message=eligibility.reason))
        self._start_button.setEnabled(True)
        self._render_status(status)

    def _refresh_start_enabled(self) -> None:
        """Compatibility name for existing lifecycle callers."""
        self._refresh_preflight_status()

    # ---- Start / Stop ---------------------------------------------------------------

    def _on_start_clicked(self) -> None:
        try:
            eligibility = self._current_eligibility()
        except Exception as error:
            status = self._state_machine.apply(
                SessionEvent(stage=SessionStage.FAILED, message=MessageKey.ERROR_UNEXPECTED, params={"detail": str(error)}),
            )
            self._render_status(status)
            return
        if eligibility.reason is MessageKey.INELIGIBLE_GPU_UNAVAILABLE:
            # Only a missing GPU actually blocks Start: it's the one precondition
            # `DubbingSessionRunner.run()` itself re-checks and cannot recover from. Never
            # render it as `FAILED` (the alarming, persistent red banner reserved for a
            # genuine attempted-session error) -- show the same neutral `IDLE`-with-reason
            # state the periodic preflight refresh already displays while waiting.
            status = self._state_machine.apply(SessionEvent(stage=SessionStage.IDLE, message=eligibility.reason))
            self._render_status(status)
            self._start_button.setEnabled(True)
            return
        # No Chrome stream yet is *not* a reason to refuse Start: the runner itself arms
        # and waits (`SessionStage.ARMED`, see `_wait_for_unique_stream`) until exactly one
        # appears, which is exactly the "press Start, then go play something in Chrome"
        # flow the user asked for -- refusing here would make that flow unreachable.

        try:
            selected_stream = self._chrome_stream_combo.currentData()
            selected_output = self._output_combo.currentData()
            current_sinks = self._available_physical_sinks()
            if not isinstance(selected_output, SinkRef) or not any(
                self._same_sink(selected_output, candidate) for candidate in current_sinks
            ):
                raise ValueError(resolve(self._language, MessageKey.ERROR_STALE_SELECTION))
            if selected_stream is not None:
                if not isinstance(selected_stream, StreamRef) or not any(
                    self._same_stream(selected_stream, candidate) for candidate in self._available_chrome_streams()
                ):
                    raise ValueError(resolve(self._language, MessageKey.ERROR_STALE_SELECTION))
        except Exception as error:
            status = self._state_machine.apply(
                SessionEvent(stage=SessionStage.FAILED, message=MessageKey.ERROR_UNEXPECTED, params={"detail": str(error)}),
            )
            self._render_status(status)
            return

        def chrome_streams() -> list[StreamRef]:
            current = PipeWireStreamDiscovery(self._pactl)()
            if selected_stream is None:
                return current
            return [stream for stream in current if self._same_stream(stream, selected_stream)]

        def chrome_sink_inputs() -> list[dict[str, object]]:
            return json.loads(self._pactl.run(("pactl", "-f", "json", "list", "sink-inputs")))

        pipeline = LocalDubbingPipeline(self._vad, self._asr, self._translator, self._tts, clock=time.monotonic)
        scheduler = FixedCapacityScheduler(capacity=_SCHEDULER_CAPACITY, max_age_seconds=_SCHEDULER_MAX_AGE_SECONDS)
        capture = PhraseCapture(PipeWireMonitorCapture(OwnedMonitorDiscovery(self._pactl), journal=self._journal))
        player = PwPlayPcmPlayer()
        observer = _SignalObserver(self._worker)

        runner = DubbingSessionRunner(
            router=self._router,
            chrome_streams=chrome_streams,
            chrome_sink_inputs=chrome_sink_inputs,
            physical_sink=selected_output,
            capture=capture,
            pipeline=pipeline,
            scheduler=scheduler,
            player=player,
            cuda_probe=cuda_available,
            observer=observer,
            clock=time.monotonic,
            original_mix_controller=self._router,
            controls=self._live_controls,
            mix_settings=AudioMixSettings(
                mode=self._preferences.audio_mode,
                original_volume=self._preferences.original_volume,
                dub_volume=self._preferences.dub_volume,
            ),
        )

        self._session_active = True
        self._start_button.setEnabled(False)
        self._stop_button.setEnabled(True)
        self._caption_overlay.set_session_active(True)
        self._set_caption_overlay_visible(self._preferences.audio_mode in (AudioMode.CAPTIONS, AudioMode.BOTH))
        self._sync_tray()
        self._launch_runner(runner)

    def _launch_runner(self, runner: DubbingSessionRunner) -> None:
        """Single Qt boundary kept separate so offscreen tests never start a worker."""
        self._worker.begin_session()
        self._start_worker.emit(runner)

    def _on_stop_clicked(self) -> None:
        self._stop_button.setEnabled(False)
        self._worker.request_stop()
        self._on_session_event(SessionEvent(stage=SessionStage.STOPPING))

    def _on_session_finished(self) -> None:
        self._session_active = False
        self._stop_button.setEnabled(False)
        self._caption_overlay.set_session_active(False)
        self._sync_tray()
        self._refresh_preflight_status()

    # ---- Rendering --------------------------------------------------------------

    def _on_session_event(self, event: SessionEvent) -> None:
        status = self._state_machine.apply(event)
        self._render_status(status)

    def _render_status(self, status: SessionStatus) -> None:
        stage_key = _STAGE_MESSAGE_KEYS[status.stage]
        if status.stage is SessionStage.PROCESSING and status.diagnostics:
            stage_key = {
                "transcription": MessageKey.STAGE_TRANSCRIBING,
                "translation": MessageKey.STAGE_TRANSLATING,
                "voice": MessageKey.STAGE_VOICE,
            }.get(status.diagnostics.active_operation, stage_key)
        self._stage_label.setText(resolve(self._language, stage_key))
        if status.stage is SessionStage.RECOVERY_FAILED:
            self._stage_label.setStyleSheet(_STAGE_LABEL_RECOVERY_FAILED_STYLE)
            self._reason_label.setStyleSheet(_REASON_LABEL_RECOVERY_FAILED_STYLE)
        else:
            self._stage_label.setStyleSheet(_STAGE_LABEL_DEFAULT_STYLE)
            self._reason_label.setStyleSheet(_REASON_LABEL_DEFAULT_STYLE)

        if status.stage is SessionStage.RECOVERY_FAILED:
            reason_text = resolve(self._language, MessageKey.RECOVERY_FAILED_MESSAGE)
        elif status.message is not None:
            reason_text = resolve(self._language, status.message, status.params)
        else:
            reason_text = ""
        if status.stage is SessionStage.FAILED and not reason_text:
            reason_text = resolve(self._language, MessageKey.SESSION_ERROR_HELP)
        if not reason_text and status.stage is SessionStage.IDLE:
            reason_text = resolve(self._language, MessageKey.START_HELP)
        if status.diagnostics and (status.diagnostics.overload_dropped or status.diagnostics.stale_dropped) and status.stage not in (SessionStage.FAILED, SessionStage.RECOVERY_FAILED):
            reason_text = resolve(self._language, MessageKey.CATCHING_UP)
        self._reason_label.setText(reason_text)

        if status.transcript is not None:
            self._transcript_view.setPlainText(status.transcript)
        if status.translation is not None:
            self._translation_view.setPlainText(status.translation)
            self._caption_overlay.set_caption(status.translation)

        diagnostics = status.diagnostics
        for key, (_, value_label) in self._diagnostics_rows.items():
            if diagnostics is None:
                value_label.setText("-")
                continue
            if key is MessageKey.DIAGNOSTICS_TIMINGS_LABEL:
                value_label.setText(" · ".join(f"{name}: {duration:.0f}" for name, duration in diagnostics.timings_ms.items()) or "-")
            elif key is MessageKey.DIAGNOSTICS_EXECUTION_MODE_LABEL:
                value_label.setText(diagnostics.execution_mode or "-")
            elif key is MessageKey.DIAGNOSTICS_BACKLOG_LABEL:
                value_label.setText(str(diagnostics.backlog))
            elif key is MessageKey.DIAGNOSTICS_OVERLOAD_DROPPED_LABEL:
                value_label.setText(str(diagnostics.overload_dropped))
            elif key is MessageKey.DIAGNOSTICS_STALE_DROPPED_LABEL:
                value_label.setText(str(diagnostics.stale_dropped))
            elif key is MessageKey.DIAGNOSTICS_LAST_ROUTING_ERROR_LABEL:
                value_label.setText(diagnostics.last_routing_error or "-")
            elif key is MessageKey.DIAGNOSTICS_RECOVERY_RESTORED_LABEL:
                value_label.setText("-" if diagnostics.recovery_restored is None else str(diagnostics.recovery_restored))
            elif key is MessageKey.DIAGNOSTICS_RECOVERY_DETAIL_LABEL:
                value_label.setText(diagnostics.recovery_detail or "-")

    # ---- Lifecycle ----------------------------------------------------------------

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt override signature)
        self._preflight_timer.stop()
        self._caption_overlay.close()
        if self._tray_icon is not None:
            self._tray_icon.hide()
        self._worker.request_stop()
        self._thread.quit()
        self._thread.wait(2000)
        super().closeEvent(event)


def _ensure_correct_launch_environment() -> None:
    """Two independent gaps only a fresh process with a corrected environment can fix --
    checked together so there is at most one `os.execve` re-exec, not two.

    1. **cuDNN/cuBLAS on the dynamic loader's search path.** CTranslate2 (faster-whisper's
       backend) dynamically loads `libcudnn_ops.so.9` and friends by SONAME *at first real
       inference*, not at model construction -- a construction-only smoke test can report
       success right before the first actual transcription call aborts the whole process
       (`Cannot load symbol cudnnCreateTensorDescriptor`, not a catchable Python exception).
       Unlike PyTorch, it has no built-in knowledge of the `nvidia-cudnn-cu12`/
       `nvidia-cublas-cu12` pip wheels' own bundled `lib/` directories, so it only finds them
       via `LD_LIBRARY_PATH`.
    2. **Running under XWayland, not native Wayland.** GNOME's Wayland compositor (Mutter)
       implements no protocol for an ordinary top-level window to request "always on top" --
       `_CaptionOverlay`'s `Qt.WindowType.WindowStaysOnTopHint` is silently ignored there
       (confirmed: under native Wayland the window is invisible to `wmctrl`/`xdotool`
       entirely). Under XWayland (Qt's `xcb` platform plugin) the same window becomes a real
       X11 client, and Mutter's X11 compatibility layer does honor the standard
       `_NET_WM_STATE_ABOVE` EWMH hint (confirmed: the window becomes visible to `wmctrl` as
       soon as `QT_QPA_PLATFORM=xcb` is set). Only forced when a Wayland session is actually
       running and `xcb` isn't already selected; an X11-native session is untouched.

    Both gaps share the same fix shape: an env var is read once by native code (the dynamic
    loader; Qt's platform-plugin selection) at process start, so writing `os.environ` from
    inside the already-running interpreter does nothing for either -- only a fresh process
    with the corrected environment works. A no-op (no re-exec) once both are already
    satisfied (this function's own prior re-exec, or an explicit launcher/session)."""
    env = dict(os.environ)
    changed = False

    try:
        import nvidia.cublas
        import nvidia.cudnn
    except ImportError:
        pass
    else:
        cudnn_lib = str(Path(nvidia.cudnn.__file__).resolve().parent / "lib")
        cublas_lib = str(Path(nvidia.cublas.__file__).resolve().parent / "lib")
        current = env.get("LD_LIBRARY_PATH", "").split(":")
        if cudnn_lib not in current or cublas_lib not in current:
            env["LD_LIBRARY_PATH"] = ":".join(filter(None, [cudnn_lib, cublas_lib, *current]))
            changed = True

    if env.get("WAYLAND_DISPLAY") and env.get("QT_QPA_PLATFORM") != "xcb":
        env["QT_QPA_PLATFORM"] = "xcb"
        changed = True

    if changed:
        os.execve(sys.executable, [sys.executable, *sys.argv], env)


def main() -> int:
    _ensure_correct_launch_environment()
    # Mirrors Blucast's own entry point (`app/control_panel.py`) verbatim, including the
    # `argv[0]`/`setDesktopFileName` pair: run as a plain script, Qt/X11 would otherwise
    # derive WM_CLASS from the script path, which GNOME's dash/taskbar can't resolve to an
    # icon -- forcing it to the same name a `.desktop` file's `StartupWMClass` declares
    # lets the taskbar icon actually appear once such a `.desktop` entry is installed.
    sys.argv[0] = "opendubstream"
    app = QApplication(sys.argv)
    app.setDesktopFileName("opendubstream")
    # The platform's native widget style (e.g. a GTK/Adwaita integration on GNOME) paints
    # some widgets -- notably QComboBox popups -- from the desktop theme instead of this
    # module's QSS, breaking the Blucast-matched dark theme. Fusion is a QSS-only style with
    # no native-theme delegation, so every widget this app styles renders identically
    # regardless of the host desktop's theme.
    app.setStyle("Fusion")
    app.setPalette(_dark_palette())
    app.setWindowIcon(QIcon(str(_ICON_PATH)))
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
