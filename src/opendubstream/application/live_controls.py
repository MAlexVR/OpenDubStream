"""One thread-safe settings snapshot shared by the UI and the active session."""
from dataclasses import dataclass
from threading import Lock

from opendubstream.domain.contracts import AudioMixSettings


@dataclass(frozen=True)
class LiveSettings:
    mix: AudioMixSettings = AudioMixSettings()
    speed: float = 1.0
    voice_enabled: bool = True


class LiveControls:
    def __init__(self, settings: LiveSettings = LiveSettings()) -> None:
        self._settings = settings
        self._lock = Lock()

    def snapshot(self) -> LiveSettings:
        with self._lock:
            return self._settings

    def update(self, settings: LiveSettings) -> None:
        if not 0.5 <= settings.speed <= 2.0:
            raise ValueError("speech speed must be between 0.5 and 2")
        with self._lock:
            self._settings = settings
