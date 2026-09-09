"""Direct coverage for the real subprocess boundary behind PCM playback.

test_physical_playback.py only exercises run_pcm_player indirectly through a
FakePlayer, so its actual timeout/exit-code/no-shell behavior was untested.
"""

from __future__ import annotations

import subprocess

import pytest

from opendubstream.infrastructure.audio import process as process_module
from opendubstream.infrastructure.audio.playback import PcmPlaybackError, PwPlayPcmPlayer
from opendubstream.infrastructure.audio.process import PcmProcessError, run_pcm_player


class FakeCompleted:
    def __init__(self, returncode: int, stderr: bytes = b"") -> None:
        self.returncode = returncode
        self.stderr = stderr


def test_run_pcm_player_rejects_non_positive_deadline() -> None:
    with pytest.raises(PcmProcessError, match="positive"):
        run_pcm_player(("pw-play", "-"), b"\x00\x00", 0.0)
    with pytest.raises(PcmProcessError, match="positive"):
        run_pcm_player(("pw-play", "-"), b"\x00\x00", -1.0)


def test_run_pcm_player_runs_without_a_shell_and_forwards_pcm_as_stdin(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_run(argv: tuple[str, ...], **kwargs: object) -> FakeCompleted:
        calls.append({"argv": argv, **kwargs})
        return FakeCompleted(0)

    monkeypatch.setattr(process_module.subprocess, "run", fake_run)

    run_pcm_player(("pw-play", "--target", "alsa_output.speakers", "-"), b"\x01\x02", 1.5)

    assert calls == [
        {
            "argv": ("pw-play", "--target", "alsa_output.speakers", "-"),
            "shell": False,
            "check": False,
            "input": b"\x01\x02",
            "capture_output": True,
            "timeout": 1.5,
        }
    ]


def test_run_pcm_player_raises_on_nonzero_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(process_module.subprocess, "run", lambda *args, **kwargs: FakeCompleted(1))

    with pytest.raises(PcmProcessError, match="exited 1"):
        run_pcm_player(("pw-play", "-"), b"\x00\x00", 1.0)


def test_run_pcm_player_includes_real_stderr_in_the_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A silently swallowed stderr cost real debugging time diagnosing a live pw-play failure."""
    stderr = b'sndfile: failed to open audio file "-": Format not recognised.\n'
    monkeypatch.setattr(process_module.subprocess, "run", lambda *args, **kwargs: FakeCompleted(1, stderr=stderr))

    with pytest.raises(PcmProcessError, match="Format not recognised"):
        run_pcm_player(("pw-play", "-"), b"\x00\x00", 1.0)


def test_run_pcm_player_raises_timeout_error_on_deadline_exceeded(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*args: object, **kwargs: object) -> FakeCompleted:
        raise subprocess.TimeoutExpired(cmd="pw-play", timeout=kwargs["timeout"])

    monkeypatch.setattr(process_module.subprocess, "run", fake_run)

    with pytest.raises(TimeoutError):
        run_pcm_player(("pw-play", "-"), b"\x00\x00", 0.05)


def test_pw_play_pcm_player_wraps_a_real_timeout_with_a_non_empty_message(monkeypatch: pytest.MonkeyPatch) -> None:
    """run_pcm_player raises a bare TimeoutError() (empty str()); the wrapper must not lose the reason."""
    from opendubstream.domain.contracts import SinkRef
    from opendubstream.infrastructure.audio import playback as playback_module

    monkeypatch.setattr(playback_module, "run_pcm_player", lambda *_args, **_kwargs: (_ for _ in ()).throw(TimeoutError()))

    with pytest.raises(PcmPlaybackError, match="timed out"):
        PwPlayPcmPlayer().play(SinkRef("alsa_output.speakers", True, "sink-serial"), b"\x00\x00", 1.0)


def test_cancelled_playback_never_launches_a_process(monkeypatch):
    monkeypatch.setattr(process_module.subprocess, 'Popen', lambda *a, **k: pytest.fail('unexpected launch'))
    run_pcm_player(('pw-play', '-'), b'pcm', 1, should_stop=lambda: True)


def test_cancellation_kills_and_reaps_in_flight_player(monkeypatch):
    stopped = False
    class Child:
        returncode = None
        killed = False
        calls = 0
        def communicate(self, input=None, timeout=None):
            nonlocal stopped
            self.calls += 1
            if self.calls == 1:
                stopped = True
                raise subprocess.TimeoutExpired('pw-play', timeout)
            return b'', b''
        def kill(self):
            self.killed = True
            self.returncode = -9
    child = Child()
    monkeypatch.setattr(process_module.subprocess, 'Popen', lambda *a, **k: child)
    run_pcm_player(('pw-play', '-'), b'pcm', 1, should_stop=lambda: stopped)
    assert child.killed and child.calls == 2


def test_stopped_player_cannot_launch_late_playback(monkeypatch):
    from opendubstream.domain.contracts import SinkRef
    from opendubstream.infrastructure.audio import playback
    seen = []
    monkeypatch.setattr(playback, 'run_pcm_player', lambda *a, **kw: seen.append(kw['should_stop']()))
    player = PwPlayPcmPlayer()
    player.stop()
    player.play(SinkRef('speakers', True, '1'), b'pcm', 1)
    assert seen == [True]
