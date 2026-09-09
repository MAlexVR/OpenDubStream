"""Narrow, literal-argv process boundary for PipeWire-Pulse commands."""

from __future__ import annotations

import re
import subprocess
import time
from collections.abc import Callable, Sequence
from threading import Event


class ProcessExecutionError(RuntimeError):
    pass


class ProcessTimeout(ProcessExecutionError):
    pass


class ProcessCancelled(ProcessExecutionError):
    pass


class ProcessOutputError(ProcessExecutionError):
    pass


class PcmProcessError(ProcessExecutionError):
    pass


_SAFE_TOKEN = re.compile(r"^[A-Za-z0-9_.:-]+$")
_ALLOWED_FORMS = {
    ("pactl", "list", "sinks", "short"),
    ("pactl", "list", "sink-inputs", "short"),
    ("pactl", "-f", "json", "list", "sink-inputs"),
    ("pactl", "-f", "json", "list", "sinks"),
    ("pactl", "-f", "json", "list", "sources"),
    ("pactl", "-f", "json", "list", "modules"),
    ("pactl", "get-default-sink"),
}


class SafePactlRunner:
    """Runs only a small fixed set of pactl forms needed by the routing adapter.

    Callers supply complete argv tuples. The runner never invokes a shell and rejects
    anything that cannot be represented as a known literal PipeWire-Pulse operation.
    """

    def __init__(
        self,
        launch: Callable[[tuple[str, ...], float], object] | None = None,
        timeout_seconds: float = 2.0,
    ) -> None:
        self._launch = launch or self._subprocess_launch
        self._timeout_seconds = timeout_seconds

    def run(self, argv: tuple[str, ...], *, cancel: Event | None = None) -> str:
        if cancel is not None and cancel.is_set():
            raise ProcessCancelled("process launch cancelled")
        self._validate(argv)
        try:
            result = self._launch(argv, self._timeout_seconds)
        except TimeoutError as error:
            raise ProcessTimeout("pactl command timed out") from error
        if cancel is not None and cancel.is_set():
            raise ProcessCancelled("process completed after cancellation")
        return self._normalize_result(result)

    def _validate(self, argv: tuple[str, ...]) -> None:
        if argv in _ALLOWED_FORMS:
            return
        if len(argv) == 3 and argv[:2] == ("pactl", "unload-module") and argv[2].isdigit():
            return
        if (
            len(argv) == 4
            and argv[:2] == ("pactl", "move-sink-input")
            and argv[2].isdigit()
            and _SAFE_TOKEN.fullmatch(argv[3])
        ):
            return
        if (
            len(argv) == 4
            and argv[:2] == ("pactl", "set-sink-input-mute")
            and argv[2].isdigit()
            and argv[3] in {"0", "1"}
        ):
            return
        if (
            len(argv) == 4
            and argv[:2] == ("pactl", "set-sink-input-volume")
            and argv[2].isdigit()
            and argv[3].endswith("%")
            and argv[3][:-1].isdigit()
            and 0 <= int(argv[3][:-1]) <= 100
        ):
            return
        if (
            len(argv) == 4
            and argv[:3] == ("pactl", "load-module", "module-null-sink")
            and argv[3].startswith("sink_name=")
            and _SAFE_TOKEN.fullmatch(argv[3].split("=", 1)[1])
        ):
            return
        if (
            len(argv) == 5
            and argv[:3] == ("pactl", "load-module", "module-loopback")
            and argv[3].startswith("source=")
            and argv[3].endswith(".monitor")
            and _SAFE_TOKEN.fullmatch(argv[3][len("source="):-len(".monitor")])
            and argv[4].startswith("sink=")
            and _SAFE_TOKEN.fullmatch(argv[4].split("=", 1)[1])
        ):
            return
        raise ProcessExecutionError("pactl argv is not an allowlisted literal routing command")

    @staticmethod
    def _normalize_result(result: object) -> str:
        if result is None:
            raise ProcessExecutionError("pactl child exited without a result")
        if isinstance(result, str):
            return result
        if not isinstance(result, tuple) or len(result) != 2 or not isinstance(result[0], int) or not isinstance(result[1], str):
            raise ProcessOutputError("pactl produced malformed process output")
        exit_code, output = result
        if exit_code != 0:
            detail = output.strip()
            raise ProcessExecutionError(f"pactl exited {exit_code}{': ' + detail if detail else ''}")
        return output

    @staticmethod
    def _subprocess_launch(argv: tuple[str, ...], timeout: float) -> tuple[int, str]:
        try:
            completed = subprocess.run(
                argv,
                shell=False,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as error:
            raise TimeoutError() from error
        if completed.returncode == 0:
            return completed.returncode, completed.stdout
        # A cleanup decision may distinguish a confirmed already-absent owned resource
        # from an actionable PipeWire outage. Preserve stderr only on failure so module
        # ID-producing success output remains the exact numeric stdout contract.
        detail = completed.stderr.strip() or completed.stdout.strip()
        return completed.returncode, detail


def run_pcm_player(
    argv: tuple[str, ...], pcm: bytes, deadline: float, *, on_started: Callable[[], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> None:
    """Write PCM to a pre-validated literal player argv without a shell."""
    if deadline <= 0:
        raise PcmProcessError("PCM playback deadline must be positive")
    if should_stop is not None and should_stop():
        return
    if on_started is None and should_stop is None:
        try:
            completed = subprocess.run(argv, shell=False, check=False, input=pcm, capture_output=True, timeout=deadline)
        except subprocess.TimeoutExpired as error:
            raise TimeoutError() from error
        if completed.returncode != 0:
            stderr = completed.stderr.decode("utf-8", errors="replace").strip() if completed.stderr else ""
            detail = f": {stderr}" if stderr else ""
            raise PcmProcessError(f"PCM player exited {completed.returncode}{detail}")
        return

    child: subprocess.Popen[bytes] | None = None
    try:
        child = subprocess.Popen(
            argv, shell=False, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        if on_started is not None:
            on_started()
        if should_stop is None:
            _, stderr_bytes = child.communicate(input=pcm, timeout=deadline)
        else:
            until = time.monotonic() + deadline
            payload = pcm
            while True:
                if should_stop():
                    child.kill()
                    child.communicate()
                    return
                remaining = until - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(argv, deadline)
                try:
                    _, stderr_bytes = child.communicate(input=payload, timeout=min(0.05, remaining))
                    break
                except subprocess.TimeoutExpired:
                    # communicate retains unsent input across timeouts. Passing it a
                    # second time would duplicate audio (and raises on CPython).
                    payload = None
    except subprocess.TimeoutExpired as error:
        if child is not None:
            child.kill()
            child.communicate()
        raise TimeoutError() from error
    except Exception:
        if child is not None:
            child.kill()
            child.communicate()
        raise
    if child.returncode != 0:
        stderr = stderr_bytes.decode("utf-8", errors="replace").strip() if stderr_bytes else ""
        detail = f": {stderr}" if stderr else ""
        raise PcmProcessError(f"PCM player exited {child.returncode}{detail}")
