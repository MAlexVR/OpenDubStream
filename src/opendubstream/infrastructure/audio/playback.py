"""Validated physical PCM playback and deterministic selected-monitor feedback checks."""

from __future__ import annotations

import json
import math
import re
import threading
from dataclasses import dataclass
from typing import Callable, Protocol

from opendubstream.domain.contracts import PhysicalSinkRequired, RouteLease, SinkRef
from opendubstream.infrastructure.audio.process import PcmProcessError, run_pcm_player

_SAFE_TARGET = re.compile(r"^[A-Za-z0-9_.:-]+$")
_SAMPLE_RATE = 16_000
_TAG_SAMPLES = 800
_SILENCE_SAMPLES = 800
_MAX_LAG = 3_200
_BASELINE_LIMIT = 0.10
_WINDOW_LIMIT = 0.10
_DELTA_LIMIT = 0.05


class PcmPlaybackError(RuntimeError):
    """Physical PCM playback could not complete safely."""


class PcmPlayer(Protocol):
    def play(self, target: SinkRef, pcm: bytes, deadline: float) -> None: ...

    def stop(self) -> None: ...


def _only(entries: list[dict[str, object]], target: SinkRef) -> dict[str, object]:
    if len(entries) != 1:
        raise PhysicalSinkRequired(f"physical target identity matched {len(entries)} entries; exactly one is required")
    entry = entries[0]
    properties = entry.get("properties")
    serial = properties.get("object.serial") if isinstance(properties, dict) else None
    if not isinstance(serial, str) or not serial:
        raise PhysicalSinkRequired("physical target has no stable serial")
    if target.serial and target.serial != serial:
        raise PhysicalSinkRequired("physical target identity changed")
    if not isinstance(properties, dict) or not isinstance(properties.get("device.bus"), str):
        raise PhysicalSinkRequired("target is not a physical playback sink")
    return entry


def resolve_physical_target(sinks_json: str, desired: SinkRef, lease: RouteLease) -> SinkRef:
    """Re-resolve a physical playback sink before every write; never infer identity by name."""
    if not desired.is_physical:
        raise PhysicalSinkRequired("generated audio requires a physical playback sink")
    if not desired.serial:
        raise PhysicalSinkRequired("physical target requires a stable serial")
    if desired.name == lease.virtual_sink or desired.name.startswith("opendubstream."):
        raise PhysicalSinkRequired("owned virtual sink is not a physical playback target")
    if desired.name.endswith(".monitor"):
        raise PhysicalSinkRequired("monitor source is not a physical playback target")
    try:
        entries = json.loads(sinks_json)
    except json.JSONDecodeError as error:
        raise PhysicalSinkRequired(f"malformed sink inventory: {error}") from error
    if not isinstance(entries, list) or not all(isinstance(entry, dict) for entry in entries):
        raise PhysicalSinkRequired("malformed sink inventory")
    _only([entry for entry in entries if entry.get("name") == desired.name], desired)
    return desired


def build_pw_play_argv(target_name: str) -> tuple[str, ...]:
    """Create the only permitted pw-play invocation for 16-kHz mono signed PCM."""
    if not _SAFE_TARGET.fullmatch(target_name):
        raise PcmPlaybackError("playback target must be a literal PipeWire name")
    return ("pw-play", "--target", target_name, "--format", "s16", "--rate", "16000", "--channels", "1", "--raw", "-")


class PwPlayPcmPlayer:
    """Physical player implementation. The OS process is always shell-free and bounded."""

    def __init__(self, on_started: Callable[[], None] | None = None) -> None:
        self._on_started = on_started
        self._stopped = threading.Event()
        self._cancel_check: Callable[[], bool] = lambda: False

    def set_cancel_check(self, check: Callable[[], bool]) -> None:
        self._cancel_check = check

    def play(self, target: SinkRef, pcm: bytes, deadline: float) -> None:
        try:
            run_pcm_player(
                build_pw_play_argv(target.name), pcm, deadline, on_started=self._on_started,
                should_stop=lambda: self._stopped.is_set() or self._cancel_check(),
            )
        except TimeoutError as error:
            raise PcmPlaybackError("PCM player process timed out") from error
        except PcmProcessError as error:
            raise PcmPlaybackError(str(error)) from error

    def stop(self) -> None:
        self._stopped.set()


class PhysicalPlaybackTarget:
    def validate_target(self, sink: SinkRef) -> SinkRef:
        if not sink.is_physical:
            raise PhysicalSinkRequired("generated audio requires a physical playback sink")
        return sink


class PhysicalPcmPlayback:
    """Combines fresh physical-target validation with a cancellable player boundary."""

    def __init__(self, inventory: Callable[[], str], player: PcmPlayer) -> None:
        self._inventory = inventory
        self._player = player

    def play(self, pcm: bytes, target: SinkRef, lease: RouteLease, *, deadline: float) -> SinkRef:
        resolved = resolve_physical_target(self._inventory(), target, lease)
        try:
            self._player.play(resolved, pcm + playback_tag(), deadline)
        except TimeoutError as error:
            self._player.stop()
            raise PcmPlaybackError("PCM player timed out") from error
        except Exception:
            self._player.stop()
            raise
        return resolved


def playback_tag() -> bytes:
    """800 samples at 997 Hz / 0.20, followed by 800 silent samples at 16 kHz."""
    tone = (
        int(0.20 * 32767 * math.sin(2 * math.pi * 997 * index / _SAMPLE_RATE)).to_bytes(2, "little", signed=True)
        for index in range(_TAG_SAMPLES)
    )
    return b"".join(tone) + b"\x00\x00" * _SILENCE_SAMPLES


def _samples(pcm: bytes) -> list[float]:
    if len(pcm) % 2:
        raise PcmPlaybackError("PCM must contain complete s16le samples")
    return [int.from_bytes(pcm[index : index + 2], "little", signed=True) / 32768.0 for index in range(0, len(pcm), 2)]


def _correlation_window_scores(tag: bytes, observed: bytes, search_range: int) -> list[tuple[int, float]]:
    """Shared normalized-correlation computation used by both the score-only and the
    score+position variants. `search_range` is the inclusive upper bound of candidate
    start positions searched, beginning at position 0. Returns an empty list on any
    empty/invalid input instead of raising -- callers decide how to fail closed."""
    if search_range < 0:
        return []
    reference = _samples(tag)[:_TAG_SAMPLES]
    captured = _samples(observed)
    reference_energy = math.sqrt(sum(value * value for value in reference))
    if not reference_energy or not captured:
        return []
    upper = min(len(captured) - len(reference), search_range)
    if upper < 0:
        return []
    scores: list[tuple[int, float]] = []
    for start in range(0, upper + 1):
        candidate = captured[start : start + len(reference)]
        energy = math.sqrt(sum(value * value for value in candidate))
        if not energy:
            continue
        score = abs(sum(left * right for left, right in zip(reference, candidate)) / (reference_energy * energy))
        scores.append((start, score))
    return scores


def _maximum_normalized_correlation(tag: bytes, observed: bytes) -> float:
    scores = _correlation_window_scores(tag, observed, _MAX_LAG)
    return max((score for _, score in scores), default=0.0)


def best_tag_correlation_score(tag: bytes, observed: bytes, search_range: int) -> float:
    """Diagnostic companion to `locate_best_tag_position`: the best score the search
    found, independent of `confidence_floor`. Lets a caller that got `None` back tell
    "nothing above any reasonable floor" apart from "a real match, floor too strict"."""
    scores = _correlation_window_scores(tag, observed, search_range)
    return max((score for _, score in scores), default=0.0)


def locate_best_tag_position(tag: bytes, observed: bytes, search_range: int, *, confidence_floor: float) -> int | None:
    """Search a wide, configurable range of candidate start positions (unlike the narrow
    `_MAX_LAG` window `_maximum_normalized_correlation` uses for feedback-window detection)
    and return the position of the best-scoring window -- not just its score. Used to
    measure real-world transport/startup latency, where realistic PipeWire/pw-play
    routing delay far exceeds `_MAX_LAG`. Fails closed to `None` below `confidence_floor`,
    on empty/short input, or on any invalid input -- never guess."""
    scores = _correlation_window_scores(tag, observed, search_range)
    if not scores:
        return None
    best_position, best_score = max(scores, key=lambda item: item[1])
    if best_score < confidence_floor:
        return None
    return best_position


@dataclass(frozen=True)
class FeedbackObservation:
    baseline_score: float
    window_score: float
    feedback_absent: bool

    @classmethod
    def measure(cls, tag: bytes, baseline: bytes, window: bytes) -> "FeedbackObservation":
        baseline_score = _maximum_normalized_correlation(tag, baseline)
        window_score = _maximum_normalized_correlation(tag, window)
        absent = (
            baseline_score <= _BASELINE_LIMIT
            and window_score <= _WINDOW_LIMIT
            and window_score - baseline_score < _DELTA_LIMIT
        )
        return cls(baseline_score, window_score, absent)

_PROTOCOL_NAMESPACE = b"opendubstream.feedback-isolation/v1"
_IMMUTABLE_FAILED_BASELINE_DIGEST = "db82ba31a54ab1a70dc0de1e1a3256c2c9993217ea05eeba4c6a35ae1bd6bc72"


@dataclass(frozen=True)
class ProtocolConfig:
    """Frozen constants for the v1 selected-monitor feedback protocol."""

    namespace: bytes = _PROTOCOL_NAMESPACE
    code_count: int = 16
    tag_samples: int = 800
    sample_rate: int = 16_000
    tag_insertion_samples: int = 3_200
    analysis_samples: int = 64_000
    max_lag: int = 3_200
    permutation_count: int = 999
    alpha: float = 0.01


def _pcm_s16le(values: list[int]) -> bytes:
    return b"".join(value.to_bytes(2, "little", signed=True) for value in values)


def deterministic_tag(config: ProtocolConfig, code_index: int) -> bytes:
    """Return one balanced, SHA-256-ordered v1 tag without runtime entropy."""
    import hashlib

    if not 0 <= code_index < config.code_count:
        raise PcmPlaybackError("code index is outside the frozen protocol family")
    positions = list(range(config.tag_samples))
    positions.sort(
        key=lambda position: hashlib.sha256(
            config.namespace + code_index.to_bytes(1, "big") + position.to_bytes(2, "big")
        ).digest()
    )
    values = [0] * config.tag_samples
    for position in positions[: config.tag_samples // 2]:
        values[position] = -6553
    for position in positions[config.tag_samples // 2 :]:
        values[position] = 6553
    return _pcm_s16le(values)


def emit_tagged_probe_pcm(synthesized_pcm: bytes, tag: bytes, config: ProtocolConfig) -> bytes:
    """Place the tag at the declared sample-clock position, never at an inferred duration."""
    if len(synthesized_pcm) % 2 or len(tag) != config.tag_samples * 2:
        raise PcmPlaybackError("probe PCM must be complete s16le and use a complete frozen tag")
    return b"\x00\x00" * config.tag_insertion_samples + tag + synthesized_pcm


def crop_tagged_monitor_window(
    captured_pcm: bytes,
    reader_sample_count_at_pw_play: int,
    transport_offset_samples: int,
    config: ProtocolConfig,
) -> bytes | None:
    """Crop only from the declared byte-clock anchor; missing samples remain inconclusive."""
    if len(captured_pcm) % 2 or reader_sample_count_at_pw_play < 0 or not isinstance(transport_offset_samples, int):
        return None
    expected = reader_sample_count_at_pw_play + config.tag_insertion_samples + transport_offset_samples
    start = expected - config.tag_insertion_samples
    stop = start + config.analysis_samples
    if start < 0 or stop > len(captured_pcm) // 2:
        return None
    return captured_pcm[start * 2 : stop * 2]


@dataclass(frozen=True)
class CalibrationScore:
    """The active-code statistic and predeclared global-max null distribution."""

    active_max: float
    permutation_global_max: tuple[float, ...]

    def corrected_p_value(self, config: ProtocolConfig) -> float | None:
        if (
            len(self.permutation_global_max) != config.permutation_count
            or not math.isfinite(self.active_max)
            or any(not math.isfinite(value) for value in self.permutation_global_max)
        ):
            return None
        return (1 + sum(value >= self.active_max for value in self.permutation_global_max)) / (config.permutation_count + 1)


@dataclass(frozen=True)
class CalibrationDecision:
    value: str
    null_p_values: tuple[float, ...]
    positive_p_value: float | None
    selected_p_value: float | None
    baseline_digest: str


def evaluate_calibration(
    *,
    null_scores: tuple[CalibrationScore, ...],
    positive_score: CalibrationScore | None,
    selected_score: CalibrationScore | None,
    baseline_digest: str,
    config: ProtocolConfig,
) -> CalibrationDecision:
    """Apply the v1 no-feedback rule; every missing or conflicting condition is no-go."""
    null_p_values = tuple(score.corrected_p_value(config) for score in null_scores)
    positive_p_value = positive_score.corrected_p_value(config) if positive_score else None
    selected_p_value = selected_score.corrected_p_value(config) if selected_score else None
    complete = (
        baseline_digest == _IMMUTABLE_FAILED_BASELINE_DIGEST
        and len(null_scores) == 3
        and len(null_p_values) == 3
        and all(value is not None for value in null_p_values)
        and positive_p_value is not None
        and selected_p_value is not None
    )
    if not complete:
        return CalibrationDecision("inconclusive", tuple(value for value in null_p_values if value is not None), positive_p_value, selected_p_value, baseline_digest)
    resolved_nulls = tuple(value for value in null_p_values if value is not None)
    no_feedback = (
        all(value > config.alpha for value in resolved_nulls)
        and positive_p_value <= config.alpha
        and selected_p_value > config.alpha
    )
    feedback_present = selected_p_value <= config.alpha and positive_p_value <= config.alpha and all(
        value > config.alpha for value in resolved_nulls
    )
    return CalibrationDecision(
        "no-feedback" if no_feedback else "feedback-present" if feedback_present else "inconclusive",
        resolved_nulls,
        positive_p_value,
        selected_p_value,
        baseline_digest,
    )


def permutation_shifts(candidate_revision: str, dataset_label: str, config: ProtocolConfig) -> tuple[int, ...]:
    """Derive the frozen circular-shift family used by max-T calibration.

    Confirmed live (2026-09-04): an unconstrained shift can circularly re-place the *same*
    localized signal back within reach of the observed statistic's own search window (the
    positions `[0, 2*max_lag+tag_samples)` a shift=0 search can read) -- inflating the
    permutation null distribution for any real signal, not just the deliberately-injected
    positive-control tag, and making `alpha` effectively unreachable regardless of signal
    strength. This is a known failure mode of shift/jitter-based permutation nulls (fixed
    the same way established shift-surrogate methods do it: a minimum-shift guard band).
    Shifts are drawn only from `[guard, analysis_samples - guard)` so no permuted search
    arc can ever overlap the observed one. `guard` is clamped so tiny (e.g. test-scale)
    configs where the ideal guard would leave no valid shifts still produce a bounded,
    deterministic, best-effort-guarded result instead of raising or looping forever."""
    import hashlib

    if not candidate_revision or not dataset_label:
        raise PcmPlaybackError("permutation inputs require candidate revision and dataset label")
    revision = candidate_revision.encode("utf-8")
    label = dataset_label.encode("utf-8")
    ideal_guard = 2 * config.max_lag + config.tag_samples
    guard = min(ideal_guard, max(0, config.analysis_samples // 2 - 1))
    span = config.analysis_samples - 2 * guard
    return tuple(
        guard
        + int.from_bytes(
            hashlib.sha256(config.namespace + revision + label + index.to_bytes(2, "big")).digest()[:8], "big"
        )
        % span
        for index in range(config.permutation_count)
    )


def _pcm_int_samples(pcm: bytes) -> tuple[int, ...]:
    if len(pcm) % 2:
        raise PcmPlaybackError("PCM must contain complete s16le samples")
    return tuple(int.from_bytes(pcm[index : index + 2], "little", signed=True) for index in range(0, len(pcm), 2))


def _circular_normalized_correlation(
    tag: tuple[int, ...], samples: tuple[int, ...], start: int, shift: int
) -> float:
    """Score a tag at one lag of a circularly shifted fixed-size dataset."""
    dot = 0
    tag_energy = 0
    sample_energy = 0
    size = len(samples)
    for index, tag_value in enumerate(tag):
        sample_value = samples[(start + index + shift) % size]
        dot += tag_value * sample_value
        tag_energy += tag_value * tag_value
        sample_energy += sample_value * sample_value
    if not tag_energy or not sample_energy:
        return 0.0
    return abs(dot / math.sqrt(tag_energy * sample_energy))


def _family_code_maxima_at_shift(
    tags: tuple[tuple[int, ...], ...], samples: tuple[int, ...], shift: int, config: ProtocolConfig,
    *, use_numpy: bool | None = None,
) -> tuple[float, ...]:
    """Evaluate every predeclared code × lag cell before any statistic is selected.

    At production `ProtocolConfig` scale (16 codes x 6,401 lags x 1000 shifts x 5 datasets)
    the pure-Python path below measured live at ~23.7h; confirmed against primary sources
    (parec/PulseAudio latency docs, this project's own prior benchmarks) that a NumPy
    batched-matmul path is ~360x faster with no loss of exactness (every intermediate sum
    stays exactly representable in float64 for the s16le-range values this protocol uses)
    and no new dependency in the real-hardware runtime. GPU was evaluated and rejected:
    non-deterministic parallel-reduction float ordering conflicts with this protocol's
    exact-reproducibility design. `use_numpy=None` auto-detects via a local import (same
    pattern as `run_confirmed`'s other heavy-dependency imports) so this stays importable
    and unit-testable with zero numpy dependency; `True`/`False` force a path for tests."""
    if use_numpy is None:
        try:
            import numpy  # noqa: F401
        except ImportError:
            use_numpy = False
        else:
            use_numpy = True
    if use_numpy:
        import numpy as np
        return _family_code_maxima_at_shift_numpy(tags, samples, shift, config, np)
    return tuple(
        max(
            _circular_normalized_correlation(tag, samples, start, shift)
            for start in range(0, 2 * config.max_lag + 1)
        )
        for tag in tags
    )


def _family_code_maxima_at_shift_numpy(
    tags: tuple[tuple[int, ...], ...], samples: tuple[int, ...], shift: int, config: ProtocolConfig, np: object,
) -> tuple[float, ...]:
    """NumPy-vectorized equivalent of the pure-Python `_family_code_maxima_at_shift` loop:
    one batched matrix multiply instead of `code_count * (2*max_lag+1)` scalar Python
    iterations. `idx`'s modulo indexing mirrors `_circular_normalized_correlation`'s
    `samples[(start+index+shift) % size]` exactly, so this is correct for every case the
    pure-Python version is -- not just the production scale it was built for."""
    if not tags or not samples:
        return tuple(0.0 for _ in tags)
    lags = 2 * config.max_lag + 1
    size = len(samples)
    tag_len = len(tags[0])
    samples_arr = np.asarray(samples, dtype=np.float64)
    idx = (np.arange(lags)[:, None] + np.arange(tag_len)[None, :] + shift) % size
    windows = samples_arr[idx]  # (lags, tag_len)
    tags_arr = np.asarray(tags, dtype=np.float64)  # (code_count, tag_len)
    tag_energy = np.sqrt((tags_arr**2).sum(axis=1))  # (code_count,)
    win_energy = np.sqrt((windows**2).sum(axis=1))  # (lags,)
    dot = tags_arr @ windows.T  # (code_count, lags), BLAS matmul
    denom = np.outer(tag_energy, win_energy)
    scores = np.divide(np.abs(dot), denom, out=np.zeros_like(dot), where=denom > 0)
    return tuple(float(v) for v in scores.max(axis=1))


def _global_maximum_at_shift(
    tags: tuple[tuple[int, ...], ...], samples: tuple[int, ...], shift: int, config: ProtocolConfig
) -> float:
    """Return the global maximum over the fully evaluated code × lag family."""
    return max(_family_code_maxima_at_shift(tags, samples, shift, config), default=0.0)


def score_pcm_dataset(
    observed_pcm: bytes, candidate_revision: str, dataset_label: str, config: ProtocolConfig
) -> CalibrationScore:
    """Compute a complete all-code/all-lag max-T score for one fixed protocol dataset."""
    samples = _pcm_int_samples(observed_pcm)
    if len(samples) != config.analysis_samples:
        raise PcmPlaybackError("calibration dataset must have the exact predeclared sample count")
    tags = tuple(_pcm_int_samples(deterministic_tag(config, code_index)) for code_index in range(config.code_count))
    # The observed statistic chooses active code 0 only *after* evaluating the full family.
    active_max = _family_code_maxima_at_shift(tags, samples, 0, config)[0]
    permutations = tuple(
        _global_maximum_at_shift(tags, samples, shift, config)
        for shift in permutation_shifts(candidate_revision, dataset_label, config)
    )
    return CalibrationScore(active_max=active_max, permutation_global_max=permutations)
