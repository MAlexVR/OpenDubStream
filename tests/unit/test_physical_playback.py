from __future__ import annotations

import json
import math
import sys

import pytest

from opendubstream.domain.contracts import PhysicalSinkRequired, RouteLease, SinkRef, StreamRef
from opendubstream.infrastructure.audio.playback import (
    FeedbackObservation,
    PhysicalPcmPlayback,
    PcmPlaybackError,
    build_pw_play_argv,
    playback_tag,
    resolve_physical_target,
)


def lease() -> RouteLease:
    return RouteLease(
        stream=StreamRef("41", "Google Chrome", "Video", "chrome-a", "alsa_output.speakers"),
        virtual_sink="opendubstream.41",
        virtual_module_id="77",
        original_sink="alsa_output.speakers",
        physical_sink="alsa_output.speakers",
    )


def sink_inventory(*sinks: dict[str, object]) -> str:
    return json.dumps(list(sinks))


def physical_sink(name: str = "alsa_output.speakers", serial: str = "sink-serial") -> dict[str, object]:
    return {"name": name, "properties": {"object.serial": serial, "device.bus": "pci"}}


@pytest.mark.parametrize(
    "inventory, desired, message",
    [
        (sink_inventory(), SinkRef("alsa_output.speakers", True, "sink-serial"), "matched 0"),
        (
            sink_inventory(physical_sink("opendubstream.41")),
            SinkRef("opendubstream.41", True, "sink-serial"),
            "owned virtual",
        ),
        (
            sink_inventory(physical_sink("alsa_output.speakers.monitor")),
            SinkRef("alsa_output.speakers.monitor", True, "sink-serial"),
            "monitor",
        ),
        (
            sink_inventory(physical_sink(), physical_sink()),
            SinkRef("alsa_output.speakers", True, "sink-serial"),
            "matched 2",
        ),
        (
            sink_inventory(physical_sink(serial="changed")),
            SinkRef("alsa_output.speakers", True, "sink-serial"),
            "changed",
        ),
        (
            sink_inventory(physical_sink()),
            SinkRef("alsa_output.speakers", True),
            "stable serial",
        ),
        (
            sink_inventory({"name": "alsa_output.speakers", "properties": {"object.serial": "sink-serial"}}),
            SinkRef("alsa_output.speakers", True, "sink-serial"),
            "physical",
        ),
    ],
)
def test_resolve_physical_target_rejects_unsafe_ambiguous_or_changed_identity(
    inventory: str, desired: SinkRef, message: str
) -> None:
    with pytest.raises(PhysicalSinkRequired, match=message):
        resolve_physical_target(inventory, desired, lease())


def test_resolve_physical_target_requires_one_matching_stable_physical_sink() -> None:
    desired = SinkRef("alsa_output.speakers", True, "sink-serial")

    assert resolve_physical_target(sink_inventory(physical_sink()), desired, lease()) == desired


def test_pw_play_argv_is_literal_and_rejects_shell_injection() -> None:
    # --raw is required: without it pw-play sniffs the input as a libsndfile-parsed file
    # (WAV/AIFF/etc.) and rejects headerless PCM on stdin with "Format not recognised"
    # (confirmed live, 2026-09-03) instead of honoring --format/--rate/--channels.
    assert build_pw_play_argv("alsa_output.speakers") == (
        "pw-play", "--target", "alsa_output.speakers", "--format", "s16", "--rate", "16000", "--channels", "1", "--raw", "-"
    )
    with pytest.raises(PcmPlaybackError, match="literal"):
        build_pw_play_argv("alsa_output.speakers;id")


class FakePlayer:
    def __init__(self, result: Exception | None = None) -> None:
        self.result = result
        self.calls: list[tuple[SinkRef, bytes, float]] = []
        self.stop_calls = 0

    def play(self, target: SinkRef, pcm: bytes, deadline: float) -> None:
        self.calls.append((target, pcm, deadline))
        if self.result is not None:
            raise self.result

    def stop(self) -> None:
        self.stop_calls += 1


def test_playback_stops_injected_player_after_failure() -> None:
    target = SinkRef("alsa_output.speakers", True, "sink-serial")
    player = FakePlayer(TimeoutError())
    playback = PhysicalPcmPlayback(lambda: sink_inventory(physical_sink()), player)

    with pytest.raises(PcmPlaybackError, match="timed out"):
        playback.play(b"\x01\x00", target, lease(), deadline=0.25)

    assert player.calls == [(target, b"\x01\x00" + playback_tag(), 0.25)]
    assert player.stop_calls == 1


def tagged_pcm() -> bytes:
    tag = [int(0.20 * 32767 * math.sin(2 * math.pi * 997 * index / 16000)) for index in range(800)]
    return b"".join(value.to_bytes(2, "little", signed=True) for value in tag) + b"\x00\x00" * 800


def test_feedback_observation_accepts_low_baseline_and_uncorrelated_capture() -> None:
    tag = tagged_pcm()
    baseline = b"\x00\x00" * 800
    window = b"\x01\x00" * 2400

    observation = FeedbackObservation.measure(tag, baseline, window)

    assert observation.feedback_absent is True
    assert observation.baseline_score == 0.0
    assert observation.window_score < 0.10


@pytest.mark.parametrize(
    "baseline, window",
    [
        (tagged_pcm(), b"\x00\x00" * 2400),
        (b"\x00\x00" * 800, b"\x00\x00" * 200 + tagged_pcm() + b"\x00\x00" * 600),
    ],
)
def test_feedback_observation_rejects_correlated_baseline_or_capture_window(baseline: bytes, window: bytes) -> None:
    observation = FeedbackObservation.measure(tagged_pcm(), baseline, window)

    assert observation.feedback_absent is False
    assert observation.window_score - observation.baseline_score >= 0.05 or observation.baseline_score > 0.10


def test_calibration_protocol_generates_balanced_frozen_codes_and_byte_clock_crop() -> None:
    from opendubstream.infrastructure.audio.playback import (
        ProtocolConfig,
        crop_tagged_monitor_window,
        deterministic_tag,
        emit_tagged_probe_pcm,
    )

    config = ProtocolConfig()
    codes = [deterministic_tag(config, index) for index in range(config.code_count)]

    assert config.tag_insertion_samples == 3_200
    assert all(len(code) == config.tag_samples * 2 for code in codes)
    assert all(code.count(b"\x67\xe6") == 400 for code in codes)
    assert all(code.count(b"\x99\x19") == 400 for code in codes)
    assert len(set(codes)) == config.code_count

    probe = emit_tagged_probe_pcm(b"\x01\x00" * 3, codes[0], config)
    assert probe[: config.tag_insertion_samples * 2] == b"\x00\x00" * config.tag_insertion_samples
    assert probe[config.tag_insertion_samples * 2 : (config.tag_insertion_samples + config.tag_samples) * 2] == codes[0]

    expected = 10_000 + config.tag_insertion_samples - 17
    captured = b"\x11\x00" * (expected - config.tag_insertion_samples) + b"\x22\x00" * config.analysis_samples
    assert crop_tagged_monitor_window(captured, 10_000, -17, config) == b"\x22\x00" * config.analysis_samples
    assert crop_tagged_monitor_window(captured[:-2], 10_000, -17, config) is None


def test_calibration_decision_uses_global_max_t_and_rejects_incomplete_or_changed_baseline() -> None:
    from opendubstream.infrastructure.audio.playback import (
        CalibrationScore,
        ProtocolConfig,
        evaluate_calibration,
    )

    config = ProtocolConfig()
    null = CalibrationScore(active_max=0.20, permutation_global_max=tuple([0.21] * config.permutation_count))
    positive = CalibrationScore(active_max=0.95, permutation_global_max=tuple([0.30] * config.permutation_count))
    selected = CalibrationScore(active_max=0.25, permutation_global_max=tuple([0.35] * config.permutation_count))

    decision = evaluate_calibration(
        null_scores=(null, null, null),
        positive_score=positive,
        selected_score=selected,
        baseline_digest="db82ba31a54ab1a70dc0de1e1a3256c2c9993217ea05eeba4c6a35ae1bd6bc72",
        config=config,
    )
    assert decision.value == "no-feedback"
    assert decision.null_p_values == (1.0, 1.0, 1.0)
    assert decision.positive_p_value == 0.001
    assert decision.selected_p_value == 1.0

    incomplete = evaluate_calibration(
        null_scores=(null, null),
        positive_score=positive,
        selected_score=selected,
        baseline_digest="db82ba31a54ab1a70dc0de1e1a3256c2c9993217ea05eeba4c6a35ae1bd6bc72",
        config=config,
    )
    changed_baseline = evaluate_calibration(
        null_scores=(null, null, null),
        positive_score=positive,
        selected_score=selected,
        baseline_digest="not-the-immutable-baseline",
        config=config,
    )
    assert incomplete.value == "inconclusive"
    assert changed_baseline.value == "inconclusive"


def test_calibration_permutation_shifts_are_revision_and_dataset_bound() -> None:
    from opendubstream.infrastructure.audio.playback import ProtocolConfig, permutation_shifts

    config = ProtocolConfig()
    first = permutation_shifts("candidate-a", "selected", config)
    second = permutation_shifts("candidate-a", "selected", config)
    other_dataset = permutation_shifts("candidate-a", "positive", config)
    other_revision = permutation_shifts("candidate-b", "selected", config)

    assert len(first) == config.permutation_count
    assert first == second
    assert first != other_dataset
    assert first != other_revision
    assert all(0 <= shift < config.analysis_samples for shift in first)


def test_calibration_permutation_shifts_never_re_reach_the_observed_search_window() -> None:
    """Confirmed live (2026-09-04, real hardware 4.1d run): an injected positive-control
    tag at a fixed position scored `positive_p_value=0.113` against `alpha=.01` -- not
    chance. Verified analytically and numerically: ~10-14% of unconstrained circular
    shifts put the *same* localized feature back within reach of the search window used
    for the observed (shift=0) statistic, inflating the permutation null distribution for
    ANY real signal regardless of where in the buffer it sits (the "selected" crop is
    anchored so a genuine signal, like the deliberately placed tag, appears near position
    `tag_insertion_samples` -- the same region the observed statistic searches). This is a
    known failure mode of shift/jitter-based permutation nulls (confirmed against
    neuroscience surrogate-testing literature): the fix is a minimum-shift guard band so no
    permutation's search arc can ever overlap the observed statistic's own search arc."""
    from opendubstream.infrastructure.audio.playback import ProtocolConfig, permutation_shifts

    config = ProtocolConfig()
    guard = 2 * config.max_lag + config.tag_samples  # width of the observed search arc
    shifts = permutation_shifts("candidate-a", "selected", config)

    for shift in shifts:
        # The permuted search arc is [shift, shift+guard) mod N; the observed (shift=0)
        # search arc is [0, guard). They overlap iff shift < guard (arc starts inside the
        # observed region) or shift + guard > N (arc wraps around and touches position 0).
        overlaps_observed_arc = shift < guard or shift + guard > config.analysis_samples
        assert not overlaps_observed_arc, f"shift {shift} re-reaches the observed search window"


def test_calibration_protocol_has_frozen_hash_vectors_for_all_codes() -> None:
    import hashlib
    from opendubstream.infrastructure.audio.playback import ProtocolConfig, deterministic_tag

    expected = (
        "c6a3bc46735babe3d9d963575b1062f4dd049ad5feca8a2d959fe451cef16f04",
        "004718b5bd4eec18d103a9f8d98136aaf90ffdaea913b16e86083acd066afbca",
        "e222980224ff6f45a006339802aba726bd780816ef3d08fb7d3022ec66a1993a",
        "685451acef90a92141fabd6d082f445238ca2686bb7de1253fbec9628cb30125",
        "cefb30d7617df2da1a604ab0cfd68811a35f09b45126a785ff74664a6dc1c9b1",
        "2ea2884a7860f964cb4187a62cf838b5a32190786087a37753434a07653b0010",
        "3ecad41edfa7af7d9d866bc112356c5b53bc001df9dec7b8bf2e16b0c38c706e",
        "b618c3ac737424e9a3314eaef6feb41add0f994774079ea85db64ed26ca27c34",
        "213f7ec5764598be94e75d8ceba40cbe2789ad5e2fe764d7217b3a2742e8e260",
        "b688852009e5b6c7a6f08fe0e2f49c1792ce3eba1a00307d3b0044cc9ccecee6",
        "6c98290fce9ade477989c8929f5e2af41ce1a5801b5b1cc6067de9f79bdd81c2",
        "68c545480e2f15c0a089a26d763564c1f98bf5db76c3b299633ac23a35918f12",
        "c98beaa1631836357df74e6484030b3b644e058fb6617ae2f6d78d5e2c73b02c",
        "3443b718bef7cce6259f7807bcb622100b5b78c6697e227b4801f0a926d28a80",
        "5047406f9b4acab6c197983c00dce1baba055ab09f02ad6b5bb85bc49fdc0e33",
        "8e26b3580daf7ab4a423879578c664cfff9821d5e29454ff8279418cf2fafa25",
    )
    config = ProtocolConfig()
    assert tuple(hashlib.sha256(deterministic_tag(config, index)).hexdigest() for index in range(config.code_count)) == expected


def test_max_t_scorer_searches_every_code_lag_and_deterministic_shift() -> None:
    from opendubstream.infrastructure.audio.playback import ProtocolConfig, deterministic_tag, score_pcm_dataset

    config = ProtocolConfig(code_count=2, tag_samples=4, analysis_samples=16, max_lag=2, permutation_count=3)
    tag = deterministic_tag(config, 0)
    observed = b"\x00\x00" * 2 + tag + b"\x00\x00" * 10

    score = score_pcm_dataset(observed, "candidate-a", "selected", config)

    assert score.active_max == pytest.approx(1.0)
    assert len(score.permutation_global_max) == 3
    assert all(0.0 <= value <= 1.0 for value in score.permutation_global_max)


def test_max_t_observed_searches_the_full_family_but_selects_active_code_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    import opendubstream.infrastructure.audio.playback as playback

    config = playback.ProtocolConfig(code_count=2, tag_samples=4, analysis_samples=16, max_lag=2, permutation_count=1)
    observed = b"\x01\x00" * config.analysis_samples
    calls: list[int] = []

    def fixed_tag(config: playback.ProtocolConfig, code_index: int) -> bytes:
        return bytes((code_index + 1, 0)) * config.tag_samples

    def record_tag(tag: tuple[int, ...], samples: tuple[int, ...], start: int, shift: int) -> float:
        calls.append(tag[0])
        return 0.25 if tag[0] == 1 else 0.90

    monkeypatch.setattr(playback, "deterministic_tag", fixed_tag)
    monkeypatch.setattr(playback, "_circular_normalized_correlation", record_tag)
    # Force the pure-Python path regardless of whether numpy happens to be installed in
    # whatever environment runs this test: `_family_code_maxima_at_shift`'s numpy path
    # (added for production-scale performance, see design.md) does not call
    # `_circular_normalized_correlation` at all, so the monkeypatch above would silently
    # have no effect if numpy's auto-detection picked the vectorized path instead.
    monkeypatch.setitem(sys.modules, "numpy", None)
    score = playback.score_pcm_dataset(observed, "candidate-a", "selected", config)

    # Code 1 has a stronger peak, but observed statistic is exactly code 0's maximum.
    assert score.active_max == 0.25
    # Observed plus its one permutation each inspect 2 codes × 5 lags.
    assert len(calls) == 2 * config.code_count * (2 * config.max_lag + 1)


def test_locate_best_tag_position_finds_the_exact_offset_within_a_wide_search_range() -> None:
    from opendubstream.infrastructure.audio.playback import ProtocolConfig, deterministic_tag, locate_best_tag_position

    config = ProtocolConfig()
    tag = deterministic_tag(config, 0)
    position = 12_000
    observed = b"\x00\x00" * position + tag + b"\x00\x00" * 4_000

    found = locate_best_tag_position(tag, observed, search_range=20_000, confidence_floor=0.5)

    assert found == position


def test_locate_best_tag_position_fails_closed_below_the_confidence_floor() -> None:
    from opendubstream.infrastructure.audio.playback import ProtocolConfig, deterministic_tag, locate_best_tag_position

    tag = deterministic_tag(ProtocolConfig(), 0)
    observed = b"\x01\x00" * 5_000  # uncorrelated content: no window scores at or above the floor

    assert locate_best_tag_position(tag, observed, search_range=5_000, confidence_floor=0.99) is None


def test_locate_best_tag_position_rejects_empty_or_short_input() -> None:
    from opendubstream.infrastructure.audio.playback import ProtocolConfig, deterministic_tag, locate_best_tag_position

    tag = deterministic_tag(ProtocolConfig(), 0)

    assert locate_best_tag_position(tag, b"", search_range=1_000, confidence_floor=0.5) is None
    assert locate_best_tag_position(tag, b"\x00\x00" * 10, search_range=1_000, confidence_floor=0.5) is None
    assert locate_best_tag_position(b"", b"\x00\x00" * 2_000, search_range=1_000, confidence_floor=0.5) is None


def test_locate_best_tag_position_rejects_a_negative_search_range() -> None:
    from opendubstream.infrastructure.audio.playback import ProtocolConfig, deterministic_tag, locate_best_tag_position

    tag = deterministic_tag(ProtocolConfig(), 0)

    assert locate_best_tag_position(tag, b"\x00\x00" * 5_000, search_range=-1, confidence_floor=0.5) is None


def test_maximum_normalized_correlation_still_matches_after_the_shared_refactor() -> None:
    """Regression guard: `_maximum_normalized_correlation` must keep its exact behavior
    (search from position 0 through `_MAX_LAG`) once it shares code with the new
    position-search variant."""
    from opendubstream.infrastructure.audio.playback import _maximum_normalized_correlation

    tag = tagged_pcm()
    assert _maximum_normalized_correlation(tag, tag) == pytest.approx(1.0)
    assert _maximum_normalized_correlation(tag, b"\x00\x00" * 2400) == 0.0


def test_family_code_maxima_at_shift_numpy_path_matches_pure_python_path() -> None:
    """Confirmed live (2026-09-03): production-scale pure-Python scoring projects ~23.7h;
    a NumPy-vectorized path (batched matmul, no GPU -- rejected for non-deterministic
    parallel-reduction float ordering) projects ~4 minutes. Every intermediate sum stays
    exactly representable in float64 (tag/sample magnitudes <=~32768, lengths <=800/64000,
    max product-sum ~8.6e11, far under 2**53), so the two paths must agree exactly, not
    just approximately."""
    numpy = pytest.importorskip("numpy")
    from opendubstream.infrastructure.audio.playback import ProtocolConfig, _family_code_maxima_at_shift

    config = ProtocolConfig(code_count=4, tag_samples=37, tag_insertion_samples=5, analysis_samples=211, max_lag=53)
    rng = numpy.random.default_rng(1234)
    tags = tuple(tuple(int(v) for v in rng.integers(-6553, 6554, size=config.tag_samples)) for _ in range(config.code_count))
    samples = tuple(int(v) for v in rng.integers(-32768, 32767, size=config.analysis_samples))

    for shift in (0, 17, config.analysis_samples - 1, config.analysis_samples // 2):
        python_result = _family_code_maxima_at_shift(tags, samples, shift, config, use_numpy=False)
        numpy_result = _family_code_maxima_at_shift(tags, samples, shift, config, use_numpy=True)
        assert numpy_result == pytest.approx(python_result, abs=1e-9)


def test_family_code_maxima_at_shift_numpy_path_handles_zero_energy_like_pure_python() -> None:
    """A silent (all-zero) window must score 0.0 on both paths, not raise or divide by zero."""
    pytest.importorskip("numpy")
    from opendubstream.infrastructure.audio.playback import ProtocolConfig, _family_code_maxima_at_shift

    config = ProtocolConfig(code_count=2, tag_samples=5, tag_insertion_samples=1, analysis_samples=20, max_lag=3)
    tags = ((100, -100, 100, -100, 100), (0, 0, 0, 0, 0))
    samples = tuple(0 for _ in range(config.analysis_samples))

    python_result = _family_code_maxima_at_shift(tags, samples, 0, config, use_numpy=False)
    numpy_result = _family_code_maxima_at_shift(tags, samples, 0, config, use_numpy=True)
    assert python_result == (0.0, 0.0)
    assert numpy_result == (0.0, 0.0)


def test_family_code_maxima_at_shift_auto_detects_numpy_when_available() -> None:
    """Default `use_numpy=None` must pick the same result as an explicit choice -- proving
    the auto-detection wiring itself (not just each path's math) is correct."""
    pytest.importorskip("numpy")
    from opendubstream.infrastructure.audio.playback import ProtocolConfig, _family_code_maxima_at_shift

    config = ProtocolConfig(code_count=2, tag_samples=5, tag_insertion_samples=1, analysis_samples=20, max_lag=3)
    tags = ((100, -100, 100, -100, 100), (50, 50, -50, -50, 50))
    samples = tuple((i * 13) % 200 - 100 for i in range(config.analysis_samples))

    auto_result = _family_code_maxima_at_shift(tags, samples, 0, config)
    explicit_result = _family_code_maxima_at_shift(tags, samples, 0, config, use_numpy=True)
    assert auto_result == explicit_result


def test_family_code_maxima_at_shift_falls_back_to_pure_python_without_numpy() -> None:
    """`use_numpy=False` (simulating an environment where numpy is not installed, like the
    lightweight `.venv` this test suite normally runs in) must still produce the correct
    result via the unchanged pure-Python path."""
    from opendubstream.infrastructure.audio.playback import ProtocolConfig, _family_code_maxima_at_shift

    config = ProtocolConfig(code_count=1, tag_samples=3, tag_insertion_samples=1, analysis_samples=10, max_lag=2)
    tags = ((1000, -1000, 1000),)
    samples = (0, 0, 1000, -1000, 1000, 0, 0, 0, 0, 0)

    result = _family_code_maxima_at_shift(tags, samples, 0, config, use_numpy=False)
    assert result[0] == pytest.approx(1.0)


def test_best_tag_correlation_score_reports_the_score_locate_best_tag_position_used_to_decide() -> None:
    """Diagnostic companion to `locate_best_tag_position`: callers that get `None` back
    need to know whether the search found nothing above the floor, or found a strong
    match that the floor itself rejected -- `locate_best_tag_position` alone can't
    distinguish those two cases from its return value."""
    from opendubstream.infrastructure.audio.playback import ProtocolConfig, best_tag_correlation_score, deterministic_tag

    config = ProtocolConfig()
    tag = deterministic_tag(config, 0)
    observed = b"\x00\x00" * 12_000 + tag + b"\x00\x00" * 4_000

    assert best_tag_correlation_score(tag, observed, search_range=20_000) == pytest.approx(1.0)
    assert best_tag_correlation_score(tag, b"", search_range=1_000) == 0.0
    assert best_tag_correlation_score(tag, b"\x00\x00" * 5_000, search_range=-1) == 0.0


def test_pw_play_player_forwards_the_process_started_observer(monkeypatch: pytest.MonkeyPatch) -> None:
    import opendubstream.infrastructure.audio.playback as playback

    events: list[str] = []

    def fake_run(argv: tuple[str, ...], pcm: bytes, deadline: float, *, on_started, should_stop) -> None:  # type: ignore[no-untyped-def]
        events.append("pw-play-process-started")
        on_started()

    monkeypatch.setattr(playback, "run_pcm_player", fake_run)
    player = playback.PwPlayPcmPlayer(on_started=lambda: events.append("snapshot-during-pw-play"))

    player.play(SinkRef("alsa_output.speakers", True, "sink-1"), b"\x00\x00", 1.0)

    assert events == ["pw-play-process-started", "snapshot-during-pw-play"]
