"""RED-first pure coverage for the active-Chrome-stream selection extracted from
`scripts/verify-real-audio-loop.py`'s `select_active_chrome_route` (task 1.1)."""

from __future__ import annotations

import pytest

from opendubstream.application.chrome_route import count_active_chrome_streams, select_active_chrome_stream
from opendubstream.domain.contracts import StreamRef


def _stream(identifier: str, application_name: str, serial: str, sink_name: str = "alsa_output.speakers") -> StreamRef:
    return StreamRef(
        identifier=identifier, application_name=application_name, media_name="Video",
        serial=serial, sink_name=sink_name,
    )


def _sink_input(serial: str, *, corked: bool) -> dict[str, object]:
    return {"corked": corked, "properties": {"object.serial": serial}}


def test_select_active_chrome_stream_raises_when_no_chrome_stream_exists() -> None:
    streams = [_stream("1", "firefox", "serial-1")]
    sink_inputs = [_sink_input("serial-1", corked=False)]

    with pytest.raises(RuntimeError, match="no actively playing"):
        select_active_chrome_stream(streams, sink_inputs)


def test_select_active_chrome_stream_selects_the_one_active_non_corked_chrome_stream() -> None:
    streams = [_stream("1", "Google Chrome", "serial-1")]
    sink_inputs = [_sink_input("serial-1", corked=False)]

    selected = select_active_chrome_stream(streams, sink_inputs)

    assert selected == streams[0]


def test_select_active_chrome_stream_excludes_a_corked_chrome_stream() -> None:
    streams = [_stream("1", "Google Chrome", "serial-1")]
    sink_inputs = [_sink_input("serial-1", corked=True)]

    with pytest.raises(RuntimeError, match="no actively playing"):
        select_active_chrome_stream(streams, sink_inputs)


def test_select_active_chrome_stream_ignores_a_non_chrome_active_stream_alongside_a_corked_chrome_one() -> None:
    streams = [_stream("1", "Google Chrome", "serial-1"), _stream("2", "firefox", "serial-2")]
    sink_inputs = [_sink_input("serial-1", corked=True), _sink_input("serial-2", corked=False)]

    with pytest.raises(RuntimeError, match="no actively playing"):
        select_active_chrome_stream(streams, sink_inputs)


def test_select_active_chrome_stream_selects_the_first_match_when_multiple_are_active() -> None:
    # Matches the pre-extraction behavior exactly: `select_active_chrome_route` never
    # rejected multiple simultaneously active Chrome streams -- it always picked the
    # first match. The UI's own multi-stream refusal lives in `eligibility.py`, a
    # separate, stricter gate; this pure selection stays byte-identical to the CLI.
    streams = [
        _stream("1", "Google Chrome", "serial-1"),
        _stream("2", "Google Chrome", "serial-2"),
    ]
    sink_inputs = [_sink_input("serial-1", corked=False), _sink_input("serial-2", corked=False)]

    selected = select_active_chrome_stream(streams, sink_inputs)

    assert selected == streams[0]


def test_select_active_chrome_stream_matches_case_insensitively_on_application_name() -> None:
    streams = [_stream("1", "google chrome", "serial-1")]
    sink_inputs = [_sink_input("serial-1", corked=False)]

    selected = select_active_chrome_stream(streams, sink_inputs)

    assert selected == streams[0]


# `count_active_chrome_streams` (task 4.4's press-time eligibility re-check needs an exact
# count, not a single fail-closed-on-zero/first-match-wins-on-multiple selection): mirrors
# the same active-non-corked-Chrome filter as `select_active_chrome_stream`, and matches
# `DubbingSessionRunner._count_active_chrome_streams`'s pre-existing private mirror exactly
# (same filter, counting instead of selecting) so a UI press-time gate can reuse one public,
# tested function instead of duplicating this filter as untested logic in `ui/app.py`.


def test_count_active_chrome_streams_is_zero_when_none_are_active() -> None:
    streams = [_stream("1", "firefox", "serial-1")]
    sink_inputs = [_sink_input("serial-1", corked=False)]

    assert count_active_chrome_streams(streams, sink_inputs) == 0


def test_count_active_chrome_streams_is_one_for_a_single_active_chrome_stream() -> None:
    streams = [_stream("1", "Google Chrome", "serial-1")]
    sink_inputs = [_sink_input("serial-1", corked=False)]

    assert count_active_chrome_streams(streams, sink_inputs) == 1


def test_count_active_chrome_streams_excludes_corked_streams() -> None:
    streams = [_stream("1", "Google Chrome", "serial-1")]
    sink_inputs = [_sink_input("serial-1", corked=True)]

    assert count_active_chrome_streams(streams, sink_inputs) == 0


def test_count_active_chrome_streams_counts_every_simultaneously_active_match() -> None:
    streams = [
        _stream("1", "Google Chrome", "serial-1"),
        _stream("2", "Google Chrome", "serial-2"),
    ]
    sink_inputs = [_sink_input("serial-1", corked=False), _sink_input("serial-2", corked=False)]

    assert count_active_chrome_streams(streams, sink_inputs) == 2
