"""Pure active, non-corked Chrome stream selection.

Extracted from `scripts/verify-real-audio-loop.py`'s `select_active_chrome_route` so this
one genuinely shared precondition -- the same route-selection logic used by both
`run_confirmed` and `measure_transport_offset` -- gets test coverage it currently lacks.
`select_active_chrome_route` now delegates to `select_active_chrome_stream` for this pure
part, with byte-identical CLI behavior: neither the exception type/message nor the
first-match-wins semantics for multiple simultaneously active Chrome streams changed.
"""

from __future__ import annotations

from opendubstream.domain.contracts import StreamRef


def select_active_chrome_stream(streams: list[StreamRef], sink_inputs: list[dict[str, object]]) -> StreamRef:
    """Select the one actively playing (non-corked) Chrome stream.

    `sink_inputs` is the already `json.loads`-decoded `pactl -f json list sink-inputs`
    payload; this function performs no I/O and no JSON parsing. When more than one
    Chrome stream is simultaneously active, the first match wins -- this pure selection
    preserves `select_active_chrome_route`'s pre-extraction behavior exactly. A UI's own,
    stricter "exactly one" gate belongs in `application/eligibility.py`, not here.
    """
    active_serials = {
        str(entry.get("properties", {}).get("object.serial"))
        for entry in sink_inputs
        if not entry.get("corked", True)
    }
    chrome = [
        stream for stream in streams
        if "chrome" in stream.application_name.lower() and stream.serial in active_serials
    ]
    if not chrome:
        raise RuntimeError("no actively playing (non-corked) Chrome stream found")
    return chrome[0]


def count_active_chrome_streams(streams: list[StreamRef], sink_inputs: list[dict[str, object]]) -> int:
    """Count every actively playing (non-corked) Chrome stream, using the exact same
    filter as `select_active_chrome_stream` -- but counting matches instead of selecting
    one. `evaluate_eligibility` (`application/eligibility.py`) needs an exact count (0/1/>1)
    for its "exactly one" gate; `select_active_chrome_stream` needs a single fail-closed
    selection. Public and pure so a UI's press-time eligibility re-check
    (`ui/app.py`, per design.md's "re-evaluated immediately before every Start" decision)
    can reuse one tested function instead of duplicating this filter as untested logic.
    """
    active_serials = {
        str(entry.get("properties", {}).get("object.serial"))
        for entry in sink_inputs
        if not entry.get("corked", True)
    }
    return sum(
        1 for stream in streams
        if "chrome" in stream.application_name.lower() and stream.serial in active_serials
    )
