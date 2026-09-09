from __future__ import annotations

import pytest

from opendubstream.domain.contracts import StreamRef
from opendubstream.infrastructure.audio.discovery import (
    DiscoveryError,
    PipeWireStreamDiscovery,
    parse_streams,
)

SINK_INPUTS_JSON = """[{"index":946,"sink":62,"properties":{"application.name":"Google Chrome","media.name":"Playback","object.serial":"946"}}]"""
SINKS_JSON = """[{"index":62,"name":"alsa_output.pci-0000_06_00.6.analog-stereo"}]"""


def test_parses_real_chrome_sink_input_into_a_stream_ref() -> None:
    streams = parse_streams(SINK_INPUTS_JSON, SINKS_JSON)

    assert streams == [
        StreamRef(
            identifier="946",
            application_name="Google Chrome",
            media_name="Playback",
            serial="946",
            sink_name="alsa_output.pci-0000_06_00.6.analog-stereo",
        )
    ]


def test_skips_entries_missing_application_identity_or_serial() -> None:
    incomplete = """[{"index":1,"sink":62,"properties":{"media.name":"Playback"}}]"""

    assert parse_streams(incomplete, SINKS_JSON) == []


def test_skips_entries_whose_sink_index_is_unknown() -> None:
    orphaned = """[{"index":946,"sink":999,"properties":{"application.name":"Google Chrome","object.serial":"946"}}]"""

    assert parse_streams(orphaned, SINKS_JSON) == []


def test_rejects_malformed_json_with_a_clear_error() -> None:
    with pytest.raises(DiscoveryError, match="sink-input listing"):
        parse_streams("not json", SINKS_JSON)
    with pytest.raises(DiscoveryError, match="sink listing"):
        parse_streams(SINK_INPUTS_JSON, "not json")


def test_pipewire_stream_discovery_calls_only_the_two_json_listing_forms() -> None:
    calls: list[tuple[str, ...]] = []
    responses = {
        ("pactl", "-f", "json", "list", "sink-inputs"): SINK_INPUTS_JSON,
        ("pactl", "-f", "json", "list", "sinks"): SINKS_JSON,
    }

    class FakePactl:
        def run(self, argv: tuple[str, ...], **_kwargs: object) -> str:
            calls.append(argv)
            return responses[argv]

    discovery = PipeWireStreamDiscovery(FakePactl())

    streams = discovery()

    assert streams == parse_streams(SINK_INPUTS_JSON, SINKS_JSON)
    assert set(calls) == set(responses)
