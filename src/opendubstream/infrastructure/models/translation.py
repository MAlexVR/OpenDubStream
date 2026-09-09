"""Mandatory local OPUS-MT English-to-Spanish translation adapter contract."""

from __future__ import annotations

from collections.abc import Callable

from opendubstream.infrastructure.models.assets import LocalAssets


class OpusMtEnglishSpanishTranslator:
    """Runs an injected local OPUS-MT inference function; it never performs HTTP calls."""

    model_id = "Helsinki-NLP/opus-mt-en-es"
    asset_name = "opus-mt-en-es"

    def __init__(self, assets: LocalAssets, infer: Callable[[str], str]) -> None:
        assets.require(self.asset_name)
        self._infer = infer

    def translate(self, text: str) -> str:
        return self._infer(text)
