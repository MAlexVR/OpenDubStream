"""RED-first coverage for Phase 3: the EN/ES localization catalog (`resolve()`) and the
persisted language preference (`PreferencesStore`). Pure/`tmp_path`-only, no Qt import --
matches `tests/unit/test_chrome_route.py`'s and `test_eligibility.py`'s pure-module pattern.

Per design.md's "Localization" decision, `resolve(language, key, params)` is total: an
unmapped key raises rather than silently falling back to English. Per the
`bilingual-desktop-control` spec's "Presentation-language selection and persistence"
requirement, every `MessageKey` MUST have both an English and a Spanish entry -- the
parity test below iterates every real `MessageKey` member for every real `Language` member,
so a catalog gap fails loudly here rather than at UI runtime.
"""

from __future__ import annotations

import pytest

from opendubstream.ui.resources import Language, MessageKey, Preferences, PreferencesStore, resolve


# ---------------------------------------------------------------------------
# Catalog completeness / parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("language", list(Language))
def test_every_message_key_resolves_for_every_language(language: Language) -> None:
    """Every `MessageKey` member MUST have an entry for every `Language` member. A stray
    format placeholder in a template that does not need `detail` is harmless: `str.format`
    ignores unused keyword arguments, so passing `detail` for every key exercises catalog
    completeness uniformly without special-casing the one key that actually uses it."""
    for key in MessageKey:
        text = resolve(language, key, params={"detail": "example-detail"})
        assert isinstance(text, str)
        assert text.strip() != ""


def test_english_and_spanish_text_differ_for_every_key() -> None:
    """Guards against a lazily copy-pasted Spanish entry that is actually just the English
    string again -- a real translation, not a silent fallback disguised as one.
    `WINDOW_TITLE` is a documented exception: it is the product name ("OpenDubStream"),
    deliberately not localized, not a missed translation. `TAB_GENERAL` is a second: "General"
    is genuinely the identical word in English and Spanish (unlike every other tab label),
    not a missed translation either."""
    for key in (MessageKey.WINDOW_TITLE, MessageKey.TAB_GENERAL):
        assert resolve(Language.ENGLISH, key) == resolve(Language.SPANISH, key)
    for key in MessageKey:
        if key in (MessageKey.WINDOW_TITLE, MessageKey.TAB_GENERAL):
            continue
        english = resolve(Language.ENGLISH, key, params={"detail": "example-detail"})
        spanish = resolve(Language.SPANISH, key, params={"detail": "example-detail"})
        assert english != spanish, f"{key} has identical EN/ES text"


def test_resolve_raises_on_an_unmapped_key_without_falling_back_to_english() -> None:
    """`resolve` is total per design.md: an unmapped key raises in tests rather than
    silently resolving to the English string. `MessageKey` is a closed `StrEnum`, so an
    "unmapped key" is simulated with a plain string that is not any real member's value --
    `resolve` has no special dispensation to fall back for it."""
    with pytest.raises(KeyError):
        resolve(Language.ENGLISH, "not-a-real-message-key")  # type: ignore[arg-type]


def test_resolve_substitutes_format_params() -> None:
    text = resolve(Language.ENGLISH, MessageKey.ERROR_UNEXPECTED, params={"detail": "boom"})
    assert "boom" in text


def test_resolve_defaults_params_to_empty_mapping() -> None:
    # A key with no placeholders resolves fine with no params supplied at all.
    assert resolve(Language.ENGLISH, MessageKey.STAGE_IDLE) != ""


# ---------------------------------------------------------------------------
# Persisted language preference
# ---------------------------------------------------------------------------


def test_preferences_store_round_trips_the_selected_language(tmp_path) -> None:
    path = tmp_path / "ui.json"
    PreferencesStore(path).save(Preferences(language=Language.SPANISH))

    reloaded = PreferencesStore(path).load()

    assert reloaded.language is Language.SPANISH


def test_preferences_store_defaults_to_english_when_the_file_is_missing(tmp_path) -> None:
    path = tmp_path / "does-not-exist" / "ui.json"

    loaded = PreferencesStore(path).load()

    assert loaded == Preferences(language=Language.ENGLISH)


def test_preferences_store_creates_parent_directories_on_save(tmp_path) -> None:
    path = tmp_path / "nested" / "dir" / "ui.json"

    PreferencesStore(path).save(Preferences(language=Language.ENGLISH))

    assert path.exists()
