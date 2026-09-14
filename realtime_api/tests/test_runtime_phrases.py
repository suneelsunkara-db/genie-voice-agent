from __future__ import annotations

from string import Formatter

from genie_voice.i18n import LANGUAGE_SPECS
from realtime_api.runtime.phrases import ENGLISH, catalog, phrase


def _fields(value: str) -> set[str]:
    return {
        str(field)
        for _, field, _, _ in Formatter().parse(value)
        if field is not None
    }


def test_runtime_catalog_covers_every_supported_language_and_key() -> None:
    loaded = catalog()
    assert set(LANGUAGE_SPECS) <= set(loaded)
    for language in LANGUAGE_SPECS:
        assert set(loaded[language]) == set(ENGLISH)
        for key, source in ENGLISH.items():
            assert loaded[language][key].strip()
            assert _fields(loaded[language][key]) == _fields(source)


def test_fixed_phrase_renders_reviewed_placeholders() -> None:
    rendered = phrase(
        "language.switch",
        language="en-US",
        detected_language="Spanish",
    )
    assert rendered == (
        "I heard Spanish. Please switch the app language to Spanish so we can continue."
    )
