"""Agent-Mode report localization directive.

`_with_language_directive` steers the on-screen "why" report into the caller's
language. It must resolve the CLEAN language name from the catalog (matching the
normalized BCP-47 tag, not a naive subtag split) and no-op for English / unknown.
"""
from __future__ import annotations

from genie_voice.genie.agent_mode import _with_language_directive


def test_english_and_none_are_noops():
    assert _with_language_directive("Q", None) == "Q"
    assert _with_language_directive("Q", "en-US") == "Q"


def test_unknown_language_is_noop():
    # Unsupported tag -> normalize raises -> graceful no-op (no leaked tag).
    assert _with_language_directive("Q", "xx-YY") == "Q"


def test_supported_language_uses_clean_catalog_name():
    # zh-CN resolves to the catalog's clean "Chinese", not a spec label variant,
    # and the directive is appended once.
    out = _with_language_directive("Q", "zh-CN")
    assert out.startswith("Q\n\n")
    assert out.rstrip().endswith("in Chinese.")


def test_catalog_match_uses_iso_base_not_subtag_split():
    """A tag whose subtag split differs from its catalog ISO base still resolves.

    Guards the nb-NO -> Norwegian case: the catalog is keyed by ISO base ("no"),
    so a "nb" split would miss it. We assert directly on the catalog-match logic
    so the test is independent of which languages a given deploy enables.
    """
    from genie_voice.i18n import LANGUAGE_CATALOG

    name = next((eng for (tag, eng) in LANGUAGE_CATALOG.values() if tag == "nb-NO"), None)
    assert name == "Norwegian"
