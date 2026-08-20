"""The refusal is the one line a caller hears when the turn could not be answered.

It has to arrive in the language of the call. Speaking English at the moment a
governed read timed out is both a language defect and a trust defect: the caller
cannot tell whether the assistant failed or the call dropped.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from realtime_api.runtime.refuse import (
    ErrorCode,
    _REFUSE_EN,
    refuse_speech,
)

CATALOG = json.loads(
    (Path(__file__).resolve().parent.parent / "phrases" / "refusals.json").read_text(
        encoding="utf-8"
    )
)


def test_every_failure_code_is_spoken_in_the_call_language():
    for code in ErrorCode:
        spoken = refuse_speech(code, language="th-TH")
        assert spoken == CATALOG["th-TH"][code.value]
        assert spoken != _REFUSE_EN[code]


@pytest.mark.parametrize("language", sorted(CATALOG))
def test_the_catalog_covers_every_code_for_every_shipped_language(language: str):
    phrases = CATALOG[language]
    for code in ErrorCode:
        text = phrases.get(code.value, "")
        assert text.strip(), f"{language} is missing {code.value}"
        # Written for TTS: markdown and bullets are read out as artifacts.
        assert not any(marker in text for marker in ("*", "#", "|", "`", "- "))


def test_a_regional_variant_falls_back_to_its_language_not_to_english():
    """A caller on es-MX must not drop to English just because es-ES is the tag."""
    spoken = refuse_speech(ErrorCode.NO_EVIDENCE, language="es-MX")
    assert spoken == CATALOG["es-ES"][ErrorCode.NO_EVIDENCE.value]


def test_an_untranslated_language_still_refuses_rather_than_going_silent():
    spoken = refuse_speech(ErrorCode.TIMEOUT, language="xx-XX")
    assert spoken == _REFUSE_EN[ErrorCode.TIMEOUT]


def test_english_and_a_missing_language_use_the_authored_source():
    for language in ("en-US", "en", "", None):
        assert refuse_speech(ErrorCode.PERMISSION, language=language) == (
            _REFUSE_EN[ErrorCode.PERMISSION]
        )


def test_an_unknown_code_degrades_to_the_unsupported_line_in_language():
    assert refuse_speech("not_a_code", language="ja-JP") == (
        CATALOG["ja-JP"][ErrorCode.UNSUPPORTED.value]
    )
    assert refuse_speech(ErrorCode.CANCELLED, language="ja-JP") == (
        CATALOG["ja-JP"][ErrorCode.CANCELLED.value]
    )
