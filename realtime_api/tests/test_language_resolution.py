"""resolve_language precedence: the caller's picker selection is authoritative.

Locks the fix for the "language flip mid-call" bug: a noisy STT detection (e.g. a
mis-heard one-word reply) must NOT override a language the caller explicitly
picked. When nothing was picked ("auto"), detection drives the reply language.
"""
from __future__ import annotations

from realtime_api.contracts import SessionStart
from realtime_api.languages import canonical_base
from realtime_api.pipelines._shared import language_mismatch, resolve_language
from realtime_api.session import VoiceSession


def _session(language: str = "auto", expected_language: str | None = None) -> VoiceSession:
    payload: dict = {"language": language, "sample_rate_hz": 16000}
    if expected_language:
        payload["expected_language"] = expected_language
    return VoiceSession(SessionStart.from_event(payload))


def test_non_auto_language_wins_over_detection():
    """A concrete picked language overrides a (possibly wrong) STT detection."""
    session = _session(language="es-ES", expected_language="es-ES")
    assert canonical_base(resolve_language(session, "en-US")) == "es"


def test_expected_language_wins_in_auto_mode():
    """In auto mode, the picker's expected_language still beats detection."""
    session = _session(language="auto", expected_language="hi")
    assert canonical_base(resolve_language(session, "en-US")) == "hi"


def test_detection_drives_when_nothing_picked():
    """Pure auto (no picker) uses the per-turn detected language."""
    session = _session(language="auto")
    assert canonical_base(resolve_language(session, "fr")) == "fr"


def test_english_last_resort_when_no_signal():
    """No picker and no detection falls back to English (last resort only)."""
    session = _session(language="auto")
    assert canonical_base(resolve_language(session, None)) == "en"


def test_explicit_selected_language_is_authoritative():
    session = VoiceSession(
        SessionStart.from_event(
            {
                "language_mode": "selected",
                "selected_language": "th-TH",
                "sample_rate_hz": 16000,
            }
        )
    )

    assert session.config.language == "th-TH"
    assert session.config.selected_language == "th-TH"
    assert resolve_language(session, "en-US") == "th-TH"
    assert language_mismatch(session, "en-US") is None


def test_explicit_auto_mode_has_no_selected_language():
    session = VoiceSession(
        SessionStart.from_event(
            {"language_mode": "auto", "sample_rate_hz": 16000}
        )
    )

    assert session.config.language == "auto"
    assert session.config.selected_language is None
    assert canonical_base(resolve_language(session, "fr")) == "fr"
