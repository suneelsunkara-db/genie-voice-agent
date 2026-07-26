"""Tests for semantic end-of-turn detection (Silero VAD + smart-turn v3)."""
from __future__ import annotations

import asyncio
import struct
from pathlib import Path

import pytest

from realtime_api.config import RealtimeSettings
from realtime_api.contracts import SessionStart
from realtime_api.endpointing import (
    _SILERO_PATH,
    _SMART_TURN_PATH,
    EndpointModels,
    SMART_TURN_LANGUAGES,
    endpointer_for,
)
from realtime_api.session import VoiceSession
from realtime_api.ws.handler import _smart_should_finalize


class _FakeModels:
    """Stand-in for EndpointModels so language routing needs no ONNX."""


def _endpointer_for(expected_language):
    return endpointer_for(
        _FakeModels(),
        sample_rate_hz=16_000,
        stop_ms=300,
        min_speech_ms=400,
        expected_language=expected_language,
    )


def test_endpointer_for_none_models_returns_none() -> None:
    assert (
        endpointer_for(None, sample_rate_hz=16_000, stop_ms=300, min_speech_ms=400, expected_language="en")
        is None
    )


@pytest.mark.parametrize("lang", ["en", "en-US", "zh-CN", "vi-VN", "ar-SA", "de-DE"])
def test_supported_languages_use_smart_turn(lang: str) -> None:
    assert _endpointer_for(lang).use_smart_turn is True


@pytest.mark.parametrize("lang", ["el-GR", "fil-PH", "ms-MY", "sv-SE", "th-TH"])
def test_unsupported_languages_fall_back_to_vad_only(lang: str) -> None:
    assert _endpointer_for(lang).use_smart_turn is False


@pytest.mark.parametrize("lang", [None, "auto", ""])
def test_auto_or_unknown_defaults_to_smart_turn(lang) -> None:
    assert _endpointer_for(lang).use_smart_turn is True


def test_smart_turn_language_set_excludes_our_fallback_langs() -> None:
    for lang in ("el", "fil", "ms", "sv", "th"):
        assert lang not in SMART_TURN_LANGUAGES


class _FakeEndpointer:
    def __init__(self, *, has_speech, use_smart_turn, silence_ms=0.0, candidate=False, complete=False):
        self.has_speech = has_speech
        self.use_smart_turn = use_smart_turn
        self.silence_ms = silence_ms
        self._candidate = candidate
        self._complete = complete

    def take_pause_candidate(self) -> bool:
        return self._candidate

    def smart_turn_complete(self, threshold: float):
        return self._complete, 0.9 if self._complete else 0.1


def _session_with(endpointer, *, turn_audio_ms=1000.0, max_turn_seconds=None, vad_silence_ms=None):
    payload = {"language": "en-US", "sample_rate_hz": 16_000}
    if max_turn_seconds is not None:
        payload["max_turn_seconds"] = max_turn_seconds
    if vad_silence_ms is not None:
        payload["vad_silence_ms"] = vad_silence_ms
    session = VoiceSession(SessionStart.from_event(payload))
    session.endpointer = endpointer
    session.turn_audio_ms = turn_audio_ms
    return session


def _finalize(session) -> bool:
    return asyncio.run(_smart_should_finalize(session, RealtimeSettings(stt_endpoint="s", llm_endpoint="l", tts_endpoint="t")))


def test_hard_cap_finalizes_even_without_speech() -> None:
    ep = _FakeEndpointer(has_speech=False, use_smart_turn=True)
    session = _session_with(ep, turn_audio_ms=20_000, max_turn_seconds=20)
    assert _finalize(session) is True


def test_no_speech_before_cap_does_not_finalize() -> None:
    ep = _FakeEndpointer(has_speech=False, use_smart_turn=True)
    session = _session_with(ep, turn_audio_ms=1_000)
    assert _finalize(session) is False


def test_smart_turn_complete_finalizes_at_pause() -> None:
    ep = _FakeEndpointer(has_speech=True, use_smart_turn=True, candidate=True, complete=True)
    session = _session_with(ep, turn_audio_ms=3_000)
    assert _finalize(session) is True


def test_smart_turn_incomplete_keeps_listening() -> None:
    ep = _FakeEndpointer(has_speech=True, use_smart_turn=True, candidate=True, complete=False)
    session = _session_with(ep, turn_audio_ms=3_000)
    assert _finalize(session) is False


def test_smart_turn_not_evaluated_without_pause_candidate() -> None:
    ep = _FakeEndpointer(has_speech=True, use_smart_turn=True, candidate=False, complete=True)
    session = _session_with(ep, turn_audio_ms=3_000)
    assert _finalize(session) is False


def test_smart_turn_long_pause_safety_net_finalizes() -> None:
    # Supported language, smart-turn keeps saying "incomplete", but a very long
    # Silero pause must still end the turn (bounds worst case to vad_silence_ms).
    ep = _FakeEndpointer(
        has_speech=True, use_smart_turn=True, candidate=True, complete=False, silence_ms=2_600
    )
    session = _session_with(ep, turn_audio_ms=8_000, vad_silence_ms=2_500)
    assert _finalize(session) is True


def test_vad_only_fallback_finalizes_on_silence_gap() -> None:
    ep = _FakeEndpointer(has_speech=True, use_smart_turn=False, silence_ms=2_600)
    session = _session_with(ep, turn_audio_ms=5_000, vad_silence_ms=2_500)
    assert _finalize(session) is True


def test_vad_only_fallback_waits_for_full_gap() -> None:
    ep = _FakeEndpointer(has_speech=True, use_smart_turn=False, silence_ms=800)
    session = _session_with(ep, turn_audio_ms=5_000, vad_silence_ms=2_500)
    assert _finalize(session) is False


def test_is_noise_timeout_true_when_no_speech_past_window() -> None:
    ep = _FakeEndpointer(has_speech=False, use_smart_turn=True)
    session = _session_with(ep, turn_audio_ms=4_500)
    assert session.is_noise_timeout(4) is True


def test_is_noise_timeout_false_with_speech() -> None:
    ep = _FakeEndpointer(has_speech=True, use_smart_turn=True)
    session = _session_with(ep, turn_audio_ms=10_000)
    assert session.is_noise_timeout(4) is False


_MODELS_PRESENT = _SILERO_PATH.exists() and _SMART_TURN_PATH.exists()


@pytest.mark.skipif(not _MODELS_PRESENT, reason="ONNX models not bundled")
def test_models_load_and_infer_on_silence() -> None:
    """Real ONNX path: silence yields no detected speech; smart-turn returns a prob."""
    models = EndpointModels.load()
    assert models is not None
    from realtime_api.endpointing import TurnEndpointer

    ep = TurnEndpointer(
        models, sample_rate_hz=16_000, stop_ms=300, min_speech_ms=400, use_smart_turn=True
    )
    silence = struct.pack("<" + "h" * 512, *([0] * 512))
    for _ in range(40):  # ~1.3s of silence
        ep.feed(silence)
    assert ep.has_speech is False
    complete, prob = ep.smart_turn_complete(0.5)
    assert 0.0 <= prob <= 1.0
    assert isinstance(complete, bool)
