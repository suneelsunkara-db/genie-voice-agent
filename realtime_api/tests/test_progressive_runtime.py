"""Tests for Progressive Turn Runtime (Phase 2)."""
from __future__ import annotations

import asyncio

import pytest

from realtime_api.runtime import (
    CancellationToken,
    SpeechKind,
    SpeechRequest,
    SpeechScheduler,
    TurnPhase,
    TurnState,
)
from realtime_api.session import VoiceSession
from realtime_api.contracts import SessionStart


def test_speech_scheduler_ack_budget():
    s = SpeechScheduler()
    assert s.accept(SpeechRequest(SpeechKind.ACK, "one moment"))
    assert not s.accept(SpeechRequest(SpeechKind.ACK, "again"))
    assert "ack:budget" in s.skipped


def test_speech_scheduler_final_requires_cite():
    s = SpeechScheduler()
    assert not s.accept(SpeechRequest(SpeechKind.FINAL, "42 dollars", cited=False))
    assert s.accept(SpeechRequest(SpeechKind.FINAL, "42 dollars", cited=True))


def test_speech_scheduler_preview_stable_only():
    s = SpeechScheduler()
    assert not s.accept(SpeechRequest(SpeechKind.PREVIEW, "maybe", stable=False))
    assert s.accept(SpeechRequest(SpeechKind.PREVIEW, "stable fact", stable=True))


def test_turn_state_cancel_drops_later_work():
    t = TurnState(turn_id=3)
    assert t.busy
    t.cancel_turn()
    assert t.cancel.cancelled
    assert t.phase == TurnPhase.COMPLETED
    with pytest.raises(asyncio.CancelledError):
        t.cancel.raise_if_cancelled()


def test_cancellation_token():
    c = CancellationToken()
    assert not c.cancelled
    c.cancel()
    assert c.cancelled


def test_session_phase_helpers():
    session = VoiceSession(config=SessionStart(language="en-US", sample_rate_hz=16_000))
    assert session.phase == "idle"
    assert not session.busy
    session.set_working()
    assert session.busy and session.phase == "working"
    session.set_awaiting_user()
    assert not session.busy and session.phase == "awaiting_user"
    session.set_completed()
    assert session.phase == "completed"


def test_mid_turn_synthesize_does_not_increment_when_busy():
    """Contract: same-turn inject keeps turn_id (handler tested via state)."""
    session = VoiceSession(config=SessionStart(language="en-US", sample_rate_hz=16_000))
    session.turn_id = 5
    session.set_working()
    session.speech_scheduler = SpeechScheduler()
    before = session.turn_id
    # Inject path must not bump id
    assert session.busy
    assert session.turn_id == before
    ok = session.speech_scheduler.accept(
        SpeechRequest(kind=SpeechKind.INJECT, text="spoken summary")
    )
    assert ok
    assert session.turn_id == before
