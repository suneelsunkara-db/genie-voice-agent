"""Unit tests for client-managed turn mode (``endpointing: false``).

Offline/batch callers (e.g. the multilingual voice benchmark) already hold a
whole utterance and mark its end with ``audio.end``. Server-side turn detection
would otherwise finalize a long clip at a natural mid-utterance pause and the
caller would only see the first fragment — the truncation artifact that inflated
benchmark WER. ``endpointing: false`` puts the server in manual mode: no
automatic finalization, the turn ends only on ``audio.end``.

These tests pin the two behaviors against each other on identical audio:
  * default session  -> energy VAD cuts the turn at the silence gap (truncated).
  * manual session   -> the whole clip is one turn (nothing dropped).
"""
from __future__ import annotations

import struct

from fastapi.testclient import TestClient

from realtime_api.app import create_app
from realtime_api.config import RealtimeSettings
from realtime_api.contracts import SessionStart
from realtime_api.pipelines import ServingBundle

_SAMPLES = 320  # 20 ms @ 16 kHz
_FRAME_BYTES = _SAMPLES * 2
_LOUD = struct.pack("<" + "h" * _SAMPLES, *([6000, -6000] * (_SAMPLES // 2)))
_SILENCE = struct.pack("<" + "h" * _SAMPLES, *([0] * _SAMPLES))
_SILENCE_FRAMES = 40  # 800 ms — far beyond the 100 ms VAD gap below.
_TOTAL_BYTES = (1 + _SILENCE_FRAMES) * _FRAME_BYTES


class LenStub:
    """STT stub that reports how many audio bytes the turn captured, so a test
    can distinguish a whole-clip turn from one truncated at the silence gap."""

    def transcribe(self, audio: bytes, *, language: str | None, sample_rate_hz: int) -> tuple[str, str | None]:
        return str(len(audio)), "en-US"


def _client() -> TestClient:
    settings = RealtimeSettings(
        stt_endpoint="stt",
        llm_endpoint="llm",
        tts_endpoint="tts",
        supported_languages=("en",),
        stt_languages=("en",),
        # Exercise the energy-VAD turn flow (square-wave frames aren't real speech
        # to Silero); the semantic endpointer has its own tests.
        endpointing_enabled=False,
        # A short gap so the default path finalizes well inside the 800 ms of
        # trailing silence, and a tiny min-speech so one loud frame qualifies.
        vad_silence_ms=100,
        min_speech_ms=20,
    )
    fake = LenStub()
    app = create_app(settings=settings, bundle_factory=lambda _s: ServingBundle(stt=fake, llm=fake, tts=fake))
    return TestClient(app)


def _first_transcript_bytes(*, endpointing: bool | None) -> int:
    """Send one loud frame + a long silence + audio.end and return the byte
    length the first finalized turn captured."""
    start = {"type": "session.start", "language": "en-US", "sample_rate_hz": 16000}
    if endpointing is not None:
        start["endpointing"] = endpointing
    with _client() as client, client.websocket_connect("/v1/speech-to-text") as ws:
        ws.send_json(start)
        assert ws.receive_json()["type"] == "session.ready"
        ws.send_bytes(_LOUD)
        for _ in range(_SILENCE_FRAMES):
            ws.send_bytes(_SILENCE)
        ws.send_json({"type": "audio.end"})
        for _ in range(8):  # skip speech.started / turn.started
            msg = ws.receive_json()
            if msg["type"] == "transcript.final":
                return int(msg["text"])
    raise AssertionError("no transcript.final received")


def test_default_session_truncates_turn_at_silence_gap() -> None:
    # The energy VAD finalizes at the 100 ms gap, so the first turn captures only
    # the leading speech + gap — the rest of the clip is dropped from this turn.
    captured = _first_transcript_bytes(endpointing=None)
    assert captured < _TOTAL_BYTES


def test_manual_session_transcribes_whole_clip_on_audio_end() -> None:
    # endpointing:false disables auto-finalization: the mid-clip silence is
    # buffered and the entire clip is transcribed as one turn on audio.end.
    captured = _first_transcript_bytes(endpointing=False)
    assert captured == _TOTAL_BYTES


def test_session_start_parses_endpointing_flag() -> None:
    assert SessionStart.from_event({"language": "en", "endpointing": False}).endpointing is False
    assert SessionStart.from_event({"language": "en", "endpointing": True}).endpointing is True
    # Absent -> None: inherit the server's endpointing.enabled default (unchanged
    # live-call behavior).
    assert SessionStart.from_event({"language": "en"}).endpointing is None


def test_session_start_rejects_non_bool_endpointing() -> None:
    import pytest

    with pytest.raises(ValueError):
        SessionStart.from_event({"language": "en", "endpointing": "false"})
