"""Audio in → transcript.final only (no LLM, no TTS)."""
from __future__ import annotations

from typing import AsyncIterator

from ..session import VoiceSession
from . import ServingBundle
from ._shared import resolve_language, transcribe


async def process_turn(
    bundle: ServingBundle,
    session: VoiceSession,
    turn_id: int,
    audio: bytes,
) -> AsyncIterator[dict]:
    transcript, detected, stt_ms = await transcribe(bundle, session, audio)
    if turn_id != session.turn_id:
        return
    if not transcript.strip():
        return
    language = resolve_language(session, detected)
    yield {
        "type": "transcript.final",
        "turn_id": turn_id,
        "text": transcript,
        "language": language,
        "stt_ms": stt_ms,
    }
