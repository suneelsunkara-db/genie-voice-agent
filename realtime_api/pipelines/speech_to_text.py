"""Audio in → transcript.final only (no LLM, no TTS)."""
from __future__ import annotations

from typing import AsyncIterator

from ..session import VoiceSession
from . import ServingBundle
from ._shared import language_mismatch, resolve_language, transcribe


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
    # Language gate: if the caller isn't speaking the selected language, don't
    # surface the off-language transcript — warn instead. This route has no TTS
    # stage, so the (spoken) switch-language prompt is the assist pipeline's job.
    mismatch = language_mismatch(session, detected)
    if mismatch:
        yield {"type": "language.mismatch", "turn_id": turn_id, **mismatch}
        return
    language = resolve_language(session, detected)
    yield {
        "type": "transcript.final",
        "turn_id": turn_id,
        "text": transcript,
        "language": language,
        "stt_ms": stt_ms,
    }
