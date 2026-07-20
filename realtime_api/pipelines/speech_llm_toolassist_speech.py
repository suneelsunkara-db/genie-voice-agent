"""Audio in → STT → LLM (+ tools) → TTS."""
from __future__ import annotations

import asyncio
import time
from typing import AsyncIterator

from ..session import VoiceSession
from . import ServingBundle
from ._shared import resolve_language, stream_tts, transcribe


async def process_turn(
    bundle: ServingBundle,
    session: VoiceSession,
    turn_id: int,
    audio: bytes,
    *,
    context: str | None = None,
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

    t = time.perf_counter()
    response_text = await asyncio.to_thread(
        bundle.llm.respond, transcript, language=language, context=context
    )
    llm_ms = round((time.perf_counter() - t) * 1000)
    if turn_id != session.turn_id:
        return
    yield {
        "type": "response.text",
        "turn_id": turn_id,
        "text": response_text,
        "llm_ms": llm_ms,
    }

    async for event in stream_tts(bundle, session, turn_id, response_text, language):
        yield event
