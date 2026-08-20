"""Text in → response.audio only (no STT, no LLM)."""
from __future__ import annotations

from typing import AsyncIterator

from ..session import VoiceSession
from . import ServingBundle
from ._shared import stream_tts


async def process_turn(
    bundle: ServingBundle,
    session: VoiceSession,
    turn_id: int,
    text: str,
    *,
    language: str | None = None,
    mark_final: bool = True,
) -> AsyncIterator[dict]:
    lang = language or session.config.language
    if not lang or lang == "auto":
        lang = "en-US"
    async for event in stream_tts(
        bundle, session, turn_id, text, lang, mark_final=mark_final
    ):
        yield event
