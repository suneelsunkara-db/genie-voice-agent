"""Audio in → STT → LLM (+ tools) → TTS."""
from __future__ import annotations

import asyncio
import logging
import time
from typing import AsyncIterator

from ..session import VoiceSession
from ..tools import ToolContext
from . import ServingBundle
from ._shared import resolve_language, stream_tts, transcribe

logger = logging.getLogger("realtime_voice")

# Maximum time (seconds) the LLM stage may take before we abort the turn and
# surface an error. Covers cold-start + multi-tool-iteration loops. Must be
# generous enough for 3 tool rounds but tight enough that a hung endpoint
# doesn't leave the user staring at a spinner forever.
_LLM_TIMEOUT_S = 50


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

    session.history.append({"role": "user", "content": transcript})

    tool_ctx = ToolContext(
        customer_id=session.config.customer_id,
        call_id=session.config.call_id,
        _detected_language=language,
    )
    t = time.perf_counter()
    respond_fn = getattr(bundle.llm, "respond_with_tools", None)
    tool_invocations: list[dict] = []
    if respond_fn:
        try:
            response_text, tool_invocations = await asyncio.wait_for(
                asyncio.to_thread(
                    respond_fn, transcript, language=language, context=context,
                    tool_ctx=tool_ctx, history=session.history[:-1],
                ),
                timeout=_LLM_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            logger.error("LLM timed out after %ds for turn %d", _LLM_TIMEOUT_S, turn_id)
            raise RuntimeError(f"LLM response timed out after {_LLM_TIMEOUT_S}s") from None
    else:
        try:
            response_text = await asyncio.wait_for(
                asyncio.to_thread(
                    bundle.llm.respond, transcript, language=language, context=context, tool_ctx=tool_ctx
                ),
                timeout=_LLM_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            logger.error("LLM timed out after %ds for turn %d", _LLM_TIMEOUT_S, turn_id)
            raise RuntimeError(f"LLM response timed out after {_LLM_TIMEOUT_S}s") from None
    llm_ms = round((time.perf_counter() - t) * 1000)
    logger.info("LLM responded in %dms (turn %d, tools=%d)", llm_ms, turn_id, len(tool_invocations))
    if turn_id != session.turn_id:
        return

    session.history.append({"role": "assistant", "content": response_text})
    # Keep history bounded to last 10 exchanges (20 messages)
    if len(session.history) > 20:
        session.history = session.history[-20:]

    for invocation in tool_invocations:
        yield {
            "type": "tool.called",
            "turn_id": turn_id,
            "name": invocation["name"],
            "arguments": invocation.get("arguments"),
            "result": invocation.get("result"),
        }

    yield {
        "type": "response.text",
        "turn_id": turn_id,
        "text": response_text,
        "llm_ms": llm_ms,
    }

    async for event in stream_tts(bundle, session, turn_id, response_text, language):
        yield event

    # After TTS completes, suppress turn finalization for 1.5s to prevent
    # speaker→mic echo from immediately triggering a false follow-up turn.
    session.set_cooldown(1.5)
