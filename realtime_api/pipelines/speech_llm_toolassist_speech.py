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

# How long to wait for the LLM before playing a spoken acknowledgment. Fast
# turns (cached / no tool calls) return well under this, so they stay snappy and
# never hear a filler. Slower turns (Lakebase + Genie tool rounds) cross it and
# the caller hears a natural "let me check" instead of dead air.
_FILLER_GRACE_S = 0.6

# Localized acknowledgments, keyed by language prefix. Kept short so TTS renders
# quickly and the clip comfortably fits inside typical LLM+tool latency.
_FILLER_PHRASES: dict[str, str] = {
    "en": "Sure, let me take a look at that for you.",
    "th": "ได้ค่ะ เดี๋ยวขอตรวจสอบให้สักครู่นะคะ",
    "id": "Baik, saya periksa dulu sebentar ya.",
    "zh": "好的，我帮您查一下，请稍等。",
    "ja": "はい、ただいま確認いたしますので少々お待ちください。",
}


def _filler_phrase(language: str) -> str | None:
    prefix = (language or "").split("-", 1)[0].lower()
    return _FILLER_PHRASES.get(prefix) or _FILLER_PHRASES.get("en")


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

    # Run the LLM (+tools) as a background task so we can cover its latency with a
    # spoken acknowledgment. The task keeps running while we stream the filler.
    if respond_fn:
        llm_task = asyncio.ensure_future(
            asyncio.to_thread(
                respond_fn, transcript, language=language, context=context,
                tool_ctx=tool_ctx, history=session.history[:-1],
            )
        )
    else:
        llm_task = asyncio.ensure_future(
            asyncio.to_thread(
                bundle.llm.respond, transcript, language=language, context=context, tool_ctx=tool_ctx
            )
        )

    try:
        await asyncio.wait({llm_task}, timeout=_FILLER_GRACE_S)
        if not llm_task.done():
            filler = _filler_phrase(language)
            if filler:
                try:
                    async for event in stream_tts(
                        bundle, session, turn_id, filler, language,
                        mark_final=False, emit_text=False,
                    ):
                        if turn_id != session.turn_id:
                            break
                        yield event
                except Exception:  # noqa: BLE001
                    logger.warning("filler TTS failed for turn %d", turn_id, exc_info=True)

        if turn_id != session.turn_id:
            llm_task.cancel()
            return

        remaining = max(1.0, _LLM_TIMEOUT_S - (time.perf_counter() - t))
        try:
            result = await asyncio.wait_for(asyncio.shield(llm_task), timeout=remaining)
        except asyncio.TimeoutError:
            logger.error("LLM timed out after %ds for turn %d", _LLM_TIMEOUT_S, turn_id)
            raise RuntimeError(f"LLM response timed out after {_LLM_TIMEOUT_S}s") from None
    finally:
        if llm_task.done() and not llm_task.cancelled():
            llm_task.exception()  # retrieve to avoid "exception never retrieved"
        else:
            llm_task.cancel()

    if respond_fn:
        response_text, tool_invocations = result
    else:
        response_text = result
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
