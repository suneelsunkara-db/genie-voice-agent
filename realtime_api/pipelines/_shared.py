"""Shared helpers for voice pipelines."""
from __future__ import annotations

import asyncio
import re
import time
from typing import AsyncIterator

from ..session import VoiceSession
from . import ServingBundle

_SENTENCE_RE = re.compile(r"[^.!?。！？…\n]+(?:[.!?。！？…]+|\n|$)", re.UNICODE)


def split_sentences(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    sentences = [match.group().strip() for match in _SENTENCE_RE.finditer(text)]
    sentences = [s for s in sentences if s]
    return sentences or [text]


def resolve_language(session: VoiceSession, detected: str | None) -> str:
    pref = session.config.language
    stt_language = None if (not pref or pref == "auto") else pref
    return detected or stt_language or "en-US"


async def transcribe(
    bundle: ServingBundle, session: VoiceSession, audio: bytes
) -> tuple[str, str | None, int]:
    pref = session.config.language
    stt_language = None if (not pref or pref == "auto") else pref
    t = time.perf_counter()
    transcript, detected = await asyncio.to_thread(
        bundle.stt.transcribe,
        audio,
        language=stt_language,
        sample_rate_hz=session.config.sample_rate_hz,
    )
    stt_ms = round((time.perf_counter() - t) * 1000)
    return transcript, detected, stt_ms


async def stream_tts(
    bundle: ServingBundle,
    session: VoiceSession,
    turn_id: int,
    text: str,
    language: str,
) -> AsyncIterator[dict]:
    if hasattr(bundle.tts, "synthesize_stream"):
        start = time.perf_counter()
        stream = bundle.tts.synthesize_stream(text, language=language)

        def _next():
            try:
                return next(stream)
            except StopIteration:
                return None

        index = 0
        pending = None
        tts_first_ms = None
        while True:
            chunk = await asyncio.to_thread(_next)
            if turn_id != session.turn_id:
                stream.close()
                return
            if chunk is None:
                break
            if tts_first_ms is None:
                tts_first_ms = round((time.perf_counter() - start) * 1000)
            if pending is not None:
                event = pending.event(turn_id, chunk_index=index, final=False, text=text if index == 0 else None)
                if index == 0:
                    event["tts_first_ms"] = tts_first_ms
                yield event
                index += 1
            pending = chunk
        if pending is not None:
            event = pending.event(turn_id, chunk_index=index, final=True, text=text if index == 0 else None)
            if index == 0:
                event["tts_first_ms"] = tts_first_ms
            yield event
        return

    sentences = split_sentences(text)
    total = len(sentences)
    for index, sentence in enumerate(sentences):
        audio_response = await asyncio.to_thread(bundle.tts.synthesize, sentence, language=language)
        if turn_id != session.turn_id:
            return
        yield audio_response.event(turn_id, chunk_index=index, final=index == total - 1, text=sentence)
