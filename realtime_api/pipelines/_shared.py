"""Shared helpers for voice pipelines."""
from __future__ import annotations

import asyncio
import base64
import io
import re
import time
import wave
from typing import AsyncIterator

from ..session import VoiceSession
from . import ServingBundle

_SENTENCE_RE = re.compile(r"[^.!?。！？…\n]+(?:[.!?。！？…]+|\n|$)", re.UNICODE)

# Seconds of the first turn's audio captured as the session's voice reference.
# 4 s is ample for VoxCPM2 timbre cloning while keeping the per-turn payload
# (the reference is re-sent to the endpoint on every later turn) small.
_VOICE_REFERENCE_SECONDS = 4.0


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
    *,
    mark_final: bool = True,
    emit_text: bool = True,
) -> AsyncIterator[dict]:
    """Stream TTS audio for `text` as response.audio events.

    mark_final=False keeps every chunk non-terminal so a preceding filler clip
    doesn't signal turn completion before the real answer streams. emit_text=False
    suppresses the transcript text carried on the first chunk (used for fillers,
    which shouldn't appear as an agent turn).
    """
    reference_b64 = session.voice_reference_b64
    # Capture this turn's audio only until the session has a locked-in voice.
    capture: bytearray | None = bytearray() if reference_b64 is None else None
    capture_sr = 48_000

    if hasattr(bundle.tts, "synthesize_stream"):
        start = time.perf_counter()
        stream = bundle.tts.synthesize_stream(text, language=language, reference_audio_b64=reference_b64)

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
            if capture is not None:
                capture_sr = chunk.sample_rate_hz
                if len(capture) < int(capture_sr * 2 * _VOICE_REFERENCE_SECONDS):
                    capture.extend(chunk.pcm)
            if pending is not None:
                first_text = text if (emit_text and index == 0) else None
                event = pending.event(turn_id, chunk_index=index, final=False, text=first_text)
                if index == 0:
                    event["tts_first_ms"] = tts_first_ms
                yield event
                index += 1
            pending = chunk
        if pending is not None:
            first_text = text if (emit_text and index == 0) else None
            event = pending.event(turn_id, chunk_index=index, final=mark_final, text=first_text)
            if index == 0:
                event["tts_first_ms"] = tts_first_ms
            yield event
        _lock_voice_reference(session, bytes(capture) if capture is not None else b"", capture_sr)
        return

    sentences = split_sentences(text)
    total = len(sentences)
    for index, sentence in enumerate(sentences):
        audio_response = await asyncio.to_thread(
            bundle.tts.synthesize, sentence, language=language, reference_audio_b64=reference_b64
        )
        if turn_id != session.turn_id:
            return
        # The non-streaming path already returns a full WAV; lock it as-is.
        if session.voice_reference_b64 is None and audio_response.audio:
            session.voice_reference_b64 = base64.b64encode(audio_response.audio).decode("ascii")
            reference_b64 = session.voice_reference_b64
        is_last = index == total - 1
        yield audio_response.event(
            turn_id,
            chunk_index=index,
            final=is_last and mark_final,
            text=sentence if emit_text else None,
        )


def _lock_voice_reference(session: VoiceSession, pcm: bytes, sample_rate_hz: int) -> None:
    """Wrap the first turn's PCM as a WAV and store it as the session's voice.

    No-op once a reference exists (locked once, reused for the whole call) or
    when the turn produced no audio (retry on the next turn).
    """
    if session.voice_reference_b64 is not None or not pcm:
        return
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate_hz)
        handle.writeframes(pcm)
    session.voice_reference_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
