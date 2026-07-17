"""Realtime session state and turn orchestration."""
from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from typing import AsyncIterator

from .contracts import SessionStart
from .services import LanguageModel, SpeechToText, TextToSpeech


# Sentence terminators for Latin and CJK scripts. Splitting the reply into
# sentences lets us synthesize + stream the first sentence while later ones are
# still generating, which slashes time-to-first-audio for the realtime UX.
_SENTENCE_RE = re.compile(r"[^.!?。！？…\n]+(?:[.!?。！？…]+|\n|$)", re.UNICODE)


@dataclass
class VoiceSession:
    config: SessionStart
    turn_id: int = 0
    audio: bytearray = field(default_factory=bytearray)
    speech_active: bool = False
    silence_ms: float = 0.0
    turn_audio_ms: float = 0.0
    voiced_ms: float = 0.0
    # True from the moment a turn is finalized until its reply fully drains. While
    # busy, incoming mic audio (the tail of the user's speech, or speaker->mic
    # echo of the assistant) must NOT finalize a new turn and cancel the reply.
    busy: bool = False

    def add_audio(self, frame: bytes) -> bool:
        """Append PCM audio and report whether this frame begins speech."""
        if len(frame) % 2:
            raise ValueError("PCM s16le audio frame must contain an even number of bytes")
        frame_ms = len(frame) / 2 / self.config.sample_rate_hz * 1000
        rms = _pcm_s16le_rms(frame)
        began_speech = bool(frame) and not self.speech_active and rms >= 250
        if rms >= 250:
            self.speech_active = True
            self.silence_ms = 0.0
            self.voiced_ms += frame_ms
        elif self.speech_active:
            self.silence_ms += frame_ms
        self.turn_audio_ms += frame_ms
        self.audio.extend(frame)
        return began_speech

    def should_finalize(self, *, silence_ms: int, max_turn_seconds: int, min_speech_ms: int = 0) -> bool:
        # Require a minimum amount of *voiced* audio so brief blips/echo don't
        # trigger a full STT->LLM->TTS turn. The hard max-turn cap still applies.
        if self.turn_audio_ms >= max_turn_seconds * 1000:
            return True
        return (
            self.speech_active
            and self.voiced_ms >= min_speech_ms
            and self.silence_ms >= silence_ms
        )

    def finish_turn(self) -> tuple[int, bytes] | None:
        if not self.audio:
            return None
        self.turn_id += 1
        audio = bytes(self.audio)
        self._reset_turn()
        return self.turn_id, audio

    def barge_in(self) -> int:
        self.turn_id += 1
        self._reset_turn()
        return self.turn_id

    def discard_buffer(self) -> None:
        """Drop audio captured while busy (echo/tail) without bumping turn_id."""
        self._reset_turn()

    def _reset_turn(self) -> None:
        self.audio.clear()
        self.speech_active = False
        self.silence_ms = 0.0
        self.turn_audio_ms = 0.0
        self.voiced_ms = 0.0


@dataclass(frozen=True)
class VoicePipeline:
    stt: SpeechToText
    llm: LanguageModel
    tts: TextToSpeech
    # In verify mode the assistant does NOT answer the question; it reports how it
    # processed the audio (identified language + verbatim transcript) and speaks
    # that back, so the language ID and transcription can be confirmed by ear/eye.
    verify_mode: bool = False

    async def process_turn(
        self, session: VoiceSession, turn_id: int, audio: bytes
    ) -> AsyncIterator[dict]:
        """Stream turn events as they are produced.

        Emits transcript -> response text -> one ``response.audio`` chunk per
        sentence (final flag on the last) so the client can start playback on the
        first sentence. Yields nothing further if the turn is superseded
        (barge-in / new turn), which lets the caller cancel cheaply.
        """
        pref = session.config.language
        stt_language = None if (not pref or pref == "auto") else pref
        sample_rate_hz = session.config.sample_rate_hz
        t = time.perf_counter()
        transcript, detected = await asyncio.to_thread(
            self.stt.transcribe, audio, language=stt_language, sample_rate_hz=sample_rate_hz
        )
        stt_ms = round((time.perf_counter() - t) * 1000)
        if turn_id != session.turn_id:
            return
        # Drop empty transcripts (silence/noise) quietly instead of driving the
        # LLM/TTS with nothing.
        if not transcript.strip():
            return
        # Follow the detected language downstream (falls back to the pref/English).
        language = detected or stt_language or "en-US"
        yield {
            "type": "transcript.final", "turn_id": turn_id, "text": transcript,
            "language": language, "stt_ms": stt_ms,
        }

        t = time.perf_counter()
        if self.verify_mode:
            response_text = _verify_message(transcript, language)
        else:
            response_text = await asyncio.to_thread(self.llm.respond, transcript, language=language)
        llm_ms = round((time.perf_counter() - t) * 1000)
        if turn_id != session.turn_id:
            return
        yield {
            "type": "response.text", "turn_id": turn_id, "text": response_text,
            "language": language, "llm_ms": llm_ms,
        }

        if hasattr(self.tts, "synthesize_stream"):
            # Realtime path: stream PCM chunks as VoxCPM2 generates them so the
            # client starts playback in a few hundred ms instead of ~3 s.
            async for event in self._stream_audio(session, turn_id, response_text, language):
                yield event
            return

        # Fallback: one WAV per sentence (endpoints without predict_stream).
        sentences = _split_sentences(response_text)
        total = len(sentences)
        for index, sentence in enumerate(sentences):
            audio_response = await asyncio.to_thread(self.tts.synthesize, sentence, language=language)
            if turn_id != session.turn_id:
                return
            yield audio_response.event(
                turn_id, chunk_index=index, final=index == total - 1, text=sentence
            )

    async def _stream_audio(
        self, session: VoiceSession, turn_id: int, text: str, language: str
    ) -> AsyncIterator[dict]:
        """Forward streamed TTS chunks, tagging only the last chunk ``final``.

        A one-chunk lookahead is used so ``final`` marks the true end of the turn.
        Each ``next()`` runs in a worker thread to keep the event loop free, and
        the generator is closed early if the turn is superseded (barge-in).
        """
        start = time.perf_counter()
        stream = self.tts.synthesize_stream(text, language=language)

        def _next():
            try:
                return next(stream)
            except StopIteration:
                return None

        index = 0
        pending = None
        tts_first_ms = None  # API-observed time to the first audio chunk
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


def _verify_message(transcript: str, language: str) -> str:
    """Echo the transcript verbatim (the detected language is shown separately)."""
    return transcript


def _split_sentences(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    sentences = [match.group().strip() for match in _SENTENCE_RE.finditer(text)]
    sentences = [s for s in sentences if s]
    return sentences or [text]


def _pcm_s16le_rms(frame: bytes) -> float:
    samples = [
        int.from_bytes(frame[index : index + 2], byteorder="little", signed=True)
        for index in range(0, len(frame), 2)
    ]
    if not samples:
        return 0.0
    return (sum(sample * sample for sample in samples) / len(samples)) ** 0.5
