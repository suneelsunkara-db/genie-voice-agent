"""Realtime session state (VAD buffering and turn lifecycle)."""
from __future__ import annotations

from dataclasses import dataclass, field

from .contracts import SessionStart


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


def _pcm_s16le_rms(frame: bytes) -> float:
    samples = [
        int.from_bytes(frame[index : index + 2], byteorder="little", signed=True)
        for index in range(0, len(frame), 2)
    ]
    if not samples:
        return 0.0
    return (sum(sample * sample for sample in samples) / len(samples)) ** 0.5
