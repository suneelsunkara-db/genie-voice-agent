"""Wire contracts for the standalone websocket voice API."""
from __future__ import annotations

from base64 import b64encode
from dataclasses import dataclass
import re
from typing import Literal


LANGUAGE_TAG_RE = re.compile(r"^[a-z]{2,3}(?:-[A-Za-z0-9]{2,8})*$")


@dataclass(frozen=True)
class AudioResponse:
    audio: bytes
    mime_type: str
    sample_rate_hz: int

    def event(
        self, turn_id: int, *, chunk_index: int = 0, final: bool = True, text: str | None = None
    ) -> dict:
        payload = {
            "type": "response.audio",
            "turn_id": turn_id,
            "chunk_index": chunk_index,
            "final": final,
            "audio_b64": b64encode(self.audio).decode("ascii"),
            "mime_type": self.mime_type,
            "sample_rate_hz": self.sample_rate_hz,
        }
        if text is not None:
            payload["text"] = text
        return payload


@dataclass(frozen=True)
class AudioChunk:
    """A streamed slice of raw PCM s16le audio (no WAV header).

    Emitted by the streaming TTS path so the client can begin playback after the
    first chunk. Chunks in one turn share a sample rate and are concatenated by
    the client in ``chunk_index`` order.
    """

    pcm: bytes
    sample_rate_hz: int
    # Server-reported timing, present only on the last chunk of a stream.
    server_ttfb_ms: float | None = None
    server_gen_ms: float | None = None

    def event(
        self, turn_id: int, *, chunk_index: int = 0, final: bool = False, text: str | None = None
    ) -> dict:
        payload = {
            "type": "response.audio",
            "turn_id": turn_id,
            "chunk_index": chunk_index,
            "final": final,
            "audio_b64": b64encode(self.pcm).decode("ascii"),
            "mime_type": "audio/pcm",
            "encoding": "pcm_s16le",
            "sample_rate_hz": self.sample_rate_hz,
        }
        if text is not None:
            payload["text"] = text
        if self.server_ttfb_ms is not None:
            payload["server_ttfb_ms"] = self.server_ttfb_ms
        if self.server_gen_ms is not None:
            payload["server_gen_ms"] = self.server_gen_ms
        return payload


@dataclass(frozen=True)
class SessionStart:
    language: str
    sample_rate_hz: int
    encoding: Literal["pcm_s16le"] = "pcm_s16le"

    @classmethod
    def from_event(cls, payload: dict) -> "SessionStart":
        # "auto" (the default) lets STT detect the spoken language per turn.
        language = str(payload.get("language") or "auto")
        if language != "auto" and not LANGUAGE_TAG_RE.fullmatch(language):
            raise ValueError("language must be 'auto' or a BCP 47 tag, for example en-US or th-TH")
        sample_rate_hz = int(payload.get("sample_rate_hz") or 16_000)
        if sample_rate_hz not in {8_000, 16_000, 24_000, 48_000}:
            raise ValueError("sample_rate_hz must be one of 8000, 16000, 24000, or 48000")
        encoding = str(payload.get("encoding") or "pcm_s16le")
        if encoding != "pcm_s16le":
            raise ValueError("Only pcm_s16le input is supported in v1")
        return cls(language=language, sample_rate_hz=sample_rate_hz, encoding=encoding)
