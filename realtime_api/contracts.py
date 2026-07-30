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


# Upper bound on session-provided textual context, to keep the LLM prompt sane.
_MAX_CONTEXT_CHARS = 8_000


@dataclass(frozen=True)
class SessionStart:
    language: str
    sample_rate_hz: int
    encoding: Literal["pcm_s16le"] = "pcm_s16le"
    max_turn_seconds: int | None = None
    vad_silence_ms: int | None = None
    context: str | None = None
    call_id: str | None = None
    customer_id: str | None = None
    # The language the agent *selected* in the UI. STT still auto-detects per
    # turn (``language`` stays "auto"); this is only used to warn the caller when
    # the detected speech doesn't match what they picked. None disables the check.
    expected_language: str | None = None
    # Assistant profile name. None (default) runs the built-in telco contact-center
    # behavior; any other value selects a registered VoiceProfile (see profiles.py).
    # The engine stays domain-agnostic — it never names a specific profile here.
    profile: str | None = None

    @classmethod
    def from_event(cls, payload: dict) -> "SessionStart":
        # "auto" (the default) lets STT detect the spoken language per turn.
        language = str(payload.get("language") or "auto")
        if language != "auto" and not LANGUAGE_TAG_RE.fullmatch(language):
            raise ValueError("language must be 'auto' or a BCP 47 tag, for example en-US or th-TH")
        expected_language = _optional_language_tag(payload.get("expected_language"))
        # Accept any plausible capture rate (the browser reports its actual
        # AudioContext rate, which may be 44100/48000 on devices that ignore a
        # 16 kHz request). The server resamples to 16 kHz where a fixed rate is
        # required (VAD/endpointing); STT receives the true rate.
        sample_rate_hz = int(payload.get("sample_rate_hz") or 16_000)
        if not 8_000 <= sample_rate_hz <= 192_000:
            raise ValueError("sample_rate_hz must be between 8000 and 192000")
        encoding = str(payload.get("encoding") or "pcm_s16le")
        if encoding != "pcm_s16le":
            raise ValueError("Only pcm_s16le input is supported in v1")
        profile = _optional_str(payload.get("profile"))
        if profile is not None:
            from .profiles import profile_names

            known = profile_names()
            if profile not in known:
                allowed = ", ".join(sorted(known)) or "(none registered)"
                raise ValueError(f"unknown profile {profile!r}; registered profiles: {allowed}")
        return cls(
            language=language,
            sample_rate_hz=sample_rate_hz,
            encoding=encoding,
            max_turn_seconds=_optional_positive_int(payload.get("max_turn_seconds"), "max_turn_seconds"),
            vad_silence_ms=_optional_positive_int(payload.get("vad_silence_ms"), "vad_silence_ms"),
            context=_optional_context(payload.get("context")),
            call_id=_optional_str(payload.get("call_id")),
            customer_id=_optional_str(payload.get("customer_id")),
            expected_language=expected_language,
            profile=profile,
        )


def _optional_positive_int(value: object, field: str) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ValueError(f"{field} must be a positive integer")
    if parsed <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return parsed


def _optional_context(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if len(text) > _MAX_CONTEXT_CHARS:
        raise ValueError(f"context must be at most {_MAX_CONTEXT_CHARS} characters")
    return text


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _optional_language_tag(value: object) -> str | None:
    """Accept a BCP 47 tag / base code (e.g. 'en' or 'en-US'); ignore 'auto'/blank."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text == "auto":
        return None
    if not LANGUAGE_TAG_RE.fullmatch(text):
        raise ValueError("expected_language must be a BCP 47 tag, for example en-US or th")
    return text
