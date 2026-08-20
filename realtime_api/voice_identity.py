"""The agent's fixed voice, shared by every session, page and language.

VoxCPM2 clones timbre from a reference clip and exposes no RNG seed, so handing
it the same clip every time is the only way to pin a voice. With no clip it
invents a speaker per synthesis — which is why an unseeded session used to sound
like a different person on every call, and why the greeting on one page never
matched the next.

Seeding sessions from one committed clip also makes the endpoint's per-``voice_id``
cache warm across every session and replica instead of being re-primed per call,
so the clip costs one upload rather than one per session.
"""
from __future__ import annotations

import base64
import hashlib
import logging
import wave
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Long enough to carry a timbre, short enough to stay cheap on the single upload.
_MIN_SECONDS = 2.0
_MAX_SECONDS = 20.0
# Relative clip paths resolve against the repo root (this file is <root>/realtime_api/).
_REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class VoiceSeed:
    """A validated reference clip plus the id the TTS endpoint caches it under."""

    reference_b64: str
    voice_id: str
    seconds: float
    sample_rate_hz: int
    source: str


def voice_id_for(wav_bytes: bytes) -> str:
    """Stable cache key for a reference clip, derived from its own bytes.

    Content-addressed rather than session-scoped so the same voice reuses one
    cache entry, and a changed clip can never collide with the previous one.
    """
    return hashlib.sha256(wav_bytes).hexdigest()[:32]


# Parsed once per path. A None entry means "unusable, already reported", which
# keeps a missing or malformed clip from re-logging on every session.start.
_cache: dict[str, VoiceSeed | None] = {}


def load_voice_seed(path: str | None) -> VoiceSeed | None:
    """Load the configured reference clip, or None to fall back to bootstrapping.

    Returning None (unset path, missing file, wrong format) leaves sessions on the
    legacy behaviour of capturing their own first turn: the voice is then only
    consistent within a call, but synthesis still works. A bad clip must never take
    the agent mute, so every failure here is a warning, not an exception.
    """
    if not path:
        return None
    if path in _cache:
        return _cache[path]
    seed = _load(path)
    _cache[path] = seed
    return seed


def _load(path: str) -> VoiceSeed | None:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = _REPO_ROOT / resolved
    if not resolved.is_file():
        logger.warning(
            "voice reference clip not found at %s — sessions will each bootstrap "
            "their own voice, so the agent will sound different per call",
            resolved,
        )
        return None
    try:
        wav_bytes, seconds, sample_rate_hz = _read_mono_pcm16(resolved)
    except ValueError as exc:
        logger.warning(
            "voice reference clip %s is unusable (%s) — sessions will each bootstrap "
            "their own voice. Convert with: "
            "ffmpeg -i in.wav -ac 1 -ar 48000 -sample_fmt s16 %s",
            resolved,
            exc,
            resolved.name,
        )
        return None
    seed = VoiceSeed(
        reference_b64=base64.b64encode(wav_bytes).decode("ascii"),
        voice_id=voice_id_for(wav_bytes),
        seconds=seconds,
        sample_rate_hz=sample_rate_hz,
        source=str(resolved),
    )
    logger.info(
        "agent voice pinned: voice_id=%s %.2fs %dHz from %s",
        seed.voice_id,
        seed.seconds,
        seed.sample_rate_hz,
        seed.source,
    )
    return seed


def _read_mono_pcm16(path: Path) -> tuple[bytes, float, int]:
    """Return (wav_bytes, seconds, sample_rate) for a mono 16-bit PCM WAV.

    Mono 16-bit is the shape the capture path already produces, so it is the
    only format proven against the endpoint's reference loader.
    """
    wav_bytes = path.read_bytes()
    try:
        with wave.open(str(path), "rb") as handle:
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            rate = handle.getframerate()
            frames = handle.getnframes()
    except wave.Error as exc:
        raise ValueError(f"not a readable WAV: {exc}") from exc
    if channels != 1:
        raise ValueError(f"expected mono, got {channels} channels")
    if width != 2:
        raise ValueError(f"expected 16-bit samples, got {width * 8}-bit")
    if not rate:
        raise ValueError("missing sample rate")
    seconds = frames / rate
    if not _MIN_SECONDS <= seconds <= _MAX_SECONDS:
        raise ValueError(
            f"duration {seconds:.2f}s outside the {_MIN_SECONDS:g}-{_MAX_SECONDS:g}s range"
        )
    return wav_bytes, seconds, rate
