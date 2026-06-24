"""Canonical, vendor-neutral data contracts.

Every STT provider normalizes its native payload into `TranscriptEvent`, and
every TTS provider produces a `SpeechResult`. All downstream code (enrichment,
medallion, serving, API, UI) depends ONLY on these types - never on a vendor's
raw schema. This is what makes Deepgram/ElevenLabs swappable: add an adapter
that maps the new vendor's payload to these shapes and nothing else changes.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class CanonicalWord:
    text: str
    start: float
    end: float
    confidence: float
    speaker: Optional[int] = None


@dataclass
class TranscriptEvent:
    """One streaming transcript event, normalized across vendors.

    `channel` follows dual-channel telephony convention (0=agent, 1=customer),
    so channel doubles as speaker identity when multichannel is used.
    `is_utterance_end` is the vendor-neutral form of Deepgram's `speech_final`.
    """
    call_id: str
    channel: int
    start: float
    end: float
    text: str
    confidence: float
    is_final: bool
    is_utterance_end: bool
    words: list[CanonicalWord] = field(default_factory=list)
    speaker: Optional[int] = None
    provider: str = "unknown"
    # Original vendor payload retained verbatim for bronze/audit/lineage.
    raw: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Utterance:
    """A complete speaker turn (finals concatenated up to utterance end)."""
    call_id: str
    channel: int
    speaker: Optional[int]
    start: float
    end: float
    text: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CharTiming:
    char: str
    start: float
    end: float


@dataclass
class SpeechResult:
    """Normalized TTS output across vendors."""
    text: str
    audio_b64: str
    audio_format: str
    provider: str
    voice: Optional[str] = None
    alignment: list[CharTiming] = field(default_factory=list)
    raw: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
