"""Provider interfaces.

A provider encapsulates EVERYTHING vendor-specific:
  - `normalize(raw)`  : map the vendor's native payload -> canonical contract
  - `mock_events(...)`: produce vendor-shaped payloads (deployment=local)
  - `stream(...)`     : connect to the live vendor (deployment=live)

Downstream code only ever sees canonical `TranscriptEvent` / `SpeechResult`,
so swapping a vendor is: implement this interface + register it. Nothing else
in the system changes.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Iterable, Iterator

from genie_voice.models.contracts import SpeechResult, TranscriptEvent


class STTProvider(ABC):
    name: ClassVar[str] = "base"

    def __init__(self, options: dict[str, Any] | None = None) -> None:
        self.options = options or {}

    @abstractmethod
    def normalize(self, raw: dict[str, Any], *, call_id: str) -> TranscriptEvent:
        """Map a single native streaming payload to a canonical event."""

    def mock_events(
        self, script: list[dict[str, Any]], render: dict[str, Any] | None = None
    ) -> Iterator[dict[str, Any]]:
        """Yield vendor-shaped raw payloads for a (vendor-neutral) call script.

        `script` is a list of turns: {"speaker": "agent"|"customer", "text": str}.
        `render` carries mock rendering options (channel map, interim step, ...).
        Default raises; vendors that support local mocking override this.
        """
        raise NotImplementedError(f"{self.name} has no mock generator")

    def stream(self, audio_chunks: Iterable[bytes]) -> Iterator[dict[str, Any]]:
        """Yield raw payloads from the live vendor (deployment=live)."""
        raise NotImplementedError(f"{self.name} live streaming not implemented")


class TTSProvider(ABC):
    name: ClassVar[str] = "base"

    def __init__(self, options: dict[str, Any] | None = None) -> None:
        self.options = options or {}

    @abstractmethod
    def synthesize(self, text: str, *, voice: str | None = None) -> SpeechResult:
        """Convert text to speech, normalized to SpeechResult."""
