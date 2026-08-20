"""Speech budget scheduler — ack/progress/preview/final with skip rules."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator

from .engagement import MAX_PROGRESS_MOMENTS


class SpeechKind(str, Enum):
    ACK = "ack"
    PROGRESS = "progress"
    PREVIEW = "preview"
    FINAL = "final"
    INJECT = "inject"  # same-turn synthesize (deep-dive spoken summary, etc.)


@dataclass(frozen=True)
class SpeechRequest:
    kind: SpeechKind
    text: str
    cited: bool = False
    stable: bool = True


@dataclass
class SpeechScheduler:
    """Enforces progressive speech budget for one turn.

    Rules (L200):
      - ack ≤ 1
      - progress ≤ the long-work cadence's own ceiling (runtime.engagement)
      - preview only when evidence is stable
      - final only when cited (cite-or-silence)
      - inject always allowed (same-turn synthesize) but never bumps turn_id
    """

    ack_emitted: int = 0
    progress_emitted: int = 0
    progress_budget: int = MAX_PROGRESS_MOMENTS
    preview_emitted: int = 0
    final_emitted: int = 0
    inject_emitted: int = 0
    skipped: list[str] = field(default_factory=list)

    def accept(self, req: SpeechRequest) -> bool:
        """Return True if this speech should be spoken; False to skip."""
        text = (req.text or "").strip()
        if not text:
            self.skipped.append(f"{req.kind.value}:empty")
            return False

        if req.kind == SpeechKind.ACK:
            if self.ack_emitted >= 1:
                self.skipped.append("ack:budget")
                return False
            self.ack_emitted += 1
            return True

        if req.kind == SpeechKind.PROGRESS:
            if self.progress_emitted >= self.progress_budget:
                self.skipped.append("progress:budget")
                return False
            self.progress_emitted += 1
            return True

        if req.kind == SpeechKind.PREVIEW:
            if not req.stable:
                self.skipped.append("preview:unstable")
                return False
            self.preview_emitted += 1
            return True

        if req.kind == SpeechKind.FINAL:
            if not req.cited:
                self.skipped.append("final:uncited")
                return False
            if self.final_emitted >= 1:
                self.skipped.append("final:budget")
                return False
            self.final_emitted += 1
            return True

        if req.kind == SpeechKind.INJECT:
            self.inject_emitted += 1
            return True

        self.skipped.append(f"{req.kind.value}:unknown")
        return False

    def drain(self, requests: Iterator[SpeechRequest]) -> list[SpeechRequest]:
        return [r for r in requests if self.accept(r)]
