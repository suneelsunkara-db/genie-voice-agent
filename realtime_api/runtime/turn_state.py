"""Turn phase and per-turn state for the Progressive Turn Runtime."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .cancellation import CancellationToken


class TurnPhase(str, Enum):
    """Replaces binary session.busy for progressive turns."""

    IDLE = "idle"
    WORKING = "working"
    AWAITING_USER = "awaiting_user"
    COMPLETED = "completed"


@dataclass
class TurnState:
    """Server-side state for one progressive turn."""

    turn_id: int
    phase: TurnPhase = TurnPhase.WORKING
    cancel: CancellationToken = field(default_factory=CancellationToken)
    committed_claims: list[dict[str, Any]] = field(default_factory=list)
    # Opaque work metadata (e.g. agent_mode timeout budget).
    meta: dict[str, Any] = field(default_factory=dict)

    def cancel_turn(self) -> None:
        self.cancel.cancel()
        self.phase = TurnPhase.COMPLETED

    @property
    def busy(self) -> bool:
        return self.phase == TurnPhase.WORKING
