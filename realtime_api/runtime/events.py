"""Ordered AgentEvent stream for the Progressive Turn Runtime."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterator


class AgentEventKind(str, Enum):
    TURN_ACCEPTED = "turn.accepted"
    ACTION_STARTED = "action.started"
    ACTION_COMPLETED = "action.completed"
    EVIDENCE_AVAILABLE = "evidence.available"
    ANSWER_PREVIEW = "answer.preview"
    ANSWER_FINAL = "answer.final"
    TURN_COMPLETED = "turn.completed"
    TURN_CANCELLED = "turn.cancelled"
    TURN_FAILED = "turn.failed"


# Stable emission order for a healthy progressive turn (cancel/fail may cut early).
EVENT_ORDER: tuple[AgentEventKind, ...] = (
    AgentEventKind.TURN_ACCEPTED,
    AgentEventKind.ACTION_STARTED,
    AgentEventKind.ACTION_COMPLETED,
    AgentEventKind.EVIDENCE_AVAILABLE,
    AgentEventKind.ANSWER_PREVIEW,
    AgentEventKind.ANSWER_FINAL,
    AgentEventKind.TURN_COMPLETED,
)


@dataclass
class AgentEvent:
    kind: AgentEventKind
    turn_id: int
    seq: int
    payload: dict[str, Any] = field(default_factory=dict)

    def envelope(self) -> dict[str, Any]:
        """Wire payload: ordered ``turn.event`` envelope for UI / MCP clients."""
        return turn_event_envelope(self)


def turn_event_envelope(event: AgentEvent) -> dict[str, Any]:
    """Build the ordered ``turn.event`` envelope (seq is authoritative)."""
    return {
        "type": "turn.event",
        "turn_id": event.turn_id,
        "seq": event.seq,
        "kind": event.kind.value,
        "payload": dict(event.payload),
    }


class EventSequencer:
    """Assigns monotonically increasing ``seq`` within one turn."""

    def __init__(self, turn_id: int) -> None:
        self.turn_id = turn_id
        self._seq = 0

    def emit(self, kind: AgentEventKind, payload: dict[str, Any] | None = None) -> AgentEvent:
        self._seq += 1
        return AgentEvent(
            kind=kind,
            turn_id=self.turn_id,
            seq=self._seq,
            payload=dict(payload or {}),
        )

    def envelopes(self, events: Iterator[AgentEvent]) -> list[dict[str, Any]]:
        return [e.envelope() for e in events]
