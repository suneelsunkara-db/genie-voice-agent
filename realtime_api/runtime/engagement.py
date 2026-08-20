"""Spoken engagement during long governed work.

This policy owns WHEN the agent may speak while a tool is still working. It does
not invent progress or findings: each stage says only what the runtime knows
(``accepted``, ``still waiting``, ``taking longer``). Product-specific wording and
translation stay in the voice pipeline.

The cadence opens quickly (an acknowledgment inside a second, then three closely
spaced reassurances) and then settles into a slow heartbeat for as long as the
work legitimately runs. Governed workspace reads are measured in minutes, so a
cadence that stopped after the third moment left the caller in silence for most
of the wait — which reads as a hung call. The heartbeat is deliberately sparse
and bounded so it stays a sign of life rather than a hold-message loop.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import cycle, islice


class EngagementKind(str, Enum):
    ACK = "ack"
    PROGRESS = "progress"


@dataclass(frozen=True)
class EngagementStage:
    key: str
    due_s: float
    kind: EngagementKind


OPENING_STAGES: tuple[EngagementStage, ...] = (
    EngagementStage("ack", 0.8, EngagementKind.ACK),
    EngagementStage("progress_1", 12.0, EngagementKind.PROGRESS),
    EngagementStage("progress_2", 30.0, EngagementKind.PROGRESS),
    EngagementStage("progress_3", 55.0, EngagementKind.PROGRESS),
)

# Seconds between heartbeats once the opening cadence is spent.
HEARTBEAT_EVERY_S = 30.0
# Ceiling on heartbeats, so a genuinely stuck upstream cannot talk forever.
MAX_HEARTBEATS = 12
# Heartbeats reuse the opening progress wording, cycling so the caller does not
# hear the same sentence twice in a row. Reusing the keys also means they hit the
# phrase cache the opening cadence already warmed — no extra generation mid-turn.
_HEARTBEAT_KEYS: tuple[str, ...] = ("progress_2", "progress_3")
# A real upstream step ("running the query") may also be narrated, one sentence per
# distinct step. Matches the step ceiling in runtime.genie_one.
MAX_UPSTREAM_STEP_NARRATIONS = 12


def _heartbeats() -> tuple[EngagementStage, ...]:
    last_opening = OPENING_STAGES[-1].due_s
    keys = islice(cycle(_HEARTBEAT_KEYS), MAX_HEARTBEATS)
    return tuple(
        EngagementStage(key, last_opening + HEARTBEAT_EVERY_S * (n + 1), EngagementKind.PROGRESS)
        for n, key in enumerate(keys)
    )


DEFAULT_STAGES: tuple[EngagementStage, ...] = OPENING_STAGES + _heartbeats()

# Every spoken progress moment a single turn may spend. The speech scheduler takes
# its budget from here so a cadence this policy allows cannot be silently dropped
# by a budget that disagrees with it.
MAX_PROGRESS_MOMENTS = (
    sum(1 for stage in OPENING_STAGES if stage.kind is EngagementKind.PROGRESS)
    + MAX_HEARTBEATS
    + MAX_UPSTREAM_STEP_NARRATIONS
)


class LongWorkEngagement:
    """Return each due engagement stage exactly once, in order."""

    def __init__(
        self,
        stages: tuple[EngagementStage, ...] = DEFAULT_STAGES,
    ) -> None:
        self._stages = stages
        self._next = 0

    @property
    def exhausted(self) -> bool:
        return self._next >= len(self._stages)

    @property
    def next_due_s(self) -> float | None:
        if self.exhausted:
            return None
        return self._stages[self._next].due_s

    def pop_due(self, elapsed_s: float) -> list[EngagementStage]:
        """Return all stages due by ``elapsed_s`` and mark them emitted.

        Only the most recent stage is returned when several came due at once: after
        a stall the caller wants one current sentence, not a backlog read out
        back-to-back.
        """
        due: list[EngagementStage] = []
        while not self.exhausted and self._stages[self._next].due_s <= elapsed_s:
            due.append(self._stages[self._next])
            self._next += 1
        return due[-1:] if len(due) > 1 else due

    def defer(self, elapsed_s: float, quiet_for_s: float = HEARTBEAT_EVERY_S) -> None:
        """Skip stages that fall inside ``quiet_for_s`` from ``elapsed_s``.

        Called when something more specific was just spoken (a real upstream step),
        so the generic heartbeat does not immediately talk over it.
        """
        until = elapsed_s + quiet_for_s
        while not self.exhausted and self._stages[self._next].due_s <= until:
            self._next += 1
