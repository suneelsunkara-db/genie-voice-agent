"""The guardrail ledger: one row per check, per turn.

Every guardrail records what it did — including when it found nothing and when it
deliberately did not run. That asymmetry is the whole point. A UI built only on
incidents shows an empty list and proves nothing; the statement worth making to an
operator is "23 checks ran on this turn, 2 were delegated to Qwen3-ASR, none
fired." Only ``fired`` entries additionally get a span, so a full roster costs a
fraction of a KB and leaves the latency waterfall readable.

This module is deliberately dependency-free (no config, no serving, no tracing
import): guards on the hot path call ``report`` and nothing else, and it stays
cheap enough to call unconditionally.

See docs/guardrails-and-observability.md §4.2.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import threading
from typing import Any, Literal

# execution = permits / modifies / blocks content (PII, injection, output checks).
# decision  = chooses an action (route to a tool, block-and-speak); it keeps its own
#             logic and only reports here. See §4.1 for why these stay separate.
Seam = Literal["execution", "decision"]

Stage = Literal["stt", "input_transcript", "routing", "observer_async", "pre_tts", "turn"]

# The UI shows guardrails; turn-integrity mechanics (empty transcript, noise
# timeout, stale turn, cooldown) are real checks but not guardrails, so they are
# tagged here rather than filtered by a denylist in the frontend (§7.0).
Surface = Literal["guardrail", "internal"]

Owner = Literal["us", "qwen"]

Outcome = Literal["passed", "fired", "delegated", "not_evaluated", "disabled", "error"]


@dataclass(frozen=True)
class GuardEntry:
    guard_id: str
    seam: Seam
    stage: Stage
    outcome: Outcome
    surface: Surface = "guardrail"
    owner: Owner = "us"
    latency_ms: float = 0.0
    # Human-readable and REDACTED. Never put raw transcript or PII here: the
    # roster is persisted to Lakebase and mirrored to Delta.
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "guard_id": self.guard_id,
            "seam": self.seam,
            "stage": self.stage,
            "surface": self.surface,
            "owner": self.owner,
            "outcome": self.outcome,
            "latency_ms": round(self.latency_ms, 3),
        }
        if self.reason:
            out["reason"] = self.reason
        return out


@dataclass
class GuardLedger:
    """Per-turn roster. Thread-safe because observers report off the hot path."""

    _entries: list[GuardEntry] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def report(
        self,
        guard_id: str,
        outcome: Outcome,
        *,
        seam: Seam = "execution",
        stage: Stage = "turn",
        surface: Surface = "guardrail",
        owner: Owner = "us",
        latency_ms: float = 0.0,
        reason: str | None = None,
    ) -> GuardEntry:
        entry = GuardEntry(
            guard_id=guard_id,
            seam=seam,
            stage=stage,
            outcome=outcome,
            surface=surface,
            owner=owner,
            latency_ms=latency_ms,
            reason=reason,
        )
        with self._lock:
            self._entries.append(entry)
        return entry

    @property
    def entries(self) -> list[GuardEntry]:
        with self._lock:
            return list(self._entries)

    def to_list(self) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self.entries]

    def report_entry(self, entry: GuardEntry) -> None:
        with self._lock:
            self._entries.append(entry)

    def summary(self) -> dict[str, int]:
        """Counts per outcome, for the UI headline ("N ran, M delegated, none fired").

        Counts only guardrail-surface rows: an operator reading "0 fired" must not
        have that number quietly include turn-integrity mechanics.
        """
        counts: dict[str, int] = {}
        for entry in self.entries:
            if entry.surface != "guardrail":
                continue
            counts[entry.outcome] = counts.get(entry.outcome, 0) + 1
        return counts


def report(
    ledger: GuardLedger | None,
    guard_id: str,
    outcome: Outcome,
    **kwargs: Any,
) -> None:
    """Report to ``ledger`` if there is one.

    Guards run on paths that sometimes have no trace attached (tests, warmup, the
    standalone realtime app), and a check must never fail because nobody was
    listening — so the null case is handled here instead of at every call site.
    """
    if ledger is not None:
        ledger.report(guard_id, outcome, **kwargs)
