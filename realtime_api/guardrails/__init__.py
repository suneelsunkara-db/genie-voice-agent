"""Voice guardrails: the shared ledger now, the execution engine in Phase 1.

See docs/guardrails-and-observability.md. Phase 0 adds no guards and changes no
behavior — it makes the checks that ALREADY run report what they did, including
the declines that currently record nothing at all.
"""
from __future__ import annotations

from .ledger import GuardEntry, GuardLedger, report

__all__ = ["GuardEntry", "GuardLedger", "report"]
