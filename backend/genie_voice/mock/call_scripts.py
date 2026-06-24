"""Vendor-neutral mock call scripts.

A script is a list of turns: {"speaker": "agent"|"customer", "text": str}.

Scripts are no longer hand-written: they are produced by the enterprise data
generator (`genie_voice.datagen`) so each conversation references a REAL
customer, agent, and invoice with consistent amounts/dates. The same generated
dataset also populates the structured Delta tables, which is what guarantees the
speech <-> text <-> table relationships Genie depends on.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any


@lru_cache(maxsize=1)
def _scripts() -> list[dict[str, Any]]:
    from genie_voice.datagen import build_dataset

    return build_dataset().call_scripts()


def get_scripts() -> list[dict[str, Any]]:
    return _scripts()


def get_script(call_id: str) -> dict[str, Any] | None:
    return next((s for s in _scripts() if s["call_id"] == call_id), None)
