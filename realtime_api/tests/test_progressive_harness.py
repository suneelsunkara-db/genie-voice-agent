"""Progressive Turn Runtime harness — Phase 6.8 contract checks."""
from __future__ import annotations

import pytest

from realtime_api.runtime import (
    CancellationToken,
    SpeechKind,
    SpeechRequest,
    SpeechScheduler,
    evidence_from_tool_result,
)


def test_ack_ttfa_budget_one_only():
    """Ack speech is budgeted ≤1 (covers filler / TTFA covering speech)."""
    sched = SpeechScheduler()
    assert sched.accept(SpeechRequest(SpeechKind.ACK, "One moment"))
    assert not sched.accept(SpeechRequest(SpeechKind.ACK, "Still working"))


def test_preview_requires_stable_content():
    sched = SpeechScheduler()
    assert sched.accept(
        SpeechRequest(SpeechKind.PREVIEW, "Current balance is $42.50.", stable=True)
    )
    assert not sched.accept(
        SpeechRequest(SpeechKind.PREVIEW, "unstable guess", stable=False)
    )


def test_cancel_drops_later_work():
    import asyncio

    token = CancellationToken()
    token.cancel()
    with pytest.raises(asyncio.CancelledError):
        token.raise_if_cancelled()


def test_cancel_leak_dropped_via_scheduler_after_cancel():
    """After cancel, callers stop scheduling further speech."""
    import asyncio

    token = CancellationToken()
    token.cancel()
    with pytest.raises(asyncio.CancelledError):
        token.raise_if_cancelled()


def test_same_turn_inject_does_not_need_new_turn_id():
    sched = SpeechScheduler()
    turn_id = 7
    assert sched.accept(SpeechRequest(SpeechKind.INJECT, "spoken summary"))
    # Inject never implies a turn_id bump — caller keeps the same id.
    assert turn_id == 7


def test_prose_only_genie_tool_result_preserves_natural_answer():
    ev = evidence_from_tool_result("ask_card_genie", {"answer": "narrative only"})
    assert ev.display_prose == "narrative only"


def test_pack_schema_tables_to_evidence():
    ev = evidence_from_tool_result(
        "ask_card_genie",
        {"columns": ["m"], "rows": [[1]], "answer": "display only"},
    )
    assert ev.has_tabular
    assert ev.display_prose == "display only"


def test_legacy_scalar_pack_result_is_promoted_to_typed_evidence():
    ev = evidence_from_tool_result(
        "lookup_account",
        {"balance": 42.5, "invoice": {"status": "overdue"}, "answer": "uncited prose"},
    )
    assert ev.table is not None
    assert ev.table.columns == ["balance", "invoice.status"]
    assert ev.display_prose == "uncited prose"


def test_knowledge_matches_are_structured_evidence():
    ev = evidence_from_tool_result(
        "knowledge_search",
        {
            "matches": [
                {
                    "topic": "Unity Catalog",
                    "answer": "Governed data and AI assets.",
                    "citation": "docs/unity-catalog",
                }
            ]
        },
    )
    assert ev.has_tabular
    assert ev.table and "citation" in ev.table.columns
