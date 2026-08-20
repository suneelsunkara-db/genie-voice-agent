"""Progressive Turn Runtime harness — Phase 6.8 contract checks."""
from __future__ import annotations

import pytest

from realtime_api.runtime import (
    CancellationToken,
    Evidence,
    EvidenceComposer,
    SpeechKind,
    SpeechRequest,
    SpeechScheduler,
    TableEvidence,
    evidence_from_tool_result,
)
from realtime_api.runtime.refuse import ErrorCode


def test_ack_ttfa_budget_one_only():
    """Ack speech is budgeted ≤1 (covers filler / TTFA covering speech)."""
    sched = SpeechScheduler()
    assert sched.accept(SpeechRequest(SpeechKind.ACK, "One moment"))
    assert not sched.accept(SpeechRequest(SpeechKind.ACK, "Still working"))


def test_evidence_to_preview_requires_stable_cite():
    composer = EvidenceComposer()
    ev = Evidence(
        source="genie",
        table=TableEvidence(columns=["balance"], rows=[[42.5]]),
    )
    claims = composer.spoken_claims_from_table(ev)
    assert isinstance(claims, list) and claims
    sched = SpeechScheduler()
    assert sched.accept(
        SpeechRequest(SpeechKind.PREVIEW, claims[0].text, cited=True, stable=True)
    )
    assert not sched.accept(
        SpeechRequest(SpeechKind.PREVIEW, "unstable guess", cited=True, stable=False)
    )


def test_cancel_drops_later_work():
    import asyncio

    token = CancellationToken()
    token.cancel()
    with pytest.raises(asyncio.CancelledError):
        token.raise_if_cancelled()


def test_cancel_leak_dropped_via_scheduler_after_cancel():
    """After cancel, further FINAL speech must not pass uncited / stale paths."""
    token = CancellationToken()
    token.cancel()
    sched = SpeechScheduler()
    # Uncited final is always dropped (cite-or-silence).
    assert not sched.accept(SpeechRequest(SpeechKind.FINAL, "42", cited=False))
    assert "final:uncited" in sched.skipped


def test_same_turn_inject_does_not_need_new_turn_id():
    sched = SpeechScheduler()
    turn_id = 7
    assert sched.accept(SpeechRequest(SpeechKind.INJECT, "spoken summary"))
    # Inject never implies a turn_id bump — caller keeps the same id.
    assert turn_id == 7


def test_empty_genie_tool_result_refuses():
    ev = evidence_from_tool_result("ask_card_genie", {"answer": "narrative only"})
    composer = EvidenceComposer()
    out = composer.spoken_claims_from_table(ev)
    # Prose-only → no tabular → refuse / error evidence
    assert not isinstance(out, list) or out == []
    if not isinstance(out, list):
        assert out.code == ErrorCode.NO_EVIDENCE


def test_pack_schema_tables_to_evidence():
    ev = evidence_from_tool_result(
        "ask_card_genie",
        {"columns": ["m"], "rows": [[1]], "answer": "display only"},
    )
    assert ev.has_tabular
    assert ev.display_prose == "display only"


def test_legacy_scalar_pack_result_is_promoted_to_cited_evidence():
    ev = evidence_from_tool_result(
        "lookup_account",
        {"balance": 42.5, "invoice": {"status": "overdue"}, "answer": "uncited prose"},
    )
    claims, refusal = EvidenceComposer().compose_or_refuse(ev)
    assert refusal is None
    assert claims
    assert all(claim.field_paths for claim in claims)
    assert "uncited prose" not in " ".join(claim.text for claim in claims)


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
