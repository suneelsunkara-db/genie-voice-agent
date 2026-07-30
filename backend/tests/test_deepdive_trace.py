"""Unit tests for the Agent-Mode deep-dive tracing helpers.

The deep-dive lane (Genie Agent Mode) is traced with the same ``TurnTrace``
machinery as the voice turns so an investigation shows up in Trace Explorer
linked to the call. These tests lock the pure event→span folding without
touching Databricks or the SSE transport.
"""
from __future__ import annotations

import pytest

pytest.importorskip("realtime_api.tracing")

from realtime_api.deep_dive import (  # noqa: E402
    DEEPDIVE_CAPABILITY,
    _norm_status,
    new_deepdive_trace,
    record_deepdive_event,
)


def test_norm_status_maps_agent_mode_vocabulary():
    assert _norm_status("completed") == "ok"
    assert _norm_status("COMPLETED") == "ok"
    assert _norm_status("success") == "ok"
    assert _norm_status(None) == "ok"
    assert _norm_status("failed") == "error"
    assert _norm_status("error") == "error"
    assert _norm_status("running") == "running"


def test_trace_folds_reasoning_sql_and_report():
    trace = new_deepdive_trace(
        "Why did my expenses jump this month?",
        call_id="card-CH-0001-123",
        session_id="sess-abc",
        customer_id="CH-0001",
    )
    events = [
        {"kind": "reasoning", "text": "Comparing this cycle to prior months"},
        {"kind": "sql", "sql": "SELECT category, SUM(amount) FROM txns ..."},
        {"kind": "sql", "sql": "SELECT * FROM statements ..."},
        {
            "kind": "report",
            "status": "completed",
            "report": "Your travel spend rose $1,800 vs your typical month.",
            "tables": [{"a": 1}],
            "sql": ["q1", "q2"],
            "reasoning": ["step"],
        },
    ]
    for ev in events:
        record_deepdive_event(trace, ev)

    d = trace.to_dict()
    assert d["capability"] == DEEPDIVE_CAPABILITY
    assert d["call_id"] == "card-CH-0001-123"
    assert d["session_id"] == "sess-abc"
    assert d["customer_id"] == "CH-0001"
    assert d["input_transcript"].startswith("Why did my expenses")
    assert d["output_text"].startswith("Your travel spend")
    assert d["status"] == "ok"

    kinds = [s["kind"] for s in d["spans"]]
    assert kinds.count("TOOL") == 2   # two SQL queries as provenance spans
    assert kinds.count("LLM") == 2    # one reasoning step + the report span

    # The two SQL statements are captured as span inputs (provenance).
    sql_inputs = [s["input"] for s in d["spans"] if s["kind"] == "TOOL"]
    assert any("txns" in (i or "") for i in sql_inputs)


def test_trace_marks_error():
    trace = new_deepdive_trace("q", call_id=None, session_id=None, customer_id=None)
    record_deepdive_event(trace, {"kind": "error", "error": {"message": "boom"}})
    d = trace.to_dict()
    assert d["status"] == "error"
    assert "boom" in (d["error"] or "")
