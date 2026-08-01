"""The /traces/guardrails rollup the Guardrails view reads.

Two properties matter more than the arithmetic:

  1. coverage is reported, not just incidents — "checks ran and passed" is the
     compliance statement, and an endpoint that only counted hits would make a
     healthy system look unmonitored;
  2. turn-integrity mechanics (empty transcript, stale turn) are excluded by their
     own declared ``surface``, so "none fired" can't be inflated by a dropped turn.
"""
from __future__ import annotations

import pytest

pytest.importorskip("httpx")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from api.app.routers import traces as traces_router  # noqa: E402

_ROWS = [
    {
        "trace_id": "t1",
        "session_id": "s1",
        "turn_id": 1,
        "language": "en-US",
        "created_at": "2026-07-31T00:00:00Z",
        "guard_roster": [
            {"guard_id": "language_id", "outcome": "delegated", "owner": "qwen", "stage": "stt"},
            {"guard_id": "language_gate", "outcome": "passed", "seam": "decision", "stage": "routing"},
            {"guard_id": "selection_length", "outcome": "passed", "seam": "decision", "stage": "routing"},
            # Not a guardrail: a dropped turn must not count as "checked".
            {"guard_id": "empty_transcript", "outcome": "fired", "surface": "internal", "stage": "turn"},
        ],
    },
    {
        "trace_id": "t2",
        "session_id": "s1",
        "turn_id": 2,
        "language": "hi-IN",
        "created_at": "2026-07-31T00:01:00Z",
        "guard_roster": [
            {"guard_id": "language_id", "outcome": "not_evaluated", "owner": "qwen", "stage": "stt",
             "reason": "session pinned language=hi-IN"},
            {
                "guard_id": "selection_length",
                "outcome": "fired",
                "seam": "decision",
                "stage": "routing",
                "reason": "11 words > 8; deferred to LLM",
            },
        ],
    },
]


@pytest.fixture()
def client(monkeypatch):
    class _FakeServing:
        def list_voice_traces(self, **kwargs):
            return _ROWS

    monkeypatch.setattr(traces_router, "serving", lambda: _FakeServing())
    app = FastAPI()
    app.include_router(traces_router.router)
    return TestClient(app)


def test_rollup_counts_coverage_not_just_incidents(client):
    body = client.get("/traces/guardrails").json()
    assert body["turns"] == 2
    # 3 guardrail rows on t1 (the internal one excluded) + 2 on t2.
    assert body["checks"] == 5
    assert body["totals"] == {"delegated": 1, "passed": 2, "not_evaluated": 1, "fired": 1}


def test_internal_mechanics_are_excluded_by_their_declared_surface(client):
    body = client.get("/traces/guardrails").json()
    assert "empty_transcript" not in {g["guard_id"] for g in body["guards"]}
    assert not any(f["guard_id"] == "empty_transcript" for f in body["recent_fired"])


def test_per_guard_rows_carry_outcomes_and_a_reason(client):
    body = client.get("/traces/guardrails").json()
    guards = {g["guard_id"]: g for g in body["guards"]}
    assert guards["language_id"]["outcomes"] == {"delegated": 1, "not_evaluated": 1}
    assert guards["language_id"]["owner"] == "qwen"
    assert guards["selection_length"]["outcomes"] == {"passed": 1, "fired": 1}
    assert "> 8" in guards["selection_length"]["last_reason"]


def test_fired_rows_deep_link_to_their_trace(client):
    fired = client.get("/traces/guardrails").json()["recent_fired"]
    assert [f["trace_id"] for f in fired] == ["t2"]
    assert fired[0]["turn_id"] == 2 and fired[0]["language"] == "hi-IN"


def test_language_breakdown_separates_pinned_from_detected(client):
    by_language = client.get("/traces/guardrails").json()["by_language"]
    assert by_language["en-US"] == {"delegated": 1, "passed": 2}
    assert by_language["hi-IN"] == {"not_evaluated": 1, "fired": 1}


def test_rollup_is_empty_not_broken_without_any_rosters(monkeypatch):
    class _Empty:
        def list_voice_traces(self, **kwargs):
            return [{"trace_id": "t0", "guard_roster": None}]

    monkeypatch.setattr(traces_router, "serving", lambda: _Empty())
    app = FastAPI()
    app.include_router(traces_router.router)
    body = TestClient(app).get("/traces/guardrails").json()
    assert body["checks"] == 0 and body["checks_per_turn"] == 0.0
    assert body["guards"] == [] and body["recent_fired"] == []


def test_guardrails_route_is_not_swallowed_by_the_trace_id_route(client):
    # /traces/{trace_id} is declared after it; a reorder would turn this rollup
    # into a 404 lookup for a trace literally named "guardrails".
    assert client.get("/traces/guardrails").status_code == 200
