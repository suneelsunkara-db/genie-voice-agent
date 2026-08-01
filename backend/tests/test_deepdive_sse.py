"""Hermetic smoke test for the Genie Agent-Mode deep-dive SSE lane.

Drives GET /card/deepdive end-to-end through the real FastAPI router + SSE
generator + background worker, with Genie Agent Mode and the trace sink MOCKED.
Verifies the streamed event contract (reasoning/sql/report/done, use_case echoed)
AND that a linked trace is submitted for observability — no Databricks needed.
"""
from __future__ import annotations

import json

import pytest

pytest.importorskip("httpx")  # fastapi TestClient transport

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from api.app.routers import card as card_router  # noqa: E402
from realtime_api import deep_dive as deep_dive_mod  # noqa: E402


class _FakeAgent:
    """Stand-in for GenieAgentModeClient that emits a canned investigation."""

    def __init__(self, *a, **k):
        pass

    def ask(self, question, on_event=None, **kwargs):
        for ev in (
            {"kind": "started"},
            {"kind": "reasoning", "text": "Comparing this cycle to prior months"},
            {"kind": "sql", "sql": "SELECT category, SUM(amount) FROM txns"},
            {
                "kind": "report",
                "status": "completed",
                "report": "Your expenses rose $2,450 vs your typical month.",
                "tables": [{"driver": "travel"}],
                "sql": ["SELECT category, SUM(amount) FROM txns"],
                "reasoning": ["Comparing this cycle to prior months"],
            },
        ):
            if on_event:
                on_event(ev)


@pytest.fixture()
def client(monkeypatch):
    import genie_voice.genie.agent_mode as agent_mode
    import realtime_api.tracing as tracing

    monkeypatch.setattr(agent_mode, "GenieAgentModeClient", _FakeAgent)
    captured: list = []
    monkeypatch.setattr(tracing, "submit_trace", lambda t: captured.append(t))
    # Stub BOTH report-rendering LLM calls so the test stays hermetic (no Databricks
    # call): the spoken "why" and the on-screen translation.
    monkeypatch.setattr(
        deep_dive_mod,
        "summarize_deepdive",
        lambda question, report_text, language: f"In short: {report_text} [{language}]",
    )
    monkeypatch.setattr(
        deep_dive_mod,
        "localize_report",
        lambda report_text, language: f"[{language}] {report_text}",
    )

    app = FastAPI()
    app.include_router(card_router.router)
    c = TestClient(app)
    c._captured_traces = captured  # type: ignore[attr-defined]
    return c


def _events(resp) -> list[dict]:
    return [json.loads(line[5:]) for line in resp.text.splitlines() if line.startswith("data:")]


def test_deepdive_streams_contract_and_echoes_use_case(client):
    resp = client.get(
        "/card/deepdive",
        params={
            "question": "Why did my expenses go up this month?",
            "use_case": "statement_insights",
            "call_id": "card-CH-0001-1",
            "session_id": "sess-1",
            "customer_id": "CH-0001",
            "language": "es-ES",
        },
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")

    evs = _events(resp)
    kinds = [e["kind"] for e in evs]
    assert "reasoning" in kinds
    assert "sql" in kinds
    assert "report" in kinds
    assert kinds[-1] == "done"

    # A meta event leads the stream carrying the single-source timeout for the
    # client's stall watchdog (server + client can't disagree).
    meta = next(e for e in evs if e["kind"] == "meta")
    assert isinstance(meta["timeout_ms"], int) and meta["timeout_ms"] > 0

    # use_case is echoed on every agent event for UI routing (not meta/done, which
    # are emitted by the SSE generator itself).
    for e in evs:
        if e["kind"] not in ("done", "meta"):
            assert e.get("use_case") == "statement_insights"

    # The report arrives in two beats: one the caller can hear immediately (the
    # agent's English text + the spoken "why"), then the translation that replaces
    # the on-screen text, so the voice never waits on the translator.
    report = next(e for e in evs if e["kind"] == "report")
    assert "$2,450" in report["report"]
    assert report["report_language"] == "en"
    assert report["localization_pending"] is True
    # The spoken "why" is generated in the caller's language for the client to speak
    # instead of reading the whole report.
    assert "$2,450" in report["spoken_summary"]
    assert "es-ES" in report["spoken_summary"]

    patch = next(e for e in evs if e["kind"] == "report_localized")
    assert patch["report"] == "[es-ES] Your expenses rose $2,450 vs your typical month."
    assert patch["report_language"] == "es-ES"
    assert evs.index(report) < evs.index(patch)


def test_deepdive_submits_linked_trace(client):
    client.get(
        "/card/deepdive",
        params={
            "question": "Why did my expenses go up this month?",
            "use_case": "statement_insights",
            "call_id": "card-CH-0001-2",
            "session_id": "sess-2",
            "customer_id": "CH-0001",
        },
    )
    captured = client._captured_traces  # type: ignore[attr-defined]
    assert len(captured) == 1
    t = captured[0]
    assert t.capability == deep_dive_mod.DEEPDIVE_CAPABILITY
    assert t.call_id == "card-CH-0001-2"
    assert t.session_id == "sess-2"
    assert t.customer_id == "CH-0001"
    assert t.status == "ok"
    assert "expenses rose" in (t.output_text or "").lower()

    d = t.to_dict()
    kinds = [s["kind"] for s in d["spans"]]
    assert kinds.count("TOOL") == 1   # the one SQL query
    assert kinds.count("LLM") == 2    # reasoning + report
