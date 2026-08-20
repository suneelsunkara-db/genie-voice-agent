"""Phase 0 of the guardrail plan: the checks that already run must REPORT.

The valuable assertions here are about declines and passes, not incidents. Before
this, a deterministic route that declined recorded nothing at all — the trace showed
an LLM turn with no explanation of why saying "Telco" didn't route — and a turn where
everything was fine was indistinguishable from a turn where nothing was checked.
"""
from __future__ import annotations

from realtime_api.guardrails import GuardLedger, report
from realtime_api.pipelines import speech_llm_toolassist_speech as pipeline
from realtime_api.tracing import TurnTrace

from .test_app import _app, _drive_turn


def _outcomes(ledger: GuardLedger) -> dict[str, str]:
    return {e.guard_id: e.outcome for e in ledger.entries}


def test_report_without_a_ledger_is_a_noop():
    # Guards also run on paths with no trace (warmup, tests); a check must not fail
    # just because nobody was listening.
    report(None, "selection_length", "passed")


def test_summary_excludes_internal_mechanics():
    ledger = GuardLedger()
    ledger.report("selection_length", "passed")
    ledger.report("empty_transcript", "fired", surface="internal")
    # An operator reading "1 check, none fired" must not have turn-integrity
    # mechanics silently folded into that number.
    assert ledger.summary() == {"passed": 1}


def _captured_rosters(monkeypatch) -> list[list[dict]]:
    """Capture what each finished turn would have persisted."""
    rosters: list[list[dict]] = []
    monkeypatch.setattr(
        pipeline, "submit_trace", lambda trace: rosters.append(trace.to_dict()["guard_roster"])
    )
    return rosters


def _roster_outcomes(roster: list[dict]) -> dict[str, str]:
    return {e["guard_id"]: e["outcome"] for e in roster}


def test_pinned_session_reports_language_id_as_not_evaluated(monkeypatch):
    # Honesty about delegation: with a pinned language, Qwen's LID does not run and
    # `detected` only echoes our own choice, so claiming "delegated" would be a lie.
    rosters = _captured_rosters(monkeypatch)
    with _app() as client, client.websocket_connect("/v1/speech-llm-toolassist-speech") as ws:
        ws.send_json({"type": "session.start", "language": "en-US", "sample_rate_hz": 16000})
        assert ws.receive_json()["type"] == "session.ready"
        _drive_turn(ws)
        while not ws.receive_json().get("final"):
            pass

    outcomes = _roster_outcomes(rosters[0])
    assert outcomes["language_id"] == "not_evaluated"
    assert outcomes["no_speech_suppression"] == "passed"
    entry = next(e for e in rosters[0] if e["guard_id"] == "language_id")
    assert entry["owner"] == "qwen" and entry["stage"] == "stt"
    assert "pinned" in entry["reason"]


def test_auto_session_reports_language_id_as_delegated(monkeypatch):
    rosters = _captured_rosters(monkeypatch)
    with _app() as client, client.websocket_connect("/v1/speech-llm-toolassist-speech") as ws:
        ws.send_json({"type": "session.start", "language": "auto", "sample_rate_hz": 16000})
        assert ws.receive_json()["type"] == "session.ready"
        _drive_turn(ws)
        while not ws.receive_json().get("final"):
            pass

    assert _roster_outcomes(rosters[0])["language_id"] == "delegated"


def test_language_gate_reports_when_it_drops_a_turn(monkeypatch):
    rosters = _captured_rosters(monkeypatch)
    with _app() as client, client.websocket_connect("/v1/speech-llm-toolassist-speech") as ws:
        # Thai selected, fake STT hears en-US: the turn is dropped.
        ws.send_json({
            "type": "session.start",
            "language": "en-US",
            "expected_language": "th-TH",
            "sample_rate_hz": 16000,
        })
        assert ws.receive_json()["type"] == "session.ready"
        _drive_turn(ws)
        for _ in range(12):
            msg = ws.receive_json()
            if msg["type"] == "response.audio" and msg.get("final"):
                break

    entry = next(e for e in rosters[0] if e["guard_id"] == "language_gate")
    assert entry["outcome"] == "fired"
    assert entry["seam"] == "decision"
    assert "expected=th-TH" in entry["reason"] and "detected=en-US" in entry["reason"]


def test_language_gate_reports_not_evaluated_without_a_selection(monkeypatch):
    # No expected_language means there is nothing to compare against; "passed"
    # would overstate what was checked.
    rosters = _captured_rosters(monkeypatch)
    with _app() as client, client.websocket_connect("/v1/speech-llm-toolassist-speech") as ws:
        ws.send_json({"type": "session.start", "language": "en-US", "sample_rate_hz": 16000})
        assert ws.receive_json()["type"] == "session.ready"
        _drive_turn(ws)
        while not ws.receive_json().get("final"):
            pass

    assert _roster_outcomes(rosters[0])["language_gate"] == "not_evaluated"


def test_trace_carries_the_roster_and_summary():
    trace = TurnTrace(session_id="s1", turn_id=1, capability="cap")
    report(
        trace.guards,
        "navigation.semantic",
        "delegated",
        seam="decision",
        stage="routing",
        owner="qwen",
    )
    report(
        trace.guards,
        "navigation.policy",
        "passed",
        seam="decision",
        stage="routing",
        owner="us",
    )
    out = trace.to_dict()
    assert [e["guard_id"] for e in out["guard_roster"]] == [
        "navigation.semantic",
        "navigation.policy",
    ]
    assert out["guard_summary"] == {"delegated": 1, "passed": 1}
    entry = out["guard_roster"][0]
    assert entry["surface"] == "guardrail" and entry["owner"] == "qwen"
