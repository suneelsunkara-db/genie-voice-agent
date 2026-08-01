"""Agent Mode must never end a run silently.

The failure this covers: Genie closed the SSE stream without a
``response.completed`` or ``response.failed`` event. The client returned normally,
the deep-dive worker put its terminal ``done``, and the UI showed a generic
"investigation ended without a result" — with nothing logged anywhere. The run was
undiagnosable after the fact, which is the one outcome an observability path must
not produce.
"""
from __future__ import annotations

import pytest

from genie_voice.genie import agent_mode as am


class _FakeResp:
    """Minimal stand-in for a streaming requests response."""

    status_code = 200

    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    def iter_lines(self, decode_unicode: bool = False):
        yield from self._lines

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _client(monkeypatch, resp) -> am.GenieAgentModeClient:
    client = am.GenieAgentModeClient.__new__(am.GenieAgentModeClient)
    monkeypatch.setattr(client, "_host", lambda: "https://example.invalid", raising=False)
    monkeypatch.setattr(client, "_auth_headers", lambda: {}, raising=False)
    monkeypatch.setattr(client, "resolve_agent_id", lambda *a, **k: "space123", raising=False)
    # `ask` imports requests locally, so the module attribute is what to patch.
    monkeypatch.setattr("requests.post", lambda *a, **k: resp)
    return client


def _events(client: am.GenieAgentModeClient, **kwargs) -> list[dict]:
    seen: list[dict] = []
    client.ask("why did my expenses rise?", on_event=seen.append, **kwargs)
    return seen


def test_stream_without_a_terminal_event_reports_an_error(monkeypatch, caplog):
    lines = [
        'data: {"type": "response.created", "response": {"conversation_id": "c1"}}',
        "",
        'data: {"type": "response.output_item.done", "item": {"type": "reasoning"}}',
        "",
        # ...and then Genie just stops. No completed, no failed.
    ]
    client = _client(monkeypatch, _FakeResp(lines))
    with caplog.at_level("WARNING"):
        events = _events(client)

    errors = [e for e in events if e["kind"] == "error"]
    assert len(errors) == 1, "an empty stream must surface exactly one error"
    assert errors[0]["status"] == "incomplete"
    assert "without a terminal response event" in errors[0]["error"]["message"]
    # The events we DID see are the only clue to why; they have to be carried.
    assert "response.created" in errors[0]["error"]["events_seen"]
    assert "no terminal event" in caplog.text


def test_completed_stream_still_reports_once_and_does_not_error(monkeypatch):
    lines = [
        'data: {"type": "response.created", "response": {"conversation_id": "c1"}}',
        "",
        'data: {"type": "response.completed", "response": {"status": "completed", "output": []}}',
        "",
    ]
    events = _events(_client(monkeypatch, _FakeResp(lines)))
    kinds = [e["kind"] for e in events]
    assert kinds.count("report") == 1
    assert "error" not in kinds


def test_failed_response_keeps_its_own_error_and_adds_no_second_one(monkeypatch):
    lines = [
        'data: {"type": "response.failed", "response": {"status": "failed", '
        '"error": {"message": "query timed out"}}}',
        "",
    ]
    events = _events(_client(monkeypatch, _FakeResp(lines)))
    errors = [e for e in events if e["kind"] == "error"]
    assert len(errors) == 1
    assert errors[0]["error"]["message"] == "query timed out"


def test_http_error_before_the_stream_is_reported(monkeypatch):
    class _Err(_FakeResp):
        status_code = 403

        def json(self):
            return {"error": {"message": "FEATURE_DISABLED"}}

    client = _client(monkeypatch, _Err([]))
    events: list[dict] = []
    client.ask("q", on_event=events.append)
    assert events[0]["kind"] == "error"
    assert events[0]["error"]["http_status"] == 403


def test_the_question_is_sent_verbatim_with_no_language_directive(monkeypatch):
    """The agent is asked in English, full stop.

    Appending "write your report in <language>" is what killed the Hindi runs: the
    agent did all its SQL, announced the Hindi report, then closed the stream with
    no terminal event. Localizing is now realtime_api.deep_dive's job, so nothing
    may be appended to the question here.
    """
    sent: dict = {}

    def _post(url, **kwargs):
        sent.update(kwargs.get("json") or {})
        return _FakeResp([
            'data: {"type": "response.completed", "response": {"status": "completed"}}',
            "",
        ])

    client = am.GenieAgentModeClient.__new__(am.GenieAgentModeClient)
    monkeypatch.setattr(client, "_host", lambda: "https://example.invalid", raising=False)
    monkeypatch.setattr(client, "_auth_headers", lambda: {}, raising=False)
    monkeypatch.setattr(client, "resolve_agent_id", lambda *a, **k: "space123", raising=False)
    monkeypatch.setattr("requests.post", _post)

    client.ask("Why did my expenses rise?")
    text = sent["input"][0]["content"][0]["text"]
    assert text == "Why did my expenses rise?"


def test_ask_does_not_accept_a_language(monkeypatch):
    # A `language=` kwarg would silently do nothing now; it must not be accepted.
    client = _client(monkeypatch, _FakeResp([]))
    with pytest.raises(TypeError):
        client.ask("q", language="hi-IN")  # type: ignore[call-arg]
