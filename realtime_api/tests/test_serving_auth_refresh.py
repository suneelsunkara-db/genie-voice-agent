"""The serving client must re-read auth on every request.

The app runs as a service principal whose OAuth token expires after 60 minutes,
and the SDK only refreshes it when ``Config.authenticate()`` is called. A client
that captures those headers once keeps presenting a dead token, so every STT,
LLM and TTS call starts returning 403 an hour after the app boots -- while the
app itself stays healthy, which makes it look like a permissions problem.
"""
from __future__ import annotations

import json

import databricks.sdk
import pytest
import requests

from realtime_api.services import _SdkDeployClient


class _FakeConfig:
    """Hands out a different token per call, the way a refresh would."""

    def __init__(self) -> None:
        self.host = "https://example.cloud.databricks.com/"
        self.authenticate_calls = 0

    def authenticate(self) -> dict[str, str]:
        self.authenticate_calls += 1
        return {"Authorization": f"Bearer token-{self.authenticate_calls}"}


class _FakeWorkspace:
    def __init__(self, profile: str | None = None) -> None:
        self.config = _FakeConfig()


class _FakeResponse:
    def __init__(self, *, payload: dict | None = None, lines: list[str] | None = None) -> None:
        self._payload = payload or {}
        self._lines = lines or []

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload

    def iter_lines(self, decode_unicode: bool = False):
        yield from self._lines

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> _SdkDeployClient:
    monkeypatch.setattr(databricks.sdk, "WorkspaceClient", _FakeWorkspace)
    return _SdkDeployClient()


@pytest.fixture
def sent(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Record the headers of every outgoing request."""
    calls: list[dict] = []

    def fake_post(url, headers=None, json=None, timeout=None, stream=False):
        calls.append({"url": url, "headers": dict(headers or {})})
        if stream:
            return _FakeResponse(lines=['data: {"ok": true}', "data: [DONE]"])
        return _FakeResponse(payload={"ok": True})

    monkeypatch.setattr(requests, "post", fake_post)
    return calls


def _tokens(sent: list[dict]) -> list[str]:
    return [c["headers"].get("Authorization") for c in sent]


def test_no_token_is_captured_at_construction(client: _SdkDeployClient) -> None:
    """Nothing is authenticated until a request actually needs it, so a client
    built at import time cannot pin a token that later goes stale."""
    assert client._w.config.authenticate_calls == 0


def test_each_predict_authenticates_again(client: _SdkDeployClient, sent: list[dict]) -> None:
    client.predict(endpoint="stt", inputs={"a": 1})
    client.predict(endpoint="stt", inputs={"a": 2})

    # Distinct tokens: the second call went back to the SDK rather than reusing
    # the first, which is what lets an expired token be replaced mid-session.
    assert _tokens(sent) == ["Bearer token-1", "Bearer token-2"]


def test_each_stream_authenticates_again(client: _SdkDeployClient, sent: list[dict]) -> None:
    for _ in range(2):
        list(client.predict_stream(endpoint="tts", inputs={"text": "hi"}))

    assert _tokens(sent) == ["Bearer token-1", "Bearer token-2"]


def test_predict_and_stream_share_one_refreshing_source(
    client: _SdkDeployClient, sent: list[dict]
) -> None:
    """Both paths build headers the same way; neither may hold a private copy."""
    client.predict(endpoint="stt", inputs={})
    list(client.predict_stream(endpoint="tts", inputs={}))
    client.predict(endpoint="llm", inputs={})

    assert _tokens(sent) == ["Bearer token-1", "Bearer token-2", "Bearer token-3"]
    assert client._w.config.authenticate_calls == 3


def test_content_type_still_accompanies_the_token(
    client: _SdkDeployClient, sent: list[dict]
) -> None:
    client.predict(endpoint="stt", inputs={})
    assert sent[0]["headers"]["Content-Type"] == "application/json"


def test_stream_still_parses_sse_payloads(client: _SdkDeployClient, sent: list[dict]) -> None:
    """Guards the refactor itself: fresh headers must not disturb SSE decoding."""
    assert list(client.predict_stream(endpoint="tts", inputs={})) == [{"ok": True}]
    assert json.loads('{"ok": true}') == {"ok": True}
