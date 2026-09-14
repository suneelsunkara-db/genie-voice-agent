"""Unity AI Gateway routing and bounded 429 retry."""
from __future__ import annotations

import requests

from genie_voice.databricks import ai_gateway


def test_is_unity_model_service() -> None:
    assert ai_gateway.is_unity_model_service("system.ai.qwen3-next-80b-a3b-instruct")
    assert ai_gateway.is_unity_model_service("model-services/system.ai.gpt-5-5")
    assert not ai_gateway.is_unity_model_service("databricks-qwen3-next-80b-a3b-instruct")
    assert not ai_gateway.is_unity_model_service("dummy_llm")
    assert not ai_gateway.is_unity_model_service("voice_stt")


def test_chat_body_sets_model_and_stream() -> None:
    body = ai_gateway.chat_body(
        "system.ai.qwen3-next-80b-a3b-instruct",
        {"messages": [], "max_tokens": 16},
        stream=True,
    )
    assert body["model"] == "system.ai.qwen3-next-80b-a3b-instruct"
    assert body["stream"] is True
    assert body["max_tokens"] == 16


def test_gateway_retries_429_then_succeeds(monkeypatch) -> None:
    calls = {"n": 0}

    class _Resp:
        def __init__(self, status: int, payload=None):
            self.status_code = status
            self.text = "rate limited" if status == 429 else "ok"
            self._payload = payload or {"choices": []}

        def json(self):
            return self._payload

    def fake_post(url, headers=None, json=None, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return _Resp(429)
        return _Resp(200, {"ok": True})

    monkeypatch.setattr(ai_gateway.time, "sleep", lambda _s: None)
    out = ai_gateway._post_json(
        fake_post,
        "https://example/ai-gateway/mlflow/v1/chat/completions",
        {},
        {"model": "system.ai.gpt-5-5"},
        5.0,
        retry_429=True,
    )
    assert calls["n"] == 2
    assert out == {"ok": True}


def test_gateway_429_exhausted_raises(monkeypatch) -> None:
    class _Resp:
        status_code = 429
        text = "rate limited"

        def json(self):
            return {}

    monkeypatch.setattr(ai_gateway.time, "sleep", lambda _s: None)
    try:
        ai_gateway._post_json(
            lambda *a, **k: _Resp(),
            "https://example/x",
            {},
            {},
            5.0,
            retry_429=True,
        )
        raise AssertionError("expected HTTPError")
    except requests.HTTPError as exc:
        assert "429" in str(exc)
