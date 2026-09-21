"""Unity AI Gateway routing and bounded 429 retry."""
from __future__ import annotations

import json

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


def test_service_policy_http_200_is_raised_as_denial() -> None:
    payload = {
        "choices": [{"message": {"role": "assistant", "content": "Blocked"}}],
        "databricks_service_policy": {
            "name": "block-unsafe-content",
            "phase": "pre_call",
            "reason": "Unsafe content was blocked.",
        },
    }
    try:
        ai_gateway._raise_policy_denial(payload)
        raise AssertionError("expected GatewayPolicyDenied")
    except ai_gateway.GatewayPolicyDenied as exc:
        assert exc.policy_name == "block-unsafe-content"
        assert exc.reason == "Unsafe content was blocked."
        assert exc.is_input_denial


def test_service_policy_post_call_is_output_denial() -> None:
    denial = ai_gateway.GatewayPolicyDenied(
        {"name": "block-hallucination", "phase": "post_call"}
    )
    assert denial.is_input_denial is False


def test_inference_context_merges_complete_request_provenance() -> None:
    with ai_gateway.inference_context(
        {
            "traffic_class": "conversation",
            "surface": "card",
            "profile": "card",
            "trace_id": "trace-1",
            "session_id": "session-1",
            "turn_id": 4,
            "capability": "voice",
            "model_role": "navigation",
            "empty": "",
        }
    ):
        headers = ai_gateway.request_headers(
            lambda: {"Authorization": "Bearer test"},
            gateway=True,
            request_tags={"model_role": "conversion"},
        )
    tags = json.loads(headers["Databricks-Ai-Gateway-Request-Tags"])
    assert tags == {
        "app": "genie-voice-agent",
        "traffic_class": "conversation",
        "surface": "card",
        "profile": "card",
        "trace_id": "trace-1",
        "session_id": "session-1",
        "turn_id": "4",
        "capability": "voice",
        "model_role": "conversion",
    }
    assert "Databricks-Ai-Gateway-Request-Tags" not in ai_gateway.request_headers(
        lambda: {}, gateway=False
    )


def test_model_serving_call_records_content_free_lineage(monkeypatch) -> None:
    events = []

    class _Resp:
        status_code = 200
        text = "ok"

        def json(self):
            return {"request_id": "req-1", "custom_outputs": {"transcript": "secret"}}

    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: _Resp())
    with ai_gateway.inference_context(
        {"traffic_class": "conversation", "model_role": "stt"},
        recorder=events.append,
    ):
        ai_gateway.invoke(
            host="https://example",
            authenticate=lambda: {},
            endpoint="custom-stt",
            inputs={"audio": "secret"},
            timeout_s=5,
        )
    assert events == [
        {
            "endpoint": "custom-stt",
            "transport": "model_serving",
            "model_role": "stt",
            "request_id": "req-1",
            "invocation_id": None,
            "duration_ms": events[0]["duration_ms"],
            "status": "ok",
        }
    ]
    assert "secret" not in json.dumps(events)
