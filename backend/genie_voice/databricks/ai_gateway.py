"""Unity AI Gateway transport for Unity Catalog model services.

Foundation-model chat uses:

    POST {host}/ai-gateway/mlflow/v1/chat/completions
    body.model = catalog.schema.service  (e.g. system.ai.qwen3-next-80b-a3b-instruct)

Custom STT/TTS (and any name that is not a UC model-service FQN) keep using:

    POST {host}/serving-endpoints/{name}/invocations

Auth is the caller's ``authenticate()`` on every request so a 60-minute app SP
token is not pinned at construction.
"""
from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator
from typing import Any

CHAT_COMPLETIONS_PATH = "/ai-gateway/mlflow/v1/chat/completions"
_RETRY_DELAYS_S = (0.4, 0.8)
_APP_TAGS = {"app": "genie-voice-agent"}


class GatewayPolicyDenied(RuntimeError):
    """A Unity AI Gateway service policy blocked a request or response."""

    def __init__(self, policy: dict[str, Any]) -> None:
        self.policy = dict(policy)
        self.policy_name = str(
            policy.get("policy_name") or policy.get("name") or policy.get("policy") or "service_policy"
        )
        self.reason = str(policy.get("reason") or policy.get("message") or "Blocked by service policy")
        self.phase = str(policy.get("phase") or policy.get("event_type") or "") or None
        self.resource: str | None = None
        super().__init__(f"{self.policy_name}: {self.reason}")

    @property
    def is_input_denial(self) -> bool:
        phase = str(self.phase or "").lower().replace("-", "_")
        return phase in {"input", "request", "on_call", "pre_call"} or not phase


def model_service_id(name: str) -> str:
    return (name or "").strip().removeprefix("model-services/")


def is_unity_model_service(name: str) -> bool:
    """True for catalog.schema.leaf (Unity model service FQN)."""
    parts = model_service_id(name).split(".")
    return len(parts) >= 3 and all(parts)


def chat_completions_url(host: str) -> str:
    return f"{host.rstrip('/')}{CHAT_COMPLETIONS_PATH}"


def serving_invocations_url(host: str, endpoint: str) -> str:
    return f"{host.rstrip('/')}/serving-endpoints/{endpoint}/invocations"


def chat_body(model: str, inputs: dict[str, Any], *, stream: bool = False) -> dict[str, Any]:
    body = dict(inputs)
    body["model"] = model_service_id(model)
    if stream:
        body["stream"] = True
    return body


def request_headers(authenticate: Callable[[], dict[str, str] | None], *, gateway: bool) -> dict[str, str]:
    headers = {**dict(authenticate() or {}), "Content-Type": "application/json"}
    if gateway:
        headers["Databricks-Ai-Gateway-Request-Tags"] = json.dumps(
            _APP_TAGS, separators=(",", ":")
        )
    return headers


def iter_sse_json(resp: Any) -> Iterator[dict[str, Any]]:
    resp.encoding = "utf-8"
    for line in resp.iter_lines(decode_unicode=True):
        if line and line.startswith("data:"):
            payload = line[len("data:") :].strip()
            if payload and payload != "[DONE]":
                try:
                    chunk = json.loads(payload)
                    _raise_policy_denial(chunk)
                    yield chunk
                except json.JSONDecodeError:
                    continue


def invoke(
    *,
    host: str,
    authenticate: Callable[[], dict[str, str] | None],
    endpoint: str,
    inputs: dict[str, Any],
    timeout_s: float,
) -> dict[str, Any]:
    """One non-streaming inference call, routed by endpoint name."""
    import requests

    gateway = is_unity_model_service(endpoint)
    url = chat_completions_url(host) if gateway else serving_invocations_url(host, endpoint)
    body = chat_body(endpoint, inputs) if gateway else inputs
    headers = request_headers(authenticate, gateway=gateway)
    payload = _post_json(requests.post, url, headers, body, timeout_s, retry_429=gateway)
    if gateway:
        _raise_policy_denial(payload)
    return payload


def invoke_stream(
    *,
    host: str,
    authenticate: Callable[[], dict[str, str] | None],
    endpoint: str,
    inputs: dict[str, Any],
    timeout_s: float,
) -> Iterator[dict[str, Any]]:
    """SSE inference, routed by endpoint name. No 429 retry after the stream opens."""
    import requests

    gateway = is_unity_model_service(endpoint)
    url = chat_completions_url(host) if gateway else serving_invocations_url(host, endpoint)
    body = chat_body(endpoint, inputs, stream=True) if gateway else {**inputs, "stream": True}
    headers = request_headers(authenticate, gateway=gateway)
    with requests.post(url, headers=headers, json=body, stream=True, timeout=timeout_s) as resp:
        _raise_http(resp)
        yield from iter_sse_json(resp)


def _post_json(post, url: str, headers: dict[str, str], body: dict[str, Any], timeout_s: float, *, retry_429: bool) -> dict[str, Any]:
    delays = _RETRY_DELAYS_S if retry_429 else ()
    last = None
    for attempt in range(len(delays) + 1):
        resp = post(url, headers=headers, json=body, timeout=timeout_s)
        last = resp
        if resp.status_code != 429 or attempt >= len(delays):
            _raise_http(resp)
            return resp.json()
        time.sleep(delays[attempt])
    _raise_http(last)
    return last.json()


def _raise_http(resp: Any) -> None:
    import requests

    if getattr(resp, "status_code", 200) < 400:
        return
    detail = ""
    try:
        detail = (resp.text or "")[:500]
    except Exception:  # noqa: BLE001
        detail = ""
    raise requests.HTTPError(f"{resp.status_code} {detail}".strip(), response=resp)


def _raise_policy_denial(payload: Any) -> None:
    """Raise for the HTTP-200 denial envelope returned by service policies."""
    if not isinstance(payload, dict):
        return
    policy = payload.get("databricks_service_policy")
    if isinstance(policy, dict):
        raise GatewayPolicyDenied(policy)
