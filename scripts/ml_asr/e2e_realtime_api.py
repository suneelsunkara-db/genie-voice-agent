"""End-to-end test of the realtime voice API against live endpoints.

Drives the real FastAPI WebSocket app (``realtime_api.app.create_app``) with an
SDK-backed serving client (no local mlflow needed), so the full pipeline runs:

    synthesized utterance -> WS audio -> STT -> LLM(+tools) -> streaming TTS -> WS

For each language it: (1) synthesizes a prompt with the TTS endpoint and
resamples it to 16 kHz PCM, (2) streams it into the WebSocket, (3) collects the
transcript, LLM reply, and streamed ``response.audio`` PCM chunks, reporting the
client-observed time-to-first-audio.

Usage:
    python scripts/ml_asr/e2e_realtime_api.py --languages en-US,th-TH
"""
from __future__ import annotations

import argparse
import base64
import json
import struct
import sys
import time
from pathlib import Path
from typing import Any

import requests

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from _realtime_config import databricks, realtime_voice, find_candidate  # noqa: E402
from realtime_api.app import create_app  # noqa: E402
from realtime_api.config import RealtimeSettings  # noqa: E402
from realtime_api.services import DatabricksServing  # noqa: E402
from realtime_api.pipelines import ServingBundle  # noqa: E402

_PROMPTS = {
    "en-US": "What time is it right now in Bangkok?",
    "th-TH": "ตอนนี้ที่กรุงเทพกี่โมงแล้ว",
    "id-ID": "Jam berapa sekarang di Bangkok?",
    "zh-CN": "现在曼谷几点了？",
}


class _SdkDeployClient:
    """Databricks-SDK client exposing predict + predict_stream (SSE)."""

    def __init__(self, workspace) -> None:
        self._w = workspace
        self._host = workspace.config.host.rstrip("/")

    def _auth_headers(self) -> dict[str, str]:
        """Fresh per request: OAuth tokens expire in 60 minutes, and a full
        multilingual run outlives that."""
        return {**dict(self._w.config.authenticate() or {}), "Content-Type": "application/json"}

    def predict(self, *, endpoint: str, inputs: dict) -> dict:
        return self._w.api_client.do("POST", f"/serving-endpoints/{endpoint}/invocations", body=inputs)

    def predict_stream(self, *, endpoint: str, inputs: dict):
        body = {**inputs, "stream": True}
        url = f"{self._host}/serving-endpoints/{endpoint}/invocations"
        with requests.post(
            url, headers=self._auth_headers(), json=body, stream=True, timeout=180
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=True):
                if line and line.startswith("data:"):
                    payload = line[len("data:"):].strip()
                    if payload and payload != "[DONE]":
                        try:
                            yield json.loads(payload)
                        except json.JSONDecodeError:
                            continue


def _synthesize_16k_pcm(client: _SdkDeployClient, tts_endpoint: str, text: str, language: str) -> bytes:
    """Synthesize a prompt and downsample to 16 kHz mono PCM s16le for STT."""
    resp = client.predict(
        endpoint=tts_endpoint,
        inputs={"input": [{"role": "user", "content": text}], "custom_inputs": {"text": text, "language": language}},
    )
    custom = (resp or {}).get("custom_outputs") or {}
    wav = base64.b64decode(str(custom.get("audio_b64") or ""))
    src_sr = int(custom.get("sample_rate_hz") or 48_000)
    body = wav[44:] if wav[:4] == b"RIFF" else wav
    samples = struct.unpack("<" + "h" * (len(body) // 2), body[: (len(body) // 2) * 2])
    factor = max(1, round(src_sr / 16_000))
    down = samples[::factor]
    return struct.pack("<" + "h" * len(down), *down)


def _run_language(app, client, tts_endpoint, language: str) -> dict[str, Any]:
    from fastapi.testclient import TestClient

    text = _PROMPTS.get(language, _PROMPTS["en-US"])
    pcm = _synthesize_16k_pcm(client, tts_endpoint, text, language)
    result: dict[str, Any] = {"language": language, "prompt": text, "input_pcm_bytes": len(pcm)}

    with TestClient(app) as http, http.websocket_connect("/v1/speech-llm-toolassist-speech") as ws:
        ws.send_json({"type": "session.start", "language": language, "sample_rate_hz": 16_000})
        assert ws.receive_json()["type"] == "session.ready"

        for offset in range(0, len(pcm), 640):  # 20 ms frames @ 16 kHz
            ws.send_bytes(pcm[offset : offset + 640])
        sent_at = time.perf_counter()
        ws.send_json({"type": "audio.end"})

        chunks = 0
        first_audio_ms = None
        audio_bytes = 0
        while True:
            event = ws.receive_json()
            etype = event.get("type")
            if etype == "transcript.final":
                result["transcript"] = event.get("text")
            elif etype == "response.text":
                result["response_text"] = event.get("text")
            elif etype == "response.audio":
                if first_audio_ms is None:
                    first_audio_ms = (time.perf_counter() - sent_at) * 1000.0
                chunks += 1
                audio_bytes += len(base64.b64decode(event.get("audio_b64") or ""))
                if event.get("final"):
                    break
            elif etype == "error":
                result["error"] = event
                break
    result["audio_chunks"] = chunks
    result["audio_bytes"] = audio_bytes
    result["client_ttfa_ms"] = round(first_audio_ms, 1) if first_audio_ms is not None else None
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--languages", default="en-US,th-TH,id-ID,zh-CN")
    args = parser.parse_args()
    languages = [x.strip() for x in args.languages.split(",") if x.strip()]

    from databricks.sdk import WorkspaceClient

    settings = RealtimeSettings.resolve()
    rv = realtime_voice()
    tts_endpoint = find_candidate(next(iter(rv.get("tts_candidates") or {})))["endpoint"]

    w = WorkspaceClient(profile=databricks().get("profile") or None)
    client = _SdkDeployClient(w)

    serving = DatabricksServing(
        client=client,
        stt_endpoint=settings.stt_endpoint,
        llm_endpoint=settings.llm_endpoint,
        tts_endpoint=settings.tts_endpoint,
        llm_temperature=settings.llm_temperature,
        llm_max_tokens=settings.llm_max_tokens,
        llm_tools_enabled=settings.llm_tools_enabled,
        llm_max_tool_iterations=settings.llm_max_tool_iterations,
        tts_inference_timesteps=settings.tts_inference_timesteps,
        tts_cfg_value=settings.tts_cfg_value,
    )
    app = create_app(settings=settings, bundle_factory=lambda _s: ServingBundle(stt=serving, llm=serving, tts=serving))

    print(f"stt={settings.stt_endpoint}  llm={settings.llm_endpoint}  tts={settings.tts_endpoint}\n")
    for language in languages:
        try:
            r = _run_language(app, client, tts_endpoint, language)
        except Exception as exc:  # noqa: BLE001
            print(f"[{language}] ERROR: {str(exc)[:160]}\n")
            continue
        if "error" in r:
            print(f"[{language}] pipeline error: {r['error']}\n")
            continue
        print(f"[{language}] prompt:     {r['prompt']}")
        print(f"          transcript: {r.get('transcript')!r}")
        print(f"          reply:      {r.get('response_text')!r}")
        print(
            f"          audio:      {r['audio_chunks']} chunks, {r['audio_bytes']} bytes, "
            f"client TTFA {r['client_ttfa_ms']} ms\n"
        )


if __name__ == "__main__":
    main()
