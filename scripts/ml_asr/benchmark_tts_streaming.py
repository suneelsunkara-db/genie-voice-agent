"""Measure the streaming win for the VoxCPM2 TTS endpoint.

Compares two serving paths on the SAME GPU:
  * non-streaming ``predict``      -> one WAV after the whole sentence (~3 s)
  * streaming ``predict_stream``   -> ~80 ms PCM chunks as generated

The metric that matters for realtime UX is time-to-first-audio (TTFB): with
streaming the client can start playback after the first chunk instead of waiting
for the full sentence. This does not need an A100 -- it is a serving-path change.

Usage:
    python scripts/ml_asr/benchmark_tts_streaming.py --languages en-US,th-TH
"""
from __future__ import annotations

import argparse
import base64
import json
import statistics
import time
from typing import Any

import requests

from _realtime_config import databricks, find_candidate, realtime_voice

_SAMPLE_TEXT = {
    "en-US": "Sure, our office opens at nine in the morning and closes at six in the evening.",
    "th-TH": "ได้ค่ะ สำนักงานของเราเปิดเวลาเก้าโมงเช้าและปิดเวลาหกโมงเย็น",
    "id-ID": "Tentu, kantor kami buka pukul sembilan pagi dan tutup pukul enam sore.",
    "zh-CN": "好的，我们的办公室早上九点开门，晚上六点关门。",
}


def _auth() -> tuple[str, dict[str, str]]:
    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient(profile=databricks().get("profile") or None)
    host = w.config.host.rstrip("/")
    headers = w.config.authenticate() or {}
    return host, dict(headers)


def _stream_call(host: str, headers: dict[str, str], endpoint: str, text: str, ci: dict[str, Any]) -> dict[str, Any]:
    url = f"{host}/serving-endpoints/{endpoint}/invocations"
    body = {"input": [{"role": "user", "content": text}], "custom_inputs": ci, "stream": True}
    started = time.perf_counter()
    ttfb: float | None = None
    chunks = 0
    total_bytes = 0
    final: dict[str, Any] = {}
    with requests.post(url, headers={**headers, "Content-Type": "application/json"}, json=body, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data:"):
                continue
            payload = raw[len("data:"):].strip()
            if payload in ("", "[DONE]"):
                continue
            try:
                event = json.loads(payload)
            except json.JSONDecodeError:
                continue
            co = event.get("custom_outputs") or {}
            audio = co.get("audio_pcm16_b64")
            if audio:
                if ttfb is None:
                    ttfb = (time.perf_counter() - started) * 1000.0
                chunks += 1
                total_bytes += len(base64.b64decode(audio))
            if co.get("final"):
                final = co
    total_ms = (time.perf_counter() - started) * 1000.0
    return {
        "ttfb_ms": round(ttfb, 1) if ttfb is not None else None,
        "total_ms": round(total_ms, 1),
        "chunks": chunks,
        "audio_bytes": total_bytes,
        "server_gen_ms": final.get("gen_ms"),
        "server_ttfb_ms": final.get("ttfb_ms"),
        "sample_rate_hz": final.get("sample_rate_hz"),
        "device": final.get("inference_device"),
    }


def _predict_call(host: str, headers: dict[str, str], endpoint: str, text: str, ci: dict[str, Any]) -> dict[str, Any]:
    url = f"{host}/serving-endpoints/{endpoint}/invocations"
    body = {"input": [{"role": "user", "content": text}], "custom_inputs": ci}
    started = time.perf_counter()
    resp = requests.post(url, headers={**headers, "Content-Type": "application/json"}, json=body, timeout=120)
    resp.raise_for_status()
    e2e = (time.perf_counter() - started) * 1000.0
    co = (resp.json() or {}).get("custom_outputs") or {}
    return {"e2e_ms": round(e2e, 1), "gen_ms": co.get("gen_ms"), "device": co.get("inference_device")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tts", default=None)
    parser.add_argument("--languages", default="en-US,th-TH,id-ID,zh-CN")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--timesteps", type=int, default=None)
    args = parser.parse_args()

    rv = realtime_voice()
    tts_id = args.tts or next(iter(rv.get("tts_candidates") or {}), None)
    endpoint = find_candidate(tts_id)["endpoint"]
    languages = [x.strip() for x in args.languages.split(",") if x.strip()]
    defaults = rv.get("tts_defaults") or {}
    ci_base = {
        "inference_timesteps": args.timesteps or int(defaults.get("inference_timesteps", 6)),
        "cfg_value": float(defaults.get("cfg_value", 2.0)),
    }

    host, headers = _auth()
    print(f"endpoint={endpoint} reps={args.reps} timesteps={ci_base['inference_timesteps']}\n")
    header = f"{'lang':7} {'stream TTFB':>12} {'stream total':>13} {'chunks':>7} {'predict e2e':>12} {'win (TTFB)':>11}"
    print(header)
    print("-" * len(header))

    for language in languages:
        text = _SAMPLE_TEXT.get(language, _SAMPLE_TEXT["en-US"])
        ci = {**ci_base, "text": text, "language": language}
        try:
            _stream_call(host, headers, endpoint, text, ci)  # warmup
            stream = [_stream_call(host, headers, endpoint, text, ci) for _ in range(args.reps)]
            predict = [_predict_call(host, headers, endpoint, text, ci) for _ in range(args.reps)]
        except Exception as exc:  # noqa: BLE001
            print(f"{language:7} ERROR {str(exc)[:70]}")
            continue

        ttfb = statistics.median([s["ttfb_ms"] for s in stream if s["ttfb_ms"] is not None] or [0])
        stotal = statistics.median([s["total_ms"] for s in stream])
        chunks = statistics.median([s["chunks"] for s in stream])
        e2e = statistics.median([p["e2e_ms"] for p in predict])
        win = f"{e2e / ttfb:.1f}x" if ttfb else "n/a"
        print(f"{language:7} {ttfb:10.0f}ms {stotal:11.0f}ms {chunks:7.0f} {e2e:10.0f}ms {win:>11}")

    print("\nTTFB = time to first audio chunk (playback can start here).")
    print("predict e2e = full-sentence non-streaming latency (previous behaviour).")


if __name__ == "__main__":
    main()
