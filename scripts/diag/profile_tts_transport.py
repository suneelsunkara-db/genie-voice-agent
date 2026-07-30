"""Measure realtime TTS first-audio overhead on an IDLE endpoint.

Goal: separate the latency our client observes from the time the model actually
needs, so we fix the real bottleneck instead of guessing.

Method notes (these matter — an earlier version of this probe got them wrong):
  * Each request is drained to completion. Breaking out early leaves the server
    generating into a closed connection, so the NEXT request queues behind it on
    a single-replica ("Small") endpoint and every later timing is inflated.
  * A settle delay runs between requests so each one starts against an idle
    replica, isolating per-request overhead from self-inflicted contention.

Reported per variant:
    cli_1st_audio : client-observed time to first PCM chunk (the dead air heard)
    srv_ttfb      : endpoint's OWN time to its first chunk (custom_outputs.ttfb_ms)
    overhead      : cli_1st_audio - srv_ttfb  (transport + serving invocation)
    srv_gen/audio : full generation time vs audio produced (RTF = gen/audio)

Variants price the ~200KB base64 voice reference we currently resend on EVERY
turn (``ref``) against no cloning at all (``noref``).

Read-only: hits the endpoint directly; changes no config and no app code.

Usage:
    python scripts/diag/profile_tts_transport.py --reps 3 --settle 5
"""
from __future__ import annotations

import argparse
import base64
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

TEXT = "I'm connecting you to Telco billing support right away."


def _one_request(
    session: Any, url: str, headers: dict[str, str], body: dict[str, Any]
) -> dict[str, Any]:
    """Stream one synthesis to completion, timing headers and first audio."""
    start = time.perf_counter()
    first_audio_ms: float | None = None
    headers_ms: float | None = None
    chunks = 0
    pcm_bytes = 0
    sample_rate = 0
    srv_ttfb: float | None = None
    srv_gen: float | None = None

    with session.post(url, headers=headers, json=body, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        headers_ms = (time.perf_counter() - start) * 1000.0
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if not payload or payload == "[DONE]":
                continue
            try:
                event = json.loads(payload)
            except ValueError:
                continue
            custom = (event or {}).get("custom_outputs") or {}
            pcm_b64 = custom.get("audio_pcm16_b64")
            if pcm_b64:
                if first_audio_ms is None:
                    first_audio_ms = (time.perf_counter() - start) * 1000.0
                chunks += 1
                pcm_bytes += len(base64.b64decode(pcm_b64))
                sample_rate = int(custom.get("sample_rate_hz") or sample_rate)
            elif custom.get("final"):
                srv_ttfb = custom.get("ttfb_ms")
                srv_gen = custom.get("gen_ms")

    total_ms = (time.perf_counter() - start) * 1000.0
    audio_ms = (pcm_bytes / 2 / sample_rate * 1000.0) if sample_rate else 0.0
    return {
        "headers_ms": round(headers_ms or 0.0, 1),
        "cli_first_audio_ms": round(first_audio_ms or 0.0, 1),
        "cli_total_ms": round(total_ms, 1),
        "srv_ttfb_ms": float(srv_ttfb) if srv_ttfb is not None else None,
        "srv_gen_ms": float(srv_gen) if srv_gen is not None else None,
        "chunks": chunks,
        "audio_ms": round(audio_ms, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--settle", type=float, default=5.0, help="Idle seconds between requests")
    args = parser.parse_args()

    import requests

    from realtime_api.serving_factory import shared_serving

    serving = shared_serving()
    client = serving.client
    url = f"{client._host}/serving-endpoints/{serving.tts_endpoint}/invocations"
    headers = dict(client._headers)

    print(f"endpoint={serving.tts_endpoint} timesteps={serving.tts_inference_timesteps} "
          f"cfg={serving.tts_cfg_value}")
    print(f"reps={args.reps} settle={args.settle}s (drain-to-completion)\n")

    print("generating voice reference...")
    ref = serving.synthesize("Hello, thanks for calling.", language="en-US")
    reference_b64 = base64.b64encode(ref.audio).decode("ascii")
    print(f"reference b64 payload = {len(reference_b64) / 1024:.0f} KB\n")

    def body(with_ref: bool) -> dict[str, Any]:
        custom: dict[str, Any] = {
            "text": TEXT,
            "language": "en-US",
            "inference_timesteps": serving.tts_inference_timesteps,
            "cfg_value": serving.tts_cfg_value,
        }
        if with_ref:
            custom["reference_audio_b64"] = reference_b64
        return {
            "input": [{"role": "user", "content": TEXT}],
            "custom_inputs": custom,
            "stream": True,
        }

    hdr = (f"{'variant':<8} {'cli_1st':>8} {'srv_ttfb':>9} {'overhead':>9} "
           f"{'srv_gen':>8} {'audio':>7} {'rtf':>5}")
    print(hdr)
    print("-" * len(hdr))

    session = requests.Session()
    try:
        for name, with_ref in (("noref", False), ("ref", True)):
            runs: list[dict[str, Any]] = []
            for _ in range(args.reps):
                time.sleep(args.settle)  # let the replica go idle first
                try:
                    runs.append(_one_request(session, url, headers, body(with_ref)))
                except Exception as exc:  # noqa: BLE001
                    print(f"{name:<8} ERROR {type(exc).__name__}: {exc}"[:160])
                    runs = []
                    break
            if not runs:
                continue

            def med(key: str) -> float | None:
                vals = [r[key] for r in runs if isinstance(r.get(key), (int, float))]
                return round(statistics.median(vals), 1) if vals else None

            cli = med("cli_first_audio_ms")
            srv = med("srv_ttfb_ms")
            gen = med("srv_gen_ms")
            aud = med("audio_ms")
            overhead = round(cli - srv, 1) if (cli is not None and srv is not None) else None
            rtf = round(gen / aud, 2) if (gen and aud) else None
            print(f"{name:<8} {cli if cli is not None else '-':>8} "
                  f"{srv if srv is not None else '-':>9} "
                  f"{overhead if overhead is not None else '-':>9} "
                  f"{gen if gen is not None else '-':>8} "
                  f"{aud if aud is not None else '-':>7} "
                  f"{rtf if rtf is not None else '-':>5}")
    finally:
        session.close()


if __name__ == "__main__":
    main()
