#!/usr/bin/env python3
"""Quick probe of realtime API routes (HTTP + WebSocket)."""
from __future__ import annotations

import argparse
import asyncio
import json
import struct
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from databricks_auth import build_token_provider  # noqa: E402
from paths import benchmark_sp_credentials, databricks_host  # noqa: E402


def _loud_frame(samples: int = 320) -> bytes:
    return struct.pack("<" + "h" * samples, *([6000, -6000] * (samples // 2)))


async def _probe_ws(url: str, token: str, mode: str) -> dict:
    import websockets

    headers = {"Authorization": f"Bearer {token}"}
    out: dict = {"url": url, "mode": mode, "ok": False}
    try:
        try:
            conn = websockets.connect(url, additional_headers=headers, ping_interval=None, open_timeout=30)
        except TypeError:
            conn = websockets.connect(url, extra_headers=headers, ping_interval=None, open_timeout=30)
        async with conn as ws:
            await ws.send(json.dumps({
                "type": "session.start",
                "language": "en",
                "sample_rate_hz": 16000,
                "encoding": "pcm_s16le",
            }))
            ready = json.loads(await asyncio.wait_for(ws.recv(), timeout=30))
            out["session_ready"] = ready
            if mode == "stt":
                await ws.send(_loud_frame())
                await ws.send(json.dumps({"type": "audio.end"}))
                events = []
                for _ in range(5):
                    msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=120))
                    events.append(msg.get("type"))
                    if msg.get("type") in {"transcript.final", "error"}:
                        break
                out["events"] = events
                out["ok"] = "transcript.final" in events and "response.text" not in events
            elif mode == "assist":
                await ws.send(_loud_frame())
                await ws.send(json.dumps({"type": "audio.end"}))
                events = []
                for _ in range(10):
                    msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=180))
                    events.append(msg.get("type"))
                    if msg.get("type") == "response.audio" and msg.get("final"):
                        break
                    if msg.get("type") == "error":
                        break
                out["events"] = events
                out["ok"] = all(t in events for t in ("transcript.final", "response.text", "response.audio"))
            elif mode == "tts":
                await ws.send(json.dumps({"type": "synthesize", "text": "Hello", "language": "en"}))
                events = []
                for _ in range(5):
                    msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=120))
                    events.append(msg.get("type"))
                    if msg.get("type") == "response.audio" and msg.get("final"):
                        break
                    if msg.get("type") == "error":
                        break
                out["events"] = events
                out["ok"] = "response.audio" in events and "transcript.final" not in events
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="https://genie-voice-agent-3644297589119053.aws.databricksapps.com")
    p.add_argument("--prefix", default="realtime")
    p.add_argument("--profile", default="fe-vm-vdm-classic-rcn6ip")
    args = p.parse_args()

    client_id, client_secret = benchmark_sp_credentials()
    provider = build_token_provider(
        host=databricks_host(),
        client_id=client_id,
        client_secret=client_secret,
        profile=args.profile,
    )
    token = provider()
    host = args.host.rstrip("/").replace("https://", "").replace("http://", "")
    prefix = "/" + args.prefix.strip("/")

    import requests

    for path in ("/healthz", "/v1/capabilities", "/v1/languages"):
        url = f"https://{host}{prefix}{path}"
        r = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=30)
        print(f"GET {path}: HTTP {r.status_code} {r.text[:200]}")

    routes = [
        ("/v1/speech-to-text", "stt"),
        ("/v1/speech-llm-toolassist-speech", "assist"),
        ("/v1/text-to-speech", "tts"),
        ("/v1/realtime/voice", "assist"),
    ]
    results = []
    for path, mode in routes:
        url = f"wss://{host}{prefix}{path}"
        print(f"\nProbing {url} ({mode})...")
        result = asyncio.run(_probe_ws(url, token, mode))
        results.append(result)
        print(json.dumps(result, indent=2))

    ok = [r for r in results if r.get("ok")]
    print(f"\nSummary: {len(ok)}/{len(results)} routes OK")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
