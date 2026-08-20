"""Diagnostic: capture the ACTUAL agent voice the billing-support path emits
from our realtime API (WebSocket), end-to-end, and measure it.

Drives the real FastAPI app (realtime_api.create_app + the shared serving) over
the SAME route and explicit profile the billing UI uses
(/realtime/v1/speech-llm-toolassist-speech, profile="billing"), runs a full
STT -> LLM -> streaming-TTS turn, and
reassembles the response.audio PCM the server sends to the browser. Reports the
objective audio signals (sample rate, peak, RMS, crest, clipping) so we compare
billing's real output against the card path on identical terms.

    python scripts/diag/capture_billing_voice.py --profile billing
    python scripts/diag/capture_billing_voice.py --profile card
"""
from __future__ import annotations

import argparse
import base64
import sys
import wave
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
for p in (str(_REPO), str(_REPO / "backend")):
    if p not in sys.path:
        sys.path.insert(0, p)

from fastapi.testclient import TestClient  # noqa: E402

from realtime_api.app import create_app  # noqa: E402
from realtime_api.config import RealtimeSettings  # noqa: E402
from realtime_api.pipelines import ServingBundle  # noqa: E402
from realtime_api.serving_factory import build_serving  # noqa: E402

FULL = 32767.0
# realtime_api.create_app serves the route directly; the "/realtime" prefix the
# browser uses is added by the api/app mount. Same pipeline code either way.
ROUTE = "/v1/speech-llm-toolassist-speech"


def analyze(pcm_bytes: bytes, sr: int, label: str) -> None:
    a = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    if a.size == 0:
        print(f"{label}: EMPTY audio")
        return
    peak = float(np.max(np.abs(a)))
    clip_pct = float(np.mean(np.abs(a) >= FULL * 0.999) * 100)
    rms = float(np.sqrt(np.mean((a / FULL) ** 2)))
    peak_db = 20 * np.log10(max(peak, 1) / FULL)
    rms_db = 20 * np.log10(max(rms, 1e-9))
    dur = a.size / sr if sr else 0
    print(
        f"{label}:\n"
        f"    sample_rate = {sr} Hz   duration = {dur:.2f}s\n"
        f"    peak        = {peak:.0f}/{FULL:.0f} ({peak_db:+.2f} dBFS)   "
        f"rms = {rms:.4f} ({rms_db:+.2f} dBFS)   crest = {peak_db - rms_db:.1f} dB\n"
        f"    clipping    = {clip_pct:.3f}% at full-scale\n"
    )


def prompt_16k_pcm(serving, text: str, language: str = "en-US") -> bytes:
    r = serving.synthesize(text, language=language)
    wav = r.audio
    body = wav[44:] if wav[:4] == b"RIFF" else wav
    a = np.frombuffer(body, dtype=np.int16)
    factor = max(1, round(r.sample_rate_hz / 16_000))
    return a[::factor].astype("<i2").tobytes()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        default="billing",
        choices=["billing", "telco", "card"],
        help="'telco' remains a CLI alias for the registered 'billing' profile",
    )
    parser.add_argument(
        "--prompt",
        default="Why is my bill higher than last month?",
        help="Customer utterance to synthesize and stream in as the caller.",
    )
    args = parser.parse_args()

    settings = RealtimeSettings.resolve()
    serving = build_serving(settings)
    app = create_app(
        settings=settings,
        bundle_factory=lambda _s: ServingBundle(stt=serving, llm=serving, tts=serving),
    )
    print(
        f"route={ROUTE}  profile={args.profile}  "
        f"tts_endpoint={settings.tts_endpoint}  cfg_value={settings.tts_cfg_value}\n"
    )

    pcm_in = prompt_16k_pcm(serving, args.prompt)

    start = {
        "type": "session.start",
        "language": "auto",
        "sample_rate_hz": 16_000,
        "encoding": "pcm_s16le",
        "call_id": "diag-billing",
        "customer_id": "CUST-0001",
    }
    start["profile"] = "card" if args.profile == "card" else "billing"

    audio_chunks: list[bytes] = []
    sr_out = 0
    chunk_rates: set[int] = set()
    events_seen: list[str] = []

    with TestClient(app) as http, http.websocket_connect(ROUTE) as ws:
        ws.send_json(start)
        ready = ws.receive_json()
        events_seen.append(ready.get("type"))
        print(f"session.ready -> {ready}")

        for off in range(0, len(pcm_in), 640):  # 20 ms @ 16 kHz
            ws.send_bytes(pcm_in[off : off + 640])
        ws.send_json({"type": "audio.end"})

        while True:
            ev = ws.receive_json()
            etype = ev.get("type")
            if etype == "transcript.final":
                print(f"transcript.final -> {ev.get('text')!r}  (lang={ev.get('language')})")
            elif etype == "response.text":
                print(f"response.text    -> {ev.get('text')!r}")
            elif etype == "tool.called":
                print(f"tool.called      -> {ev.get('name')}")
            elif etype == "response.audio":
                b = ev.get("audio_b64")
                if b:
                    audio_chunks.append(base64.b64decode(b))
                    sr_out = int(ev.get("sample_rate_hz") or sr_out)
                    chunk_rates.add(int(ev.get("sample_rate_hz") or 0))
                if ev.get("final"):
                    if ev.get("server_gen_ms") is not None:
                        print(
                            f"response.audio   -> FINAL  server_gen_ms={ev.get('server_gen_ms')} "
                            f"server_ttfb_ms={ev.get('server_ttfb_ms')}"
                        )
                    break
            elif etype == "error":
                print(f"ERROR -> {ev}")
                break

    pcm = b"".join(audio_chunks)
    print(
        f"\ncaptured {len(audio_chunks)} response.audio chunks, "
        f"{len(pcm)} PCM bytes, chunk sample-rates={sorted(chunk_rates)}\n"
    )
    out = f"/tmp/agent_voice_{args.profile}.wav"
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr_out or 48_000)
        w.writeframes(pcm)
    analyze(pcm, sr_out or 48_000, f"AGENT VOICE via API ({args.profile})")
    print(f"    wrote {out}")


if __name__ == "__main__":
    main()
