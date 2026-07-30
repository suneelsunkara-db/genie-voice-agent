"""Measure realtime TTS first-audio latency through the REAL app code path.

``profile_tts_transport.py`` posts to the endpoint directly, so it prices the
endpoint and the wire but not our own client-side delay. This probe instead calls
``DatabricksServing.synthesize_stream`` -- the same method ``stream_tts`` uses --
and builds the voice reference exactly the way a live session does, so the
numbers are directly comparable to the ``tts_first_ms`` we record in traces.

Reference fidelity matters: ``_lock_voice_reference`` captures
``_VOICE_REFERENCE_SECONDS`` (4.0s) of 48kHz mono PCM16 and wraps it as a WAV,
which is ~512KB once base64-encoded. Probing with a shorter clip understates the
per-turn upload cost, so this script reproduces the real capture and also runs a
smaller clip to confirm the size -> latency relationship.

Each request is drained to completion with a settle delay in between; breaking
out early leaves the replica generating and makes every later timing queue
behind it on a single-replica ("Small") endpoint.

Read-only: no config, endpoint or app changes.

Usage:
    python scripts/diag/profile_tts_app_path.py --reps 3 --settle 6
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import statistics
import sys
import time
import wave
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

TEXT = "I'm connecting you to Telco billing support right away."
SEED_TEXT = (
    "Hello, thanks for calling. I can help you with your account today, "
    "and I'll walk you through whatever you need."
)


def _capture_reference(serving: Any, seconds: float) -> tuple[str, int]:
    """Build a session voice reference the way ``_lock_voice_reference`` does.

    Streams a first utterance, keeps the leading ``seconds`` of PCM, and wraps it
    as a mono 16-bit WAV -- byte-for-byte the shape the live pipeline uploads.
    """
    capture = bytearray()
    sample_rate = 48_000
    for chunk in serving.synthesize_stream(SEED_TEXT, language="en-US"):
        sample_rate = chunk.sample_rate_hz or sample_rate
        if len(capture) < int(sample_rate * 2 * seconds):
            capture.extend(chunk.pcm)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(bytes(capture))
    wav = buffer.getvalue()
    return base64.b64encode(wav).decode("ascii"), len(wav)


def _measure(
    serving: Any, reference_b64: str | None, voice_id: str | None = None
) -> dict[str, Any]:
    """Time one synthesis through the app's own streaming method."""
    start = time.perf_counter()
    first_ms: float | None = None
    chunks = 0
    pcm_bytes = 0
    sample_rate = 0
    srv_ttfb: float | None = None
    srv_gen: float | None = None

    for chunk in serving.synthesize_stream(
        TEXT, language="en-US", reference_audio_b64=reference_b64, voice_id=voice_id
    ):
        if first_ms is None:
            first_ms = (time.perf_counter() - start) * 1000.0
        chunks += 1
        pcm_bytes += len(chunk.pcm)
        sample_rate = chunk.sample_rate_hz or sample_rate
        if chunk.server_ttfb_ms is not None:
            srv_ttfb = chunk.server_ttfb_ms
        if chunk.server_gen_ms is not None:
            srv_gen = chunk.server_gen_ms

    total_ms = (time.perf_counter() - start) * 1000.0
    audio_ms = (pcm_bytes / 2 / sample_rate * 1000.0) if sample_rate else 0.0
    return {
        "cli_first_ms": round(first_ms or 0.0, 1),
        "cli_total_ms": round(total_ms, 1),
        "srv_ttfb_ms": srv_ttfb,
        "srv_gen_ms": srv_gen,
        "chunks": chunks,
        "audio_ms": round(audio_ms, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--settle", type=float, default=6.0)
    args = parser.parse_args()

    from realtime_api.pipelines._shared import _VOICE_REFERENCE_SECONDS
    from realtime_api.serving_factory import shared_serving

    serving = shared_serving()
    print(f"endpoint={serving.tts_endpoint} timesteps={serving.tts_inference_timesteps} "
          f"cfg={serving.tts_cfg_value}")
    print(f"via DatabricksServing.synthesize_stream (app path), reps={args.reps} "
          f"settle={args.settle}s\n")

    print(f"capturing app-exact reference ({_VOICE_REFERENCE_SECONDS}s @48kHz)...")
    full_ref, full_wav = _capture_reference(serving, _VOICE_REFERENCE_SECONDS)
    time.sleep(args.settle)
    print(f"capturing short reference (1.0s) for comparison...")
    short_ref, short_wav = _capture_reference(serving, 1.0)
    print(f"  app-exact : wav={full_wav / 1024:.0f}KB  b64={len(full_ref) / 1024:.0f}KB")
    print(f"  short     : wav={short_wav / 1024:.0f}KB  b64={len(short_ref) / 1024:.0f}KB\n")

    variants = [
        ("no reference", None),
        ("short ref", short_ref),
        ("app ref (4s)", full_ref),
    ]

    hdr = (f"{'variant':<14} {'ref_kb':>7} {'cli_1st':>8} {'srv_ttfb':>9} "
           f"{'overhead':>9} {'srv_gen':>8} {'audio':>7} {'rtf':>5}")
    print(hdr)
    print("-" * len(hdr))

    def _row(name: str, runs: list[dict[str, Any]], ref_kb: int) -> None:
        def med(key: str) -> float | None:
            vals = [r[key] for r in runs if isinstance(r.get(key), (int, float))]
            return round(statistics.median(vals), 1) if vals else None

        cli, srv = med("cli_first_ms"), med("srv_ttfb_ms")
        gen, aud = med("srv_gen_ms"), med("audio_ms")
        overhead = round(cli - srv, 1) if (cli is not None and srv is not None) else None
        rtf = round(gen / aud, 2) if (gen and aud) else None
        print(f"{name:<14} {ref_kb:>7} {cli if cli is not None else '-':>8} "
              f"{srv if srv is not None else '-':>9} "
              f"{overhead if overhead is not None else '-':>9} "
              f"{gen if gen is not None else '-':>8} "
              f"{aud if aud is not None else '-':>7} "
              f"{rtf if rtf is not None else '-':>5}")

    def _collect(name: str, ref: str | None, reps: int, voice_id: str | None = None) -> list[dict[str, Any]]:
        runs: list[dict[str, Any]] = []
        for _ in range(reps):
            time.sleep(args.settle)
            try:
                runs.append(_measure(serving, ref, voice_id))
            except Exception as exc:  # noqa: BLE001
                print(f"{name:<14} ERROR {type(exc).__name__}: {exc}"[:160])
                return []
        return runs

    for name, ref in variants:
        runs = _collect(name, ref, args.reps)
        if runs:
            _row(name, runs, round(len(ref) / 1024) if ref else 0)

    # voice_id path: the clip is uploaded on the first turn only, then addressed
    # by id. Turn 1 is reported separately from steady state because they are
    # deliberately different amounts of work.
    print()
    print("with voice_id (clip uploaded once, then cached at the endpoint):")
    print("-" * len(hdr))
    voice_id = hashlib.sha256(full_ref.encode("ascii")).hexdigest()[:32]
    serving._voice_ids_sent.discard(voice_id)
    first = _collect("turn 1", full_ref, 1, voice_id)
    if first:
        _row("turn 1 upload", first, round(len(full_ref) / 1024))
    cached = _collect("cached", full_ref, args.reps, voice_id)
    if cached:
        _row("turns 2+ id", cached, 0)


if __name__ == "__main__":
    main()
