"""Smoke-test the realtime voice ResponsesAgent endpoints.

Queries the STT and/or TTS agent endpoints with the same Responses shape the
FastAPI adapter uses (``input`` + ``custom_inputs``), verifies structured output
in ``custom_outputs``, and records per-call latency. Endpoints are resolved from
the merged ``realtime_voice:`` config block.

Default mode is a multilingual round-trip: for each language the TTS endpoint
synthesizes a sample sentence, then that speech is fed back into the STT endpoint.
This validates both models with real speech (no external audio files) and yields a
per-language quality signal.

Examples:
    # multilingual round-trip (default 4 validation languages)
    python scripts/ml_asr/smoke_realtime_voice_agents.py

    # single direction
    python scripts/ml_asr/smoke_realtime_voice_agents.py --mode separate --languages en-US
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import struct
import time
import wave
from typing import Any

from _realtime_config import databricks, find_candidate, realtime_voice

_SAMPLE_TEXT = {
    "en-US": "Hello, this is a realtime voice quality check.",
    "th-TH": "สวัสดีครับ นี่คือการทดสอบคุณภาพเสียงแบบเรียลไทม์",
    "id-ID": "Halo, ini adalah pemeriksaan kualitas suara secara real time.",
    "zh-CN": "你好，这是一次实时语音质量测试。",
    "es-ES": "Hola, esta es una prueba de calidad de voz en tiempo real.",
    "ja-JP": "こんにちは、これはリアルタイム音声品質のテストです。",
}


def _tone_wav(seconds: float = 0.6, sample_rate_hz: int = 16_000, freq: float = 180.0) -> bytes:
    frames = int(seconds * sample_rate_hz)
    pcm = b"".join(
        struct.pack("<h", int(0.2 * 32767 * math.sin(2 * math.pi * freq * n / sample_rate_hz)))
        for n in range(frames)
    )
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate_hz)
        handle.writeframes(pcm)
    return buffer.getvalue()


def _wav_sample_rate(wav_bytes: bytes) -> int:
    with wave.open(io.BytesIO(wav_bytes), "rb") as handle:
        return handle.getframerate()


def _predict(client: Any, endpoint: str, text: str, custom_inputs: dict[str, Any]) -> tuple[dict[str, Any], float]:
    start = time.perf_counter()
    response = client.api_client.do(
        "POST",
        f"/serving-endpoints/{endpoint}/invocations",
        body={"input": [{"role": "user", "content": text}], "custom_inputs": custom_inputs},
    )
    elapsed_ms = (time.perf_counter() - start) * 1000
    return (response if isinstance(response, dict) else dict(response)), elapsed_ms


def _custom(response: dict[str, Any]) -> dict[str, Any]:
    custom = response.get("custom_outputs")
    return custom if isinstance(custom, dict) else {}


def _synthesize(client: Any, endpoint: str, text: str, language: str) -> tuple[dict[str, Any], float]:
    response, elapsed = _predict(client, endpoint, text, {"text": text, "language": language})
    return _custom(response), elapsed


def _transcribe(client: Any, endpoint: str, audio_b64: str, language: str, sample_rate_hz: int) -> tuple[dict[str, Any], float]:
    response, elapsed = _predict(
        client, endpoint, "transcribe",
        {"audio_b64": audio_b64, "language": language, "sample_rate_hz": sample_rate_hz},
    )
    return _custom(response), elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stt", default=None, help="STT candidate id (default: first configured)")
    parser.add_argument("--tts", default=None, help="TTS candidate id (default: first configured)")
    parser.add_argument("--languages", default="en-US,th-TH,id-ID,zh-CN", help="Comma-separated BCP 47 tags")
    parser.add_argument("--mode", choices=["roundtrip", "separate"], default="roundtrip")
    args = parser.parse_args()

    rv = realtime_voice()
    stt_id = args.stt or next(iter(rv.get("stt_candidates") or {}), None)
    tts_id = args.tts or next(iter(rv.get("tts_candidates") or {}), None)
    stt_ep = find_candidate(stt_id)["endpoint"] if stt_id else None
    tts_ep = find_candidate(tts_id)["endpoint"] if tts_id else None
    languages = [x.strip() for x in args.languages.split(",") if x.strip()]

    from databricks.sdk import WorkspaceClient

    client = WorkspaceClient(profile=databricks().get("profile") or None)
    report: dict[str, Any] = {"mode": args.mode, "stt_endpoint": stt_ep, "tts_endpoint": tts_ep, "languages": {}}

    for language in languages:
        text = _SAMPLE_TEXT.get(language, _SAMPLE_TEXT["en-US"])
        entry: dict[str, Any] = {}
        audio_b64, sr = "", 16_000

        if tts_ep:
            try:
                custom, ms = _synthesize(client, tts_ep, text, language)
                audio_b64 = str(custom.get("audio_b64") or "")
                sr = int(custom.get("sample_rate_hz") or 0) or (_wav_sample_rate(base64.b64decode(audio_b64)) if audio_b64 else 16_000)
                entry["tts"] = {
                    "latency_ms": round(ms, 1),
                    "device": custom.get("inference_device"),
                    "audio_bytes": len(base64.b64decode(audio_b64)) if audio_b64 else 0,
                    "sample_rate_hz": sr,
                    "ok": bool(audio_b64),
                }
            except Exception as exc:  # noqa: BLE001
                entry["tts"] = {"ok": False, "error": str(exc)[:200]}

        if stt_ep:
            try:
                if args.mode == "roundtrip" and audio_b64:
                    custom, ms = _transcribe(client, stt_ep, audio_b64, language, sr)
                    source = "tts_roundtrip"
                else:
                    custom, ms = _transcribe(client, stt_ep, base64.b64encode(_tone_wav()).decode("ascii"), language, 16_000)
                    source = "tone"
                entry["stt"] = {
                    "latency_ms": round(ms, 1),
                    "device": custom.get("inference_device"),
                    "source": source,
                    "reference_text": text,
                    "transcript": custom.get("transcript"),
                    "detected_language": custom.get("detected_language"),
                    "ok": "transcript" in custom,
                }
            except Exception as exc:  # noqa: BLE001
                entry["stt"] = {"ok": False, "error": str(exc)[:200]}

        report["languages"][language] = entry

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
