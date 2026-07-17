"""Profile VoxCPM2 TTS latency vs quality across ``inference_timesteps``.

For each language and each timestep setting the script:
  1. calls the TTS endpoint N times (warm), recording server ``gen_ms``/``rtf``
     and wall-clock end-to-end latency (median + p95),
  2. feeds the synthesized audio back through the STT endpoint once and scores a
     character-level similarity against the reference text as a quality signal.

The endpoint exposes ``inference_timesteps``/``cfg_value`` via ``custom_inputs``,
so the whole sweep runs live against one deployed model (no redeploy per point).

Examples:
    python scripts/ml_asr/profile_tts_latency.py
    python scripts/ml_asr/profile_tts_latency.py --timesteps 4,6,8,10 --reps 3 --languages en-US,zh-CN
"""
from __future__ import annotations

import argparse
import base64
import difflib
import json
import statistics
import time
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


def _predict(client: Any, endpoint: str, text: str, custom_inputs: dict[str, Any]) -> tuple[dict[str, Any], float]:
    start = time.perf_counter()
    response = client.api_client.do(
        "POST",
        f"/serving-endpoints/{endpoint}/invocations",
        body={"input": [{"role": "user", "content": text}], "custom_inputs": custom_inputs},
    )
    elapsed_ms = (time.perf_counter() - start) * 1000
    payload = response if isinstance(response, dict) else dict(response)
    custom = payload.get("custom_outputs")
    return (custom if isinstance(custom, dict) else {}), elapsed_ms


def _similarity(reference: str, hypothesis: str) -> float:
    """Character-level ratio, whitespace-insensitive (0..1). CJK has no spaces."""
    ref = "".join((reference or "").split())
    hyp = "".join((hypothesis or "").split())
    if not ref:
        return 0.0
    return round(difflib.SequenceMatcher(None, ref, hyp).ratio(), 3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tts", default=None, help="TTS candidate id (default: first configured)")
    parser.add_argument("--stt", default=None, help="STT candidate id for quality roundtrip (default: first)")
    parser.add_argument("--languages", default="en-US,th-TH,id-ID,zh-CN")
    parser.add_argument("--timesteps", default="4,6,8,10")
    parser.add_argument("--cfg", type=float, default=2.0)
    parser.add_argument("--reps", type=int, default=3, help="Warm reps per (language, timesteps) point")
    parser.add_argument("--no-quality", action="store_true", help="Skip the STT roundtrip quality check")
    args = parser.parse_args()

    rv = realtime_voice()
    tts_id = args.tts or next(iter(rv.get("tts_candidates") or {}), None)
    stt_id = args.stt or next(iter(rv.get("stt_candidates") or {}), None)
    tts_ep = find_candidate(tts_id)["endpoint"] if tts_id else None
    stt_ep = find_candidate(stt_id)["endpoint"] if stt_id else None
    if not tts_ep:
        parser.error("No TTS endpoint resolved")
    languages = [x.strip() for x in args.languages.split(",") if x.strip()]
    timesteps = [int(x) for x in args.timesteps.split(",") if x.strip()]

    from databricks.sdk import WorkspaceClient

    client = WorkspaceClient(profile=databricks().get("profile") or None)
    report: dict[str, Any] = {
        "tts_endpoint": tts_ep,
        "stt_endpoint": stt_ep if not args.no_quality else None,
        "cfg_value": args.cfg,
        "reps": args.reps,
        "results": [],
    }

    for language in languages:
        text = _SAMPLE_TEXT.get(language, _SAMPLE_TEXT["en-US"])
        for steps in timesteps:
            e2e: list[float] = []
            gen: list[float] = []
            rtf: list[float] = []
            last_audio_b64, last_sr = "", 0
            device = None
            error = None
            for _ in range(args.reps):
                try:
                    custom, ms = _predict(
                        client, tts_ep, text,
                        {"text": text, "language": language, "inference_timesteps": steps, "cfg_value": args.cfg},
                    )
                    e2e.append(ms)
                    if custom.get("gen_ms") is not None:
                        gen.append(float(custom["gen_ms"]))
                    if custom.get("rtf") is not None:
                        rtf.append(float(custom["rtf"]))
                    last_audio_b64 = str(custom.get("audio_b64") or "")
                    last_sr = int(custom.get("sample_rate_hz") or 0)
                    device = custom.get("inference_device")
                except Exception as exc:  # noqa: BLE001
                    error = str(exc)[:200]
                    break

            point: dict[str, Any] = {"language": language, "inference_timesteps": steps, "device": device}
            if error:
                point["error"] = error
                report["results"].append(point)
                continue

            point["e2e_ms_median"] = round(statistics.median(e2e), 1)
            point["e2e_ms_max"] = round(max(e2e), 1)
            if gen:
                point["gen_ms_median"] = round(statistics.median(gen), 1)
            if rtf:
                point["rtf_median"] = round(statistics.median(rtf), 3)

            if not args.no_quality and stt_ep and last_audio_b64:
                try:
                    stt_custom, stt_ms = _predict(
                        client, stt_ep, "transcribe",
                        {"audio_b64": last_audio_b64, "language": language, "sample_rate_hz": last_sr or 48_000},
                    )
                    transcript = str(stt_custom.get("transcript") or "")
                    point["quality"] = {
                        "stt_ms": round(stt_ms, 1),
                        "similarity": _similarity(text, transcript),
                        "transcript": transcript,
                    }
                except Exception as exc:  # noqa: BLE001
                    point["quality"] = {"error": str(exc)[:200]}

            report["results"].append(point)
            _print_row(point)

    print(json.dumps(report, indent=2, ensure_ascii=False))


def _print_row(p: dict[str, Any]) -> None:
    q = p.get("quality") or {}
    print(
        f"[{p['language']:>5}] steps={p['inference_timesteps']:>2} "
        f"gen={p.get('gen_ms_median', '?'):>7}ms e2e={p.get('e2e_ms_median', '?'):>7}ms "
        f"rtf={p.get('rtf_median', '?'):>5} sim={q.get('similarity', '-')}"
    )


if __name__ == "__main__":
    main()
