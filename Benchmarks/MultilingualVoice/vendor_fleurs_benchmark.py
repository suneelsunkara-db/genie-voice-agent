"""Run measured vendor baselines on FLEURS without touching the Genie run.

This script writes separate dataset ids into the existing benchmark Delta tables:

  - fleurs_deepgram_stt     : Deepgram STT on original FLEURS audio
  - fleurs_elevenlabs_tts   : ElevenLabs TTS on FLEURS text, transcribed by Deepgram

Keeping distinct dataset ids avoids changing the table schema or merge key while
the main Genie FLEURS job is running.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import time
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def _resolve_benchmark_dir() -> Path:
    raw = os.environ.get("MLV_BENCHMARK_DIR")
    if raw:
        return Path(raw)
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd()


_BENCHMARK_DIR = _resolve_benchmark_dir()
sys.path.insert(0, str(_BENCHMARK_DIR))

from evaluators import EVALUATORS, primary_metric, primary_metric_name  # noqa: E402
import languages as langmap  # noqa: E402
from paths import benchmark_results_dir  # noqa: E402
from results_store import write_run  # noqa: E402
from staging import load_staged  # noqa: E402
from volume_logging import setup_run_logging  # noqa: E402


DEEPGRAM_LISTEN_URL = "https://api.deepgram.com/v1/listen"
ELEVENLABS_TTS_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/with-timestamps"


def _apply_job_wiring(args: argparse.Namespace) -> None:
    if args.benchmark_dir:
        os.environ["MLV_BENCHMARK_DIR"] = args.benchmark_dir
    if args.config:
        os.environ["GENIE_CONFIG"] = args.config
    if args.databricks_host:
        os.environ["DATABRICKS_HOST"] = args.databricks_host
    if args.secret_scope:
        from databricks_auth import read_workspace_secret

        if args.deepgram_secret_key:
            os.environ["DEEPGRAM_API_KEY"] = read_workspace_secret(args.secret_scope, args.deepgram_secret_key)
        if args.elevenlabs_secret_key:
            os.environ["ELEVENLABS_API_KEY"] = read_workspace_secret(args.secret_scope, args.elevenlabs_secret_key)
        if args.hf_secret_key:
            token = read_workspace_secret(args.secret_scope, args.hf_secret_key)
            os.environ["HF_TOKEN"] = token
            os.environ["HUGGING_FACE_HUB_TOKEN"] = token


def _pcm16_wav(pcm: bytes, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


def _deepgram_transcribe(
    audio: bytes,
    *,
    content_type: str,
    language: str,
    model: str,
    api_key: str,
) -> dict[str, Any]:
    params = {
        "model": model,
        "language": language,
        "smart_format": "true",
        "punctuate": "true",
    }
    req = Request(
        f"{DEEPGRAM_LISTEN_URL}?{urlencode(params)}",
        data=audio,
        method="POST",
    )
    req.add_header("Authorization", f"Token {api_key}")
    req.add_header("Content-Type", content_type)
    req.add_header("Accept", "application/json")
    started = time.perf_counter()
    with urlopen(req, timeout=120) as resp:
        raw = json.loads(resp.read().decode("utf-8"))
    latency_ms = round((time.perf_counter() - started) * 1000)
    channels = raw.get("results", {}).get("channels") or []
    alternatives = channels[0].get("alternatives") if channels else []
    alt = alternatives[0] if alternatives else {}
    return {
        "transcript": str(alt.get("transcript") or "").strip(),
        "confidence": alt.get("confidence"),
        "latency_ms": latency_ms,
        "raw": raw,
    }


def _elevenlabs_synthesize(
    text: str,
    *,
    api_key: str,
    model_id: str,
    voice_id: str,
    output_format: str,
) -> dict[str, Any]:
    url = ELEVENLABS_TTS_URL.format(voice_id=voice_id) + "?" + urlencode({"output_format": output_format})
    payload = json.dumps({"text": text, "model_id": model_id}).encode("utf-8")
    req = Request(url, data=payload, method="POST")
    req.add_header("xi-api-key", api_key)
    req.add_header("content-type", "application/json")
    req.add_header("accept", "application/json")
    started = time.perf_counter()
    with urlopen(req, timeout=120) as resp:
        raw = json.loads(resp.read().decode("utf-8"))
    latency_ms = round((time.perf_counter() - started) * 1000)
    audio = base64.b64decode(raw.get("audio_base64") or "")
    return {"audio": audio, "latency_ms": latency_ms, "raw": raw}


def _deepgram_language(language: str) -> str:
    """Deepgram accepts ISO-ish language ids; keep known FLEURS base tags simple."""
    return {"zh": "zh", "fil": "tl"}.get(language, language)


def _percentiles(values: list[int]) -> dict[str, int]:
    if not values:
        return {}
    sorted_v = sorted(values)
    n = len(sorted_v)
    return {
        "p50": sorted_v[n // 2],
        "p95": sorted_v[min(n - 1, int(n * 0.95))],
        "p99": sorted_v[min(n - 1, int(n * 0.99))],
        "mean": round(sum(sorted_v) / n),
    }


def _staged_samples(args: argparse.Namespace, lang: str):
    out_dir = Path(args.out_dir) if args.out_dir else benchmark_results_dir()
    return load_staged("fleurs", lang, args.limit, out_dir=out_dir)


def _run_deepgram_stt(args: argparse.Namespace, lang: str) -> tuple[dict, list[dict]]:
    api_key = os.environ["DEEPGRAM_API_KEY"]
    rows: list[dict] = []
    latencies: list[int] = []
    errors = 0
    for sample in _staged_samples(args, lang):
        try:
            wav = _pcm16_wav(sample["pcm"], sample["sample_rate"])
            result = _deepgram_transcribe(
                wav,
                content_type="audio/wav",
                language=_deepgram_language(lang),
                model=args.deepgram_model,
                api_key=api_key,
            )
            transcript = result["transcript"]
            stt_ms = result["latency_ms"]
            latencies.append(stt_ms)
            error = ""
        except Exception as exc:  # noqa: BLE001
            transcript = ""
            stt_ms = None
            error = f"{type(exc).__name__}: {exc}"
            errors += 1
        rows.append({
            "dataset": "fleurs_deepgram_stt",
            "language": lang,
            "reference": sample.get("reference"),
            "transcript": transcript,
            "response": "",
            "detected_language": lang,
            "tts_audio_bytes": 0,
            "tts_roundtrip": {},
            "error": error,
            "stt_ms": stt_ms,
            "llm_ms": None,
            "tts_first_ms": None,
            "client_ttfa_ms": None,
            "total_ms": stt_ms,
        })
    scores = EVALUATORS["asr"](rows)
    scores["primary_metric"] = primary_metric_name("asr", lang)
    run = {
        "dataset": "fleurs_deepgram_stt",
        "language": lang,
        "evaluator": "asr",
        "samples": len(rows),
        "errors": errors,
        "issues": {"count": errors, "by_kind": {"vendor_error": errors} if errors else {}},
        "primary_score": primary_metric("asr", scores, language=lang),
        "scores": scores,
        "latency_ms": {"stt_ms": _percentiles(latencies)},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return run, rows


def _run_elevenlabs_tts(args: argparse.Namespace, lang: str) -> tuple[dict, list[dict]]:
    eleven_key = os.environ["ELEVENLABS_API_KEY"]
    deepgram_key = os.environ["DEEPGRAM_API_KEY"]
    rows: list[dict] = []
    stt_latencies: list[int] = []
    tts_latencies: list[int] = []
    errors = 0
    for sample in _staged_samples(args, lang):
        reference = str(sample.get("reference") or "").strip()
        try:
            synth = _elevenlabs_synthesize(
                reference,
                api_key=eleven_key,
                model_id=args.elevenlabs_model,
                voice_id=args.elevenlabs_voice,
                output_format=args.elevenlabs_output_format,
            )
            heard = _deepgram_transcribe(
                synth["audio"],
                content_type="audio/mpeg",
                language=_deepgram_language(lang),
                model=args.deepgram_model,
                api_key=deepgram_key,
            )
            transcript = heard["transcript"]
            stt_ms = heard["latency_ms"]
            tts_ms = synth["latency_ms"]
            stt_latencies.append(stt_ms)
            tts_latencies.append(tts_ms)
            audio_bytes = len(synth["audio"])
            error = ""
        except Exception as exc:  # noqa: BLE001
            transcript = ""
            stt_ms = None
            tts_ms = None
            audio_bytes = 0
            error = f"{type(exc).__name__}: {exc}"
            errors += 1
        rows.append({
            "dataset": "fleurs_elevenlabs_tts",
            "language": lang,
            "reference": reference,
            "transcript": transcript,
            "response": reference,
            "detected_language": lang,
            "tts_audio_bytes": audio_bytes,
            "tts_roundtrip": {"spoken_text": reference, "reheard_text": transcript, "error": error or None},
            "error": error,
            "stt_ms": stt_ms,
            "llm_ms": None,
            "tts_first_ms": tts_ms,
            "client_ttfa_ms": tts_ms,
            "total_ms": (stt_ms + tts_ms) if stt_ms is not None and tts_ms is not None else None,
        })
    scores = EVALUATORS["asr"](rows)
    scores["tts_roundtrip_wer"] = scores.get("wer")
    scores["tts_roundtrip_cer"] = scores.get("cer")
    scores["tts_audio_rate"] = (sum(1 for r in rows if r.get("tts_audio_bytes")) / len(rows)) if rows else None
    scores["primary_metric"] = primary_metric_name("asr", lang)
    run = {
        "dataset": "fleurs_elevenlabs_tts",
        "language": lang,
        "evaluator": "asr",
        "samples": len(rows),
        "errors": errors,
        "issues": {"count": errors, "by_kind": {"vendor_error": errors} if errors else {}},
        "primary_score": primary_metric("asr", scores, language=lang),
        "scores": scores,
        "latency_ms": {"stt_ms": _percentiles(stt_latencies), "tts_first_ms": _percentiles(tts_latencies)},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return run, rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vendor FLEURS baseline job")
    parser.add_argument("--vendors", default="deepgram,elevenlabs")
    parser.add_argument("--vendor", default="", choices=["", "deepgram", "elevenlabs"])
    parser.add_argument("--languages", default="")
    parser.add_argument("--language", default="")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--benchmark-dir", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--databricks-host", default=None)
    parser.add_argument("--secret-scope", default=None)
    parser.add_argument("--deepgram-secret-key", default=None)
    parser.add_argument("--elevenlabs-secret-key", default=None)
    parser.add_argument("--hf-secret-key", default=None)
    parser.add_argument("--deepgram-model", default="nova-3")
    parser.add_argument("--elevenlabs-model", default="eleven_turbo_v2_5")
    parser.add_argument("--elevenlabs-voice", default="Rachel")
    parser.add_argument("--elevenlabs-output-format", default="mp3_44100_128")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _apply_job_wiring(args)
    requested = [args.language.strip()] if args.language.strip() else [
        c.strip() for c in args.languages.split(",") if c.strip()
    ] or None
    languages = langmap.resolve_languages("fleurs", requested)
    vendors = {args.vendor.strip()} if args.vendor.strip() else {v.strip() for v in args.vendors.split(",") if v.strip()}
    out_dir = Path(args.out_dir) if args.out_dir else benchmark_results_dir()
    label_bits = [args.run_id, args.vendor or "vendors", args.language or "all"]
    log_mgr = setup_run_logging(out_dir, run_label="_".join(label_bits))

    for lang in languages:
        if "deepgram" in vendors:
            run, rows = _run_deepgram_stt(args, lang)
            write_run(args.run_id, run, sample_rows=rows)
            log_mgr.checkpoint_to_volume()
        if "elevenlabs" in vendors:
            run, rows = _run_elevenlabs_tts(args, lang)
            write_run(args.run_id, run, sample_rows=rows)
            log_mgr.checkpoint_to_volume()
    return 0


if __name__ == "__main__":
    _code = main()
    if _code:
        raise SystemExit(_code)
