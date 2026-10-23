"""Benchmark VoxCPM2 streaming latency *and spoken-output integrity*.

Compares the two transport contracts on the SAME GPU:
  * streaming ``predict_stream`` -> validated waveform packetized over SSE
  * non-streaming ``predict``    -> the same validated waveform returned as WAV

Every generated waveform is sent through the configured Qwen STT endpoint. The
result is compared with the exact source text using multilingual character edits.
Insertion rate is the direct "random rambling" signal: words spoken by TTS that
were never present in the intended response. This is an evaluation-only loop;
production audio is not delayed or persisted by this script.

Usage:
    python scripts/ml_asr/benchmark_tts_streaming.py --languages all --reps 5
"""
from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path
import statistics
import time
import unicodedata
import wave
from collections.abc import Callable
from typing import Any

import requests

from _realtime_config import databricks, find_candidate, realtime_voice

_REPO = Path(__file__).resolve().parents[2]
_LOCALES = _REPO / "frontend" / "src" / "locales"
_DEFAULT_OUTPUT = _REPO / ".run" / "tts_integrity_benchmark.json"
_SAMPLE_TEXT = {
    "en-US": "Sure, our office opens at nine in the morning and closes at six in the evening.",
    "th-TH": "ได้ค่ะ สำนักงานของเราเปิดเวลาเก้าโมงเช้าและปิดเวลาหกโมงเย็น",
    "id-ID": "Tentu, kantor kami buka pukul sembilan pagi dan tutup pukul enam sore.",
    "zh-CN": "好的，我们的办公室早上九点开门，晚上六点关门。",
}


def _locale_texts() -> dict[str, str]:
    """Use the app's maintained translations as the 24-language gold set."""
    texts = dict(_SAMPLE_TEXT)
    for path in sorted(_LOCALES.glob("*.json")):
        language = "en-US" if path.stem == "en" else path.stem
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        text = str(payload.get("voiceStackDesc") or "").strip()
        if text:
            texts[language] = text
    return texts


def _normalize(text: str) -> str:
    folded = unicodedata.normalize("NFKC", text or "").casefold()
    return "".join(ch for ch in folded if ch.isalnum())


def _character_edits(reference: str, hypothesis: str) -> dict[str, Any]:
    """Levenshtein counts with deterministic backtracking for CER diagnostics."""
    ref = _normalize(reference)
    hyp = _normalize(hypothesis)
    rows, cols = len(ref) + 1, len(hyp) + 1
    distance = [[0] * cols for _ in range(rows)]
    operation = [[""] * cols for _ in range(rows)]
    for i in range(1, rows):
        distance[i][0], operation[i][0] = i, "delete"
    for j in range(1, cols):
        distance[0][j], operation[0][j] = j, "insert"
    for i in range(1, rows):
        for j in range(1, cols):
            if ref[i - 1] == hyp[j - 1]:
                distance[i][j], operation[i][j] = distance[i - 1][j - 1], "equal"
                continue
            candidates = (
                (distance[i - 1][j - 1] + 1, "substitute"),
                (distance[i - 1][j] + 1, "delete"),
                (distance[i][j - 1] + 1, "insert"),
            )
            distance[i][j], operation[i][j] = min(candidates, key=lambda item: item[0])
    counts = {"insertions": 0, "deletions": 0, "substitutions": 0}
    i, j = len(ref), len(hyp)
    while i or j:
        op = operation[i][j]
        if op == "equal":
            i, j = i - 1, j - 1
        elif op == "substitute":
            counts["substitutions"] += 1
            i, j = i - 1, j - 1
        elif op == "delete":
            counts["deletions"] += 1
            i -= 1
        else:
            counts["insertions"] += 1
            j -= 1
    denominator = max(len(ref), 1)
    edits = sum(counts.values())
    return {
        **counts,
        "reference_characters": len(ref),
        "hypothesis_characters": len(hyp),
        "cer": round(edits / denominator, 4),
        "insertion_rate": round(counts["insertions"] / denominator, 4),
        "exact_normalized": ref == hyp,
    }


def _auth() -> tuple[str, Callable[[], dict[str, str]]]:
    """Return the host and a factory for fresh auth headers.

    A factory rather than a dict: OAuth tokens live 60 minutes, and a full
    sweep across languages and reps runs longer than that.
    """
    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient(profile=databricks().get("profile") or None)
    host = w.config.host.rstrip("/")
    return host, lambda: {**dict(w.config.authenticate() or {}), "Content-Type": "application/json"}


def _stream_call(host: str, auth: Callable[[], dict[str, str]], endpoint: str, text: str, ci: dict[str, Any]) -> dict[str, Any]:
    url = f"{host}/serving-endpoints/{endpoint}/invocations"
    body = {"input": [{"role": "user", "content": text}], "custom_inputs": ci, "stream": True}
    started = time.perf_counter()
    ttfb: float | None = None
    chunks = 0
    pcm = bytearray()
    final: dict[str, Any] = {}
    with requests.post(url, headers=auth(), json=body, stream=True, timeout=120) as resp:
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
                pcm.extend(base64.b64decode(audio))
            if co.get("final"):
                final = co
    total_ms = (time.perf_counter() - started) * 1000.0
    return {
        "ttfb_ms": round(ttfb, 1) if ttfb is not None else None,
        "total_ms": round(total_ms, 1),
        "chunks": chunks,
        "audio_bytes": len(pcm),
        "audio_pcm": bytes(pcm),
        "server_gen_ms": final.get("gen_ms"),
        "server_ttfb_ms": final.get("ttfb_ms"),
        "sample_rate_hz": final.get("sample_rate_hz"),
        "device": final.get("inference_device"),
        "integrity_validated": bool(final.get("integrity_validated")),
    }


def _predict_call(host: str, auth: Callable[[], dict[str, str]], endpoint: str, text: str, ci: dict[str, Any]) -> dict[str, Any]:
    url = f"{host}/serving-endpoints/{endpoint}/invocations"
    body = {"input": [{"role": "user", "content": text}], "custom_inputs": ci}
    started = time.perf_counter()
    resp = requests.post(url, headers=auth(), json=body, timeout=120)
    resp.raise_for_status()
    e2e = (time.perf_counter() - started) * 1000.0
    co = (resp.json() or {}).get("custom_outputs") or {}
    return {
        "e2e_ms": round(e2e, 1),
        "gen_ms": co.get("gen_ms"),
        "device": co.get("inference_device"),
        "audio_b64": str(co.get("audio_b64") or ""),
        "sample_rate_hz": int(co.get("sample_rate_hz") or 0),
        "integrity_validated": bool(co.get("integrity_validated")),
    }


def _wav_b64(pcm: bytes, sample_rate_hz: int) -> str:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate_hz)
        handle.writeframes(pcm)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _quality_call(
    host: str,
    auth: Callable[[], dict[str, str]],
    endpoint: str,
    *,
    audio_b64: str,
    language: str,
    sample_rate_hz: int,
    reference: str,
) -> dict[str, Any]:
    url = f"{host}/serving-endpoints/{endpoint}/invocations"
    body = {
        "input": [{"role": "user", "content": "transcribe"}],
        "custom_inputs": {
            "audio_b64": audio_b64,
            "language": language,
            "sample_rate_hz": sample_rate_hz,
        },
    }
    started = time.perf_counter()
    response = requests.post(url, headers=auth(), json=body, timeout=120)
    response.raise_for_status()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    custom = (response.json() or {}).get("custom_outputs") or {}
    transcript = str(custom.get("transcript") or "")
    return {
        "stt_ms": round(elapsed_ms, 1),
        "transcript": transcript,
        **_character_edits(reference, transcript),
    }


def _median(values: list[float | int | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return round(statistics.median(clean), 2) if clean else None


def _summary(rows: list[dict[str, Any]], *, max_cer: float, max_insertion_rate: float) -> dict[str, Any]:
    scored = [row for row in rows if isinstance(row.get("quality"), dict)]
    failed = [
        row
        for row in scored
        if row["quality"]["cer"] > max_cer
        or row["quality"]["insertion_rate"] > max_insertion_rate
    ]
    return {
        "samples": len(scored),
        "failed_samples": len(failed),
        "pass_rate": round((len(scored) - len(failed)) / max(len(scored), 1), 4),
        "cer_median": _median([row["quality"]["cer"] for row in scored]),
        "insertion_rate_max": max(
            (row["quality"]["insertion_rate"] for row in scored), default=None
        ),
        "ttfb_ms_median": _median([row.get("ttfb_ms") for row in rows]),
        "total_ms_median": _median(
            [row.get("total_ms", row.get("e2e_ms")) for row in rows]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tts", default=None)
    parser.add_argument("--stt", default=None)
    parser.add_argument("--languages", default="all")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--max-cer", type=float, default=0.25)
    parser.add_argument("--max-insertion-rate", type=float, default=0.08)
    parser.add_argument("--no-quality", action="store_true")
    parser.add_argument("--output", default=str(_DEFAULT_OUTPUT))
    args = parser.parse_args()

    rv = realtime_voice()
    tts_id = args.tts or next(iter(rv.get("tts_candidates") or {}), None)
    stt_id = args.stt or next(iter(rv.get("stt_candidates") or {}), None)
    endpoint = find_candidate(tts_id)["endpoint"]
    stt_endpoint = find_candidate(stt_id)["endpoint"] if stt_id else None
    samples = _locale_texts()
    languages = (
        sorted(samples)
        if args.languages.strip().lower() == "all"
        else [x.strip() for x in args.languages.split(",") if x.strip()]
    )
    defaults = rv.get("tts_defaults") or {}
    ci_base = {
        "inference_timesteps": args.timesteps or int(defaults.get("inference_timesteps", 6)),
        "cfg_value": float(defaults.get("cfg_value", 2.0)),
    }

    host, auth = _auth()
    report: dict[str, Any] = {
        "tts_endpoint": endpoint,
        "stt_endpoint": None if args.no_quality else stt_endpoint,
        "reps": args.reps,
        "thresholds": {
            "max_cer": args.max_cer,
            "max_insertion_rate": args.max_insertion_rate,
        },
        "results": [],
    }
    print(f"tts={endpoint} stt={stt_endpoint} reps={args.reps} timesteps={ci_base['inference_timesteps']}\n")
    header = f"{'lang':7} {'mode':9} {'TTFB':>8} {'total':>9} {'CER':>8} {'insert':>8} {'pass':>6}"
    print(header)
    print("-" * len(header))

    for language in languages:
        text = samples.get(language)
        if not text:
            print(f"{language:7} ERROR no localized evaluation text")
            continue
        ci = {**ci_base, "text": text, "language": language}
        try:
            _stream_call(host, auth, endpoint, text, ci)  # warmup
            stream = []
            predict = []
            for rep in range(1, args.reps + 1):
                streamed = _stream_call(host, auth, endpoint, text, ci)
                streamed.update({"language": language, "mode": "stream", "rep": rep})
                if not args.no_quality and stt_endpoint and streamed["audio_pcm"]:
                    streamed["quality"] = _quality_call(
                        host,
                        auth,
                        stt_endpoint,
                        audio_b64=_wav_b64(
                            streamed.pop("audio_pcm"), streamed["sample_rate_hz"]
                        ),
                        language=language,
                        sample_rate_hz=streamed["sample_rate_hz"],
                        reference=text,
                    )
                else:
                    streamed.pop("audio_pcm", None)
                stream.append(streamed)

                buffered = _predict_call(host, auth, endpoint, text, ci)
                buffered.update({"language": language, "mode": "buffered", "rep": rep})
                if not args.no_quality and stt_endpoint and buffered["audio_b64"]:
                    buffered["quality"] = _quality_call(
                        host,
                        auth,
                        stt_endpoint,
                        audio_b64=buffered["audio_b64"],
                        language=language,
                        sample_rate_hz=buffered["sample_rate_hz"],
                        reference=text,
                    )
                buffered.pop("audio_b64", None)
                predict.append(buffered)
        except Exception as exc:  # noqa: BLE001
            print(f"{language:7} ERROR {str(exc)[:70]}")
            continue

        report["results"].extend(stream + predict)
        for mode, rows in (("stream", stream), ("buffered", predict)):
            quality = [row["quality"] for row in rows if row.get("quality")]
            cer = _median([item["cer"] for item in quality])
            insertion = max((item["insertion_rate"] for item in quality), default=None)
            passed = bool(quality) and all(
                item["cer"] <= args.max_cer
                and item["insertion_rate"] <= args.max_insertion_rate
                for item in quality
            )
            pass_label = "-" if args.no_quality else "yes" if passed else "NO"
            ttfb = _median([row.get("ttfb_ms") for row in rows])
            total = _median(
                [row.get("total_ms", row.get("e2e_ms")) for row in rows]
            )
            print(
                f"{language:7} {mode:9} "
                f"{(f'{ttfb:.0f}ms' if ttfb is not None else '-'):>8} "
                f"{(f'{total:.0f}ms' if total is not None else '-'):>9} "
                f"{(f'{cer:.3f}' if cer is not None else '-'):>8} "
                f"{(f'{insertion:.3f}' if insertion is not None else '-'):>8} "
                f"{pass_label:>6}"
            )

    report["summary"] = {
        mode: _summary(
            [row for row in report["results"] if row["mode"] == mode],
            max_cer=args.max_cer,
            max_insertion_rate=args.max_insertion_rate,
        )
        for mode in ("stream", "buffered")
    }
    for mode in ("stream", "buffered"):
        rows = [row for row in report["results"] if row["mode"] == mode]
        report["summary"][mode]["integrity_validated"] = bool(rows) and all(
            row.get("integrity_validated") is True for row in rows
        )
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {output}")
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
    invalid_modes = [
        mode
        for mode in ("stream", "buffered")
        if not report["summary"][mode]["integrity_validated"]
    ]
    if invalid_modes:
        print(
            "\nWARNING: the deployed TTS endpoint did not certify integrity for: "
            + ", ".join(invalid_modes)
        )


if __name__ == "__main__":
    main()
