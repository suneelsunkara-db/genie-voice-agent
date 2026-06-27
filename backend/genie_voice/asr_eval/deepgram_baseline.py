"""Run the locked Deepgram baseline for ASR gold clips.

Example:
    python -m genie_voice.asr_eval.deepgram_baseline \
      --manifest docs/asr_model_training_manifest.example.jsonl \
      --output /tmp/asr_model_training/deepgram_nova3_baseline.jsonl
"""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .manifest import ASRGoldClip, load_manifest
from .metrics import score_transcript


DEEPGRAM_LISTEN_URL = "https://api.deepgram.com/v1/listen"


def run_baseline(
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    api_key: str | None = None,
    model: str = "nova-3",
    language: str = "en-US",
    smart_format: bool = True,
    punctuate: bool = True,
    splits: list[str] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Transcribe all manifest clips with Deepgram and write JSONL results."""
    key = api_key or _deepgram_api_key()
    clips = load_manifest(manifest_path, splits=splits)
    if limit is not None:
        clips = clips[:limit]

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    with output.open("w", encoding="utf-8") as f:
        for clip in clips:
            result = transcribe_clip(
                clip,
                api_key=key,
                model=model,
                language=language,
                smart_format=smart_format,
                punctuate=punctuate,
            )
            score = score_transcript(
                clip.reference_transcript,
                result["transcript"],
                clip.expected_entities,
            )
            row = {
                "clip_id": clip.clip_id,
                "call_id": clip.call_id,
                "speaker": clip.speaker,
                "audio_path": clip.audio_path,
                "provider": "deepgram",
                "baseline_name": f"deepgram_{model.replace('-', '_')}_baseline",
                "model": model,
                "language": language,
                "transcript": result["transcript"],
                "reference_transcript": clip.reference_transcript,
                "latency_ms": result["latency_ms"],
                "confidence": result["confidence"],
                "score": score.to_dict(),
                "raw": result["raw"],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            results.append(row)
    return results


def transcribe_clip(
    clip: ASRGoldClip,
    *,
    api_key: str,
    model: str,
    language: str,
    smart_format: bool,
    punctuate: bool,
) -> dict[str, Any]:
    audio_bytes = _read_audio(clip.audio_path)
    mime_type = clip.audio_format or mimetypes.guess_type(clip.audio_path)[0] or "audio/wav"
    params = {
        "model": model,
        "language": language,
        "smart_format": str(smart_format).lower(),
        "punctuate": str(punctuate).lower(),
    }
    req = Request(
        f"{DEEPGRAM_LISTEN_URL}?{urlencode(params)}",
        data=audio_bytes,
        method="POST",
    )
    req.add_header("Authorization", f"Token {api_key}")
    req.add_header("Content-Type", mime_type)
    req.add_header("Accept", "application/json")

    started = time.perf_counter()
    with urlopen(req, timeout=120) as resp:
        raw = json.loads(resp.read().decode("utf-8"))
    latency_ms = round((time.perf_counter() - started) * 1000)

    transcript, confidence = _extract_deepgram_result(raw)
    return {
        "transcript": transcript,
        "confidence": confidence,
        "latency_ms": latency_ms,
        "raw": raw,
    }


def _extract_deepgram_result(raw: dict[str, Any]) -> tuple[str, float | None]:
    channels = raw.get("results", {}).get("channels") or []
    alternatives = channels[0].get("alternatives") if channels else []
    alt = alternatives[0] if alternatives else {}
    transcript = str(alt.get("transcript") or "").strip()
    confidence = alt.get("confidence")
    return transcript, float(confidence) if confidence is not None else None


def _read_audio(path: str) -> bytes:
    if path.startswith("data:"):
        _, _, encoded = path.partition(",")
        return base64.b64decode(encoded)
    if path.startswith("/Volumes/") or path.startswith("dbfs:/Volumes/"):
        return _read_volume_audio(path)
    if path.startswith("file://"):
        path = path.removeprefix("file://")
    return Path(path).read_bytes()


def _read_volume_audio(path: str) -> bytes:
    """Read UC Volume audio from a local process via `databricks fs cp`."""
    uri = path if path.startswith("dbfs:") else f"dbfs:{path}"
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        cmd = ["databricks", "fs", "cp", uri, str(tmp_path), "--overwrite"]
        profile = os.environ.get("ASR_DATABRICKS_PROFILE") or os.environ.get("DATABRICKS_CONFIG_PROFILE")
        if profile:
            cmd = ["databricks", "--profile", profile, "fs", "cp", uri, str(tmp_path), "--overwrite"]
        env = {
            **os.environ,
            "DATABRICKS_AUTH_STORAGE": os.environ.get("DATABRICKS_AUTH_STORAGE", "plaintext"),
        }
        subprocess.run(cmd, check=True, text=True, capture_output=True, timeout=120, env=env)
        return tmp_path.read_bytes()
    except FileNotFoundError as exc:
        raise RuntimeError("Databricks CLI is required to read /Volumes audio paths locally") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        raise RuntimeError(f"Failed to copy audio from {uri}: {detail}") from exc
    finally:
        tmp_path.unlink(missing_ok=True)


def _deepgram_api_key() -> str:
    key = os.environ.get("DEEPGRAM_API_KEY", "").strip()
    if key:
        return key

    try:
        from genie_voice.config import get_settings

        key = get_settings().secrets.deepgram_api_key.strip()
    except Exception:  # noqa: BLE001
        key = ""
    if not key:
        raise RuntimeError("DEEPGRAM_API_KEY is not configured")
    return key


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Deepgram ASR baseline over a gold manifest.")
    parser.add_argument("--manifest", required=True, help="Path to ASR gold manifest JSONL.")
    parser.add_argument("--output", required=True, help="Path to write baseline JSONL results.")
    parser.add_argument("--model", default="nova-3", help="Deepgram model name.")
    parser.add_argument("--language", default="en-US", help="Deepgram language code.")
    parser.add_argument("--split", action="append", dest="splits", help="Manifest split to include.")
    parser.add_argument("--limit", type=int, help="Maximum number of clips to process.")
    args = parser.parse_args()

    results = run_baseline(
        args.manifest,
        args.output,
        model=args.model,
        language=args.language,
        splits=args.splits,
        limit=args.limit,
    )
    summary = _summarize(results)
    print(json.dumps(summary, indent=2))


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {"clips": 0}
    scores = [row["score"] for row in results]
    return {
        "clips": len(results),
        "avg_wer": sum(score["wer"] for score in scores) / len(scores),
        "avg_cer": sum(score["cer"] for score in scores) / len(scores),
        "avg_entity_accuracy": _avg_present(score["entity_accuracy"] for score in scores),
        "avg_latency_ms": sum(row["latency_ms"] for row in results) / len(results),
    }


def _avg_present(values) -> float | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return sum(present) / len(present)


if __name__ == "__main__":
    main()
