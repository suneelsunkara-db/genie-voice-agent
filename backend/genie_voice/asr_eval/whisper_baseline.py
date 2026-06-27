"""Run a Whisper baseline for ASR model-training clips.

This module intentionally keeps Whisper dependencies optional. Install
`transformers`, `torch`, and an audio backend such as `librosa`/`soundfile` on a
Databricks GPU cluster or ML workstation, then run the same manifest used by
the Deepgram baseline.

Example:
    python -m genie_voice.asr_eval.whisper_baseline \
      --manifest /Volumes/.../asr_training_gold_v1.jsonl \
      --output /Volumes/.../whisper_small_baseline.jsonl \
      --model openai/whisper-small.en
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from .manifest import ASRGoldClip, load_manifest
from .metrics import score_transcript


def run_baseline(
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    model: str = "openai/whisper-small.en",
    language: str = "english",
    task: str = "transcribe",
    device: int | str | None = None,
    splits: list[str] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Transcribe manifest clips with Whisper and write JSONL results."""
    clips = load_manifest(manifest_path, splits=splits)
    if limit is not None:
        clips = clips[:limit]

    transcriber = _load_pipeline(model=model, device=device)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    with output.open("w", encoding="utf-8") as f:
        for clip in clips:
            result = transcribe_clip(
                clip,
                transcriber=transcriber,
                model=model,
                language=language,
                task=task,
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
                "provider": "whisper",
                "baseline_name": f"whisper_{_model_slug(model)}_baseline",
                "model": model,
                "language": language,
                "transcript": result["transcript"],
                "reference_transcript": clip.reference_transcript,
                "latency_ms": result["latency_ms"],
                "confidence": None,
                "score": score.to_dict(),
                "raw": result["raw"],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            results.append(row)
    return results


def transcribe_clip(
    clip: ASRGoldClip,
    *,
    transcriber: Any,
    model: str,
    language: str,
    task: str,
) -> dict[str, Any]:
    with _materialized_audio(clip.audio_path) as audio_path:
        generate_kwargs = _whisper_generate_kwargs(model, language, task)

        started = time.perf_counter()
        raw = transcriber(
            str(audio_path),
            return_timestamps=False,
            generate_kwargs=generate_kwargs,
        )
        latency_ms = round((time.perf_counter() - started) * 1000)

    transcript = str(raw.get("text") or "").strip()
    return {
        "transcript": transcript,
        "latency_ms": latency_ms,
        "raw": {
            "model": model,
            "pipeline_result": raw,
        },
    }


def _load_pipeline(*, model: str, device: int | str | None) -> Any:
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise RuntimeError(
            "Whisper baseline requires optional ML dependencies. Install on a "
            "Databricks GPU cluster or ML workstation: pip install transformers "
            "torch accelerate librosa soundfile"
        ) from exc

    pipeline_kwargs: dict[str, Any] = {
        "task": "automatic-speech-recognition",
        "model": model,
    }
    if device is not None:
        pipeline_kwargs["device"] = int(device)
    return pipeline(**pipeline_kwargs)


class _materialized_audio:
    def __init__(self, path: str) -> None:
        self.path = path
        self._tmp_path: Path | None = None

    def __enter__(self) -> Path:
        path = self.path
        if path.startswith("/Volumes/") or path.startswith("dbfs:/Volumes/"):
            return self._copy_volume_audio(path)
        if path.startswith("file://"):
            path = path.removeprefix("file://")
        return Path(path)

    def __exit__(self, *_exc_info: object) -> None:
        if self._tmp_path is not None:
            self._tmp_path.unlink(missing_ok=True)

    def _copy_volume_audio(self, path: str) -> Path:
        uri = path if path.startswith("dbfs:") else f"dbfs:{path}"
        with tempfile.NamedTemporaryFile(suffix=Path(path).suffix or ".wav", delete=False) as tmp:
            self._tmp_path = Path(tmp.name)
        cmd = ["databricks", "fs", "cp", uri, str(self._tmp_path), "--overwrite"]
        profile = os.environ.get("ASR_DATABRICKS_PROFILE") or os.environ.get("DATABRICKS_CONFIG_PROFILE")
        if profile:
            cmd = ["databricks", "--profile", profile, "fs", "cp", uri, str(self._tmp_path), "--overwrite"]
        try:
            subprocess.run(cmd, check=True, text=True, capture_output=True, timeout=120)
        except FileNotFoundError as exc:
            raise RuntimeError("Databricks CLI is required to read /Volumes audio paths locally") from exc
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or "").strip()
            raise RuntimeError(f"Failed to copy audio from {uri}: {detail}") from exc
        return self._tmp_path


def _model_slug(model: str) -> str:
    return model.replace("/", "_").replace("-", "_").replace(".", "_")


def _whisper_generate_kwargs(model: str, language: str, task: str) -> dict[str, str]:
    """English-only Whisper checkpoints reject explicit language/task hints."""
    if model.endswith(".en"):
        return {}
    kwargs = {"task": task}
    if language:
        kwargs["language"] = language
    return kwargs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Whisper ASR baseline over a gold manifest.")
    parser.add_argument("--manifest", required=True, help="Path to ASR gold manifest JSONL.")
    parser.add_argument("--output", required=True, help="Path to write baseline JSONL results.")
    parser.add_argument("--model", default="openai/whisper-small.en", help="Hugging Face Whisper model id.")
    parser.add_argument("--language", default="english", help="Whisper language hint.")
    parser.add_argument("--task", default="transcribe", choices=["transcribe", "translate"], help="Whisper task.")
    parser.add_argument("--device", help="Transformers pipeline device, e.g. 0 for first GPU or -1 for CPU.")
    parser.add_argument("--split", action="append", dest="splits", help="Manifest split to include.")
    parser.add_argument("--limit", type=int, help="Maximum number of clips to process.")
    args = parser.parse_args()

    try:
        results = run_baseline(
            args.manifest,
            args.output,
            model=args.model,
            language=args.language,
            task=args.task,
            device=args.device,
            splits=args.splits,
            limit=args.limit,
        )
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(_summarize(results), indent=2))


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
