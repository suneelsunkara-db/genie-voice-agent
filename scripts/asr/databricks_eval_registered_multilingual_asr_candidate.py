"""Evaluate a registered multilingual ASR candidate against a manifest."""
from __future__ import annotations

import argparse
import base64
import io
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
import soundfile as sf
from huggingface_hub import hf_hub_download, list_repo_files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["evaluate", "bootstrap-public-manifest"])
    parser.add_argument("--registered-model", default="")
    parser.add_argument("--candidate-id", default="")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--language", choices=["", "th", "id", "zh"], default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--summary-output", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--public-limit", type=int, default=1)
    parser.add_argument("--scaffold-dir", default="")
    args = parser.parse_args()
    if args.action == "bootstrap-public-manifest":
        bootstrap_public_manifest(args)
        return
    required = {
        "--registered-model": args.registered_model,
        "--candidate-id": args.candidate_id,
        "--language": args.language,
        "--output": args.output,
        "--summary-output": args.summary_output,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ValueError(f"Missing required evaluation arguments: {missing}")

    rows = [
        row
        for row in read_jsonl(args.manifest)
        if row.get("language") == args.language and row.get("split") == "holdout"
    ]
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError(f"No holdout rows for language={args.language} in {args.manifest}")

    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    model_uri = f"models:/{args.registered_model}@candidate"
    model = mlflow.pyfunc.load_model(model_uri)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result_rows: list[dict[str, Any]] = []
    with output.open("w", encoding="utf-8") as f:
        for row in rows:
            model_input = build_model_input(row)
            started = time.perf_counter()
            prediction = model.predict(pd.DataFrame([model_input]))
            latency_ms = round((time.perf_counter() - started) * 1000)
            result = first_prediction(prediction)
            transcript = str(result.get("transcript") or result.get("raw_transcript") or "").strip()
            scored = score_row(
                {
                    "language": args.language,
                    "candidate": args.candidate_id,
                    "clip_id": row.get("clip_id"),
                    "reference_transcript": row.get("reference_transcript"),
                    "transcript": transcript,
                    "latency_ms": latency_ms,
                    "expected_entities": row.get("expected_entities") or {},
                    "raw": {
                        "model_uri": model_uri,
                        "registered_model": args.registered_model,
                        "prediction": result,
                    },
                }
            )
            f.write(json.dumps(scored, ensure_ascii=False) + "\n")
            result_rows.append(scored)

    summary = summarize(result_rows, args.registered_model, args.candidate_id, args.language)
    Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_output).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def bootstrap_public_manifest(args: argparse.Namespace) -> None:
    configs = {"th": "th_th", "id": "id_id", "zh": "cmn_hans_cn"}
    base_dir = Path(args.scaffold_dir or str(Path(args.manifest).parent)) / "public_smoke_audio"
    base_dir.mkdir(parents=True, exist_ok=True)
    cache_root = Path(args.scaffold_dir or str(Path(args.manifest).parent)) / "hf_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    repo_files = list_repo_files("google/fleurs", repo_type="dataset")
    rows: list[dict[str, Any]] = []
    languages = [args.language] if args.language else ["th", "id", "zh"]
    for code in languages:
        config = configs[code]
        parquet_candidates = [
            name
            for name in repo_files
            if name.endswith(".parquet") and f"/{config}/" in name and "validation" in name
        ]
        if not parquet_candidates:
            raise RuntimeError(f"No FLEURS validation parquet found for {config}")
        parquet_path = hf_hub_download(
            repo_id="google/fleurs",
            repo_type="dataset",
            filename=sorted(parquet_candidates)[0],
            cache_dir=str(cache_root / "hub"),
        )
        dataset = pd.read_parquet(parquet_path)
        written = 0
        for _, item in dataset.iterrows():
            if written >= args.public_limit:
                break
            clip_id = f"{code}_fleurs_{written + 1:04d}"
            audio_path = write_public_audio(item.get("audio"), base_dir, clip_id)
            rows.append(
                {
                    "clip_id": clip_id,
                    "audio_path": str(audio_path),
                    "reference_transcript": str(item.get("transcription") or item.get("raw_transcription") or "").strip(),
                    "language": code,
                    "split": "holdout",
                    "scenario": "public_fleurs_smoke",
                    "duration_seconds": None,
                    "expected_entities": {},
                    "metadata": {"source": "google/fleurs", "config": config, "id": safe_json_value(item.get("id"))},
                }
            )
            written += 1
    output = Path(args.manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"manifest": str(output), "rows": len(rows), "languages": languages}, indent=2))


def write_public_audio(audio: Any, base_dir: Path, clip_id: str) -> Path:
    if isinstance(audio, dict):
        path_hint = str(audio.get("path") or "")
        suffix = Path(path_hint).suffix or ".wav"
        if audio.get("bytes"):
            output = base_dir / f"{clip_id}{suffix}"
            output.write_bytes(bytes(audio["bytes"]))
            return output
        if audio.get("array") is not None and audio.get("sampling_rate"):
            output = base_dir / f"{clip_id}.wav"
            sf.write(output, audio["array"], int(audio["sampling_rate"]))
            return output
    if isinstance(audio, (bytes, bytearray, memoryview)):
        output = base_dir / f"{clip_id}.wav"
        output.write_bytes(bytes(audio))
        return output
    raise RuntimeError(f"Unsupported FLEURS audio payload for {clip_id}: {type(audio).__name__}")


def safe_json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return str(value)


def build_model_input(row: dict[str, Any]) -> dict[str, Any]:
    audio_path = Path(str(row["audio_path"]))
    if audio_path.exists():
        return {
            "audio_b64": base64.b64encode(audio_path.read_bytes()).decode("ascii"),
            "mime_type": mime_type_for(audio_path),
            "speaker": int(row.get("speaker") or 1) if str(row.get("speaker") or "1").isdigit() else 1,
            "language": row.get("language"),
        }
    return {
        "audio_path": str(row["audio_path"]),
        "mime_type": mime_type_for(audio_path),
        "speaker": int(row.get("speaker") or 1) if str(row.get("speaker") or "1").isdigit() else 1,
        "language": row.get("language"),
    }


def first_prediction(prediction: Any) -> dict[str, Any]:
    if isinstance(prediction, list):
        return prediction[0]
    if hasattr(prediction, "to_dict"):
        return prediction.to_dict(orient="records")[0]
    if isinstance(prediction, dict):
        return prediction
    raise TypeError(f"Unsupported prediction type: {type(prediction).__name__}")


def score_row(row: dict[str, Any]) -> dict[str, Any]:
    reference = str(row.get("reference_transcript") or "")
    hypothesis = str(row.get("transcript") or "")
    ref_words = normalize_words(reference)
    hyp_words = normalize_words(hypothesis)
    ref_chars = normalize_chars(reference)
    hyp_chars = normalize_chars(hypothesis)
    word_errors = edit_distance(ref_words, hyp_words)
    char_errors = edit_distance(list(ref_chars), list(hyp_chars))
    entity = entity_accuracy(row.get("expected_entities") or {}, hypothesis)
    unsafe_reasons = []
    if not hypothesis.strip():
        unsafe_reasons.append("empty_transcript")
    if entity is not None and entity < 1.0:
        unsafe_reasons.append("missing_expected_entity")
    return {
        **row,
        "wer": ratio(word_errors, len(ref_words)),
        "cer": ratio(char_errors, len(ref_chars)),
        "critical_entity_accuracy": entity,
        "unsafe_for_resolution": bool(unsafe_reasons),
        "unsafe_reasons": unsafe_reasons,
    }


def summarize(
    rows: list[dict[str, Any]],
    registered_model: str,
    candidate_id: str,
    language: str,
) -> dict[str, Any]:
    latencies = sorted(float(row["latency_ms"]) for row in rows)
    entity_values = [row["critical_entity_accuracy"] for row in rows if row["critical_entity_accuracy"] is not None]
    unsafe = [row for row in rows if row.get("unsafe_for_resolution")]
    reasons: Counter[str] = Counter(reason for row in rows for reason in row.get("unsafe_reasons") or [])
    return {
        "registered_model": registered_model,
        "candidate": candidate_id,
        "language": language,
        "clips": len(rows),
        "avg_wer": avg(row["wer"] for row in rows),
        "avg_cer": avg(row["cer"] for row in rows),
        "avg_critical_entity_accuracy": avg(entity_values) if entity_values else None,
        "unsafe_for_resolution_rate": len(unsafe) / len(rows) if rows else None,
        "p95_latency_ms": percentile(latencies, 0.95),
        "empty_transcript_rate": sum(1 for row in rows if not str(row.get("transcript") or "").strip()) / len(rows),
        "unsafe_reason_counts": dict(reasons),
    }


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def normalize_words(text: str) -> list[str]:
    return re.findall(r"[\w\u0e00-\u0e7f\u3400-\u9fff]+", text.lower())


def normalize_chars(text: str) -> str:
    return re.sub(r"\s+", "", text.lower())


def edit_distance(left: list[Any], right: list[Any]) -> int:
    previous = list(range(len(right) + 1))
    for i, left_item in enumerate(left, start=1):
        current = [i]
        for j, right_item in enumerate(right, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (left_item != right_item),
                )
            )
        previous = current
    return previous[-1]


def ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def entity_accuracy(expected_entities: dict[str, Any], transcript: str) -> float | None:
    expected = [
        str(value)
        for values in expected_entities.values()
        if isinstance(values, list)
        for value in values
        if str(value).strip()
    ]
    if not expected:
        return None
    hits = sum(1 for value in expected if multilingual_entity_present(value, transcript))
    return hits / len(expected)


def multilingual_entity_present(expected: str, transcript: str) -> bool:
    expected_norm = normalize_loose_entity(expected)
    transcript_norm = normalize_loose_entity(transcript)
    if not expected_norm:
        return True
    if expected_norm in transcript_norm:
        return True
    invoice_match = re.search(r"inv(\d+)", expected_norm)
    if invoice_match and invoice_match.group(1) in transcript_norm:
        return True
    digits = re.sub(r"\D", "", expected)
    return bool(len(digits) >= 3 and digits in re.sub(r"\D", "", transcript))


def normalize_loose_entity(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", "", text)
    return re.sub(r"[^\w\u0e00-\u0e7f\u3400-\u9fff]", "", text)


def avg(values: Any) -> float | None:
    collected = [value for value in values if value is not None]
    return sum(collected) / len(collected) if collected else None


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    index = min(len(values) - 1, max(0, round((len(values) - 1) * pct)))
    return values[index]


def mime_type_for(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".wav":
        return "audio/wav"
    if suffix == ".mp3":
        return "audio/mpeg"
    if suffix in {".webm", ".weba"}:
        return "audio/webm"
    if suffix == ".m4a":
        return "audio/mp4"
    return "application/octet-stream"


if __name__ == "__main__":
    main()
