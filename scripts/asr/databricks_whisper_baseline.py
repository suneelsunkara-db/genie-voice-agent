"""Self-contained Whisper baseline runner for Databricks job clusters.

This file is uploaded to the ASR model-training Volume and executed as a
Databricks spark_python_task. It intentionally avoids importing the local
backend package so the job can run from a clean GPU ML runtime.
"""
from __future__ import annotations

import argparse
import json
import re
import string
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any


PUNCT_TABLE = str.maketrans("", "", string.punctuation.replace("$", ""))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Whisper ASR baseline on Databricks.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="openai/whisper-small.en")
    parser.add_argument("--language", default="english")
    parser.add_argument("--task", default="transcribe")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--split", action="append", dest="splits")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    clips = load_manifest(Path(args.manifest), splits=args.splits)
    if args.limit is not None:
        clips = clips[: args.limit]

    from transformers import pipeline

    transcriber = pipeline(
        task="automatic-speech-recognition",
        model=args.model,
        device=args.device,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    with output.open("w", encoding="utf-8") as f:
        for clip in clips:
            started = time.perf_counter()
            generate_kwargs = whisper_generate_kwargs(args.model, args.language, args.task)
            raw = transcriber(
                clip["audio_path"],
                return_timestamps=False,
                generate_kwargs=generate_kwargs,
            )
            latency_ms = round((time.perf_counter() - started) * 1000)
            transcript = str(raw.get("text") or "").strip()
            score = score_transcript(
                str(clip["reference_transcript"]),
                transcript,
                clip.get("expected_entities") or {},
            )
            row = {
                "clip_id": clip["clip_id"],
                "call_id": clip.get("call_id"),
                "speaker": clip.get("speaker"),
                "audio_path": clip["audio_path"],
                "provider": "databricks_whisper",
                "baseline_name": f"databricks_whisper_{model_slug(args.model)}_baseline",
                "model": args.model,
                "language": args.language,
                "transcript": transcript,
                "reference_transcript": clip["reference_transcript"],
                "latency_ms": latency_ms,
                "confidence": None,
                "score": score,
                "raw": {"model": args.model, "pipeline_result": raw},
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.append(row)
            print(json.dumps({"clip_id": clip["clip_id"], "latency_ms": latency_ms}))

    print(json.dumps(summarize(rows), indent=2))


def load_manifest(path: Path, *, splits: Iterable[str] | None = None) -> list[dict[str, Any]]:
    wanted = {str(split) for split in splits} if splits else None
    clips = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            row = json.loads(text)
            for key in ("clip_id", "audio_path", "reference_transcript"):
                if not row.get(key):
                    raise ValueError(f"Manifest row {line_no} missing {key}")
            if wanted and row.get("split") not in wanted:
                continue
            clips.append(row)
    return clips


def score_transcript(reference: str, hypothesis: str, expected_entities: dict[str, Any]) -> dict[str, Any]:
    ref_words = normalize_words(reference)
    hyp_words = normalize_words(hypothesis)
    word_errors = edit_distance(ref_words, hyp_words)

    ref_chars = normalize_chars(reference)
    hyp_chars = normalize_chars(hypothesis)
    char_errors = edit_distance(list(ref_chars), list(hyp_chars))

    entity_scores = score_entities(hypothesis, expected_entities)
    entity_expected = sum(score["expected"] for score in entity_scores.values())
    entity_matched = sum(score["matched"] for score in entity_scores.values())
    entity_accuracy = None if entity_expected == 0 else entity_matched / entity_expected
    return {
        "wer": ratio(word_errors, len(ref_words)),
        "cer": ratio(char_errors, len(ref_chars)),
        "word_errors": word_errors,
        "reference_words": len(ref_words),
        "char_errors": char_errors,
        "reference_chars": len(ref_chars),
        "entity_scores": entity_scores,
        "entity_accuracy": entity_accuracy,
    }


def score_entities(hypothesis: str, expected_entities: dict[str, Any]) -> dict[str, dict[str, Any]]:
    normalized_hypothesis = normalize_entity_text(hypothesis)
    scores = {}
    for group in (
        "invoice_ids",
        "amounts",
        "dates",
        "billing_actions",
        "confirmations",
        "refusals",
        "account_terms",
    ):
        values = string_list(expected_entities.get(group))
        missing = [value for value in values if not entity_present(value, normalized_hypothesis)]
        scores[group] = {
            "expected": len(values),
            "matched": len(values) - len(missing),
            "missing": missing,
            "accuracy": None if not values else (len(values) - len(missing)) / len(values),
        }
    return scores


def normalize_words(text: str) -> list[str]:
    text = text.lower().translate(PUNCT_TABLE)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split() if text else []


def normalize_chars(text: str) -> str:
    text = text.lower().translate(PUNCT_TABLE)
    return re.sub(r"\s+", "", text)


def normalize_entity_text(text: str) -> str:
    text = text.lower()
    text = text.replace("$", " dollars ")
    text = re.sub(r"([a-z]+)-(\d+)", r"\1 \2", text)
    text = re.sub(r"(\d+)\.(\d+)", r"\1 \2", text)
    text = text.translate(PUNCT_TABLE)
    text = re.sub(r"\s+", " ", text).strip()
    return f" {text} "


def entity_present(expected: str, normalized_hypothesis: str) -> bool:
    normalized_expected = normalize_entity_text(expected).strip()
    if not normalized_expected:
        return True
    if f" {normalized_expected} " in normalized_hypothesis:
        return True

    invoice_match = re.search(r"(?:inv|invoice)\s*(\d+)", normalized_expected)
    if invoice_match:
        invoice_number = invoice_match.group(1)
        return bool(re.search(rf"\b(?:inv|invoice)?\s*{re.escape(invoice_number)}\b", normalized_hypothesis))

    amount_match = re.search(r"\b(\d+)\s+dollars?(?:\s+and\s+)?(?:\s*(\d+)\s+cents?)?", normalized_expected)
    if amount_match:
        dollars = amount_match.group(1)
        cents = amount_match.group(2)
        if cents:
            return dollars in normalized_hypothesis and cents in normalized_hypothesis
        return bool(re.search(rf"\b{re.escape(dollars)}\b", normalized_hypothesis))

    return False


def edit_distance(left: list[str], right: list[str]) -> int:
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for i, left_item in enumerate(left, start=1):
        current = [i]
        for j, right_item in enumerate(right, start=1):
            cost = 0 if left_item == right_item else 1
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost))
        previous = current
    return previous[-1]


def ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0 if numerator == 0 else 1.0
    return numerator / denominator


def string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"clips": 0}
    scores = [row["score"] for row in rows]
    return {
        "clips": len(rows),
        "avg_wer": sum(score["wer"] for score in scores) / len(scores),
        "avg_cer": sum(score["cer"] for score in scores) / len(scores),
        "avg_entity_accuracy": avg(score["entity_accuracy"] for score in scores),
        "avg_latency_ms": sum(row["latency_ms"] for row in rows) / len(rows),
    }


def avg(values: Iterable[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return None if not present else sum(present) / len(present)


def model_slug(model: str) -> str:
    return model.replace("/", "_").replace("-", "_").replace(".", "_")


def whisper_generate_kwargs(model: str, language: str, task: str) -> dict[str, str]:
    """English-only Whisper checkpoints reject explicit language/task hints."""
    if model.endswith(".en"):
        return {}
    return {"language": language, "task": task}


if __name__ == "__main__":
    main()
