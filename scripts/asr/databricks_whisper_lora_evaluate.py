"""Self-contained Whisper LoRA evaluator for Databricks GPU clusters."""
from __future__ import annotations

import argparse
import json
import re
import string
import time
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any


PUNCT_TABLE = str.maketrans("", "", string.punctuation.replace("$", ""))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Whisper LoRA adapter on an ASR manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--base-model", default="openai/whisper-small.en")
    parser.add_argument("--language", default="english")
    parser.add_argument("--task", default="transcribe")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--split", action="append", dest="splits")
    args = parser.parse_args()

    clips = load_manifest(Path(args.manifest), splits=args.splits)
    if args.limit is not None:
        clips = clips[: args.limit]
    if not clips:
        raise SystemExit("No clips selected for LoRA evaluation.")

    import torch
    import librosa
    from peft import PeftModel
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    adapter_dir = Path(args.adapter_dir)
    processor_dir = adapter_dir.parent / "processor"
    processor = WhisperProcessor.from_pretrained(
        str(processor_dir) if processor_dir.exists() else args.base_model
    )
    model = WhisperForConditionalGeneration.from_pretrained(args.base_model)
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    with output.open("w", encoding="utf-8") as f:
        for clip in clips:
            started = time.perf_counter()
            transcript = transcribe_clip(
                clip["audio_path"],
                model=model,
                processor=processor,
                device=device,
                model_name=args.base_model,
                language=args.language,
                task=args.task,
            )
            latency_ms = round((time.perf_counter() - started) * 1000)
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
                "provider": "databricks_whisper_lora",
                "baseline_name": f"databricks_whisper_lora_{model_slug(args.base_model)}",
                "model": args.base_model,
                "adapter_dir": str(adapter_dir),
                "language": args.language,
                "transcript": transcript,
                "reference_transcript": clip["reference_transcript"],
                "latency_ms": latency_ms,
                "confidence": None,
                "score": score,
                "raw": {"base_model": args.base_model, "adapter_dir": str(adapter_dir)},
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.append(row)
            print(json.dumps({"clip_id": clip["clip_id"], "latency_ms": latency_ms}))

    summary = summarize(rows)
    summary_output = Path(args.summary_output)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def transcribe_clip(
    audio_path: str,
    *,
    model: Any,
    processor: Any,
    device: Any,
    model_name: str,
    language: str,
    task: str,
) -> str:
    import torch
    import librosa

    audio, sample_rate = librosa.load(audio_path, sr=16000, mono=True)
    inputs = processor.feature_extractor(
        audio,
        sampling_rate=sample_rate,
        return_tensors="pt",
    )
    input_features = inputs.input_features.to(device)
    generate_kwargs = whisper_generate_kwargs(model_name, language, task)
    with torch.no_grad():
        predicted_ids = model.generate(input_features, **generate_kwargs)
    transcript = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcript.strip()


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

    invoice_match = re.search(r"(?:inv|invoice|i\s+nv)\s*(\d+)", normalized_expected)
    if invoice_match:
        return invoice_present(invoice_match.group(1), normalized_hypothesis)

    if expected.strip().startswith("$"):
        return amount_present(expected, normalized_hypothesis)

    date_match = date_parts(normalized_expected)
    if date_match:
        month, day = date_match
        return date_present(month, day, normalized_hypothesis)

    amount_match = re.search(r"\b(\d+)\s+dollars?(?:\s+and\s+)?(?:\s*(\d+)\s+cents?)?", normalized_expected)
    if amount_match:
        dollars = amount_match.group(1)
        cents = amount_match.group(2)
        if cents:
            return dollars in normalized_hypothesis and cents in normalized_hypothesis
        return bool(re.search(rf"\b{re.escape(dollars)}\b", normalized_hypothesis))

    return False


def invoice_present(invoice_number: str, normalized_hypothesis: str) -> bool:
    number = re.escape(invoice_number)
    prefix = r"(?:inv|invoice|i\s*nv|nv|envoic\w*|envy|at\s+nv)"
    if re.search(rf"\b{prefix}\s*{number}\b", normalized_hypothesis):
        return True
    return bool(re.search(rf"\b{number}\b", normalized_hypothesis))


def amount_present(expected: str, normalized_hypothesis: str) -> bool:
    match = re.search(r"\$(\d[\d,]*)(?:\.(\d{2}))?", expected)
    if not match:
        return False
    dollars = match.group(1).replace(",", "")
    cents = match.group(2) or "00"
    if not re.search(rf"\b{re.escape(dollars)}\b", normalized_hypothesis):
        return False
    if cents == "00":
        return True
    return bool(re.search(rf"\b{re.escape(cents)}\b", normalized_hypothesis))


def date_parts(normalized_expected: str) -> tuple[str, str] | None:
    months = (
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
    )
    match = re.search(rf"\b({'|'.join(months)})\s+(\d{{1,2}})\b", normalized_expected)
    if match:
        return match.group(1), match.group(2)
    return None


def date_present(month: str, day: str, normalized_hypothesis: str) -> bool:
    day_pattern = rf"{re.escape(day)}(?:st|nd|rd|th)?"
    month_pattern = re.escape(month)
    return bool(
        re.search(rf"\b{month_pattern}\s+(?:the\s+)?{day_pattern}\b", normalized_hypothesis)
        or re.search(rf"\b(?:the\s+)?{day_pattern}\s+(?:of\s+)?{month_pattern}\b", normalized_hypothesis)
    )


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
    entity_totals: dict[str, dict[str, int]] = defaultdict(lambda: {"expected": 0, "matched": 0})
    for row in rows:
        for group, score in (row.get("score", {}).get("entity_scores") or {}).items():
            entity_totals[group]["expected"] += score.get("expected") or 0
            entity_totals[group]["matched"] += score.get("matched") or 0
    return {
        "clips": len(rows),
        "provider": sorted({row.get("provider") for row in rows if row.get("provider")}),
        "models": sorted({row.get("model") for row in rows if row.get("model")}),
        "adapters": sorted({row.get("adapter_dir") for row in rows if row.get("adapter_dir")}),
        "avg_wer": sum(score["wer"] for score in scores) / len(scores),
        "avg_cer": sum(score["cer"] for score in scores) / len(scores),
        "avg_entity_accuracy": avg(score["entity_accuracy"] for score in scores),
        "avg_latency_ms": sum(row["latency_ms"] for row in rows) / len(rows),
        "entity_groups": {
            group: {
                **counts,
                "accuracy": None if counts["expected"] == 0 else counts["matched"] / counts["expected"],
            }
            for group, counts in sorted(entity_totals.items())
        },
    }


def avg(values: Iterable[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return None if not present else sum(present) / len(present)


def model_slug(model: str) -> str:
    return model.replace("/", "_").replace("-", "_").replace(".", "_")


def whisper_generate_kwargs(model: str, language: str, task: str) -> dict[str, str]:
    if model.endswith(".en"):
        return {}
    return {"language": language, "task": task}


if __name__ == "__main__":
    main()
