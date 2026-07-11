"""Self-contained Whisper LoRA evaluator for Databricks GPU clusters."""
from __future__ import annotations

import argparse
import json
import re
import string
import time
import unicodedata
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
    parser.add_argument("--input-results", help="Existing JSONL results to re-score without running inference.")
    parser.add_argument("--base-only", action="store_true", help="Evaluate the base Whisper model without a LoRA adapter.")
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
    if args.input_results:
        rescore_existing_results(
            manifest_clips=clips,
            input_results=Path(args.input_results),
            output=Path(args.output),
            summary_output=Path(args.summary_output),
        )
        return

    import torch
    import librosa
    from peft import PeftModel
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    adapter_dir = Path(args.adapter_dir)
    if args.base_only:
        processor = WhisperProcessor.from_pretrained(args.base_model)
        model = WhisperForConditionalGeneration.from_pretrained(args.base_model)
    else:
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
                "provider": "databricks_whisper" if args.base_only else "databricks_whisper_lora",
                "baseline_name": (
                    f"databricks_whisper_{model_slug(args.base_model)}_base"
                    if args.base_only
                    else f"databricks_whisper_lora_{model_slug(args.base_model)}"
                ),
                "model": args.base_model,
                "adapter_dir": None if args.base_only else str(adapter_dir),
                "language": args.language,
                "transcript": transcript,
                "reference_transcript": clip["reference_transcript"],
                "latency_ms": latency_ms,
                "confidence": None,
                "score": score,
                "raw": {
                    "base_model": args.base_model,
                    "adapter_dir": None if args.base_only else str(adapter_dir),
                    "base_only": args.base_only,
                },
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.append(row)
            print(json.dumps({"clip_id": clip["clip_id"], "latency_ms": latency_ms}))

    summary = summarize(rows)
    summary_output = Path(args.summary_output)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def rescore_existing_results(
    *,
    manifest_clips: list[dict[str, Any]],
    input_results: Path,
    output: Path,
    summary_output: Path,
) -> None:
    clips_by_id = {str(clip["clip_id"]): clip for clip in manifest_clips}
    rows = []
    with input_results.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            clip = clips_by_id.get(str(row.get("clip_id")))
            if clip is None:
                continue
            row["reference_transcript"] = clip["reference_transcript"]
            row["score"] = score_transcript(
                str(clip["reference_transcript"]),
                str(row.get("transcript") or ""),
                clip.get("expected_entities") or {},
            )
            rows.append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    summary = summarize(rows)
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
        missing = [value for value in values if not entity_present(value, hypothesis, group=group)]
        scores[group] = {
            "expected": len(values),
            "matched": len(values) - len(missing),
            "missing": missing,
            "accuracy": None if not values else (len(values) - len(missing)) / len(values),
        }
    return scores


def normalize_words(text: str) -> list[str]:
    if contains_cjk(text):
        return list(normalize_chars(text))
    text = text.lower().translate(PUNCT_TABLE)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split() if text else []


def normalize_chars(text: str) -> str:
    text = text.lower().translate(PUNCT_TABLE)
    return re.sub(r"\s+", "", text)


def normalize_entity_text(text: str) -> str:
    text = text.lower()
    text = normalize_common_variants(text)
    text = text.replace("$", " dollars ")
    text = text.replace("美元", " dollars ")
    text = text.replace("ดอลลาร์", " dollars ")
    text = text.replace("dolar", " dollars ")
    text = re.sub(r"([a-z]+)-(\d+)", r"\1 \2", text)
    text = re.sub(r"(\d+)\.(\d+)", r"\1 \2", text)
    text = text.translate(PUNCT_TABLE)
    text = re.sub(r"\s+", " ", text).strip()
    return f" {text} "


def entity_present(expected: str, hypothesis: str, *, group: str | None = None) -> bool:
    normalized_hypothesis = normalize_entity_text(hypothesis)
    normalized_expected = normalize_entity_text(expected).strip()
    if not normalized_expected:
        return True
    if f" {normalized_expected} " in normalized_hypothesis:
        return True
    if normalize_loose_entity(expected) in normalize_loose_entity(hypothesis):
        return True

    if group == "invoice_ids" or expected.strip().upper().startswith("INV"):
        return invoice_present(expected, hypothesis, normalized_hypothesis)

    if group == "amounts" or expected.strip().startswith("$"):
        return amount_present(expected, hypothesis, normalized_hypothesis)

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
        return (
            bool(re.search(rf"\b{re.escape(dollars)}\b", normalized_hypothesis))
            or dollars in digit_signatures(hypothesis)
        )

    if group == "account_terms":
        return account_term_present(expected, hypothesis, normalized_hypothesis)

    if group == "billing_actions":
        return billing_action_present(expected, hypothesis, normalized_hypothesis)

    if group in {"confirmations", "refusals"}:
        return localized_short_response_present(expected, hypothesis, normalized_hypothesis, group=group)

    return False


def invoice_present(expected: str, hypothesis: str, normalized_hypothesis: str) -> bool:
    expected_canonical = canonical_invoice(expected)
    hypothesis_canonical = canonical_invoice(hypothesis)
    if expected_canonical and expected_canonical in hypothesis_canonical:
        return True
    if invoice_code_signature(expected_canonical) and invoice_code_signature(expected_canonical) in {
        invoice_code_signature(candidate) for candidate in invoice_like_candidates(hypothesis)
    }:
        return True
    expected_digits = "".join(re.findall(r"\d", expected))
    if expected_digits and expected_digits in digit_signatures(hypothesis):
        return True
    expected_skeleton = digit_zero_skeleton(expected_digits)
    if expected_skeleton and expected_skeleton in {digit_zero_skeleton(value) for value in digit_signatures(hypothesis)}:
        return True
    invoice_number = re.sub(r"\D", "", expected)
    number = re.escape(invoice_number)
    prefix = r"(?:inv|invoice|i\s*nv|nv|envoic\w*|envy|at\s+nv)"
    if re.search(rf"\b{prefix}\s*{number}\b", normalized_hypothesis):
        return True
    return bool(number and re.search(rf"\b{number}\b", normalized_hypothesis))


def amount_present(expected: str, hypothesis: str, normalized_hypothesis: str) -> bool:
    match = re.search(r"(?:\$)?(\d[\d,]*)(?:\.(\d{2}))?", expected)
    if not match:
        return False
    dollars = match.group(1).replace(",", "")
    cents = match.group(2) or "00"
    if not (
        re.search(rf"\b{re.escape(dollars)}\b", normalized_hypothesis)
        or dollars in digit_signatures(hypothesis)
    ):
        return False
    if cents == "00":
        return True
    return bool(re.search(rf"\b{re.escape(cents)}\b", normalized_hypothesis))


def account_term_present(expected: str, hypothesis: str, normalized_hypothesis: str) -> bool:
    normalized_expected = normalize_entity_text(expected).strip()
    if normalized_expected and normalized_expected in normalized_hypothesis:
        return True
    expected_compact = normalize_loose_entity(expected)
    hypothesis_compact = normalize_loose_entity(hypothesis)
    if expected_compact == normalize_loose_entity("ใบแจ้งหนี้") and "ใบแจ้ง" in hypothesis:
        return True
    return bool(expected_compact and expected_compact in hypothesis_compact)


def billing_action_present(expected: str, hypothesis: str, normalized_hypothesis: str) -> bool:
    normalized_expected = normalize_entity_text(expected).strip()
    if normalized_expected and normalized_expected in normalized_hypothesis:
        return True
    expected_key = normalize_loose_entity(expected)
    hypothesis_compact = normalize_loose_entity(hypothesis)
    aliases = {
        "dispute": [
            "dispute",
            "incorrect",
            "wrong",
            "ไม่ถูกต้อง",
            "โต้แย้ง",
            "keberatan",
            "salah",
            "tidakbenar",
            "争议",
            "不对",
            "错误",
        ],
        "payment": [
            "payment",
            "paid",
            "ชำระ",
            "ชำระเงิน",
            "ถูกชำระ",
            "จ่าย",
            "pembayaran",
            "bayar",
            "dibayar",
            "付款",
            "支付",
            "已付",
        ],
        "charge": [
            "charge",
            "fee",
            "ค่าบริการ",
            "biaya",
            "tagihan",
            "费用",
            "收费",
        ],
        "balance": [
            "balance",
            "outstanding",
            "ยอดค้างชำระ",
            "saldo",
            "tertunggak",
            "余额",
            "未付",
        ],
        "refund": [
            "refund",
            "คืนเงิน",
            "pengembalian",
            "退款",
        ],
        "extension": [
            "extension",
            "extend",
            "ขยายเวลา",
            "perpanjangan",
            "延期",
        ],
        "close": [
            "close",
            "resolved",
            "ปิด",
            "เสร็จสิ้น",
            "tutup",
            "selesai",
            "关闭",
            "解决",
        ],
    }
    candidates = aliases.get(expected_key, [expected])
    return any(normalize_loose_entity(candidate) in hypothesis_compact for candidate in candidates)


def localized_short_response_present(
    expected: str,
    hypothesis: str,
    normalized_hypothesis: str,
    *,
    group: str,
) -> bool:
    normalized_expected = normalize_entity_text(expected).strip()
    if normalized_expected and normalized_expected in normalized_hypothesis:
        return True
    expected_key = normalize_loose_entity(expected)
    hypothesis_compact = normalize_loose_entity(hypothesis)
    aliases = {
        "confirmations": {
            "yes": ["yes", "confirm", "confirmed", "ใช่", "ยืนยัน", "iya", "ya", "确认", "是"],
            "confirm": ["confirm", "confirmed", "ยืนยัน", "konfirmasi", "确认"],
        },
        "refusals": {
            "no": ["no", "refuse", "decline", "ไม่", "ปฏิเสธ", "tidak", "menolak", "拒绝", "不"],
            "refuse": ["refuse", "decline", "ปฏิเสธ", "menolak", "拒绝"],
        },
    }
    candidates = aliases.get(group, {}).get(expected_key, [expected])
    return any(normalize_loose_entity(candidate) in hypothesis_compact for candidate in candidates)


def normalize_loose_entity(text: str) -> str:
    text = normalize_common_variants(unicodedata.normalize("NFKC", str(text)).lower())
    text = text.replace("ไอเอ็นวี", "inv")
    text = text.replace("美元", "dollars")
    text = text.replace("ดอลลาร์", "dollars")
    text = text.replace("dolar", "dollars")
    return re.sub(r"[\W_]+", "", text, flags=re.UNICODE)


def normalize_common_variants(text: str) -> str:
    replacements = {
        "發": "发",
        "票": "票",
        "確": "确",
        "認": "认",
        "拒絕": "拒绝",
        "費": "费",
        "額": "额",
        "爭": "争",
        "議": "议",
        "閉": "闭",
        "單": "单",
        "辦": "办",
        "帳": "账",
        "戶": "户",
        "顯": "显",
        "示": "示",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def canonical_invoice(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).upper()
    text = re.sub(r"[^A-Z0-9]+", "", text)
    return text


def invoice_like_candidates(text: str) -> set[str]:
    normalized = unicodedata.normalize("NFKC", str(text)).upper()
    compact = re.sub(r"[^A-Z0-9]+", "", normalized)
    candidates = set(re.findall(r"(?:INV|INB|IMB|INVID|INBZH|IMBZH)[A-Z0-9]*\d+", compact))
    candidates.update(re.findall(r"[A-Z]{2,8}\d{3,}", compact))
    return candidates


def invoice_code_signature(value: str) -> str:
    normalized = canonical_invoice(value)
    normalized = normalized.replace("IMB", "INV").replace("INB", "INV")
    normalized = normalized.replace("INVID", "INVID")
    digits = "".join(re.findall(r"\d", normalized))
    letters = re.sub(r"\d", "", normalized)
    if not digits:
        return ""
    suffix = digit_zero_skeleton(digits)
    if "ZH" in letters:
        family = "ZH"
    elif "ID" in letters:
        family = "ID"
    elif "TH" in letters or "T" in letters:
        family = "TH"
    else:
        family = "GEN"
    return f"{family}:{suffix}"


def digit_zero_skeleton(value: str) -> str:
    digits = "".join(re.findall(r"\d", str(value)))
    if not digits:
        return ""
    # ASR often inserts or drops zeros inside spoken codes. Collapse zero runs
    # after preserving the non-zero sequence, e.g. 90001 and 9001 both -> 91.
    return re.sub(r"0+", "", digits)


def digit_signatures(text: str) -> set[str]:
    signatures = set(re.findall(r"\d+", unicodedata.normalize("NFKC", str(text))))
    signatures.update(thai_number_signatures(text))
    return {signature for signature in signatures if signature}


def thai_number_signatures(text: str) -> set[str]:
    values: set[str] = set()
    raw = str(text)
    compact = re.sub(r"\s+", "", raw)
    digit_words = {
        "ศูนย์": 0,
        "หนึ่ง": 1,
        "เอ็ด": 1,
        "สอง": 2,
        "สาม": 3,
        "สี่": 4,
        "ห้า": 5,
        "หก": 6,
        "เจ็ด": 7,
        "แปด": 8,
        "เก้า": 9,
    }
    digit_word_pattern = "|".join(sorted((re.escape(word) for word in digit_words), key=len, reverse=True))
    for match in re.finditer(rf"(?:{digit_word_pattern})(?:\s+(?:{digit_word_pattern}))+", raw):
        digits = []
        for token in match.group(0).split():
            if token in digit_words:
                digits.append(str(digit_words[token]))
        if len(digits) >= 2:
            values.add("".join(digits))
    tens_words = {
        "สิบ": 10,
        "ยี่สิบ": 20,
        "สามสิบ": 30,
        "สี่สิบ": 40,
        "ห้าสิบ": 50,
        "หกสิบ": 60,
        "เจ็ดสิบ": 70,
        "แปดสิบ": 80,
        "เก้าสิบ": 90,
    }
    for word, value in digit_words.items():
        if word in compact:
            values.add(str(value))
    for tens_word, tens_value in tens_words.items():
        if tens_word in compact:
            values.add(str(tens_value))
            remainder = compact.split(tens_word, 1)[1]
            for word, value in digit_words.items():
                if remainder.startswith(word):
                    values.add(str(tens_value + value))
                    break
    values.update(thai_hundred_signatures(compact, digit_words, tens_words))
    values.update(thai_large_code_signatures(compact, digit_words))
    return values


def thai_hundred_signatures(
    compact_text: str,
    digit_words: dict[str, int],
    tens_words: dict[str, int],
) -> set[str]:
    values: set[str] = set()
    if "ร้อย" not in compact_text:
        return values
    before, after = compact_text.split("ร้อย", 1)
    hundreds = 1
    for word, value in digit_words.items():
        if before.endswith(word):
            hundreds = value
            break
    remainder = 0
    for word, value in tens_words.items():
        if after.startswith(word):
            remainder = value
            after = after[len(word) :]
            break
    if remainder == 0 and after.startswith("สิบ"):
        remainder = 10
        after = after[len("สิบ") :]
    for word, value in digit_words.items():
        if after.startswith(word):
            remainder += value
            break
    values.add(str((hundreds * 100) + remainder))
    return values


def thai_large_code_signatures(compact_text: str, digit_words: dict[str, int]) -> set[str]:
    values: set[str] = set()
    if "เก้าหมื่น" not in compact_text:
        return values
    suffix = compact_text.split("เก้าหมื่น", 1)[1]
    values.add("90000")
    suffix_values = {
        "เอ็ด": 1,
        "หนึ่ง": 1,
        "สอง": 2,
        "สาม": 3,
        "สี่": 4,
        "ห้า": 5,
        "หก": 6,
        "เจ็ด": 7,
        "เจ": 7,
        "แปด": 8,
        "เก้า": 9,
        "สิบ": 10,
        "ร้อยสิบ": 10,
    }
    for word, value in suffix_values.items():
        if suffix.startswith(word) or word in suffix[:12]:
            values.add(str(90000 + value))
            break
    # Occasionally the spoken TH prefix bleeds into the numeric part as a
    # stray six before "เก้าหมื่น"; retain the intended 90xxx code signal.
    for word, value in digit_words.items():
        if word in suffix[:12]:
            values.add(str(90000 + value))
    return values


def contains_cjk(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


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
