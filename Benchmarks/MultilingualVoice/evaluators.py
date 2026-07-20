"""Judge-free evaluators for the multilingual voice benchmark.

- ASR   : WER + CER via Levenshtein edit distance (FLEURS, TTS round-trip).
- MCQ   : answer-letter/number extraction + accuracy (2M-Belebele).
- QA    : normalized match + optional GPT judge correctness (CCFQA).

Pure-Python (no jiwer/torch dependency) so it runs anywhere the API client runs.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Iterable

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_SPACE_RE = re.compile(r"\s+", re.UNICODE)


def normalize_text(text: str, *, keep_spaces: bool = True) -> str:
    text = unicodedata.normalize("NFKC", str(text or "")).lower().strip()
    text = _PUNCT_RE.sub(" ", text)
    text = _SPACE_RE.sub(" ", text).strip()
    if not keep_spaces:
        text = text.replace(" ", "")
    return text


def _edit_distance(ref: list[str], hyp: list[str]) -> int:
    if not ref:
        return len(hyp)
    if not hyp:
        return len(ref)
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, start=1):
            cost = 0 if r == h else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[-1]


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref = normalize_text(reference).split()
    hyp = normalize_text(hypothesis).split()
    if not ref:
        return 0.0 if not hyp else 1.0
    return _edit_distance(ref, hyp) / len(ref)


def char_error_rate(reference: str, hypothesis: str) -> float:
    ref = list(normalize_text(reference, keep_spaces=False))
    hyp = list(normalize_text(hypothesis, keep_spaces=False))
    if not ref:
        return 0.0 if not hyp else 1.0
    return _edit_distance(ref, hyp) / len(ref)


def _mean(values: Iterable[float]) -> float | None:
    vals = [v for v in values if isinstance(v, (int, float))]
    return sum(vals) / len(vals) if vals else None


def evaluate_asr(rows: list[dict[str, Any]], *, reference_key: str = "reference", hyp_key: str = "transcript") -> dict[str, Any]:
    """WER/CER of the API's STT transcript vs the dataset reference."""
    scored = []
    for row in rows:
        ref = row.get(reference_key)
        hyp = row.get(hyp_key)
        if not ref:
            continue
        scored.append((word_error_rate(ref, hyp or ""), char_error_rate(ref, hyp or "")))
    return {
        "wer": _mean(w for w, _ in scored),
        "cer": _mean(c for _, c in scored),
        "scored": len(scored),
    }


def evaluate_tts_roundtrip(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """WER/CER of re-transcribed TTS audio vs the spoken text (intelligibility)."""
    scored = []
    produced = 0
    for row in rows:
        rt = row.get("tts_roundtrip") or {}
        spoken = rt.get("spoken_text") or row.get("response")
        heard = rt.get("reheard_text")
        if row.get("tts_audio_bytes"):
            produced += 1
        if not spoken or heard is None:
            continue
        scored.append((word_error_rate(spoken, heard), char_error_rate(spoken, heard)))
    return {
        "tts_roundtrip_wer": _mean(w for w, _ in scored),
        "tts_roundtrip_cer": _mean(c for _, c in scored),
        "tts_audio_rate": (produced / len(rows)) if rows else None,
        "scored": len(scored),
    }


_MCQ_LETTERS = ["a", "b", "c", "d"]


def extract_mcq_choice(response: str) -> int | None:
    """Return a 1-based choice index (1..4) or None."""
    text = normalize_text(response)
    if not text:
        return None
    for idx, letter in enumerate(_MCQ_LETTERS, start=1):
        for pat in (
            f"answer is {letter}", f"answer {letter}", f"option {letter}",
            f"choice {letter}", f"the answer is {letter}", f"correct answer is {letter}",
        ):
            if pat in text:
                return idx
    for idx, num in enumerate(["1", "2", "3", "4"], start=1):
        if text.strip() == num or text.startswith(f"{num} "):
            return idx
        for pat in (
            f"answer is {num}", f"answer {num}", f"option {num}",
            f"choice {num}", f"correct answer is {num}", f"number {num}",
        ):
            if pat in text:
                return idx
    tokens = text.split()
    if tokens and tokens[0] in _MCQ_LETTERS:
        return _MCQ_LETTERS.index(tokens[0]) + 1
    if tokens and tokens[0] in {"1", "2", "3", "4"}:
        return int(tokens[0])
    return None


def evaluate_mcq(rows: list[dict[str, Any]]) -> dict[str, Any]:
    correct = 0
    total = 0
    unparsed = 0
    for row in rows:
        gold = row.get("correct_choice")
        if gold is None:
            continue
        total += 1
        pred = extract_mcq_choice(row.get("response") or "")
        if pred is None:
            unparsed += 1
            continue
        if int(pred) == int(gold):
            correct += 1
    return {
        "acc": (100.0 * correct / total) if total else None,
        "parse_fail_rate": (100.0 * unparsed / total) if total else None,
        "scored": total,
    }


def evaluate_qa(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Judge-free QA correctness: normalized reference-substring / token overlap.

    If rows carry a GPT-judge ``score`` (list of yes/no), that takes precedence.
    """
    judged = [row for row in rows if row.get("score")]
    if judged:
        yes = 0
        for row in judged:
            votes = [str(s).strip().lower() for s in row["score"]]
            if votes.count("yes") >= votes.count("no"):
                yes += 1
        return {"gpt_correct": 100.0 * yes / len(judged), "scored": len(judged), "method": "gpt_judge"}

    hits = 0
    total = 0
    for row in rows:
        ref = row.get("reference")
        resp = row.get("response")
        if not ref:
            continue
        total += 1
        ref_norm = normalize_text(ref)
        resp_norm = normalize_text(resp or "")
        if ref_norm and (ref_norm in resp_norm or resp_norm in ref_norm):
            hits += 1
    return {
        "match_acc": (100.0 * hits / total) if total else None,
        "scored": total,
        "method": "normalized_match",
    }


EVALUATORS = {
    "asr": evaluate_asr,
    "mcq": evaluate_mcq,
    "qa": evaluate_qa,
}


# Scripts that don't delimit words with whitespace: word-level WER is meaningless
# for them (the whole utterance is one "word"), so character error rate is the
# correct primary ASR metric. Covers Chinese, Japanese, Thai, Lao, Khmer,
# Burmese, and Cantonese.
_CER_LANGS = {"zh", "ja", "th", "lo", "km", "my", "yue"}


def _is_cer_lang(language: str | None) -> bool:
    return bool(language) and language.split("-")[0].lower() in _CER_LANGS


def primary_metric(evaluator: str, scores: dict[str, Any], *, language: str | None = None) -> float | None:
    if evaluator == "asr":
        return scores.get("cer") if _is_cer_lang(language) else scores.get("wer")
    if evaluator == "mcq":
        return scores.get("acc")
    if evaluator == "qa":
        return scores.get("gpt_correct", scores.get("match_acc"))
    return None


def primary_metric_name(evaluator: str, language: str | None = None) -> str | None:
    """Name of the metric ``primary_metric`` returns (for self-documenting rows)."""
    if evaluator == "asr":
        return "cer" if _is_cer_lang(language) else "wer"
    if evaluator == "mcq":
        return "acc"
    if evaluator == "qa":
        return "gpt_correct/match_acc"
    return None
