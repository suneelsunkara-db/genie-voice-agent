"""ASR scoring for generic transcript quality and billing-specific entities."""
from __future__ import annotations

import re
import string
from dataclasses import asdict, dataclass, field
from typing import Any

from .manifest import ExpectedEntities


_PUNCT_TABLE = str.maketrans("", "", string.punctuation.replace("$", ""))


@dataclass(frozen=True)
class EntityGroupScore:
    expected: int
    matched: int
    missing: list[str] = field(default_factory=list)

    @property
    def accuracy(self) -> float | None:
        if self.expected == 0:
            return None
        return self.matched / self.expected

    def to_dict(self) -> dict[str, Any]:
        return {
            "expected": self.expected,
            "matched": self.matched,
            "missing": self.missing,
            "accuracy": self.accuracy,
        }


@dataclass(frozen=True)
class ASRScore:
    wer: float
    cer: float
    word_errors: int
    reference_words: int
    char_errors: int
    reference_chars: int
    entity_scores: dict[str, EntityGroupScore]
    entity_accuracy: float | None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["entity_scores"] = {
            key: score.to_dict() for key, score in self.entity_scores.items()
        }
        return data


def score_transcript(
    reference: str,
    hypothesis: str,
    expected_entities: ExpectedEntities | None = None,
) -> ASRScore:
    """Score one ASR hypothesis against the human reference transcript."""
    ref_words = _normalize_words(reference)
    hyp_words = _normalize_words(hypothesis)
    word_errors = _edit_distance(ref_words, hyp_words)

    ref_chars = _normalize_chars(reference)
    hyp_chars = _normalize_chars(hypothesis)
    char_errors = _edit_distance(list(ref_chars), list(hyp_chars))

    entity_scores = score_entities(hypothesis, expected_entities or ExpectedEntities())
    entity_expected = sum(score.expected for score in entity_scores.values())
    entity_matched = sum(score.matched for score in entity_scores.values())
    entity_accuracy = None if entity_expected == 0 else entity_matched / entity_expected

    return ASRScore(
        wer=_ratio(word_errors, len(ref_words)),
        cer=_ratio(char_errors, len(ref_chars)),
        word_errors=word_errors,
        reference_words=len(ref_words),
        char_errors=char_errors,
        reference_chars=len(ref_chars),
        entity_scores=entity_scores,
        entity_accuracy=entity_accuracy,
    )


def score_entities(
    hypothesis: str,
    expected_entities: ExpectedEntities,
) -> dict[str, EntityGroupScore]:
    """Score whether expected business entities appear in the transcript."""
    normalized_hypothesis = _normalize_entity_text(hypothesis)
    scores: dict[str, EntityGroupScore] = {}
    for group, expected_values in expected_entities.groups().items():
        missing: list[str] = []
        for value in expected_values:
            if not _entity_present(value, normalized_hypothesis, group=group):
                missing.append(value)
        scores[group] = EntityGroupScore(
            expected=len(expected_values),
            matched=len(expected_values) - len(missing),
            missing=missing,
        )
    return scores


def _normalize_words(text: str) -> list[str]:
    text = text.lower().translate(_PUNCT_TABLE)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split() if text else []


def _normalize_chars(text: str) -> str:
    text = text.lower().translate(_PUNCT_TABLE)
    return re.sub(r"\s+", "", text)


def _normalize_entity_text(text: str) -> str:
    text = text.lower()
    text = text.replace("$", " dollars ")
    text = re.sub(r"([a-z]+)-(\d+)", r"\1 \2", text)
    text = re.sub(r"(\d+)\.(\d+)", r"\1 \2", text)
    text = text.translate(_PUNCT_TABLE)
    text = re.sub(r"\s+", " ", text).strip()
    return f" {text} "


def _entity_present(expected: str, normalized_hypothesis: str, *, group: str | None = None) -> bool:
    normalized_expected = _normalize_entity_text(expected).strip()
    if not normalized_expected:
        return True
    if f" {normalized_expected} " in normalized_hypothesis:
        return True
    if _known_entity_variant_present(normalized_expected, normalized_hypothesis, group=group):
        return True

    # Common billing normalization: INV-10482, invoice 10482, and inv 10482
    # should be treated as equivalent.
    invoice_match = re.search(r"(?:inv|invoice|i\s+nv)\s*(\d+)", normalized_expected)
    if invoice_match:
        invoice_number = invoice_match.group(1)
        return _invoice_present(invoice_number, normalized_hypothesis)

    if expected.strip().startswith("$"):
        return _amount_present(expected, normalized_hypothesis)

    date_match = _date_parts(normalized_expected)
    if date_match:
        month, day = date_match
        return _date_present(month, day, normalized_hypothesis)

    amount_match = re.search(r"\b(\d+)\s+dollars?(?:\s+and\s+)?(?:\s*(\d+)\s+cents?)?", normalized_expected)
    if amount_match:
        dollars = amount_match.group(1)
        cents = amount_match.group(2)
        if cents:
            return dollars in normalized_hypothesis and cents in normalized_hypothesis
        return bool(re.search(rf"\b{re.escape(dollars)}\b", normalized_hypothesis))

    return False


def _known_entity_variant_present(
    normalized_expected: str,
    normalized_hypothesis: str,
    *,
    group: str | None,
) -> bool:
    if normalized_expected == "autopay":
        return bool(re.search(r"\bauto\s*pay\b", normalized_hypothesis))
    if normalized_expected == "waive":
        return bool(re.search(r"\bwaiv(?:e|ed|ing|er)\b", normalized_hypothesis))
    if group == "account_terms" and normalized_expected == "invoice":
        return bool(
            re.search(r"\b(?:invoice|invoic\w+|envoic\w+|envoy\s+site)\b", normalized_hypothesis)
            or re.search(r"\b(?:i\s*nv|inv|nv)\s*\d+\b", normalized_hypothesis)
        )
    if group == "account_terms" and normalized_expected == "payment":
        return bool(re.search(r"\b(?:payments?|paid|pay)\b", normalized_hypothesis))
    return False


def _invoice_present(invoice_number: str, normalized_hypothesis: str) -> bool:
    number = re.escape(invoice_number)
    prefix = r"(?:inv|invoice|i\s*nv|nv|envoic\w*|envy|at\s+nv)"
    if re.search(rf"\b{prefix}\s*{number}\b", normalized_hypothesis):
        return True
    return bool(re.search(rf"\b{number}\b", normalized_hypothesis))


def _amount_present(expected: str, normalized_hypothesis: str) -> bool:
    match = re.search(r"\$(\d[\d,]*)(?:\.(\d{2}))?", expected)
    if not match:
        return False
    dollars = match.group(1).replace(",", "")
    cents = match.group(2) or "00"
    if not re.search(rf"\b{re.escape(dollars)}\b", normalized_hypothesis):
        return False
    # Treat "$89", "$89.00", and "89 dollars" as equivalent.
    if cents == "00":
        return True
    return bool(re.search(rf"\b{re.escape(cents)}\b", normalized_hypothesis))


def _date_parts(normalized_expected: str) -> tuple[str, str] | None:
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
    month_pattern = "|".join(months)
    match = re.search(rf"\b({month_pattern})\s+(\d{{1,2}})\b", normalized_expected)
    if match:
        return match.group(1), match.group(2)
    ordinal_pattern = "|".join(re.escape(word) for word in _ORDINAL_WORD_TO_DAY)
    match = re.search(rf"\b({month_pattern})\s+({ordinal_pattern})\b", normalized_expected)
    if match:
        return match.group(1), _ORDINAL_WORD_TO_DAY[match.group(2)]
    return None


def _date_present(month: str, day: str, normalized_hypothesis: str) -> bool:
    day_words = [word for word, value in _ORDINAL_WORD_TO_DAY.items() if value == str(int(day))]
    day_variants = [rf"{re.escape(str(int(day)))}(?:st|nd|rd|th)?"]
    day_variants.extend(re.escape(word) for word in day_words)
    day_pattern = rf"(?:{'|'.join(day_variants)})"
    month_pattern = re.escape(month)
    month_then_day = rf"\b{month_pattern}\s+(?:the\s+)?{day_pattern}\b"
    day_then_month = rf"\b(?:the\s+)?{day_pattern}\s+(?:of\s+)?{month_pattern}\b"
    return bool(
        re.search(month_then_day, normalized_hypothesis)
        or re.search(day_then_month, normalized_hypothesis)
    )


_ORDINAL_WORD_TO_DAY = {
    "first": "1",
    "second": "2",
    "third": "3",
    "fourth": "4",
    "fifth": "5",
    "sixth": "6",
    "seventh": "7",
    "eighth": "8",
    "ninth": "9",
    "tenth": "10",
    "eleventh": "11",
    "twelfth": "12",
    "thirteenth": "13",
    "fourteenth": "14",
    "fifteenth": "15",
    "sixteenth": "16",
    "seventeenth": "17",
    "eighteenth": "18",
    "nineteenth": "19",
    "twentieth": "20",
    "twenty first": "21",
    "twenty second": "22",
    "twenty third": "23",
    "twenty fourth": "24",
    "twenty fifth": "25",
    "twenty sixth": "26",
    "twenty seventh": "27",
    "twenty eighth": "28",
    "twenty ninth": "29",
    "thirtieth": "30",
    "thirty first": "31",
}


def _edit_distance(left: list[str], right: list[str]) -> int:
    if not left:
        return len(right)
    if not right:
        return len(left)

    previous = list(range(len(right) + 1))
    for i, left_item in enumerate(left, start=1):
        current = [i]
        for j, right_item in enumerate(right, start=1):
            cost = 0 if left_item == right_item else 1
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + cost,
                )
            )
        previous = current
    return previous[-1]


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0 if numerator == 0 else 1.0
    return numerator / denominator
