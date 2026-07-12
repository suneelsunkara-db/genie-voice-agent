from __future__ import annotations

import re
from typing import Any

from genie_voice.asr_eval.manifest import ASRGoldClip, ExpectedEntities

CRITICAL_GROUPS = ("invoice_ids", "amounts", "billing_actions", "confirmations", "refusals")
NEGATION_TERMS = re.compile(
    r"\b(?:no|not|never|don't|dont|cannot|can't|cant|won't|wont|refuse|decline|tidak|ไม่|不)\b",
    re.IGNORECASE,
)


def assess_readiness(clip: ASRGoldClip, transcript: str, score: dict[str, Any]) -> dict[str, Any]:
    entities = clip.expected_entities
    entity_scores = score.get("entity_scores") or {}
    reasons: list[str] = []
    transcript = (transcript or "").strip()
    if not transcript:
        reasons.append("empty_transcript")
    if entities.invoice_ids and (entity_scores.get("invoice_ids") or {}).get("matched", 0) == 0:
        reasons.append("missing_invoice_id")
    if entities.amounts and (entity_scores.get("amounts") or {}).get("matched", 0) == 0:
        reasons.append("missing_amount")
    if entities.billing_actions and (entity_scores.get("billing_actions") or {}).get("matched", 0) == 0:
        reasons.append("missing_billing_action")
    if (entities.confirmations or entities.refusals) and not _decision_phrase_match(entities, entity_scores):
        reasons.append("missing_customer_decision_phrase")
    if entities.amounts and not re.search(r"\d", transcript):
        reasons.append("missing_numeric_token")
    return {
        "unsafe_for_resolution": bool(reasons),
        "unsafe_reasons": reasons,
        "critical_entity_accuracy": _critical_entity_accuracy(entities, entity_scores),
        "negation_match": _negation_match(clip.reference_transcript, transcript),
    }


def _critical_entity_accuracy(entities: ExpectedEntities, entity_scores: dict[str, Any]) -> float | None:
    expected = 0
    matched = 0
    for group in CRITICAL_GROUPS:
        group_score = entity_scores.get(group) or {}
        expected += int(group_score.get("expected") or 0)
        matched += int(group_score.get("matched") or 0)
    if expected == 0:
        return None
    return matched / expected


def _decision_phrase_match(entities: ExpectedEntities, entity_scores: dict[str, Any]) -> bool:
    confirmations = (entity_scores.get("confirmations") or {}).get("matched", 0)
    refusals = (entity_scores.get("refusals") or {}).get("matched", 0)
    if entities.confirmations and confirmations > 0:
        return True
    if entities.refusals and refusals > 0:
        return True
    return confirmations > 0 or refusals > 0


def _negation_match(reference: str, hypothesis: str) -> bool | None:
    ref = bool(NEGATION_TERMS.search(reference))
    hyp = bool(NEGATION_TERMS.search(hypothesis))
    if not ref:
        return None
    return ref == hyp
