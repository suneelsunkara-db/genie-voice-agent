from __future__ import annotations

import re
from typing import Any

from genie_voice.i18n import LanguageCode

from genie_voice.ml_asr.manifest import empty_entities

SCENARIO_KEYWORDS: dict[str, dict[LanguageCode, tuple[str, ...]]] = {
    "billing_dispute": {
        "en-US": ("dispute", "incorrect charge", "overcharge", "billing error", "wrong amount"),
        "th-TH": ("โต้แย้ง", "เรียกเก็บเกิน", "ยอดไม่ถูก"),
        "id-ID": ("keberatan tagihan", "tagihan salah", "kenaikan biaya"),
        "zh-CN": ("账单异议", "多收费", "金额不对"),
    },
    "payment_lookup": {
        "en-US": ("payment status", "check payment", "payment received", "track payment"),
        "th-TH": ("ตรวจสอบการชำระ", "สถานะการชำระ"),
        "id-ID": ("status pembayaran", "periksa pembayaran"),
        "zh-CN": ("查询付款", "付款状态", "到账"),
    },
    "payment_confirmation": {
        "en-US": ("confirm payment", "payment confirmed", "paid in full", "payment settled"),
        "th-TH": ("ยืนยันการชำระ", "ชำระแล้ว"),
        "id-ID": ("konfirmasi pembayaran", "sudah bayar"),
        "zh-CN": ("确认付款", "已付清"),
    },
    "charge_refusal": {
        "en-US": ("refuse to pay", "will not pay", "do not charge", "decline charge", "stop charging"),
        "th-TH": ("ปฏิเสธการชำระ", "ไม่ยอมจ่าย", "อย่าเรียกเก็บ"),
        "id-ID": ("tolak membayar", "tidak mau bayar", "tolak tagihan"),
        "zh-CN": ("拒绝付款", "不要扣费", "拒付"),
    },
    "account_balance": {
        "en-US": ("account balance", "outstanding balance", "amount due", "balance due"),
        "th-TH": ("ยอดคงเหลือ", "ยอดค้างชำระ"),
        "id-ID": ("saldo rekening", "sisa tagihan"),
        "zh-CN": ("账户余额", "欠款金额"),
    },
    "refund_request": {
        "en-US": ("refund request", "money back", "issue refund"),
        "th-TH": ("ขอคืนเงิน", "คืนเงิน"),
        "id-ID": ("minta refund", "pengembalian dana"),
        "zh-CN": ("申请退款", "退回款项"),
    },
    "payment_extension": {
        "en-US": ("payment extension", "extend deadline", "more time to pay"),
        "th-TH": ("ขยายเวลาชำระ", "เลื่อนกำหนดชำระ"),
        "id-ID": ("perpanjangan pembayaran", "tunda pembayaran"),
        "zh-CN": ("延期付款", "延长付款"),
    },
    "case_close": {
        "en-US": ("issue resolved", "all set now", "that works for me"),
        "th-TH": ("แก้ไขเรียบร้อย", "เรียบร้อยแล้ว"),
        "id-ID": ("masalah selesai", "sudah beres"),
        "zh-CN": ("问题已解决", "处理好了"),
    },
}

BILLING_TERMS: dict[LanguageCode, tuple[str, ...]] = {
    "en-US": ("invoice", "billing", "payment", "refund", "balance", "account balance", "amount due"),
    "th-TH": ("ใบแจ้งหนี้", "ชำระเงิน", "ยอดค้าง", "เรียกเก็บ"),
    "id-ID": ("tagihan", "pembayaran", "saldo", "faktur"),
    "zh-CN": ("发票", "账单", "付款", "余额", "扣款"),
}

STRICT_INVOICE_PATTERNS = (
    r"\bINV[-\s]?\d[\w-]*\b",
    r"\binvoice\s*#?\s*\d[\w-]*\b",
    r"\b(?:bill|invoice)\s+(?:number\s+)?[A-Z0-9-]{4,}\b",
)

PLAUSIBLE_AMOUNT_PATTERNS = (
    r"\$\s?\d[\d,]*(?:\.\d{2})?",
    r"\b\d[\d,]*(?:\.\d{2})?\s?(?:dollars?|usd|baht|rupiah|yuan|元|ดอลลาร์)\b",
    r"\b\d{1,3}(?:,\d{3})+(?:\.\d{2})?\b",
    r"\b\d{3,}\b",
)

NUMERIC_ENTITY_SCENARIO = "numeric_entity_holdout"

FALSE_POSITIVE_PHRASES = (
    "criminal charge",
    "hostage crisis",
    "king louis",
    "marie antoinette",
    "soviet invasion",
    "camp david",
    "multiple sclerosis",
    "skiing",
    "super g",
    "number one beer",
)

YEAR_AMOUNT_RE = re.compile(r"^(?:1\d{3}|20\d{2})$")
SMALL_INTEGER_RE = re.compile(r"^\d{1,2}$")
MIN_HOLDOUT_QUALITY = 2
MIN_SCENARIO_HITS = 1
MIN_TRANSCRIPT_WORDS = 8


def is_business_relevant(transcript: str, language: LanguageCode) -> bool:
    text = _normalize_text(transcript)
    if not text or len(text.split()) < MIN_TRANSCRIPT_WORDS:
        return False
    if _contains_false_positive_phrase(text):
        return False
    entities = extract_entities(transcript, language)
    return bool(
        entities["amounts"]
        or entities["invoice_ids"]
        or entities["refusals"]
        or entities["confirmations"]
        or (_has_billing_term(text, language) and entities["billing_actions"])
    )


def classify_scenario(transcript: str, language: LanguageCode) -> str:
    text = _normalize_text(transcript)
    if _contains_false_positive_phrase(text):
        return "general_billing"

    best = "general_billing"
    best_score = 0
    for scenario, by_lang in SCENARIO_KEYWORDS.items():
        score = scenario_keyword_hits(transcript, scenario, language)
        if score > best_score:
            best = scenario
            best_score = score

    if best_score < MIN_SCENARIO_HITS:
        return "general_billing"
    return best


def scenario_keyword_hits(transcript: str, scenario: str, language: LanguageCode) -> int:
    keywords = SCENARIO_KEYWORDS.get(scenario, {}).get(language, ())
    if not keywords:
        keywords = SCENARIO_KEYWORDS.get(scenario, {}).get("en-US", ())
    lowered = _normalize_text(transcript)
    return sum(1 for keyword in keywords if keyword.lower() in lowered)


def extract_entities(transcript: str, language: LanguageCode) -> dict[str, list[str]]:
    entities = empty_entities()
    text = transcript.strip()
    lowered = _normalize_text(transcript)

    for pattern in STRICT_INVOICE_PATTERNS:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            value = match.group(0).strip()
            if value not in entities["invoice_ids"]:
                entities["invoice_ids"].append(value)

    for pattern in PLAUSIBLE_AMOUNT_PATTERNS:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            value = match.group(0).strip()
            if _is_plausible_amount(value) and value not in entities["amounts"]:
                entities["amounts"].append(value)

    scenario = classify_scenario(transcript, language)
    if scenario != "general_billing":
        action = scenario.replace("_", " ")
        entities["billing_actions"].append(action)

    confirm_terms = SCENARIO_KEYWORDS["payment_confirmation"].get(language, ())
    refuse_terms = SCENARIO_KEYWORDS["charge_refusal"].get(language, ())
    for term in confirm_terms:
        if term.lower() in lowered and term not in entities["confirmations"]:
            entities["confirmations"].append(term)
    for term in refuse_terms:
        if term.lower() in lowered and term not in entities["refusals"]:
            entities["refusals"].append(term)

    for term in BILLING_TERMS.get(language, BILLING_TERMS["en-US"]):
        if term.lower() in lowered and term not in entities["account_terms"]:
            entities["account_terms"].append(term)

    return entities


def entity_quality_score(entities: dict[str, list[str]]) -> int:
    score = 0
    if entities.get("amounts"):
        score += 2
    if entities.get("invoice_ids"):
        score += 2
    if entities.get("billing_actions"):
        score += 1
    if entities.get("confirmations") or entities.get("refusals"):
        score += 1
    if entities.get("account_terms"):
        score += 1
    return score


def passes_business_holdout_bar(
    transcript: str,
    language: LanguageCode,
) -> tuple[bool, dict[str, list[str]], str, int]:
    if not is_business_relevant(transcript, language):
        return False, empty_entities(), "general_billing", 0

    entities = extract_entities(transcript, language)
    quality = entity_quality_score(entities)
    scenario = classify_scenario(transcript, language)
    if scenario == "general_billing" and entities["amounts"]:
        scenario = NUMERIC_ENTITY_SCENARIO
        if "numeric entity holdout" not in entities["billing_actions"]:
            entities["billing_actions"].append("numeric entity holdout")

    if quality < MIN_HOLDOUT_QUALITY:
        return False, entities, scenario, quality
    if scenario == "general_billing":
        return False, entities, scenario, quality
    if scenario != NUMERIC_ENTITY_SCENARIO and scenario_keyword_hits(transcript, scenario, language) < MIN_SCENARIO_HITS:
        return False, entities, scenario, quality
    if _has_suspicious_labels(entities):
        return False, entities, scenario, quality
    if not entities["amounts"] and not entities["invoice_ids"] and not (
        entities["refusals"] or entities["confirmations"]
    ):
        return False, entities, scenario, quality
    return True, entities, scenario, quality


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _has_billing_term(text: str, language: LanguageCode) -> bool:
    terms = BILLING_TERMS.get(language, BILLING_TERMS["en-US"])
    return any(term.lower() in text for term in terms)


def _contains_false_positive_phrase(text: str) -> bool:
    return any(phrase in text for phrase in FALSE_POSITIVE_PHRASES)


def _has_suspicious_labels(entities: dict[str, list[str]]) -> bool:
    for amount in entities.get("amounts", []):
        token = amount.strip().replace("$", "").strip()
        if YEAR_AMOUNT_RE.match(token) or SMALL_INTEGER_RE.match(token):
            return True
    for invoice_id in entities.get("invoice_ids", []):
        if not re.search(r"\d", invoice_id):
            return True
    return False


def _is_plausible_amount(value: str) -> bool:
    token = value.strip().replace("$", "").strip()
    if YEAR_AMOUNT_RE.match(token):
        return False
    if SMALL_INTEGER_RE.match(token):
        return False
    if re.fullmatch(r"\d{3,4}", token):
        if YEAR_AMOUNT_RE.match(token):
            return False
        try:
            if int(token) < 1000:
                return False
        except ValueError:
            return False
        return True
    if re.fullmatch(r"\d{1,3}(?:,\d{3})+", token):
        return True
    if "$" in value or re.search(r"(?:dollars?|usd|baht|rupiah|yuan|元|ดอลลาร์)", value, re.IGNORECASE):
        return True
    return bool(re.fullmatch(r"\d[\d,]*(?:\.\d{2})?", token) and "," in token)
