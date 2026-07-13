"""Shared detection of customer billing intents from utterance text."""
from __future__ import annotations

import re

_WAIVER_RE = re.compile(
    r"\b(waive|remove|reverse|forgive|drop)\b.*\b(late fee|fee)\b|"
    r"\b(late fee|fee)\b.*\b(waive|remove|reverse|forgive|drop)\b|"
    r"(ยกเว้น|ยกเลิก|ยกโทษ).*(ค่าธรรมเนียม|ค่าปรับ)|"
    r"(hapus|menghapus|bebaskan|dibebaskan).*(biaya keterlambatan|denda)|"
    r"(免除|减免|取消).*(滞纳金|逾期费用|费用)",
    re.IGNORECASE,
)
_PAYMENT_PLAN_RE = re.compile(
    r"\b(payment plan|payment arrangement|installment|instalment)\b|"
    r"(แผนการชำระ|ผ่อนชำระ|แบ่งชำระ)|"
    r"(rencana pembayaran|cicilan|angsuran)|"
    r"(付款计划|分期付款|还款计划|付款计划)",
    re.IGNORECASE,
)


def detect_waiver_plan_request(text: str) -> tuple[bool, bool]:
    msg = text or ""
    return bool(_WAIVER_RE.search(msg)), bool(_PAYMENT_PLAN_RE.search(msg))
