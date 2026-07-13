"""Deterministic agent reply action plan from enrichment + account facts.

Mirrors the cockpit's `guidance.recommend()` contract: structured analytics pick
the action; FM prose only phrases that plan — it does not invent a new action.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from genie_voice.assist.billing import primary_overdue_invoice
from genie_voice.assist.billing_intent import detect_waiver_plan_request
from genie_voice.i18n import LanguageCode, content_language, normalize_language


def _money(value: Any) -> str:
    try:
        return f"${float(str(value).replace(',', '')):.2f}"
    except (TypeError, ValueError):
        return "$0.00"


def _nudge_intents(nudge: dict[str, Any]) -> set[str]:
    values = [nudge.get("primary_intent"), nudge.get("next_best_action")]
    values.extend(nudge.get("all_intents") or [])
    return {str(v) for v in values if v}


@dataclass(frozen=True)
class ReplyActionPlan:
    next_best_action: str
    offer_waiver: bool
    offer_payment_plan: bool
    process_refund: bool
    escalate_retention: bool
    title: str
    detail: str
    invoice_id: str | None = None
    late_fee_usd: float | None = None
    overdue_amount_usd: float | None = None
    plan_balance_usd: float | None = None


def build_reply_action_plan(
    nudge: dict[str, Any],
    account: dict[str, Any] | None,
    resolution: dict[str, Any] | None = None,
    *,
    customer_message: str = "",
) -> ReplyActionPlan:
    """Map enrichment NBA + resolution flags + account facts to a concrete action."""
    actions = dict((resolution or {}).get("actions") or {})
    intents = _nudge_intents(nudge)
    nba = str(nudge.get("next_best_action") or "continue")

    waiver_text, plan_text = detect_waiver_plan_request(customer_message)
    waiver_flag = bool(nudge.get("waiver_requested") or waiver_text)
    plan_flag = bool(nudge.get("payment_plan_requested") or plan_text)

    overdue = primary_overdue_invoice(account)
    summary = (account or {}).get("summary") or {}
    customer = (account or {}).get("customer") or {}

    invoice_id = str(overdue.get("invoice_id")) if overdue and overdue.get("invoice_id") else None
    late_fee = None
    plan_balance = None
    if overdue:
        try:
            late_fee = float(str(overdue.get("late_fee") or 0).replace(",", ""))
        except (TypeError, ValueError):
            late_fee = 0.0
        try:
            amount = float(str(overdue.get("amount") or 0).replace(",", ""))
        except (TypeError, ValueError):
            amount = 0.0
        plan_balance = max(amount - (late_fee or 0.0), 0.0) if late_fee else amount

    try:
        overdue_amount = float(str(summary.get("overdue_amount") or 0).replace(",", ""))
    except (TypeError, ValueError):
        overdue_amount = 0.0

    offer_waiver = waiver_flag or nba == "offer_fee_waiver" or "offer_fee_waiver" in intents
    offer_plan = (
        plan_flag
        or nba == "set_up_payment_plan"
        or "set_up_payment_plan" in intents
        or "payment_arrangement" in intents
    )
    process_refund = nba == "process_refund" or "refund" in intents
    escalate = nba == "escalate_retention_offer" or "cancellation_risk" in intents

    if waiver_flag and plan_flag:
        offer_waiver = True
        offer_plan = True

    if escalate:
        title = "Escalate with a retention offer"
        detail = (
            f"{customer.get('full_name') or 'Customer'} is a {customer.get('tenure_months') or '?'} month "
            f"{customer.get('plan') or ''} customer flagged {customer.get('status') or 'at risk'}. "
            "Loop in retention and lead with a loyalty credit or plan discount before they ask to cancel."
        )
    elif offer_waiver and offer_plan and overdue:
        title = f"Offer late fee waiver and payment plan on {invoice_id}"
        detail = (
            f"Waive the {_money(late_fee)} late fee on {invoice_id} and set up a payment plan for "
            f"the {_money(plan_balance)} balance. Ask the customer to confirm before applying."
        )
    elif offer_waiver:
        title = (
            f"Offer to waive the {_money(late_fee)} late fee on {invoice_id}"
            if overdue and invoice_id
            else "Offer to waive the late fee"
        )
        detail = (
            f"{invoice_id} is overdue — {_money(overdue.get('amount'))} including {_money(late_fee)} late fee. "
            "Waiving the fee resolves the dispute. Ask the customer to confirm before applying."
            if overdue and invoice_id
            else "Acknowledge the late fee and offer a one-time goodwill waiver. Ask to confirm before applying."
        )
    elif offer_plan:
        title = "Set up a payment plan"
        detail = (
            f"Offer to split the {_money(overdue_amount)} overdue balance into instalments. "
            "Ask the customer to confirm before applying."
            if overdue_amount > 0
            else "Offer to split the outstanding balance into instalments. Ask to confirm before applying."
        )
    elif process_refund:
        title = "Process a refund"
        detail = (
            f"Validate the disputed charge on {invoice_id} ({_money(overdue.get('amount') if overdue else overdue_amount)}) "
            "and issue a refund or credit."
            if invoice_id
            else "Validate the disputed charge and issue a refund or account credit."
        )
    else:
        title = "Continue assisting — answer the customer's question"
        detail = (
            "Address what the customer asked using validated account facts. "
            "Do not offer unrelated lookups (declined-payment date ranges, extra reports)."
        )

    return ReplyActionPlan(
        next_best_action=nba,
        offer_waiver=offer_waiver,
        offer_payment_plan=offer_plan,
        process_refund=process_refund,
        escalate_retention=escalate,
        title=title,
        detail=detail,
        invoice_id=invoice_id,
        late_fee_usd=late_fee,
        overdue_amount_usd=overdue_amount if overdue_amount > 0 else None,
        plan_balance_usd=plan_balance,
    )


def compose_action_instructions(plan: ReplyActionPlan) -> str:
    """English anchor instructions for FM prose — action is fixed, FM only phrases it."""
    lines = [
        f"NEXT_BEST_ACTION: {plan.next_best_action}",
        f"AGENT_ACTION_TITLE: {plan.title}",
        f"AGENT_ACTION_DETAIL: {plan.detail}",
        "Phrase the reply in the interaction language. Preserve invoice IDs and USD amounts exactly.",
    ]
    if plan.offer_waiver and plan.offer_payment_plan:
        lines.append(
            "Required reply shape: empathy → state overdue invoice facts → offer to waive the late fee "
            f"on {plan.invoice_id or 'the overdue invoice'} AND set up a payment plan for "
            f"{_money(plan.plan_balance_usd)} → ask whether to proceed now."
        )
    elif plan.offer_waiver:
        lines.append(
            "Required reply shape: empathy → state facts → offer to waive the late fee → ask whether to proceed now."
        )
    elif plan.offer_payment_plan:
        lines.append(
            "Required reply shape: empathy → state facts → offer a payment plan for the overdue balance → "
            "ask whether to proceed now."
        )
    elif plan.process_refund:
        lines.append("Required reply shape: empathy → validate the disputed charge → explain refund next steps.")
    elif plan.escalate_retention:
        lines.append("Required reply shape: empathy → acknowledge risk → outline retention escalation steps.")
    else:
        lines.append(
            "Required reply shape: empathy → answer the customer's question with validated facts → "
            "one clear next step. Do NOT offer unrelated account lookups."
        )
    return "\n".join(lines)


def render_deterministic_reply(
    plan: ReplyActionPlan,
    *,
    language: str | None,
    opener: str,
    customer_name: str | None = None,
) -> str | None:
    """Render billing-offer replies from the action plan without FM action selection."""
    if not plan.invoice_id:
        return None

    code = content_language(language)
    inv = plan.invoice_id
    late = _money(plan.late_fee_usd)
    balance = _money(plan.plan_balance_usd)
    amount = _money(
        (plan.plan_balance_usd or 0) + (plan.late_fee_usd or 0)
        if plan.late_fee_usd
        else plan.overdue_amount_usd
    )
    name = (customer_name or "").strip()
    name_clause = f"{name}，" if name and code == "zh-CN" else (f"{name}, " if name else "")

    if plan.offer_waiver and plan.offer_payment_plan:
        templates: dict[LanguageCode, str] = {
            "en-US": (
                f"{opener} {name_clause}I see overdue invoice {inv} for {amount}, including a {late} late fee. "
                f"I can waive the {late} late fee on {inv} and set up a payment plan for the remaining {balance}. "
                "Would you like me to proceed with both now?"
            ),
            "zh-CN": (
                f"{opener}{name_clause}您目前有一张逾期发票 {inv}，金额为 {amount}，其中包含 {late} 的滞纳金。"
                f"我可以为您免除这笔滞纳金，并为剩余 {balance} 设置付款计划。"
                "请问您是否同意我现在为您办理？"
            ),
            "th-TH": (
                f"{opener} {name_clause}มีใบแจ้งหนี้ค้างชำระ {inv} จำนวน {amount} รวมค่าปรับ {late} "
                f"ผมสามารถยกเว้นค่าปรับ {late} และจัดแผนผ่อนชำระยอด {balance} ให้ได้ "
                "คุณต้องการให้ดำเนินการทั้งสองอย่างตอนนี้เลยไหม?"
            ),
            "id-ID": (
                f"{opener} {name_clause}Ada invoice jatuh tempo {inv} sebesar {amount} termasuk denda {late}. "
                f"Saya bisa membebaskan denda {late} dan mengatur cicilan untuk sisa {balance}. "
                "Apakah Anda setuju saya lanjutkan keduanya sekarang?"
            ),
        }
        return templates[code].strip()

    if plan.offer_waiver:
        templates = {
            "en-US": (
                f"{opener} {name_clause}I can waive the {late} late fee on {inv}. "
                "Would you like me to proceed?"
            ),
            "zh-CN": (
                f"{opener}{name_clause}我可以为您免除发票 {inv} 上 {late} 的滞纳金。"
                "请问您是否同意我现在为您办理？"
            ),
            "th-TH": (
                f"{opener} {name_clause}ผมสามารถยกเว้นค่าปรับ {late} สำหรับ {inv} ได้ "
                "คุณต้องการให้ดำเนินการตอนนี้เลยไหม?"
            ),
            "id-ID": (
                f"{opener} {name_clause}Saya bisa membebaskan denda {late} untuk {inv}. "
                "Apakah Anda setuju saya lanjutkan sekarang?"
            ),
        }
        return templates[code].strip()

    if plan.offer_payment_plan:
        templates = {
            "en-US": (
                f"{opener} {name_clause}I can set up a payment plan for the {amount} overdue balance on {inv}. "
                "Would you like me to proceed?"
            ),
            "zh-CN": (
                f"{opener}{name_clause}我可以为发票 {inv} 的逾期金额 {amount} 设置付款计划。"
                "请问您是否同意我现在为您办理？"
            ),
            "th-TH": (
                f"{opener} {name_clause}ผมสามารถจัดแผนผ่อนชำระยอดค้าง {amount} สำหรับ {inv} ได้ "
                "คุณต้องการให้ดำเนินการตอนนี้เลยไหม?"
            ),
            "id-ID": (
                f"{opener} {name_clause}Saya bisa mengatur cicilan untuk saldo jatuh tempo {amount} pada {inv}. "
                "Apakah Anda setuju saya lanjutkan sekarang?"
            ),
        }
        return templates[code].strip()

    return None
