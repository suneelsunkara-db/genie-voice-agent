"""Scenario engine.

Generates each call's dialogue FROM the customer's real data state, so the
conversation references a real invoice, the real amount, and consistent dates.
The dialogue is generated so the enrichment pipeline can faithfully derive
`gold_call_insights` from it, and so the call links correctly to the seeded
customers/invoices/agents (via call_facts).

This consistency is what makes Genie answers trustworthy: e.g. "customers who
called about a billing_dispute AND have an overdue invoice" is only correct if
the call that mentions the dispute is actually linked to that overdue invoice.
"""
from __future__ import annotations

import random
from datetime import date
from typing import Any

_MONTHS = ["January", "February", "March", "April", "May", "June",
           "July", "August", "September", "October", "November", "December"]


def _usd(x: float) -> str:
    return f"${x:,.2f}"


def _human_date(iso: str) -> str:
    y, m, d = (int(p) for p in iso.split("-"))
    return f"{_MONTHS[m - 1]} {d}"


def _sentiment_label(score: float) -> str:
    if score <= -0.34:
        return "negative"
    if score >= 0.34:
        return "positive"
    return "neutral"


def _pick_subject_invoice(invoices: list[dict]) -> dict | None:
    if not invoices:
        return None
    priority = {"disputed": 0, "overdue": 1, "open": 2, "paid": 3, "refunded": 4}
    return sorted(invoices, key=lambda i: (priority.get(i["status"], 9), i["due_date"]))[0]


def build_call(
    *,
    call_id: str,
    customer: dict,
    invoices: list[dict],
    payments: list[dict],
    agent: dict,
    rng: random.Random,
) -> dict[str, Any]:
    inv = _pick_subject_invoice(invoices)
    at_risk = customer["status"] == "at_risk"
    agent_first = agent["full_name"].split()[0]
    cust_first = customer["full_name"].split()[0]
    acct = customer["customer_id"].split("-")[1]

    if inv is None:
        return _general(call_id, customer, agent, agent_first, cust_first, rng)

    status = inv["status"]
    if status == "disputed":
        call = _double_charge(call_id, customer, agent, inv, payments, agent_first, cust_first, acct, rng)
    elif status == "overdue" and customer["autopay_enabled"]:
        call = _autopay_failed(call_id, customer, agent, inv, agent_first, cust_first, acct, rng)
    elif status == "overdue":
        call = _late_fee(call_id, customer, agent, inv, agent_first, cust_first, acct, rng)
    elif status == "open":
        call = _billing_question(call_id, customer, agent, inv, agent_first, cust_first, acct, rng)
    else:
        return _general(call_id, customer, agent, agent_first, cust_first, rng)

    # Escalate to retention if the customer is at risk and unhappy.
    if at_risk and call["sentiment_score"] < 0:
        call["all_intents"] = sorted(set(call["all_intents"] + ["cancellation_risk"]))
        call["disposition"] = "escalated"
        call["next_best_action"] = "escalate_retention_offer"
        call["turns"].insert(
            len(call["turns"]) - 1,
            {"speaker": "customer", "text": "Honestly, if this keeps happening I'm going to cancel my account."},
        )
        call["turns"].insert(
            len(call["turns"]) - 1,
            {"speaker": "agent", "text": "I really don't want to lose you. Let me also apply a loyalty credit to your next bill."},
        )
    return call


# --------------------------------------------------------------------------- #
# Scenario builders. Each returns coherent call turns tied to real account facts.
# --------------------------------------------------------------------------- #
def _finish(call_id, customer, agent, inv, *, turns, primary, intents, disposition,
            resolution, sentiment, nba, mentioned_amount, summary, csat):
    return {
        "call_id": call_id,
        "customer_id": customer["customer_id"],
        "agent_id": agent["agent_id"],
        "primary_intent": primary,
        "all_intents": sorted(set(intents)),
        "disposition": disposition,
        "resolution_status": resolution,
        "sentiment_score": round(sentiment, 3),
        "sentiment_label": _sentiment_label(sentiment),
        "next_best_action": nba,
        "csat": csat,
        "mentioned_invoice_id": inv["invoice_id"] if inv else None,
        "mentioned_amount": mentioned_amount,
        "summary": summary,
        "turns": turns,
    }


def _double_charge(call_id, customer, agent, inv, payments, agent_first, cust_first, acct, rng):
    single = round(float(inv["amount"]) - float(inv["late_fee"]), 2)
    turns = [
        {"speaker": "agent", "text": f"Thank you for calling billing support, my name is {agent_first}. How can I help?"},
        {"speaker": "customer", "text": f"Hi, I'm {cust_first}. I was charged twice on invoice {inv['invoice_id']}, two charges of {_usd(single)}."},
        {"speaker": "agent", "text": f"I'm sorry about that. Can you confirm your account number?"},
        {"speaker": "customer", "text": f"Yes, it's account {acct}."},
        {"speaker": "agent", "text": f"Thanks. I can see two successful charges of {_usd(single)} on {_human_date(inv['paid_date'])}. That's clearly a duplicate."},
        {"speaker": "customer", "text": "Right, this is the second time it's happened and it's frustrating."},
        {"speaker": "agent", "text": f"I've issued a refund of {_usd(single)} to your original payment method; it posts in three to five business days."},
        {"speaker": "customer", "text": "Okay, thank you for sorting that out."},
    ]
    return _finish(call_id, customer, agent, inv, turns=turns, primary="refund",
                   intents=["refund", "billing_dispute"], disposition="resolved",
                   resolution="resolved", sentiment=-0.2, nba="process_refund",
                   mentioned_amount=single, csat=rng.choice([4, 4, 5]),
                   summary=f"Duplicate charge on {inv['invoice_id']}; refund of {_usd(single)} issued.")


def _autopay_failed(call_id, customer, agent, inv, agent_first, cust_first, acct, rng):
    base = round(float(inv["amount"]) - float(inv["late_fee"]), 2)
    turns = [
        {"speaker": "agent", "text": f"Hello, you've reached billing, this is {agent_first}. How can I assist?"},
        {"speaker": "customer", "text": f"My autopay failed and now invoice {inv['invoice_id']} is overdue."},
        {"speaker": "agent", "text": f"Let me check account {acct}. I see the autopay attempt for {_usd(base)} was declined, due {_human_date(inv['due_date'])}."},
        {"speaker": "customer", "text": "Yes, I have a new card now. Can I set up a payment arrangement?"},
        {"speaker": "agent", "text": f"Absolutely. I can split the {_usd(base)} into two payments and waive the {_usd(float(inv['late_fee']))} late fee."},
        {"speaker": "customer", "text": "That works, thank you so much."},
    ]
    return _finish(call_id, customer, agent, inv, turns=turns, primary="autopay_issue",
                   intents=["autopay_issue", "payment_arrangement", "late_fee"],
                   disposition="resolved", resolution="resolved", sentiment=0.1,
                   nba="set_up_payment_plan", mentioned_amount=base,
                   csat=rng.choice([4, 5]),
                   summary=f"Autopay declined on {inv['invoice_id']}; payment plan set, late fee waived.")


def _late_fee(call_id, customer, agent, inv, agent_first, cust_first, acct, rng):
    fee = float(inv["late_fee"])
    turns = [
        {"speaker": "agent", "text": f"Thanks for calling, my name is {agent_first}. How can I help you today?"},
        {"speaker": "customer", "text": f"I have a question about invoice {inv['invoice_id']}, it looks too high this month."},
        {"speaker": "agent", "text": f"Can you confirm your account number please?"},
        {"speaker": "customer", "text": f"Sure, account {acct}, the total was {_usd(float(inv['amount']))}."},
        {"speaker": "agent", "text": f"I see a late fee of {_usd(fee)} applied on {_human_date(inv['due_date'])}."},
        {"speaker": "customer", "text": "That's frustrating, I paid on time, I'd like it waived."},
        {"speaker": "agent", "text": f"I understand. I've removed the {_usd(fee)} late fee, your new balance is {_usd(float(inv['amount']) - fee)}."},
        {"speaker": "customer", "text": "Okay, that's better, thank you."},
    ]
    return _finish(call_id, customer, agent, inv, turns=turns, primary="late_fee",
                   intents=["late_fee", "billing_dispute"], disposition="resolved",
                   resolution="resolved", sentiment=-0.1, nba="offer_fee_waiver",
                   mentioned_amount=round(float(inv["amount"]), 2), csat=rng.choice([3, 4, 4]),
                   summary=f"Late fee dispute on {inv['invoice_id']}; {_usd(fee)} fee waived.")


def _billing_question(call_id, customer, agent, inv, agent_first, cust_first, acct, rng):
    amt = float(inv["amount"])
    turns = [
        {"speaker": "agent", "text": f"Good afternoon, this is {agent_first}. How may I help?"},
        {"speaker": "customer", "text": f"I just want to understand invoice {inv['invoice_id']} for {_usd(amt)}, due {_human_date(inv['due_date'])}."},
        {"speaker": "agent", "text": f"Of course. On account {acct} that's your standard {customer['plan']} plan charge for the period {inv['period']}."},
        {"speaker": "customer", "text": "Got it, that makes sense. I'll pay it today."},
        {"speaker": "agent", "text": "Great, thank you. Anything else I can help with?"},
        {"speaker": "customer", "text": "No, that's all, thanks for the help."},
    ]
    return _finish(call_id, customer, agent, inv, turns=turns, primary="billing_inquiry",
                   intents=["billing_inquiry"], disposition="resolved", resolution="resolved",
                   sentiment=0.4, nba="continue", mentioned_amount=round(amt, 2),
                   csat=5, summary=f"Billing question on {inv['invoice_id']} ({_usd(amt)}); explained, customer satisfied.")


def _general(call_id, customer, agent, agent_first, cust_first, rng):
    turns = [
        {"speaker": "agent", "text": f"Hi, thanks for calling, this is {agent_first}. How can I help?"},
        {"speaker": "customer", "text": f"I'm thinking about upgrading from my {customer['plan']} plan, what are the options?"},
        {"speaker": "agent", "text": "Happy to help. The next tier adds more usage and priority support."},
        {"speaker": "customer", "text": "Sounds good, I'll consider it. Thanks."},
    ]
    return _finish(call_id, customer, agent, None, turns=turns, primary="plan_inquiry",
                   intents=["plan_inquiry"], disposition="resolved", resolution="resolved",
                   sentiment=0.5, nba="offer_plan_upgrade", mentioned_amount=None,
                   csat=5, summary=f"Plan upgrade inquiry from {customer['plan']} plan; info provided.")
