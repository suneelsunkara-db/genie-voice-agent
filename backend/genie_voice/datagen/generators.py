"""Deterministic generation of the reference entities.

All randomness flows from a single seed so datasets are reproducible. Entities
are generated parents-first so foreign keys are always valid:
    agents, customers -> invoices -> payments

The generated state is intentionally *coherent*: invoice statuses, late fees,
and payment outcomes are internally consistent (e.g. an overdue invoice has a
declined autopay payment), so the conversation layer can reference real facts.
"""
from __future__ import annotations

import random
from datetime import date, timedelta
from typing import Any

FIRST = ["Dana", "Marcus", "Priya", "Liam", "Sofia", "Noah", "Aisha", "Diego",
         "Mia", "Omar", "Hana", "Lucas", "Zoe", "Ravi", "Elena", "Tom",
         "Yuki", "Carla", "Ben", "Nadia"]
LAST = ["Park", "Reyes", "Singh", "Olsen", "Costa", "Khan", "Romano", "Mbeki",
        "Nguyen", "Haddad", "Ito", "Silva", "Adams", "Patel", "Novak", "Lopez"]

SEGMENTS = ["consumer", "smb", "enterprise"]
REGIONS = ["NA", "EMEA", "APAC"]
PLANS = {"basic": 49.0, "pro": 99.0, "premium": 199.0}
TEAMS = ["billing", "retention", "technical"]


def _name(rng: random.Random) -> str:
    return f"{rng.choice(FIRST)} {rng.choice(LAST)}"


def gen_agents(rng: random.Random, n: int) -> list[dict[str, Any]]:
    agents = []
    for i in range(n):
        agents.append({
            "agent_id": f"AG-{100 + i}",
            "full_name": _name(rng),
            "team": TEAMS[i % len(TEAMS)],
            "hire_date": (date(2022, 1, 1) + timedelta(days=rng.randint(0, 1200))).isoformat(),
        })
    return agents


def gen_customers(rng: random.Random, n: int) -> list[dict[str, Any]]:
    customers = []
    for i in range(n):
        plan = rng.choices(list(PLANS), weights=[5, 4, 2])[0]
        segment = rng.choices(SEGMENTS, weights=[6, 3, 2])[0]
        tenure = rng.randint(2, 60)
        status = rng.choices(["active", "at_risk", "churned"], weights=[8, 2, 1])[0]
        cid = f"CUST-{4000 + i}"
        name = _name(rng)
        customers.append({
            "customer_id": cid,
            "full_name": name,
            "segment": segment,
            "region": rng.choice(REGIONS),
            "plan": plan,
            "monthly_charge": round(PLANS[plan] * (1.0 if segment != "enterprise" else 1.5), 2),
            "tenure_months": tenure,
            "status": status,
            "autopay_enabled": rng.random() < 0.65,
            "email": f"{name.split()[0].lower()}.{name.split()[1].lower()}@example.com",
            "signup_date": (date.today() - timedelta(days=tenure * 30)).isoformat(),
        })
    return customers


def gen_invoices_and_payments(
    rng: random.Random, customers: list[dict], months: int
) -> tuple[list[dict], list[dict]]:
    """Generate coherent invoices + payment attempts per customer."""
    invoices: list[dict[str, Any]] = []
    payments: list[dict[str, Any]] = []
    inv_seq = 90000
    pay_seq = 50000
    today = date.today()

    for cust in customers:
        cid = cust["customer_id"]
        base = float(cust["monthly_charge"])
        for m in range(months):
            issue = date(today.year, today.month, 1) - timedelta(days=30 * (months - m))
            due = issue + timedelta(days=21)
            inv_id = f"INV-{inv_seq}"
            inv_seq += 1

            # Decide a coherent outcome for this invoice.
            roll = rng.random()
            late_fee = 0.0
            paid_date = None
            status = "paid"
            method = "autopay" if cust["autopay_enabled"] else rng.choice(["card", "bank_transfer"])

            is_recent = m >= months - 2  # only recent invoices stay unpaid/disputed
            if is_recent and roll < 0.16:
                # Autopay/payment declined -> overdue, late fee.
                status = "overdue"
                late_fee = 40.0
                payments.append(_payment(pay_seq, inv_id, cid, base, due, method, "declined")); pay_seq += 1
            elif is_recent and roll < 0.26:
                # Disputed (e.g. double charge) -> two succeeded charges.
                status = "disputed"
                paid_date = (due - timedelta(days=2)).isoformat()
                for _ in range(2):
                    payments.append(_payment(pay_seq, inv_id, cid, base, due - timedelta(days=2), method, "succeeded")); pay_seq += 1
            elif is_recent and roll < 0.34:
                # Still open / not yet paid.
                status = "open"
            else:
                # Paid cleanly.
                status = "paid"
                paid_date = (due - timedelta(days=rng.randint(1, 15))).isoformat()
                payments.append(_payment(pay_seq, inv_id, cid, base, due - timedelta(days=3), method, "succeeded")); pay_seq += 1

            amount = round(base + late_fee, 2)
            invoices.append({
                "invoice_id": inv_id,
                "customer_id": cid,
                "period": issue.strftime("%Y-%m"),
                "issue_date": issue.isoformat(),
                "due_date": due.isoformat(),
                "amount": amount,
                "late_fee": round(late_fee, 2),
                "status": status,
                "paid_date": paid_date,
            })
    return invoices, payments


def _payment(seq, inv_id, cid, amount, when, method, status) -> dict[str, Any]:
    return {
        "payment_id": f"PAY-{seq}",
        "invoice_id": inv_id,
        "customer_id": cid,
        "amount": round(float(amount), 2),
        "payment_date": (when if isinstance(when, str) else when.isoformat()),
        "method": method,
        "status": status,
    }
