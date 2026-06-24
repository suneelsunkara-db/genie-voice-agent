"""Persist billing resolution (waiver / payment plan) to governed stores."""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any


def primary_overdue_invoice(account: dict[str, Any] | None) -> dict[str, Any] | None:
    if not account:
        return None
    for inv in account.get("invoices") or []:
        if str(inv.get("status")) == "overdue":
            return inv
    return None


def compute_invoice_adjustment(
    invoice: dict[str, Any],
    *,
    waiver_applied: bool,
    payment_plan_applied: bool,
) -> dict[str, Any]:
    amount_before = float(str(invoice.get("amount") or 0).replace(",", ""))
    late_fee_before = float(str(invoice.get("late_fee") or 0).replace(",", ""))
    status_before = str(invoice.get("status") or "overdue")
    late_fee_after = 0.0 if waiver_applied and late_fee_before > 0 else late_fee_before
    amount_after = max(amount_before - late_fee_before, 0.0) if waiver_applied and late_fee_before > 0 else amount_before
    status_after = "open" if (payment_plan_applied or waiver_applied) else status_before
    return {
        "invoice_id": invoice.get("invoice_id"),
        "amount_before": round(amount_before, 2),
        "late_fee_before": round(late_fee_before, 2),
        "status_before": status_before,
        "amount_after": round(amount_after, 2),
        "late_fee_after": round(late_fee_after, 2),
        "status_after": status_after,
        "waiver_applied": waiver_applied,
        "payment_plan_applied": payment_plan_applied,
    }


def prepare_billing_adjustment(
    call_id: str,
    customer_id: str,
    resolution: dict[str, Any],
    account: dict[str, Any],
) -> dict[str, Any]:
    """Build a billing adjustment payload without persisting it."""
    actions = resolution.get("actions") or {}
    invoice = primary_overdue_invoice(account)
    if not invoice:
        return {"ok": False, "reason": "no_overdue_invoice"}
    waiver = bool(actions.get("waiver_applied") or actions.get("waiver_requested"))
    plan = bool(actions.get("payment_plan_applied") or actions.get("payment_plan_requested"))
    adjustment = compute_invoice_adjustment(
        invoice,
        waiver_applied=waiver,
        payment_plan_applied=plan,
    )
    adjustment_id = f"{call_id}-{adjustment['invoice_id']}"
    adjustment.update(
        {
            "adjustment_id": adjustment_id,
            "call_id": call_id,
            "customer_id": customer_id,
            "applied_at": datetime.now(UTC).isoformat(),
            "reverted_at": None,
        }
    )
    return {"ok": True, "adjustment": adjustment}
