"""Cross-store alignment checks for resolution + billing paths."""

from __future__ import annotations

from typing import Any


def alignment_report(
    *,
    call_id: str,
    customer_id: str | None,
    resolution: dict[str, Any] | None,
    lakebase_adjustments: list[dict[str, Any]],
    account_summary: dict[str, Any] | None,
    resolution_events: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize whether Lakebase resolution state matches account facts."""
    resolution = resolution or {}
    summary = account_summary or {}
    status = str(resolution.get("status") or summary.get("issue_status") or "open")
    active_adj = [
        a for a in lakebase_adjustments
        if not a.get("reverted_at") and str(a.get("call_id") or "") == call_id
    ]
    issues: list[str] = []

    if status == "closed" and not active_adj:
        issues.append("issue closed but no active billing_adjustments row in Lakebase")
    if status != "closed" and active_adj:
        issues.append("billing_adjustments active while issue is not closed")
    if status == "closed" and int(summary.get("overdue_invoice_count") or 0) > 0:
        issues.append("issue closed but summary still shows overdue invoices")
    if resolution_events:
        latest = resolution_events[0]
        if str(latest.get("issue_status")) != status:
            issues.append(
                f"resolution_events latest status {latest.get('issue_status')} != call_state {status}"
            )

    return {
        "call_id": call_id,
        "customer_id": customer_id,
        "issue_status": status,
        "active_adjustments": len(active_adj),
        "resolution_events": len(resolution_events),
        "overdue_invoice_count": summary.get("overdue_invoice_count"),
        "aligned": len(issues) == 0,
        "issues": issues,
    }
