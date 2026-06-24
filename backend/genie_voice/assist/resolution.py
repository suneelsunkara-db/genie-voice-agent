"""FM-driven issue resolution transitions (no keyword heuristics)."""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from genie_voice.assist.validation import validate_close_eligible
from genie_voice.config import Settings, get_settings

_CLOSED_NOTE = (
    "Issue closed: payment arrangement confirmed and waiver flow applied. "
    "Update will reflect on next statement."
)


def resolution_event_for_transition(
    previous: dict[str, Any] | None,
    current: dict[str, Any],
) -> dict[str, Any] | None:
    """Return one timeline event when issue status changes; skip note-only noise."""
    previous = previous or {}
    prev_status = str(previous.get("status") or "open")
    new_status = str(current.get("status") or "open")
    if prev_status == new_status:
        return None
    return {
        "event_type": "status_changed",
        "issue_status": new_status,
        "note": current.get("note"),
        "actions": dict(current.get("actions") or {}),
    }


def evaluate_resolution(
    inner: dict,
    text: str,
    speaker: int | None,
    account: dict[str, object] | None,
    nudge: dict[str, Any],
    settings: Settings | None = None,
) -> dict:
    """Advance issue resolution using a single FM enrichment call for customer turns."""
    settings = settings or get_settings()
    existing = (inner.get("resolution") or {}).copy()
    status = str(existing.get("status") or "open")
    actions = dict(existing.get("actions") or {})
    msg = (text or "").strip()
    if not msg or speaker != 1:
        existing["status"] = status
        existing["actions"] = actions
        return existing

    if not nudge.get("available"):
        actions["close_blocked"] = True
        actions["close_block_reason"] = "FM enrichment unavailable for resolution transition"
        existing["status"] = status
        existing["actions"] = actions
        existing["resolution_source"] = "unavailable"
        return existing

    customer_signal = str(nudge.get("customer_signal") or "neutral")

    if status == "open" and customer_signal == "request_help":
        status = "in_progress"
        actions["payment_plan_requested"] = bool(nudge.get("payment_plan_requested"))
        actions["waiver_requested"] = bool(nudge.get("waiver_requested"))

    if status == "in_progress" and customer_signal == "confirm_proceed":
        can_close, block_reason = validate_close_eligible(actions, account)
        if can_close:
            actions["pending_close"] = True
        else:
            actions["close_blocked"] = True
            actions["close_block_reason"] = block_reason

    if customer_signal == "escalate":
        actions["escalation_requested"] = True
        if status == "open":
            status = "in_progress"

    existing["status"] = status
    existing["actions"] = actions
    existing["resolution_source"] = "fm"
    if account and account.get("found") and not existing.get("note") and status != "open":
        summary = account.get("summary") or {}
        overdue_amount = summary.get("overdue_amount")
        existing["note"] = (
            f"Issue {status}: guided by Genie and account context"
            + (f" (overdue amount ${overdue_amount})." if overdue_amount is not None else ".")
        )
    return existing


def finalize_resolution_after_billing(
    resolution: dict[str, Any],
    billing_result: dict[str, Any] | None,
) -> dict[str, Any]:
    """Commit close only after governed billing writes succeed."""
    out = dict(resolution)
    actions = dict(out.get("actions") or {})
    if not actions.pop("pending_close", False):
        out["actions"] = actions
        return out

    if billing_result and billing_result.get("applied"):
        out["status"] = "closed"
        actions["payment_plan_applied"] = bool(actions.get("payment_plan_requested"))
        actions["waiver_applied"] = bool(actions.get("waiver_requested"))
        actions.pop("close_blocked", None)
        actions.pop("close_block_reason", None)
        out["resolved_at"] = datetime.now(UTC).isoformat()
        out["note"] = _CLOSED_NOTE
    else:
        out["status"] = "in_progress"
        actions["close_blocked"] = True
        actions["close_block_reason"] = str(
            (billing_result or {}).get("reason") or "billing_write_failed"
        )

    out["actions"] = actions
    return out
