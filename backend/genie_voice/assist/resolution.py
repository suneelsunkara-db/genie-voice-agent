"""FM-driven issue resolution transitions with guarded multilingual normalization."""
from __future__ import annotations

from datetime import UTC, datetime
import re
from typing import Any

from genie_voice.assist.billing_intent import detect_waiver_plan_request
from genie_voice.assist.validation import validate_close_eligible
from genie_voice.config import Settings, get_settings

_CLOSED_NOTE = (
    "Issue closed: payment arrangement confirmed and waiver flow applied. "
    "Update will reflect on next statement."
)

_CONFIRM_SHORT = frozenset(
    {
        "yes",
        "yep",
        "yeah",
        "ok",
        "okay",
        "proceed",
        "continue",
        "confirm",
        "好",
        "好的",
        "继续",
        "确认",
        "可以",
        "行的",
        "同意",
        "嗯",
        "ตกลง",
        "ยืนยัน",
        "ได้",
        "ใช่",
        "ya",
        "setuju",
        "baik",
        "oke",
    }
)
_CONFIRM_RE = re.compile(
    r"\b(yes|yep|yeah|ok|okay|proceed|continue|go ahead|confirm|approved?|sounds good)\b|"
    r"(ดำเนินการต่อ|ต่อเลย|ตกลง|ยืนยัน|ได้เลย|ครับ|ค่ะ|ได้|ใช่)|"
    r"\b(lanjutkan|lanjut|setuju|ya|silakan|baik|oke)\b|"
    r"(继续|確認|确认|可以|好的|行的|同意|嗯)",
    re.IGNORECASE,
)
_CONFIRM_LONG_RE = re.compile(
    r"\b(yes|yep|yeah|ok|okay|proceed|continue|go ahead|confirm|approved?|sounds good)\b|"
    r"\b(lanjutkan|lanjut)\b",
    re.IGNORECASE,
)


def _nudge_intents(nudge: dict[str, Any]) -> set[str]:
    values = [nudge.get("primary_intent"), nudge.get("next_best_action")]
    values.extend(nudge.get("all_intents") or [])
    return {str(v) for v in values if v}


def _requested_actions_from_signal(
    text: str,
    nudge: dict[str, Any],
    existing_actions: dict[str, Any],
) -> dict[str, bool]:
    intents = _nudge_intents(nudge)
    waiver_text, plan_text = detect_waiver_plan_request(text or "")
    payment_plan = bool(
        existing_actions.get("payment_plan_requested")
        or nudge.get("payment_plan_requested")
        or "payment_arrangement" in intents
        or "set_up_payment_plan" in intents
        or plan_text
    )
    waiver = bool(
        existing_actions.get("waiver_requested")
        or nudge.get("waiver_requested")
        or "offer_fee_waiver" in intents
        or waiver_text
    )
    return {
        "payment_plan_requested": payment_plan,
        "waiver_requested": waiver,
    }


def _is_proceed_confirmation(text: str) -> bool:
    msg = (text or "").strip()
    if not msg:
        return False
    if msg.lower() in _CONFIRM_SHORT or msg in _CONFIRM_SHORT:
        return True
    if len(msg) <= 20:
        return bool(_CONFIRM_RE.fullmatch(msg) or _CONFIRM_RE.search(msg))
    return bool(_CONFIRM_LONG_RE.search(msg))


def _customer_signal(text: str, status: str, nudge: dict[str, Any], actions: dict[str, Any]) -> str:
    signal = str(nudge.get("customer_signal") or "neutral")
    has_offer = bool(actions.get("payment_plan_requested") or actions.get("waiver_requested"))
    if signal == "confirm_proceed":
        if status == "in_progress" and has_offer:
            return signal
        return "neutral"
    if status == "in_progress" and has_offer and _is_proceed_confirmation(text):
        return "confirm_proceed"
    return signal


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
    """Advance issue resolution from FM enrichment on agent or customer turns."""
    settings = settings or get_settings()
    existing = (inner.get("resolution") or {}).copy()
    status = str(existing.get("status") or "open")
    actions = dict(existing.get("actions") or {})
    msg = (text or "").strip()
    if not msg:
        existing["status"] = status
        existing["actions"] = actions
        return existing

    if speaker != 1:
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

    requested = _requested_actions_from_signal(msg, nudge, actions)
    if requested["payment_plan_requested"]:
        actions["payment_plan_requested"] = True
    if requested["waiver_requested"]:
        actions["waiver_requested"] = True
    customer_signal = _customer_signal(msg, status, nudge, actions)

    if status == "open" and customer_signal == "request_help":
        status = "in_progress"

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
