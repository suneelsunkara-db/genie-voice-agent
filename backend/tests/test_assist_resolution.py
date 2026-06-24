from genie_voice.assist.resolution import (
    evaluate_resolution,
    finalize_resolution_after_billing,
    resolution_event_for_transition,
)


def test_evaluate_resolution_moves_to_in_progress_from_fm_nudge():
    inner = {"resolution": {"status": "open", "actions": {}}}
    nudge = {
        "available": True,
        "customer_signal": "request_help",
        "payment_plan_requested": True,
        "waiver_requested": True,
        "primary_intent": "payment_arrangement",
        "all_intents": ["payment_arrangement", "late_fee"],
        "next_best_action": "set_up_payment_plan",
    }
    account = {
        "found": True,
        "summary": {"overdue_invoice_count": 1, "overdue_amount": 239.0},
        "invoices": [{"status": "overdue", "late_fee": "40.00"}],
    }
    out = evaluate_resolution(
        inner,
        "I need a payment plan and want the late fee waived.",
        1,
        account,
        nudge,
    )
    assert out["status"] == "in_progress"
    assert out["actions"]["payment_plan_requested"] is True
    assert out["actions"]["waiver_requested"] is True


def test_evaluate_resolution_sets_pending_close_on_confirm():
    inner = {
        "resolution": {
            "status": "in_progress",
            "actions": {"payment_plan_requested": True, "waiver_requested": True},
        }
    }
    nudge = {"available": True, "customer_signal": "confirm_proceed"}
    account = {
        "found": True,
        "summary": {"overdue_invoice_count": 1, "overdue_amount": 239.0},
        "invoices": [{"status": "overdue", "late_fee": "40.00"}],
    }
    out = evaluate_resolution(inner, "Yes, please proceed.", 1, account, nudge)
    assert out["status"] == "in_progress"
    assert out["actions"]["pending_close"] is True


def test_evaluate_resolution_does_not_transition_when_fm_unavailable():
    inner = {"resolution": {"status": "open", "actions": {}}}
    nudge = {"available": False}
    out = evaluate_resolution(inner, "please proceed", 1, None, nudge)
    assert out["status"] == "open"
    assert out["actions"]["close_block_reason"]


def test_finalize_resolution_commits_close_only_after_billing():
    resolution = {
        "status": "in_progress",
        "actions": {
            "pending_close": True,
            "payment_plan_requested": True,
            "waiver_requested": True,
        },
    }
    closed = finalize_resolution_after_billing(
        resolution,
        {"applied": True, "adjustment": {"invoice_id": "INV-1"}},
    )
    assert closed["status"] == "closed"
    assert closed["actions"]["payment_plan_applied"] is True
    assert closed["actions"]["waiver_applied"] is True
    assert "pending_close" not in closed["actions"]


def test_finalize_resolution_blocks_close_when_billing_fails():
    resolution = {
        "status": "in_progress",
        "actions": {"pending_close": True, "waiver_requested": True},
    }
    blocked = finalize_resolution_after_billing(
        resolution,
        {"applied": False, "reason": "uc_write_failed"},
    )
    assert blocked["status"] == "in_progress"
    assert blocked["actions"]["close_blocked"] is True
    assert blocked["actions"]["close_block_reason"] == "uc_write_failed"


def test_resolution_event_only_on_status_change():
    assert resolution_event_for_transition(
        {"status": "open"},
        {"status": "in_progress", "note": "Issue in_progress: guided by Genie."},
    ) == {
        "event_type": "status_changed",
        "issue_status": "in_progress",
        "note": "Issue in_progress: guided by Genie.",
        "actions": {},
    }
    assert resolution_event_for_transition(
        {"status": "in_progress", "note": "old"},
        {"status": "in_progress", "note": "new"},
    ) is None


def test_append_resolution_event_skips_duplicate():
    from genie_voice.serve.lakebase import LakebaseServing, _LOCK, _MEM_EVENTS

    svc = LakebaseServing()
    svc.enabled = False
    call_id = "CALL-DEDUPE-TEST"
    with _LOCK:
        _MEM_EVENTS[call_id] = []
    try:
        assert svc.append_resolution_event(call_id, "status_changed", "in_progress", "note", {}) is True
        assert len(svc.list_resolution_events(call_id)) == 1
        assert svc.append_resolution_event(call_id, "status_changed", "in_progress", "note", {}) is False
        assert len(svc.list_resolution_events(call_id)) == 1
        assert svc.clear_resolution_events(call_id) == 1
        assert svc.list_resolution_events(call_id) == []
    finally:
        with _LOCK:
            _MEM_EVENTS.pop(call_id, None)
