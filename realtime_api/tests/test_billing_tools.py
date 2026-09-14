"""Billing preparation/confirmation contracts without external service calls."""
from __future__ import annotations

import json
import sys
from types import SimpleNamespace

from realtime_api import tools
from realtime_api.runtime import CapabilityId, classifier_capabilities
from realtime_api.runtime.pack_schema import evidence_from_tool_result
from realtime_api.tool_registry import ToolContext


OMAR_ACCOUNT = {
    "found": True,
    "customer": {
        "customer_id": "CUST-4028",
        "full_name": "Omar Patel",
        "status": "at_risk",
    },
    "summary": {"overdue_amount": 239.0, "overdue_invoice_count": 1},
    "invoices": [
        {
            "invoice_id": "INV-90114",
            "status": "overdue",
            "amount": "239.00",
            "late_fee": "40.00",
        }
    ],
}


class _Serving:
    def __init__(self, account=None):
        self.account = account or OMAR_ACCOUNT
        self.applied = 0

    def get_account_facts(self, customer_id):
        assert customer_id == "CUST-4028"
        return self.account

    def apply_billing_resolution(self, call_id, customer_id, resolution, account):
        self.applied += 1
        assert call_id == "CALL-2028"
        assert customer_id == "CUST-4028"
        assert resolution["actions"]["waiver_applied"] is True
        assert account is self.account
        return {
            "applied": True,
            "adjustment": {
                "invoice_id": "INV-90114",
                "amount_before": 239.0,
                "amount_after": 199.0,
                "late_fee_before": 40.0,
                "late_fee_after": 0.0,
                "status_after": "overdue",
            },
        }


def _ctx() -> ToolContext:
    return ToolContext(
        customer_id="CUST-4028",
        call_id="CALL-2028",
        _detected_language="en-US",
        account_store={},
        profile_state={},
    )


def _install_serving(monkeypatch, service) -> None:
    monkeypatch.setitem(
        sys.modules,
        "api.app.deps",
        SimpleNamespace(serving=lambda: service),
    )


def test_billing_profile_registers_separate_prepare_and_mutate_tools():
    names = {item["function"]["name"] for item in tools._billing_tools_spec()}
    assert {
        "lookup_account",
        "prepare_billing_action",
        "apply_billing_action",
        "ask_genie",
        "get_current_time",
    } <= names


def test_prepare_reads_known_customer_and_binds_exact_offer(monkeypatch):
    service = _Serving()
    _install_serving(monkeypatch, service)
    ctx = _ctx()

    result = json.loads(
        tools._run_prepare_billing_action({"action": "waive_late_fee"}, ctx)
    )

    assert result["proposal_text"].startswith(
        "Omar Patel, I can waive the $40.00 late fee on INV-90114"
    )
    assert result["confirmation_required"] is True
    assert ctx.profile_state["pending_billing_offer"] == {
        "action": "waive_late_fee",
        "customer_id": "CUST-4028",
        "invoice_id": "INV-90114",
        "late_fee_usd": 40.0,
        "overdue_amount_usd": 239.0,
        "plan_balance_usd": 199.0,
    }

    evidence = evidence_from_tool_result("prepare_billing_action", result)
    assert evidence.has_attributed_prose
    assert evidence.prose is not None
    assert "Would you like me to proceed?" in evidence.prose.text


def test_apply_fails_closed_without_matching_prepared_offer():
    missing = json.loads(
        tools._run_apply_billing_action({"action": "waive_late_fee"}, _ctx())
    )
    assert "No exact billing offer" in missing["error"]

    ctx = _ctx()
    ctx.profile_state["pending_confirm_mutate"] = {
        "action": "payment_plan",
        "customer_id": "CUST-4028",
        "invoice_id": "INV-90114",
    }
    mismatch = json.loads(
        tools._run_apply_billing_action({"action": "waive_late_fee"}, ctx)
    )
    assert "does not match" in mismatch["error"]


def test_apply_rechecks_snapshot_before_mutation(monkeypatch):
    changed = {
        **OMAR_ACCOUNT,
        "invoices": [{**OMAR_ACCOUNT["invoices"][0], "late_fee": "45.00"}],
    }
    service = _Serving(changed)
    _install_serving(monkeypatch, service)
    ctx = _ctx()
    ctx.profile_state["pending_confirm_mutate"] = {
        "action": "waive_late_fee",
        "customer_id": "CUST-4028",
        "invoice_id": "INV-90114",
        "late_fee_usd": 40.0,
    }

    result = json.loads(
        tools._run_apply_billing_action({"action": "waive_late_fee"}, ctx)
    )

    assert "late fee changed" in result["error"]
    assert service.applied == 0


def test_apply_matching_snapshot_mutates_once(monkeypatch):
    service = _Serving()
    _install_serving(monkeypatch, service)
    monkeypatch.setattr(tools, "_close_resolution_after_billing", lambda *args: None)
    ctx = _ctx()
    ctx.profile_state["pending_confirm_mutate"] = {
        "action": "waive_late_fee",
        "customer_id": "CUST-4028",
        "invoice_id": "INV-90114",
        "late_fee_usd": 40.0,
    }

    result = json.loads(
        tools._run_apply_billing_action({"action": "waive_late_fee"}, ctx)
    )

    assert result["applied"] is True
    assert service.applied == 1


def test_prepare_capability_is_billing_only():
    assert CapabilityId.BILLING_ACTION_PREPARE in {
        item.id for item in classifier_capabilities("billing")
    }
    for profile in ("card", "knowledge", "concierge"):
        assert CapabilityId.BILLING_ACTION_PREPARE not in {
            item.id for item in classifier_capabilities(profile)
        }
