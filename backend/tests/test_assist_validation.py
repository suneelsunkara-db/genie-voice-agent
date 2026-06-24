from genie_voice.assist.validation import (
    AccountMetrics,
    cross_validate_metrics,
    metrics_from_account,
    parse_genie_metrics,
    validate_close_eligible,
    validate_reply_against_metrics,
)


def test_metrics_from_account():
    account = {
        "found": True,
        "customer": {"status": "at_risk"},
        "summary": {
            "overdue_invoice_count": 1,
            "overdue_amount": 239.0,
            "recent_declined_payments": 1,
            "status": "at_risk",
        },
    }
    metrics = metrics_from_account(account)
    assert metrics is not None
    assert metrics.overdue_invoice_count == 1
    assert metrics.overdue_amount == 239.0


def test_cross_validate_flags_genie_mismatch():
    lakebase = AccountMetrics(1, 239.0, 1, "at_risk")
    genie = AccountMetrics(4, 956.0, 4, "at_risk")
    result = cross_validate_metrics(lakebase, genie)
    assert result.authoritative.overdue_invoice_count == 1
    assert result.genie_validated is False
    assert len(result.mismatches) == 3


def test_parse_genie_metrics_from_answer():
    genie = {
        "answer": (
            "For CUST-4028: overdue_invoice_count = 1, overdue_amount_usd = $239.00, "
            "recent_declined_payments = 1, customer_status = at_risk"
        ),
        "rows": [],
    }
    metrics = parse_genie_metrics(genie)
    assert metrics is not None
    assert metrics.overdue_invoice_count == 1
    assert metrics.overdue_amount == 239.0


def test_validate_reply_rejects_wrong_overdue_count():
    metrics = AccountMetrics(1, 239.0, 1)
    ok, issues = validate_reply_against_metrics(
        "Based on Genie insights, I see 4 overdue invoices totaling $956.00.",
        metrics,
    )
    assert ok is False
    assert issues


def test_validate_reply_accepts_matching_numbers():
    metrics = AccountMetrics(1, 239.0, 1)
    ok, issues = validate_reply_against_metrics(
        "Based on Genie insights, I see 1 overdue invoice totaling $239.00.",
        metrics,
    )
    assert ok is True
    assert not issues


def test_validate_close_requires_requested_actions():
    ok, reason = validate_close_eligible({}, None)
    assert ok is False
    assert reason


def test_validate_close_requires_overdue_for_payment_plan():
    account = {
        "found": True,
        "summary": {"overdue_invoice_count": 0, "overdue_amount": 0},
        "invoices": [],
    }
    ok, reason = validate_close_eligible({"payment_plan_requested": True}, account)
    assert ok is False
    assert "overdue" in (reason or "").lower()


def test_validate_close_requires_account_facts():
    ok, reason = validate_close_eligible({"waiver_requested": True}, None)
    assert ok is False
    assert "account" in (reason or "").lower()


def test_validate_close_allows_waiver_when_late_fee_exists():
    account = {
        "found": True,
        "summary": {"overdue_invoice_count": 1, "overdue_amount": 239.0},
        "invoices": [{"status": "overdue", "late_fee": "40.00"}],
    }
    ok, _ = validate_close_eligible({"waiver_requested": True}, account)
    assert ok is True


def test_bad_agent_reply_markers_reject_genie_scope_text():
    from api.app.routers.agent_assist import _looks_like_bad_agent_reply

    bad = (
        "This request is unrelated to the database schema and data analysis. "
        "I am only able to answer questions about the data in the provided tables."
    )
    assert _looks_like_bad_agent_reply(bad) is True
    assert _looks_like_bad_agent_reply("Thanks for confirming — your waiver is applied.") is False
