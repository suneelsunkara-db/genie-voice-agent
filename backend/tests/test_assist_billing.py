from genie_voice.assist.billing import compute_invoice_adjustment, primary_overdue_invoice
from genie_voice.assist.validation import parse_genie_metrics


def test_parse_genie_metrics_with_columns():
    genie = {
        "columns": [
            "overdue_invoice_count",
            "overdue_amount_usd",
            "recent_declined_payments",
            "customer_status",
        ],
        "rows": [[1, 239.0, 1, "at_risk"]],
        "answer": "",
    }
    metrics = parse_genie_metrics(genie)
    assert metrics is not None
    assert metrics.overdue_invoice_count == 1
    assert metrics.overdue_amount == 239.0
    assert metrics.recent_declined_payments == 1
    assert metrics.status == "at_risk"


def test_compute_invoice_adjustment_waiver_only():
    invoice = {"invoice_id": "INV-1", "status": "overdue", "amount": "239.00", "late_fee": "40.00"}
    adj = compute_invoice_adjustment(
        invoice,
        waiver_applied=True,
        payment_plan_applied=False,
    )
    assert adj["amount_after"] == 199.0
    assert adj["late_fee_after"] == 0.0
    assert adj["status_after"] == "open"


def test_compute_invoice_adjustment_waiver_and_plan():
    invoice = {"invoice_id": "INV-1", "status": "overdue", "amount": "239.00", "late_fee": "40.00"}
    adj = compute_invoice_adjustment(
        invoice,
        waiver_applied=True,
        payment_plan_applied=True,
    )
    assert adj["amount_after"] == 199.0
    assert adj["late_fee_after"] == 0.0
    assert adj["status_after"] == "open"


def test_primary_overdue_invoice():
    account = {
        "invoices": [
            {"invoice_id": "INV-1", "status": "paid"},
            {"invoice_id": "INV-2", "status": "overdue"},
        ]
    }
    assert primary_overdue_invoice(account)["invoice_id"] == "INV-2"
