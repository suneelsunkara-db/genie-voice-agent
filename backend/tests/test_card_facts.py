"""Unit tests for the card fast-lane fact shaping.

``CardLakebaseServing._shape_cardholder_facts`` is the pure function that turns
raw statements / spending / rewards rows into the summary the UI renders (hero
expense metrics + rewards-leakage numbers). It is deterministic and DB-free, so
we can lock the contract here — this is also what guarantees the fast-lane
numbers reconcile (the "why would they diverge" concern): the summary is a plain
aggregation of the same rows Genie queries.
"""
from __future__ import annotations

from genie_voice.serve.card_lakebase import CardLakebaseServing

shape = CardLakebaseServing._shape_cardholder_facts


def _statements() -> list[dict]:
    # Newest first (matches the `ORDER BY cycle DESC` the serving query uses).
    latest = {
        "cycle": "2026-01", "purchases": 3200.0, "fees": 100.0, "interest": 100.0,
        "new_balance": 3400.0, "prev_balance": 900.0, "min_payment": 68.0,
        "due_date": "2026-02-15", "paid_in_full": False,
    }
    prior = [
        {"cycle": f"2025-{m:02d}", "purchases": 950.0, "fees": 0.0, "interest": 0.0,
         "new_balance": 950.0, "prev_balance": 900.0, "min_payment": 25.0,
         "due_date": "2025-01-15", "paid_in_full": True}
        for m in range(2, 13)  # 11 prior months
    ]
    return [latest, *prior]


def _cardholder() -> dict:
    return {"status": "active", "points_balance": 42000, "credit_limit": 15000.0}


def test_expense_summary_reconciles():
    facts = shape("CH-0001", _cardholder(), _statements(), spending=[], rewards=[])
    s = facts["summary"]
    assert facts["found"] is True
    assert s["current_cycle"] == "2026-01"
    # this month = purchases + fees + interest
    assert s["this_month_expenses"] == 3400.0
    # avg of the 11 prior months (all 950) = 950
    assert s["avg_monthly_expenses"] == 950.0
    # change is exactly the difference — the number the spike waterfall must match
    assert s["expense_change"] == round(3400.0 - 950.0, 2)


def test_avg_uses_all_prior_months_not_just_recent():
    # Regression: earlier bug averaged only the last 5 (LIMIT 6) prior months.
    stmts = _statements()
    facts = shape("CH-0001", _cardholder(), stmts, [], [])
    assert facts["summary"]["avg_monthly_expenses"] == 950.0
    assert len(facts["recent_statements"]) == 12


def test_rewards_leakage_summary():
    rewards = [
        {"cycle": "2026-01", "category": "travel", "points_earned": 200,
         "points_possible": 4000, "reversed_points": 0, "expired_points": 0,
         "missed_reason": "inactive_bonus"},
        {"cycle": "2026-01", "category": "dining", "points_earned": 300,
         "points_possible": 300, "reversed_points": 50, "expired_points": 25,
         "missed_reason": None},
    ]
    facts = shape("CH-0001", _cardholder(), _statements(), [], rewards)
    r = facts["summary"]["rewards"]
    assert r["cycle"] == "2026-01"
    assert r["points_earned"] == 500
    assert r["points_possible"] == 4300
    assert r["reversed_points"] == 50
    assert r["expired_points"] == 25
    # gap = unmet-possible + reversed + expired = (4300-500) + 50 + 25
    assert r["points_gap"] == (4300 - 500) + 50 + 25


def test_rewards_scoped_to_current_cycle():
    rewards = [
        {"cycle": "2026-01", "category": "travel", "points_earned": 100,
         "points_possible": 1000, "reversed_points": 0, "expired_points": 0,
         "missed_reason": "inactive_bonus"},
        {"cycle": "2025-11", "category": "travel", "points_earned": 999,
         "points_possible": 999, "reversed_points": 0, "expired_points": 0,
         "missed_reason": None},
    ]
    facts = shape("CH-0001", _cardholder(), _statements(), [], rewards)
    # Only the 2026-01 row feeds the current-cycle summary.
    assert facts["summary"]["rewards"]["points_possible"] == 1000
    assert len(facts["rewards_ledger"]) == 1


def test_missing_cardholder_marks_not_found():
    facts = shape("CH-9999", None, [], [], [])
    assert facts["found"] is False
    assert facts["summary"].get("current_cycle") is None
