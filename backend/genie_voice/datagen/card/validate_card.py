"""Prove the credit-card dataset supports the two voice stories.

Two layers of validation:

  1. Derivability (runs now, no Databricks): recompute each story's headline
     numbers STRAIGHT FROM the generated rows and assert they equal the
     hand-seeded ``GROUND_TRUTH``. If a plain aggregation can reconstruct the
     "why", Genie Agent mode (which writes richer SQL) can too — this is the
     necessary condition, checked in CI/local before anything is loaded.

  2. Live gate (later): ``DRY_RUN_SQL`` holds the decomposition queries a
     competent analyst — or Genie Agent mode — would run. After the data is
     loaded and the Genie Agent is curated, run the anchor questions through the
     Agent mode API and diff the report's drivers/figures against GROUND_TRUTH.

Run:  python -m genie_voice.datagen.card.validate_card
"""
from __future__ import annotations

from dataclasses import dataclass

from .build_card import CardDataset, build_card_dataset
from .generators_card import CURRENT_CYCLE, GROUND_TRUTH, PRIOR_CYCLES


@dataclass
class Check:
    name: str
    expected: float
    actual: float

    @property
    def ok(self) -> bool:
        return abs(round(self.expected, 2) - round(self.actual, 2)) < 0.01


def _rows(dataset: CardDataset, table: str, customer_id: str, cycle: str | None = None) -> list[dict]:
    out = [r for r in dataset.table(table) if r.get("customer_id") == customer_id]
    if cycle is not None:
        out = [r for r in out if r.get("cycle") == cycle]
    return out


def check_statement_insights(dataset: CardDataset) -> list[Check]:
    gt = GROUND_TRUTH["statement_insights"]
    cid = gt["customer_id"]
    txns = _rows(dataset, "transactions", cid, CURRENT_CYCLE)

    def s(pred) -> float:
        return round(sum(t["amount"] for t in txns if pred(t)), 2)

    is_charge = lambda t: t["txn_type"] in ("purchase", "fee", "interest", "cash_advance") and t["amount"] > 0
    expenses = s(is_charge)
    flight = s(lambda t: t["category"] == "travel" and not t["is_foreign"] and t["txn_type"] == "purchase")
    foreign_trip = s(lambda t: t["is_foreign"] and t["txn_type"] == "purchase")
    annual_fee = s(lambda t: t["fee_type"] == "annual_fee")
    foreign_tx_fee = s(lambda t: t["fee_type"] == "foreign_tx")
    interest = s(lambda t: t["txn_type"] == "interest")
    everyday = round(expenses - flight - foreign_trip - annual_fee - foreign_tx_fee - interest, 2)

    prior = [st for st in _rows(dataset, "statements", cid) if st["cycle"] in PRIOR_CYCLES]
    prior_expenses = [round(st["purchases"] + st["fees"] + st["interest"], 2) for st in prior]
    typical = round(sum(prior_expenses) / len(prior_expenses), 2) if prior_expenses else 0.0
    increase = round(expenses - typical, 2)

    d = gt["expense_drivers"]
    return [
        Check("current_cycle_expenses", gt["current_cycle_expenses"], expenses),
        Check("typical_monthly_expenses", gt["typical_monthly_expenses"], typical),
        Check("expense_increase", gt["expense_increase"], increase),
        Check("everyday_baseline", gt["everyday_baseline"], everyday),
        Check("driver.flight", d["flight"], flight),
        Check("driver.foreign_trip_spend", d["foreign_trip_spend"], foreign_trip),
        Check("driver.annual_fee", d["annual_fee"], annual_fee),
        Check("driver.foreign_tx_fee", d["foreign_tx_fee"], foreign_tx_fee),
        Check("driver.interest", d["interest"], interest),
        # The drivers must fully explain the expense increase (no unexplained residual).
        Check("drivers_sum==increase", increase, round(sum(d.values()), 2)),
    ]


def check_rewards_optimizer(dataset: CardDataset) -> list[Check]:
    gt = GROUND_TRUTH["rewards_optimizer"]
    cid = gt["customer_id"]
    ledger = _rows(dataset, "rewards_ledger", cid, CURRENT_CYCLE)

    earned = sum(r["points_earned"] for r in ledger)
    possible = sum(r["points_possible"] for r in ledger)
    reversed_pts = sum(r["reversed_points"] for r in ledger)
    gap = (possible - earned) + reversed_pts

    def missed(category: str) -> int:
        return sum(r["points_possible"] - r["points_earned"] for r in ledger if r["category"] == category)

    lk = gt["leakage"]
    return [
        Check("points_earned", gt["points_earned"], earned),
        Check("points_possible", gt["points_possible"], possible),
        Check("reversed_points", gt["reversed_points"], reversed_pts),
        Check("points_gap", gt["points_gap"], gap),
        Check("leakage.dining_wrong_card", lk["dining_wrong_card"], missed("dining")),
        Check("leakage.inactive_grocery_bonus", lk["inactive_grocery_bonus"], missed("groceries")),
        Check("leakage.reversed_points", lk["reversed_points"], reversed_pts),
    ]


def check_rewards_optimizer_suneel(dataset: CardDataset) -> list[Check]:
    """Suneel (CH-0001) also carries a rewards story: the un-activated Q4 travel
    bonus on his $1,900 trip left 3,800 points on the table. Same recompute-from-
    rows tie-out as Maya, so the on-screen persona supports BOTH voice use cases."""
    gt = GROUND_TRUTH["rewards_optimizer_suneel"]
    cid = gt["customer_id"]
    ledger = _rows(dataset, "rewards_ledger", cid, CURRENT_CYCLE)

    earned = sum(r["points_earned"] for r in ledger)
    possible = sum(r["points_possible"] for r in ledger)
    reversed_pts = sum(r["reversed_points"] for r in ledger)
    gap = (possible - earned) + reversed_pts

    def missed(category: str) -> int:
        return sum(r["points_possible"] - r["points_earned"] for r in ledger if r["category"] == category)

    lk = gt["leakage"]
    return [
        Check("suneel.points_earned", gt["points_earned"], earned),
        Check("suneel.points_possible", gt["points_possible"], possible),
        Check("suneel.reversed_points", gt["reversed_points"], reversed_pts),
        Check("suneel.points_gap", gt["points_gap"], gap),
        Check("suneel.leakage.inactive_travel_bonus", lk["inactive_travel_bonus"], missed("travel")),
    ]


def validate(dataset: CardDataset | None = None) -> tuple[bool, list[Check]]:
    dataset = dataset or build_card_dataset()
    checks = (
        check_statement_insights(dataset)
        + check_rewards_optimizer(dataset)
        + check_rewards_optimizer_suneel(dataset)
    )
    return all(c.ok for c in checks), checks


# The queries the live Genie Agent-mode gate diffs against GROUND_TRUTH. Written
# against bare table names; qualify with the card-issuer catalog.schema at run time.
DRY_RUN_SQL: dict[str, str] = {
    "statement_insights_drivers": (
        "SELECT\n"
        "  CASE\n"
        "    WHEN t.fee_type = 'annual_fee'                              THEN 'annual_fee'\n"
        "    WHEN t.fee_type = 'foreign_tx'                              THEN 'foreign_tx_fee'\n"
        "    WHEN t.txn_type = 'interest'                                THEN 'interest'\n"
        "    WHEN t.category = 'travel' AND NOT t.is_foreign             THEN 'flight'\n"
        "    WHEN t.is_foreign AND t.txn_type = 'purchase'              THEN 'foreign_trip_spend'\n"
        "    ELSE 'everyday'\n"
        "  END AS driver,\n"
        "  round(sum(t.amount), 2) AS amount\n"
        "FROM transactions t\n"
        "WHERE t.customer_id = 'CH-0001' AND t.cycle = '2025-12'\n"
        "  AND t.txn_type IN ('purchase','fee','interest','cash_advance') AND t.amount > 0\n"
        "GROUP BY 1 ORDER BY amount DESC"
    ),
    "rewards_optimizer_leakage": (
        "SELECT category, missed_reason,\n"
        "       sum(points_possible - points_earned) AS points_missed,\n"
        "       sum(reversed_points) AS points_reversed\n"
        "FROM rewards_ledger\n"
        "WHERE customer_id = 'CH-0002' AND cycle = '2025-12'\n"
        "GROUP BY category, missed_reason\n"
        "HAVING sum(points_possible - points_earned) > 0 OR sum(reversed_points) > 0\n"
        "ORDER BY points_missed DESC"
    ),
}


def main() -> int:
    dataset = build_card_dataset()
    ok, checks = validate(dataset)
    counts = {name: len(dataset.table(name)) for name in dataset.as_dict()}
    print("Credit-card dataset row counts:")
    for name, count in counts.items():
        print(f"  {name:<24} {count:>5}")
    print("\nStory validation (recomputed from generated rows):")
    for c in checks:
        mark = "ok  " if c.ok else "FAIL"
        print(f"  [{mark}] {c.name:<32} expected={c.expected:<12} actual={c.actual}")
    print("\nPASS" if ok else "\nFAIL: dataset does not match GROUND_TRUTH")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
