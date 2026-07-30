"""Deterministic generation of the credit-card issuer dataset.

All randomness flows from a single seed so datasets are reproducible. Two
archetype cardholders carry HAND-SEEDED, tie-out ground truth for the two voice
use cases; a deterministic random population adds realistic volume.

  - Suneel Sunkara (Statement Insights): this cycle's balance is ~$2,450 higher
    than a typical cycle, and that increase decomposes EXACTLY into flight +
    foreign trip spend + annual fee + foreign-tx fee + interest.
  - Maya Rivera (Rewards Optimizer): earned 12,400 pts but 19,000 were possible —
    a 6,600-pt gap that decomposes EXACTLY into wrong-card dining + an un-activated
    grocery bonus + points reversed on a return.

``GROUND_TRUTH`` records those expected numbers; ``validate_card`` recomputes
them straight from the generated rows (and, later, from the live Genie Agent
mode report) to prove the data supports the story.
"""
from __future__ import annotations

import random
from datetime import date
from typing import Any

# Fixed cycles so the seeded ground truth never drifts with the wall clock.
CURRENT_CYCLE = "2025-12"
# 11 prior monthly cycles (Jan–Nov 2025) + the current spike cycle give a full
# 12-month trend for the UI. The prior-cycle EXPENSE AVERAGE is engineered to be
# exactly $950 (see _PRIOR_MONTH_TOTALS) so the Dec spike breaks a clean baseline.
PRIOR_CYCLES = [f"2025-{m:02d}" for m in range(1, 12)]


def _cycle_dates(cycle: str) -> tuple[date, date]:
    """Statement close date (15th) and payment due date (9th of next month)."""
    y, m = (int(p) for p in cycle.split("-"))
    close = date(y, m, 15)
    due = date(y + 1, 1, 9) if m == 12 else date(y, m + 1, 9)
    return close, due

FIRST = ["Dana", "Marcus", "Priya", "Liam", "Sofia", "Noah", "Aisha", "Diego",
         "Mia", "Omar", "Hana", "Lucas", "Zoe", "Ravi", "Elena", "Tom",
         "Yuki", "Carla", "Ben", "Nadia"]
LAST = ["Park", "Reyes", "Singh", "Olsen", "Costa", "Khan", "Romano", "Mbeki",
        "Nguyen", "Haddad", "Ito", "Silva", "Adams", "Patel", "Novak", "Lopez"]
SEGMENTS = ["consumer", "affluent", "private"]
REGIONS = ["NA", "EMEA", "APAC"]
POP_CATEGORIES = ["dining", "groceries", "gas", "streaming", "other"]


# --------------------------------------------------------------------------- #
# Reference: products + per-category earn rates
# --------------------------------------------------------------------------- #
def gen_products() -> list[dict[str, Any]]:
    return [
        {"product_id": "PROD-CORE", "product_name": "EveryCard Core", "tier": "core",
         "annual_fee": 0.0, "base_earn_rate": 1.0, "foreign_tx_fee_pct": 3.0,
         "signup_bonus_points": 0, "apr_pct": 24.99},
        {"product_id": "PROD-PREF", "product_name": "EveryCard Preferred", "tier": "preferred",
         "annual_fee": 95.0, "base_earn_rate": 1.0, "foreign_tx_fee_pct": 3.0,
         "signup_bonus_points": 20000, "apr_pct": 22.99},
        {"product_id": "PROD-RESERVE", "product_name": "TravelPlus Reserve", "tier": "reserve",
         "annual_fee": 550.0, "base_earn_rate": 1.0, "foreign_tx_fee_pct": 0.0,
         "signup_bonus_points": 60000, "apr_pct": 21.99},
    ]


def gen_reward_category_rates() -> list[dict[str, Any]]:
    """Base per-category multipliers by product (activation bonuses live in
    reward_activations, not here)."""
    rows: list[dict[str, Any]] = []
    spec = {
        "PROD-CORE": {"dining": 1.0, "groceries": 1.0, "gas": 1.0, "travel": 1.0, "streaming": 1.0},
        "PROD-PREF": {"dining": 3.0, "groceries": 1.0, "gas": 1.0, "travel": 2.0, "streaming": 2.0},
        "PROD-RESERVE": {"dining": 4.0, "groceries": 1.0, "gas": 1.0, "travel": 3.0, "streaming": 1.0},
    }
    n = 1
    for product_id, cats in spec.items():
        for category, mult in cats.items():
            rows.append({
                "rate_id": f"RATE-{n:03d}",
                "product_id": product_id,
                "category": category,
                "earn_multiplier": mult,
                "quarterly_cap": None,
                "requires_activation": False,
            })
            n += 1
    return rows


def _name(rng: random.Random) -> str:
    return f"{rng.choice(FIRST)} {rng.choice(LAST)}"


def _stmt(customer_id: str, card_id: str, cycle: str, *, prev_balance: float,
          purchases: float, fees: float, interest: float, payments: float,
          paid_in_full: bool) -> dict[str, Any]:
    sd, dd = _cycle_dates(cycle)
    new_balance = round(prev_balance - payments + purchases + fees + interest, 2)
    min_payment = round(max(35.0, new_balance * 0.02), 2)
    return {
        "statement_id": f"STMT-{customer_id}-{cycle}",
        "customer_id": customer_id, "card_id": card_id, "cycle": cycle,
        "statement_date": sd.isoformat(), "due_date": dd.isoformat(),
        "prev_balance": round(prev_balance, 2), "purchases": round(purchases, 2),
        "fees": round(fees, 2), "interest": round(interest, 2),
        "payments": round(payments, 2), "new_balance": new_balance,
        "min_payment": min_payment,
        "paid_amount": new_balance if paid_in_full else (min_payment if payments else 0.0),
        "paid_in_full": paid_in_full,
    }


# --------------------------------------------------------------------------- #
# Archetype 1 — Suneel Sunkara (Statement Insights / Expense Spike)
# --------------------------------------------------------------------------- #
# The story: Suneel's monthly expenses are normally ~$950. This cycle they spiked
# to ~$3,400 because of a Japan trip (flights + hotel + dining + electronics),
# annual fee, foreign-tx fees, and interest on his revolving balance. The UI shows
# per-category spending over time, and the spike in "travel" and "dining" is the
# visual hook that makes the user ask "why did my expenses jump?"

# 11 prior monthly PURCHASE totals (USD). They sum to 10,175 -> average $925/mo
# (+ $25 interest = $950 typical monthly expenses, the baseline the Dec spike
# breaks). Totals vary month to month so the trend chart reads as real spending
# rather than a flat line, but the AVERAGE is exact so the ground-truth tie-out
# (typical = $950, increase = $2,450) holds regardless.
_PRIOR_MONTH_TOTALS = [870, 910, 945, 890, 960, 900, 1010, 855, 980, 925, 930]

# Baseline category weights for a normal (non-trip) month (sum = 1.0). Groceries
# is the largest and absorbs rounding so per-month amounts sum EXACTLY to target.
_BASELINE_WEIGHTS = [
    ("groceries", 0.36),
    ("dining", 0.24),
    ("other", 0.19),
    ("gas", 0.14),
    ("streaming", 0.07),
]
_BASELINE_MERCHANTS: dict[str, list[str]] = {
    "groceries": ["FreshMart", "Costco", "GreenGrocer", "WholeFoods"],
    "dining": ["Local Bistro", "Sushi Bar", "Corner Cafe", "Pizza Place", "Thai Kitchen"],
    "other": ["City Pharmacy", "Target", "Amazon", "Hardware Depot"],
    "gas": ["Shell", "Chevron", "BP", "Exxon"],
    "streaming": ["MusicPlus", "CloudStore", "StreamFlix"],
}


def _suneel_prior_category_amounts(total: float, month_idx: int) -> list[tuple[str, str, float]]:
    """Split a month's purchase total across baseline categories with mild,
    deterministic per-month variation. Amounts sum EXACTLY to ``total`` (the
    largest category, groceries, absorbs the rounding remainder)."""
    # Deterministic jitter (-0.03, 0, +0.03 cycling) gives each month a slightly
    # different category mix, so the stacked bars vary in composition.
    jittered = [
        (cat, max(0.02, w + 0.03 * ((month_idx + i) % 3 - 1)))
        for i, (cat, w) in enumerate(_BASELINE_WEIGHTS)
    ]
    wsum = sum(w for _, w in jittered)
    # Everything except groceries (index 0) computed normally; groceries = remainder.
    tail: list[tuple[str, str, float]] = []
    running = 0.0
    for i in range(1, len(jittered)):
        cat, w = jittered[i]
        amt = round(total * w / wsum, 2)
        running += amt
        merchant = _BASELINE_MERCHANTS[cat][(month_idx + i) % len(_BASELINE_MERCHANTS[cat])]
        tail.append((merchant, cat, amt))
    gcat = jittered[0][0]
    gmerch = _BASELINE_MERCHANTS[gcat][month_idx % len(_BASELINE_MERCHANTS[gcat])]
    return [(gmerch, gcat, round(total - running, 2)), *tail]


def _alex(tables: dict[str, list[dict]]) -> None:
    cid, card = "CH-0001", "CARD-0001A"
    tables["cardholders"].append({
        "customer_id": cid, "full_name": "Suneel Sunkara", "segment": "affluent",
        "region": "NA", "primary_product_id": "PROD-PREF", "credit_limit": 15000.0,
        "apr_pct": 22.99, "autopay_type": "minimum", "points_balance": 48000,
        "status": "active", "tenure_months": 41,
        "email": "suneel.sunkara@example.com", "signup_date": "2022-07-01",
    })
    tables["cardholder_cards"].append({
        "holding_id": f"HOLD-{cid}-1", "customer_id": cid, "product_id": "PROD-PREF",
        "card_id": card, "opened_date": "2022-07-01", "is_primary": True,
    })

    n = 1

    def txn(cyc, merchant, category, amount, *, foreign=False, ttype="purchase",
            fee_type="none", posted="2025-12-05") -> str:
        nonlocal n
        txn_id = f"TXN-{cid}-{n:03d}"
        n += 1
        tables["transactions"].append({
            "txn_id": txn_id, "customer_id": cid, "card_id": card, "product_id": "PROD-PREF",
            "cycle": cyc, "posted_date": posted, "merchant": merchant, "category": category,
            "mcc": "0000", "amount": round(amount, 2), "currency": "JPY" if foreign else "USD",
            "is_foreign": foreign, "txn_type": ttype, "fee_type": fee_type,
            "is_reversed": False, "reversal_of_txn_id": None,
        })
        return txn_id

    # Generate FULL transactions for all 11 prior cycles (not just the current
    # one) so we have per-category spending for every month. This is what makes
    # the 12-month expense trend and category breakdown rich.
    prev = 1180.0
    for m_idx, cyc in enumerate(PRIOR_CYCLES):
        total = float(_PRIOR_MONTH_TOTALS[m_idx])
        total_purchases = 0.0
        for merchant_name, cat, amt in _suneel_prior_category_amounts(total, m_idx):
            txn(cyc, merchant_name, cat, amt, posted=f"{cyc}-10")
            total_purchases += amt
        # Interest on revolving balance
        txn(cyc, "Interest Charge", "fees", 25.0, ttype="interest", fee_type="interest",
            posted=f"{cyc}-15")
        # Payment
        txn(cyc, "Payment - Thank You", "other", -(prev * 0.9), ttype="payment",
            posted=f"{cyc}-02")

        st = _stmt(cid, card, cyc, prev_balance=prev, purchases=round(total_purchases, 2),
                   fees=0.0, interest=25.0, payments=prev * 0.9, paid_in_full=False)
        tables["statements"].append(st)
        prev = st["new_balance"]

    # Current cycle (2025-12): the SPIKE month. Travel + foreign + fees.
    cyc = CURRENT_CYCLE
    # Everyday (same baseline as prior months: $950 total)
    everyday = [
        ("FreshMart", "groceries", 210.0), ("Costco", "groceries", 150.0),
        ("Shell", "gas", 65.0), ("Chevron", "gas", 60.0),
        ("Local Bistro", "dining", 95.0), ("Sushi Bar", "dining", 100.0),
        ("Corner Cafe", "dining", 90.0),
        ("StreamFlix", "streaming", 20.0), ("MusicPlus", "streaming", 15.0),
        ("CloudStore", "streaming", 15.0),
        ("City Pharmacy", "other", 70.0), ("Hardware Depot", "other", 60.0),
    ]
    for merchant_name, category, amount in everyday:
        txn(cyc, merchant_name, category, amount, posted="2025-12-05")
    stream_txn = None
    for t in tables["transactions"]:
        if t["customer_id"] == cid and t["merchant"] == "StreamFlix":
            stream_txn = t["txn_id"]

    # The SPIKE: Japan trip expenses
    txn(cyc, "SkyJet Airways", "travel", 1400.0, posted="2025-12-12")
    txn(cyc, "Tokyo Grand Hotel", "travel", 500.0, foreign=True, posted="2025-12-18")
    txn(cyc, "Ginza Dining", "dining", 250.0, foreign=True, posted="2025-12-19")
    txn(cyc, "Akihabara Electronics", "electronics", 150.0, foreign=True, posted="2025-12-20")
    # Fees triggered by the trip
    txn(cyc, "Foreign Transaction Fee", "fees", 27.0, ttype="fee", fee_type="foreign_tx", posted="2025-12-20")
    txn(cyc, "Annual Membership Fee", "fees", 95.0, ttype="fee", fee_type="annual_fee", posted="2025-12-15")
    txn(cyc, "Interest Charge", "fees", 28.0, ttype="interest", fee_type="interest", posted="2025-12-15")
    txn(cyc, "Payment - Thank You", "other", -250.0, ttype="payment", posted="2025-12-02")

    # Current statement: expenses 3400 = purchases 3250 + fees 122 + interest 28.
    tables["statements"].append(_stmt(
        cid, card, cyc, prev_balance=prev, purchases=3250.0, fees=122.0,
        interest=28.0, payments=250.0, paid_in_full=False,
    ))
    tables["subscriptions"].append({
        "subscription_id": f"SUB-{cid}-1", "customer_id": cid, "card_id": card,
        "merchant": "StreamFlix", "category": "streaming", "monthly_amount": 20.0,
        "started_cycle": cyc, "is_active": True, "first_txn_id": stream_txn,
    })

    # Rewards ledger for Suneel this cycle (PROD-PREF earn rates: dining 3x,
    # travel 2x, streaming 2x, groceries/gas/other/electronics 1x). The SAME
    # Japan trip that spiked expenses ALSO cost rewards: a Q4 travel bonus he
    # never activated means his $1,900 travel earned 2x instead of 4x — 3,800
    # points left on the table. eligible_spend per category ties out to the
    # $3,250 current-cycle purchases (unifying the two voice stories).
    def _rl(idx: int, category: str, spend: float, earned: int, possible: int,
            reason: str = "none") -> None:
        tables["rewards_ledger"].append({
            "ledger_id": f"RL-{cid}-{idx}", "customer_id": cid, "card_id": card,
            "cycle": cyc, "category": category, "eligible_spend": round(spend, 2),
            "points_earned": earned, "points_possible": possible,
            "reversed_points": 0, "expired_points": 0, "missed_reason": reason,
        })

    _rl(1, "dining", 535.0, 1605, 1605)                             # 3x, optimal
    _rl(2, "travel", 1900.0, 3800, 7600, reason="inactive_bonus")   # 2x earned, 4x possible
    _rl(3, "groceries", 360.0, 360, 360)                            # 1x, optimal
    _rl(4, "gas", 125.0, 125, 125)                                  # 1x, optimal
    _rl(5, "streaming", 50.0, 100, 100)                             # 2x, optimal
    _rl(6, "electronics", 150.0, 150, 150)                          # 1x, optimal
    _rl(7, "other", 130.0, 130, 130)                                # 1x, optimal

    # The rotating Q4 travel bonus he did NOT activate (why travel earned 2x, not 4x).
    tables["reward_activations"].append({
        "activation_id": f"ACT-{cid}-Q4", "customer_id": cid, "product_id": "PROD-PREF",
        "promo_id": "Q4-TRAVEL-4X", "category": "travel", "quarter": "2025-Q4",
        "bonus_multiplier": 2.0, "window_start": "2025-10-01", "window_end": "2025-12-31",
        "activated": False, "activated_date": None,
    })


# --------------------------------------------------------------------------- #
# Archetype 2 — Maya Rivera (Rewards Optimizer)
# --------------------------------------------------------------------------- #
def _maya(tables: dict[str, list[dict]]) -> None:
    cid = "CH-0002"
    core, reserve = "CARD-0002A", "CARD-0002B"
    tables["cardholders"].append({
        "customer_id": cid, "full_name": "Maya Rivera", "segment": "affluent",
        "region": "NA", "primary_product_id": "PROD-CORE", "credit_limit": 22000.0,
        "apr_pct": 24.99, "autopay_type": "full_balance", "points_balance": 71000,
        "status": "active", "tenure_months": 33,
        "email": "maya.rivera@example.com", "signup_date": "2023-03-01",
    })
    tables["cardholder_cards"] += [
        {"holding_id": f"HOLD-{cid}-1", "customer_id": cid, "product_id": "PROD-CORE",
         "card_id": core, "opened_date": "2023-03-01", "is_primary": True},
        {"holding_id": f"HOLD-{cid}-2", "customer_id": cid, "product_id": "PROD-RESERVE",
         "card_id": reserve, "opened_date": "2024-01-10", "is_primary": False},
    ]
    cyc = CURRENT_CYCLE
    n = 1

    def txn(merchant, category, amount, *, ttype="purchase", reversal_of=None,
            reversed_=False, posted="2025-12-08", card=core) -> str:
        nonlocal n
        txn_id = f"TXN-{cid}-{n:03d}"
        n += 1
        tables["transactions"].append({
            "txn_id": txn_id, "customer_id": cid, "card_id": card, "product_id": "PROD-CORE",
            "cycle": cyc, "posted_date": posted, "merchant": merchant, "category": category,
            "mcc": "0000", "amount": round(amount, 2), "currency": "USD", "is_foreign": False,
            "txn_type": ttype, "fee_type": "none", "is_reversed": reversed_,
            "reversal_of_txn_id": reversal_of,
        })
        return txn_id

    # Dining $1,500 put on the CORE card (1x) instead of the 4x RESERVE card.
    for amount in (600.0, 500.0, 400.0):
        txn("Fine Dining Co", "dining", amount)
    # Groceries $500 with an un-activated 5x bonus.
    txn("GreenGrocer", "groceries", 500.0)
    # A returned laptop from the prior cycle claws back 600 points this cycle.
    orig = txn("MegaElectronics", "electronics", 600.0, posted="2025-11-20", reversed_=True)
    txn("MegaElectronics", "electronics", -600.0, ttype="return", reversal_of=orig, posted="2025-12-04")
    # Everyday "other" spend that already earns its optimal rate.
    txn("Various Retail", "other", 10400.0)

    # Rewards ledger — the ground-truth accounting the story cites.
    def ledger(idx, category, spend, earned, possible, reversed_pts=0, reason="none", card=core):
        tables["rewards_ledger"].append({
            "ledger_id": f"RL-{cid}-{idx}", "customer_id": cid, "card_id": card, "cycle": cyc,
            "category": category, "eligible_spend": round(spend, 2),
            "points_earned": earned, "points_possible": possible,
            "reversed_points": reversed_pts, "expired_points": 0, "missed_reason": reason,
        })

    ledger(1, "dining", 1500.0, 1500, 6000, reason="wrong_card")        # missed 4,500
    ledger(2, "groceries", 500.0, 500, 2000, reason="inactive_bonus")   # missed 1,500
    ledger(3, "other", 10400.0, 10400, 10400, reason="none")            # optimal
    ledger(4, "electronics", 0.0, 0, 0, reversed_pts=600, reason="returned_purchase")

    # Rotating quarterly grocery bonus she did NOT activate.
    tables["reward_activations"].append({
        "activation_id": f"ACT-{cid}-Q4", "customer_id": cid, "product_id": "PROD-CORE",
        "promo_id": "Q4-GROCERY-5X", "category": "groceries", "quarter": "2025-Q4",
        "bonus_multiplier": 3.0, "window_start": "2025-10-01", "window_end": "2025-12-31",
        "activated": False, "activated_date": None,
    })

    # A simple current statement (Maya pays in full; the story is about rewards).
    tables["statements"].append(_stmt(
        cid, core, cyc, prev_balance=0.0, purchases=13400.0, fees=0.0, interest=0.0,
        payments=13400.0, paid_in_full=True,
    ))


# --------------------------------------------------------------------------- #
# Random population (deterministic) for realistic aggregate reasoning
# --------------------------------------------------------------------------- #
def gen_population(rng: random.Random, n: int) -> dict[str, list[dict]]:
    tables: dict[str, list[dict]] = {
        "cardholders": [], "cardholder_cards": [], "transactions": [],
        "statements": [], "rewards_ledger": [], "reward_activations": [], "subscriptions": [],
    }
    products = ["PROD-CORE", "PROD-PREF", "PROD-RESERVE"]
    for i in range(n):
        cid = f"CH-{i + 3:04d}"
        card = f"CARD-{i + 3:04d}A"
        product = rng.choice(products)
        tenure = rng.randint(4, 72)
        tables["cardholders"].append({
            "customer_id": cid, "full_name": _name(rng), "segment": rng.choice(SEGMENTS),
            "region": rng.choice(REGIONS), "primary_product_id": product,
            "credit_limit": float(rng.choice([5000, 8000, 12000, 20000])),
            "apr_pct": rng.choice([21.99, 22.99, 24.99]),
            "autopay_type": rng.choice(["none", "minimum", "fixed", "full_balance"]),
            "points_balance": rng.randint(2000, 90000),
            "status": rng.choices(["active", "at_risk", "delinquent"], weights=[8, 2, 1])[0],
            "tenure_months": tenure, "email": f"user{i + 3}@example.com",
            "signup_date": (date(2025, 12, 15).replace(day=1)).isoformat(),
        })
        tables["cardholder_cards"].append({
            "holding_id": f"HOLD-{cid}-1", "customer_id": cid, "product_id": product,
            "card_id": card, "opened_date": "2023-01-01", "is_primary": True,
        })
        # 4 statements with mild variation.
        prev = float(rng.randint(200, 1500))
        for cyc in PRIOR_CYCLES + [CURRENT_CYCLE]:
            purchases = float(rng.randint(400, 2500))
            fees = float(rng.choice([0, 0, 0, 95]))
            interest = 0.0 if rng.random() < 0.6 else round(prev * 0.02, 2)
            paid_full = interest == 0.0 and rng.random() < 0.7
            st = _stmt(cid, card, cyc, prev_balance=prev, purchases=purchases, fees=fees,
                       interest=interest, payments=prev * (1.0 if paid_full else 0.5),
                       paid_in_full=paid_full)
            tables["statements"].append(st)
            prev = st["new_balance"]
        # A few current-cycle transactions + rewards ledger with a modest gap.
        m = 1
        for category in POP_CATEGORIES:
            spend = float(rng.randint(30, 400))
            earned = int(spend)
            possible = earned + (rng.randint(0, int(spend)) if rng.random() < 0.4 else 0)
            reason = "wrong_card" if possible > earned else "none"
            tables["transactions"].append({
                "txn_id": f"TXN-{cid}-{m:03d}", "customer_id": cid, "card_id": card,
                "product_id": product, "cycle": CURRENT_CYCLE, "posted_date": "2025-12-10",
                "merchant": f"{category.title()} Merchant", "category": category, "mcc": "0000",
                "amount": spend, "currency": "USD", "is_foreign": False, "txn_type": "purchase",
                "fee_type": "none", "is_reversed": False, "reversal_of_txn_id": None,
            })
            tables["rewards_ledger"].append({
                "ledger_id": f"RL-{cid}-{m}", "customer_id": cid, "card_id": card,
                "cycle": CURRENT_CYCLE, "category": category, "eligible_spend": spend,
                "points_earned": earned, "points_possible": possible, "reversed_points": 0,
                "expired_points": 0, "missed_reason": reason,
            })
            m += 1
    return tables


# --------------------------------------------------------------------------- #
# Ground truth the validator (and, later, the live Agent-mode report) must match
# --------------------------------------------------------------------------- #
GROUND_TRUTH: dict[str, Any] = {
    "statement_insights": {
        "customer_id": "CH-0001",
        "cycle": CURRENT_CYCLE,
        # Framing: EXPENSES (total charges) spiked vs the typical monthly spend.
        "current_cycle_expenses": 3400.0,
        "typical_monthly_expenses": 950.0,   # trailing average of prior cycles
        "expense_increase": 2450.0,
        "everyday_baseline": 950.0,
        # What CAUSED the spike (fully decomposes the $2,450 increase):
        "expense_drivers": {
            "flight": 1400.0,             # SkyJet Airways domestic flight
            "foreign_trip_spend": 900.0,  # Tokyo hotel + Ginza dining + Akihabara
            "annual_fee": 95.0,           # Annual membership fee
            "foreign_tx_fee": 27.0,       # 3% on $900 foreign purchases
            "interest": 28.0,             # Revolving balance interest
        },
        "new_subscription": {"merchant": "StreamFlix", "monthly_amount": 20.0},
        # Category-level changes for the UI chart:
        "category_spike": {
            "travel": {"this_cycle": 1900.0, "prior_avg": 0.0, "change": "+1900"},
            "dining": {"this_cycle": 530.0, "prior_avg": 170.0, "change": "+212%"},
            "electronics": {"this_cycle": 150.0, "prior_avg": 0.0, "change": "new"},
            "fees": {"this_cycle": 150.0, "prior_avg": 25.0, "change": "+500%"},
        },
    },
    # Suneel's rewards story (same persona as statement_insights): the Japan trip's
    # un-activated Q4 travel bonus left 3,800 points on the table this cycle.
    "rewards_optimizer_suneel": {
        "customer_id": "CH-0001",
        "cycle": CURRENT_CYCLE,
        "points_earned": 6270,
        "points_possible": 10070,
        "reversed_points": 0,
        "points_gap": 3800,
        "leakage": {
            "inactive_travel_bonus": 3800,
        },
    },
    "rewards_optimizer": {
        "customer_id": "CH-0002",
        "cycle": CURRENT_CYCLE,
        "points_earned": 12400,
        "points_possible": 18400,            # optimal, excluding reversals
        "reversed_points": 600,
        "points_gap": 6600,                  # (possible - earned) + reversed
        "leakage": {
            "dining_wrong_card": 4500,
            "inactive_grocery_bonus": 1500,
            "reversed_points": 600,
        },
    },
}
