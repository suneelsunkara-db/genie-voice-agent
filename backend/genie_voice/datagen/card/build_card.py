"""Assemble the full credit-card issuer dataset (single source of truth).

Parents-first so foreign keys are always valid:
    card_products, reward_category_rates
        -> cardholders -> cardholder_cards
        -> transactions, statements, rewards_ledger, reward_activations, subscriptions

The two archetypes (Suneel, Maya) are seeded FIRST so their identifiers are stable
regardless of population size; the deterministic random population follows.
"""
from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .generators_card import (
    _alex,
    _maya,
    gen_population,
    gen_products,
    gen_reward_category_rates,
)
from .schema_card import CARD_REFERENCE_TABLES

if TYPE_CHECKING:  # avoid importing config just to build data
    from genie_voice.config import Settings

# Default population size (archetypes are always included on top of this).
DEFAULT_POPULATION = 30


@dataclass
class CardDataset:
    card_products: list[dict] = field(default_factory=list)
    reward_category_rates: list[dict] = field(default_factory=list)
    cardholders: list[dict] = field(default_factory=list)
    cardholder_cards: list[dict] = field(default_factory=list)
    transactions: list[dict] = field(default_factory=list)
    statements: list[dict] = field(default_factory=list)
    rewards_ledger: list[dict] = field(default_factory=list)
    reward_activations: list[dict] = field(default_factory=list)
    subscriptions: list[dict] = field(default_factory=list)
    spending_by_category: list[dict] = field(default_factory=list)

    def table(self, logical_name: str) -> list[dict]:
        return getattr(self, logical_name)

    def as_dict(self) -> dict[str, list[dict]]:
        return {name: self.table(name) for name in CARD_REFERENCE_TABLES}


def _derive_spending_by_category(transactions: list[dict]) -> list[dict]:
    """Aggregate transactions into per-customer, per-cycle, per-category spending.

    Computes totals, transaction counts, largest merchant per bucket, and
    percent change vs the prior cycle — giving the UI charts and Genie rich
    decomposition data without needing to scan raw transactions.
    """
    # Group: (customer_id, cycle, category) -> list of amounts + merchants
    buckets: dict[tuple[str, str, str], list[tuple[float, str]]] = defaultdict(list)
    for t in transactions:
        if t["txn_type"] not in ("purchase", "fee", "interest", "cash_advance"):
            continue
        if t["amount"] <= 0:
            continue
        key = (t["customer_id"], t["cycle"], t["category"])
        buckets[key].append((t["amount"], t["merchant"]))

    # Build rows
    rows: list[dict] = []
    n = 0
    # For pct_change, we need prior cycle lookup
    cycle_totals: dict[tuple[str, str, str], float] = {}
    for (cid, cycle, cat), items in buckets.items():
        total = round(sum(amt for amt, _ in items), 2)
        cycle_totals[(cid, cycle, cat)] = total

    all_cycles = sorted({cyc for _, cyc, _ in buckets.keys()})
    cycle_idx = {c: i for i, c in enumerate(all_cycles)}

    for (cid, cycle, cat), items in sorted(buckets.items()):
        total = round(sum(amt for amt, _ in items), 2)
        count = len(items)
        largest_amt = max(items, key=lambda x: x[0])
        # Find prior cycle
        idx = cycle_idx.get(cycle, 0)
        prior_cycle = all_cycles[idx - 1] if idx > 0 else None
        prior_total = cycle_totals.get((cid, prior_cycle, cat)) if prior_cycle else None
        pct_change = None
        is_new = False
        if prior_total is not None and prior_total > 0:
            pct_change = round(((total - prior_total) / prior_total) * 100, 2)
        elif prior_total is None or prior_total == 0:
            is_new = prior_cycle is not None  # new category if prior cycle existed

        n += 1
        rows.append({
            "spend_cat_id": f"SC-{cid}-{cycle}-{cat}",
            "customer_id": cid,
            "cycle": cycle,
            "category": cat,
            "total_amount": total,
            "txn_count": count,
            "largest_merchant": largest_amt[1],
            "largest_amount": round(largest_amt[0], 2),
            "is_new_category": is_new,
            "pct_change_vs_prior": pct_change,
        })
    return rows


def build_card_dataset(
    settings: "Settings | None" = None, *, population: int | None = None
) -> CardDataset:
    seed = 42
    pop = DEFAULT_POPULATION if population is None else population
    if settings is not None:
        seed = settings.datagen.seed
        pop = getattr(settings.datagen, "num_customers", pop) if population is None else pop
    rng = random.Random(seed)

    tables: dict[str, list[dict]] = {name: [] for name in CARD_REFERENCE_TABLES}
    tables["card_products"] = gen_products()
    tables["reward_category_rates"] = gen_reward_category_rates()

    # Archetypes first (stable ids), then the random population.
    _alex(tables)
    _maya(tables)
    for name, rows in gen_population(rng, pop).items():
        tables[name].extend(rows)

    # Derive spending_by_category from all transactions
    tables["spending_by_category"] = _derive_spending_by_category(tables["transactions"])

    return CardDataset(**tables)
