"""Dynamic Genie Agent management for the credit-card issuer domain.

A card-issuer analogue of ``genie/space.py``: reconcile a Genie space BY NAME
(``card_issuer.genie_space_name``) that points at the card-issuer UC tables, with
entity matching on categorical columns, concise instructions, example SQL, and
benchmark questions. Because the card tables declare informational PK/FK
constraints, Genie infers joins from them; the instructions restate the join map
and the domain semantics the two voice use cases rely on.

The example SQL / benchmarks for the two ANCHOR questions target the ground-truth
cardholders (CH-0001 Statement Insights, CH-0002 Rewards Optimizer) so the Agent's
answers can be diffed directly against ``card.validate_card.GROUND_TRUTH``.

CLI:  python -m genie_voice.genie.space_card
"""
from __future__ import annotations

import json
import secrets
from typing import Any

from genie_voice.config import Settings, get_settings
from genie_voice.datagen.card.schema_card import (
    CARD_MODEL,
    CARD_REFERENCE_TABLES,
    CARD_SAMPLE_QUESTIONS,
)

# Categorical columns to enable entity/format matching on (lifts NL accuracy).
_ENTITY_COLUMNS: dict[str, list[str]] = {
    "cardholders": ["segment", "region", "status", "autopay_type", "primary_product_id"],
    "card_products": ["tier"],
    "reward_category_rates": ["category"],
    "cardholder_cards": ["product_id", "is_primary"],
    "transactions": ["category", "txn_type", "fee_type", "currency", "is_foreign"],
    "rewards_ledger": ["category", "missed_reason"],
    "reward_activations": ["category", "quarter", "activated"],
    "subscriptions": ["category", "is_active"],
}

# Expose every card table: all nine are analytics-shaped and needed to answer the
# two use cases (statement decomposition + rewards leakage).
_SPACE_TABLES = list(CARD_REFERENCE_TABLES)

_TEXT_INSTRUCTIONS = [
    "Domain: credit-card issuer account assistant. transactions is the transaction-level ledger (purchases, fees, interest, payments, refunds/returns); statements is one monthly statement per card with the balance roll-forward; rewards_ledger is per-cycle, per-category points earned vs possible; reward_activations are rotating quarterly bonuses that must be ACTIVATED; subscriptions are detected recurring merchants; card_products + reward_category_rates define earn rates per product/category.\n",
    "Joins: transactions.customer_id = cardholders.customer_id; statements.customer_id = cardholders.customer_id; rewards_ledger.customer_id = cardholders.customer_id; cardholder_cards.customer_id = cardholders.customer_id; cardholder_cards.product_id = card_products.product_id; reward_category_rates.product_id = card_products.product_id; reward_activations.customer_id = cardholders.customer_id; subscriptions.customer_id = cardholders.customer_id; transactions.product_id = card_products.product_id.\n",
    "Statement roll-forward: for one statement, new_balance = prev_balance - payments + purchases + fees + interest. A cycle is identified by the 'YYYY-MM' cycle column. 'This cycle' = the most recent cycle for that customer; 'typical' or 'usual' = the average of that customer's prior cycles' total charges (purchases + fees + interest).\n",
    "Charge vs payment: in transactions, positive amount = a charge (txn_type in purchase, fee, interest, cash_advance); negative amount = a payment/refund/return. Total charges this cycle = sum(amount) where amount > 0 and txn_type <> 'payment'. Statement-increase drivers decompose current-cycle charges into: flight = category='travel' AND is_foreign=false; foreign_trip_spend = is_foreign=true AND txn_type='purchase'; annual_fee = fee_type='annual_fee'; foreign_tx_fee = fee_type='foreign_tx'; interest = txn_type='interest'; everyday = the remainder. The increase vs a typical cycle is fully explained by the non-everyday drivers.\n",
    "Rewards semantics: for a cycle, points_earned = points actually earned and points_possible = the optimal points achievable with the cardholder's best held card + any activatable bonus. Points left on the table by category = points_possible - points_earned; missed_reason explains why (wrong_card = spent on a lower-earning held card; inactive_bonus = an un-activated reward_activations bonus; returned_purchase = points clawed back on a return, tracked as reversed_points). Total points gap = sum(points_possible - points_earned) + sum(reversed_points).\n",
    "Foreign-transaction fee = card_products.foreign_tx_fee_pct percent of foreign purchase amount; the Reserve tier charges 0%. Interest posts when the prior statement was not paid in full (paid_in_full=false).\n",
    "Units: all money columns are USD; round money to 2 decimals. points_earned / points_possible / reversed_points / points_balance are integer reward points. Round percentages to 1 decimal.\n",
    "Clarification: NEVER ask for a period when the question is about a single named customer_id - answer directly using that customer's most recent cycle and their prior cycles. ONLY ask to specify a period for an AGGREGATE question across many cardholders that omits any time range.\n",
    "When summarizing: cite the table/column names used, itemize the dollar or points drivers, and make the driver amounts sum to the stated total (no unexplained residual).\n",
]


def _example_sqls(fq) -> list[dict[str, Any]]:
    return [
        {
            "question": [
                "Why did customer CH-0001's statement balance increase this cycle versus a "
                "typical cycle? Itemize the drivers."
            ],
            "sql": [
                "SELECT\n",
                "  CASE\n",
                "    WHEN t.fee_type = 'annual_fee'                   THEN 'annual_fee'\n",
                "    WHEN t.fee_type = 'foreign_tx'                   THEN 'foreign_tx_fee'\n",
                "    WHEN t.txn_type = 'interest'                     THEN 'interest'\n",
                "    WHEN t.category = 'travel' AND NOT t.is_foreign  THEN 'flight'\n",
                "    WHEN t.is_foreign AND t.txn_type = 'purchase'    THEN 'foreign_trip_spend'\n",
                "    ELSE 'everyday'\n",
                "  END AS driver,\n",
                "  round(sum(t.amount), 2) AS amount_usd\n",
                f"FROM {fq('transactions')} t\n",
                "WHERE t.customer_id = 'CH-0001' AND t.cycle = '2025-12'\n",
                "  AND t.txn_type IN ('purchase','fee','interest','cash_advance') AND t.amount > 0\n",
                "GROUP BY 1 ORDER BY amount_usd DESC",
            ],
        },
        {
            "question": [
                "How much higher are customer CH-0001's charges this cycle than their prior-cycle average?"
            ],
            "sql": [
                "WITH s AS (\n",
                "  SELECT cycle, round(purchases + fees + interest, 2) AS charges\n",
                f"  FROM {fq('statements')} WHERE customer_id = 'CH-0001'\n",
                ")\n",
                "SELECT\n",
                "  (SELECT charges FROM s WHERE cycle = '2025-12') AS this_cycle_charges,\n",
                "  round((SELECT avg(charges) FROM s WHERE cycle < '2025-12'), 2) AS typical_charges,\n",
                "  round((SELECT charges FROM s WHERE cycle = '2025-12')\n",
                "        - (SELECT avg(charges) FROM s WHERE cycle < '2025-12'), 2) AS increase_usd",
            ],
        },
        {
            "question": [
                "Where is customer CH-0002 losing rewards this cycle and why? Quantify the points gap by reason."
            ],
            "sql": [
                "SELECT category, missed_reason,\n",
                "       sum(points_possible - points_earned) AS points_missed,\n",
                "       sum(reversed_points) AS points_reversed\n",
                f"FROM {fq('rewards_ledger')}\n",
                "WHERE customer_id = 'CH-0002' AND cycle = '2025-12'\n",
                "GROUP BY category, missed_reason\n",
                "HAVING sum(points_possible - points_earned) > 0 OR sum(reversed_points) > 0\n",
                "ORDER BY points_missed DESC",
            ],
        },
        {
            "question": ["What is customer CH-0002's total points gap this cycle (earned vs possible plus reversals)?"],
            "sql": [
                "SELECT sum(points_earned) AS earned,\n",
                "       sum(points_possible) AS possible,\n",
                "       sum(reversed_points) AS reversed,\n",
                "       sum(points_possible - points_earned) + sum(reversed_points) AS points_gap\n",
                f"FROM {fq('rewards_ledger')}\n",
                "WHERE customer_id = 'CH-0002' AND cycle = '2025-12'",
            ],
        },
        {
            "question": ["How much did customer CH-0001 pay in fees this cycle, broken down by fee type?"],
            "sql": [
                "SELECT fee_type, round(sum(amount), 2) AS fees_usd\n",
                f"FROM {fq('transactions')}\n",
                "WHERE customer_id = 'CH-0001' AND cycle = '2025-12'\n",
                "  AND txn_type IN ('fee','interest') AND fee_type <> 'none'\n",
                "GROUP BY fee_type ORDER BY fees_usd DESC",
            ],
        },
        {
            "question": ["Which subscriptions started this cycle for customer CH-0001?"],
            "sql": [
                "SELECT merchant, category, monthly_amount, started_cycle\n",
                f"FROM {fq('subscriptions')}\n",
                "WHERE customer_id = 'CH-0001' AND is_active = true AND started_cycle = '2025-12'\n",
                "ORDER BY monthly_amount DESC",
            ],
        },
        {
            "question": ["How much did customer CH-0001 spend on foreign purchases and what foreign-transaction fees did that incur?"],
            "sql": [
                "SELECT\n",
                "  round(sum(CASE WHEN is_foreign AND txn_type = 'purchase' THEN amount ELSE 0 END), 2) AS foreign_spend_usd,\n",
                "  round(sum(CASE WHEN fee_type = 'foreign_tx' THEN amount ELSE 0 END), 2) AS foreign_tx_fees_usd\n",
                f"FROM {fq('transactions')}\n",
                "WHERE customer_id = 'CH-0001' AND cycle = '2025-12'",
            ],
        },
        {
            "question": ["For each spend category this cycle, how many points did customer CH-0002 earn versus the possible maximum?"],
            "sql": [
                "SELECT category, sum(points_earned) AS earned, sum(points_possible) AS possible\n",
                f"FROM {fq('rewards_ledger')}\n",
                "WHERE customer_id = 'CH-0002' AND cycle = '2025-12'\n",
                "GROUP BY category ORDER BY (sum(points_possible) - sum(points_earned)) DESC",
            ],
        },
        {
            "question": ["Which cardholders have the largest gap between points earned and points possible this cycle?"],
            "sql": [
                "SELECT r.customer_id, c.full_name,\n",
                "       sum(r.points_possible - r.points_earned) + sum(r.reversed_points) AS points_gap\n",
                f"FROM {fq('rewards_ledger')} r\n",
                f"JOIN {fq('cardholders')} c ON r.customer_id = c.customer_id\n",
                "WHERE r.cycle = '2025-12'\n",
                "GROUP BY r.customer_id, c.full_name\n",
                "ORDER BY points_gap DESC LIMIT 10",
            ],
        },
        {
            "question": ["Which rotating bonus categories did customer CH-0002 fail to activate this quarter?"],
            "sql": [
                "SELECT promo_id, category, quarter, bonus_multiplier\n",
                f"FROM {fq('reward_activations')}\n",
                "WHERE customer_id = 'CH-0002' AND activated = false",
            ],
        },
        {
            "question": ["What were customer CH-0001's largest transactions this cycle?"],
            "sql": [
                "SELECT merchant, category, amount, is_foreign, txn_type\n",
                f"FROM {fq('transactions')}\n",
                "WHERE customer_id = 'CH-0001' AND cycle = '2025-12' AND amount > 0\n",
                "ORDER BY amount DESC LIMIT 10",
            ],
        },
    ]


def build_card_serialized_space(settings: Settings) -> str:
    fq = settings.card_fqtn

    tables = sorted(
        [
            {
                "identifier": fq(name),
                "description": [CARD_MODEL[name].comment],
                "column_configs": sorted(
                    [
                        {
                            "column_name": col,
                            "enable_entity_matching": True,
                            "enable_format_assistance": True,
                        }
                        for col in _ENTITY_COLUMNS.get(name, [])
                    ],
                    key=lambda x: x["column_name"],
                ),
            }
            for name in _SPACE_TABLES
        ],
        key=lambda x: x["identifier"],
    )

    q_ids = sorted(secrets.token_hex(16) for _ in CARD_SAMPLE_QUESTIONS)
    sample_questions = sorted(
        [{"id": q_ids[i], "question": [CARD_SAMPLE_QUESTIONS[i]]} for i in range(len(CARD_SAMPLE_QUESTIONS))],
        key=lambda x: x["id"],
    )

    examples = _example_sqls(fq)
    ex_ids = sorted(secrets.token_hex(16) for _ in examples)
    example_question_sqls = sorted(
        [
            {"id": ex_ids[i], "question": examples[i]["question"], "sql": examples[i]["sql"]}
            for i in range(len(examples))
        ],
        key=lambda x: x["id"],
    )

    benchmarks = sorted(
        [
            {
                "id": secrets.token_hex(16),
                "question": [ex["question"][0]],
                "answer": [{"format": "SQL", "content": ex["sql"]}],
            }
            for ex in examples
        ],
        key=lambda x: x["id"],
    )

    config = {
        "version": 2,
        "config": {"sample_questions": sample_questions},
        "data_sources": {"tables": tables},
        "instructions": {
            "text_instructions": [{"id": secrets.token_hex(16), "content": _TEXT_INSTRUCTIONS}],
            "example_question_sqls": example_question_sqls,
        },
        "benchmarks": {"questions": benchmarks},
    }
    return json.dumps(config)


def ensure_card_space(settings: Settings | None = None) -> str | None:
    """Recreate the card-issuer Genie space by configured name.

    Story correctness is gated by ``card.validate_card`` (a plain-aggregation
    tie-out against GROUND_TRUTH), so there is no separate telco-style DQ gate here.
    """
    settings = settings or get_settings()
    if not settings.card_issuer.enabled:
        print("card_issuer.enabled is false; skipping card Genie space.")
        return None

    name = settings.card_issuer.genie_space_name
    wh = settings.databricks.sql_warehouse_id
    if not wh:
        raise RuntimeError("card genie: sql_warehouse_id is required to create a Genie space.")

    from genie_voice.databricks.client import current_user, get_workspace_client
    from genie_voice.genie.space import find_space_ids

    client = get_workspace_client(settings)
    serialized = build_card_serialized_space(settings)
    parent = f"/Users/{settings.databricks.run_as or current_user(client)}"

    # Recreate by name so reruns never accumulate duplicate active spaces.
    trash_failures: list[str] = []
    for existing in find_space_ids(client, name):
        try:
            client.genie.trash_space(existing)
            print(f"card genie: trashed existing space '{name}' ({existing})")
        except Exception as exc:  # noqa: BLE001
            try:
                client.api_client.do("DELETE", f"/api/2.0/genie/spaces/{existing}")
                print(f"card genie: trashed existing space '{name}' ({existing})")
            except Exception as exc2:  # noqa: BLE001
                print(f"card genie: could not trash existing space '{name}' ({existing}): {exc2 or exc}")
                trash_failures.append(existing)
    if trash_failures:
        raise RuntimeError(
            "Refusing to create a new card Genie space because existing matching spaces "
            "could not be trashed: " + ", ".join(trash_failures)
        )

    try:
        space = client.genie.create_space(
            warehouse_id=wh, serialized_space=serialized, title=name, parent_path=parent,
        )
        sid = getattr(space, "space_id", None)
        if not sid:
            raise RuntimeError("Genie create_space returned no space_id")
        print(f"card genie: created space '{name}' ({sid})")
        return sid
    except Exception as exc:  # noqa: BLE001
        try:
            resp = client.api_client.do(
                "POST", "/api/2.0/genie/spaces",
                body={"serialized_space": serialized, "warehouse_id": wh,
                      "title": name, "parent_path": parent},
            )
            sid = resp.get("space_id") if isinstance(resp, dict) else None
            if not sid:
                raise RuntimeError("Genie create_space REST response returned no space_id")
            print(f"card genie: created space '{name}' ({sid})")
            return sid
        except Exception as exc2:  # noqa: BLE001
            raise RuntimeError(f"card genie: could not create space '{name}': {exc2 or exc}") from exc2


def main() -> None:
    sid = ensure_card_space()
    if sid:
        s = get_settings()
        host = s.databricks_host.rstrip("/")
        print(f"card genie space ready: {host}/genie/rooms/{sid}")


if __name__ == "__main__":
    main()
