"""Live validation: does the deployed card Genie space reproduce the story?

Asks the two anchor questions (plus supporting ones) against the REAL card Genie
space via the Conversation API and diffs the generated SQL's result rows against
``card.generators_card.GROUND_TRUTH``. The Conversation API returns structured
rows + the generated SQL, so this is a deterministic gate; because Agent mode
reasons over the SAME curated tables, a pass here means the data + curation are
sound for the deeper "why" lane too.

Run:  python -m genie_voice.genie.validate_card_live
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from genie_voice.config import get_settings
from genie_voice.datagen.card.generators_card import GROUND_TRUTH
from genie_voice.genie.client import GenieClient


@dataclass
class LiveCheck:
    label: str
    question: str
    expected_numbers: list[int]
    result: dict[str, Any] = field(default_factory=dict)

    def haystack(self) -> str:
        """All text/rows Genie returned, comma-stripped, for numeric containment."""
        blob = json.dumps(
            {
                "answer": self.result.get("answer"),
                "description": self.result.get("description"),
                "rows": self.result.get("rows"),
                "sql": self.result.get("sql"),
            },
            default=str,
        )
        return re.sub(r"(?<=\d),(?=\d)", "", blob)

    def missing(self) -> list[int]:
        hay = self.haystack()
        # Match each number as a standalone integer/decimal token (avoid 1400 in 11400).
        return [n for n in self.expected_numbers if not re.search(rf"(?<!\d){n}(?!\d)", hay)]

    @property
    def ok(self) -> bool:
        return self.result.get("rows") is not None and not self.missing()


def _checks() -> list[LiveCheck]:
    si = GROUND_TRUTH["statement_insights"]
    ro = GROUND_TRUTH["rewards_optimizer"]
    d = si["drivers"]
    lk = ro["leakage"]
    return [
        LiveCheck(
            "statement_insights_drivers",
            "For cardholder CH-0001, why did the statement balance increase this cycle "
            "(cycle 2025-12) compared with their prior cycles? Break the increase down by "
            "driver with the dollar amount for each driver.",
            [int(d["flight"]), int(d["foreign_trip_spend"]), int(d["annual_fee"]),
             int(d["foreign_tx_fee"]), int(d["interest"])],
        ),
        LiveCheck(
            "rewards_optimizer_leakage",
            "For cardholder CH-0002, where are they losing rewards points this cycle "
            "(cycle 2025-12) and why? Quantify the points missed by reason.",
            [lk["dining_wrong_card"], lk["inactive_grocery_bonus"], lk["reversed_points"]],
        ),
        LiveCheck(
            "statement_fees_breakdown",
            "How much did cardholder CH-0001 pay in fees this cycle (2025-12), broken down "
            "by fee type?",
            [int(d["annual_fee"]), int(d["foreign_tx_fee"]), int(d["interest"])],
        ),
        LiveCheck(
            "rewards_points_gap_total",
            "What is cardholder CH-0002's total rewards points gap this cycle (2025-12): "
            "points earned versus possible, including reversed points?",
            [ro["points_earned"], ro["points_possible"], ro["points_gap"]],
        ),
    ]


def main() -> int:
    settings = get_settings()
    # Pin the client to the CARD space BY NAME via the constructor (no private
    # _space_id poke) — the same seam the production card path uses.
    space_name = settings.card_issuer.genie_space_name
    gc = GenieClient(settings, space_name=space_name)
    from genie_voice.databricks.client import get_workspace_client
    from genie_voice.genie.space import find_space_ids

    client = get_workspace_client(settings)
    matches = find_space_ids(client, space_name)
    if not matches:
        print(f"No Genie space named '{space_name}'. Run space_card first.")
        return 1
    if len(matches) > 1:
        print(f"Multiple spaces named '{space_name}': {matches}. Reconcile first.")
        return 1
    print(f"Validating card Genie space {matches[0]} ('{space_name}')\n")

    checks = _checks()
    conversation_id: str | None = None
    for chk in checks:
        try:
            # Fresh conversation per check so context never bleeds across customers.
            chk.result = gc.ask(chk.question)
        except Exception as exc:  # noqa: BLE001
            chk.result = {"error": str(exc)}
        mark = "ok  " if chk.ok else "FAIL"
        print(f"[{mark}] {chk.label}")
        print(f"    Q: {chk.question}")
        if chk.result.get("error"):
            print(f"    ERROR: {chk.result['error']}")
        else:
            ans = (chk.result.get("answer") or chk.result.get("description") or "").strip()
            print(f"    answer: {ans[:400]}")
            print(f"    sql:    {(chk.result.get('sql') or '').strip()[:400]}")
            print(f"    rows:   {chk.result.get('rows')}")
            miss = chk.missing()
            if miss:
                print(f"    MISSING expected numbers: {miss}")
        print()

    passed = sum(1 for c in checks if c.ok)
    print(f"{'PASS' if passed == len(checks) else 'PARTIAL/FAIL'}: {passed}/{len(checks)} live checks matched ground truth")
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
