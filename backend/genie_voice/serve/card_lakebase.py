"""Lakebase fast-facts cache for the credit-card issuer domain.

Subclasses ``LakebaseServing`` so it reuses ALL of the connection machinery
(OAuth token minting, endpoint discovery, pooling) but points at the card-issuer
Postgres serving schema and serves the CARD hot-path facts.

Data flow (same pattern as billing support ``snapshot_reference_tables``):
  1. UC Delta tables are the source of truth (populated by reference_ingest_card)
  2. ``snapshot_card_reference_tables()`` reads Delta via SQL warehouse and upserts
     into Lakebase Postgres (one-off setup/refresh cost, NOT the hot path)
  3. ``get_cardholder_facts()`` reads from Lakebase Postgres (sub-ms, hot path)

This mirrors the billing support's ``_REFERENCE_TABLE_SPECS`` +
``_ensure_reference_tables()`` + ``snapshot_reference_tables()`` pattern exactly.

When ``lakebase.enabled`` is false (offline/local), facts are served straight
from the deterministic ``build_card_dataset()`` so the end-to-end local flow
still works with no Databricks.

CLI:  python -m genie_voice.serve.card_lakebase   # ensure + snapshot
"""
from __future__ import annotations

from typing import Any

from genie_voice.config import Settings, get_settings
from genie_voice.databricks.warehouse_sql import warehouse_configured
from genie_voice.serve.lakebase import LakebaseServing, _pg_ident


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


class CardLakebaseServing(LakebaseServing):
    """Card-issuer Lakebase serving cache (mirrors billing support pattern).

    - ``_ensure_card_reference_tables()``: DDL for the Postgres serving tables
    - ``snapshot_card_reference_tables()``: Delta → Lakebase snapshot reverse-ETL
    - ``get_cardholder_facts()``: hot-path read from the Lakebase cache
    """

    def _table(self, table: str) -> str:
        """Resolve into the CARD serving schema (not the telco lakebase schema)."""
        return f"{_pg_ident(self.settings.card_issuer.schema_name)}.{_pg_ident(table)}"

    # Reference table -> (primary key, [(column, postgres_cast_type), ...]).
    # Same structure as billing support's _REFERENCE_TABLE_SPECS.
    # Column names MUST match the UC Delta table schemas exactly.
    _CARD_REFERENCE_SPECS: dict[str, tuple[str, list[tuple[str, str]]]] = {
        "cardholders": ("customer_id", [
            ("customer_id", "TEXT"), ("full_name", "TEXT"), ("segment", "TEXT"),
            ("region", "TEXT"), ("primary_product_id", "TEXT"),
            ("credit_limit", "NUMERIC(10,2)"), ("apr_pct", "NUMERIC(5,2)"),
            ("autopay_type", "TEXT"), ("points_balance", "INTEGER"),
            ("status", "TEXT"), ("tenure_months", "INTEGER"),
            ("email", "TEXT"), ("signup_date", "DATE"),
        ]),
        "statements": ("statement_id", [
            ("statement_id", "TEXT"), ("customer_id", "TEXT"), ("card_id", "TEXT"),
            ("cycle", "TEXT"), ("statement_date", "DATE"), ("due_date", "DATE"),
            ("prev_balance", "NUMERIC(10,2)"), ("purchases", "NUMERIC(10,2)"),
            ("fees", "NUMERIC(10,2)"), ("interest", "NUMERIC(10,2)"),
            ("payments", "NUMERIC(10,2)"), ("new_balance", "NUMERIC(10,2)"),
            ("min_payment", "NUMERIC(10,2)"), ("paid_amount", "NUMERIC(10,2)"),
            ("paid_in_full", "BOOLEAN"),
        ]),
        "spending_by_category": ("spend_cat_id", [
            ("spend_cat_id", "TEXT"), ("customer_id", "TEXT"), ("cycle", "TEXT"),
            ("category", "TEXT"), ("total_amount", "NUMERIC(10,2)"),
            ("txn_count", "INTEGER"), ("largest_merchant", "TEXT"),
            ("largest_amount", "NUMERIC(10,2)"), ("is_new_category", "BOOLEAN"),
            ("pct_change_vs_prior", "NUMERIC(10,2)"),
        ]),
        "rewards_ledger": ("ledger_id", [
            ("ledger_id", "TEXT"), ("customer_id", "TEXT"), ("card_id", "TEXT"),
            ("cycle", "TEXT"), ("category", "TEXT"), ("eligible_spend", "NUMERIC(10,2)"),
            ("points_earned", "BIGINT"), ("points_possible", "BIGINT"),
            ("reversed_points", "BIGINT"), ("expired_points", "BIGINT"),
            ("missed_reason", "TEXT"),
        ]),
    }

    # ---- DDL (same pattern as billing _ensure_reference_tables) ----------- #
    def _ensure_card_reference_tables(self, cur) -> None:
        """Create the card serving cache tables (idempotent).

        Same pattern as billing support's _ensure_reference_tables: create plain
        Postgres tables keyed by PK, with indexes on customer_id for hot-path
        lookups. Deliberately left at DEFAULT replica identity (not FULL).
        """
        schema = _pg_ident(self.settings.card_issuer.schema_name)
        cur.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")

        cardholders_tbl = self._table("cardholders")
        statements_tbl = self._table("statements")
        spending_tbl = self._table("spending_by_category")

        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {cardholders_tbl} (
                customer_id        TEXT PRIMARY KEY,
                full_name          TEXT,
                segment            TEXT,
                region             TEXT,
                primary_product_id TEXT,
                credit_limit       NUMERIC(10,2),
                apr_pct            NUMERIC(5,2),
                autopay_type       TEXT,
                points_balance     INTEGER,
                status             TEXT,
                tenure_months      INTEGER,
                email              TEXT,
                signup_date        DATE
            )
            """
        )
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {statements_tbl} (
                statement_id  TEXT PRIMARY KEY,
                customer_id   TEXT NOT NULL,
                card_id       TEXT,
                cycle         TEXT,
                statement_date DATE,
                due_date      DATE,
                prev_balance  NUMERIC(10,2),
                purchases     NUMERIC(10,2),
                fees          NUMERIC(10,2),
                interest      NUMERIC(10,2),
                payments      NUMERIC(10,2),
                new_balance   NUMERIC(10,2),
                min_payment   NUMERIC(10,2),
                paid_amount   NUMERIC(10,2),
                paid_in_full  BOOLEAN
            )
            """
        )
        cur.execute(
            f"CREATE INDEX IF NOT EXISTS statements_customer_idx "
            f"ON {statements_tbl} (customer_id)"
        )
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {spending_tbl} (
                spend_cat_id      TEXT PRIMARY KEY,
                customer_id       TEXT NOT NULL,
                cycle             TEXT,
                category          TEXT,
                total_amount      NUMERIC(10,2),
                txn_count         INTEGER,
                largest_merchant  TEXT,
                largest_amount    NUMERIC(10,2),
                is_new_category   BOOLEAN,
                pct_change_vs_prior NUMERIC(10,2)
            )
            """
        )
        cur.execute(
            f"CREATE INDEX IF NOT EXISTS spending_customer_idx "
            f"ON {spending_tbl} (customer_id)"
        )
        rewards_tbl = self._table("rewards_ledger")
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {rewards_tbl} (
                ledger_id       TEXT PRIMARY KEY,
                customer_id     TEXT NOT NULL,
                card_id         TEXT,
                cycle           TEXT,
                category        TEXT,
                eligible_spend  NUMERIC(10,2),
                points_earned   BIGINT,
                points_possible BIGINT,
                reversed_points BIGINT,
                expired_points  BIGINT,
                missed_reason   TEXT
            )
            """
        )
        cur.execute(
            f"CREATE INDEX IF NOT EXISTS rewards_customer_idx "
            f"ON {rewards_tbl} (customer_id)"
        )

    # ---- snapshot reverse-ETL (same pattern as billing snapshot_reference_tables) #
    def snapshot_card_reference_tables(self) -> dict[str, int]:
        """Snapshot reverse-ETL: copy card UC Delta reference tables into Lakebase.

        Same pattern as billing support's snapshot_reference_tables():
        reads from UC Delta via SQL warehouse (one-off refresh cost), upserts into
        the Lakebase serving cache. Idempotent (INSERT ... ON CONFLICT DO UPDATE).
        """
        if not self.enabled:
            return {}
        if not warehouse_configured(self.settings):
            raise RuntimeError("snapshot_card_reference_tables requires databricks.sql_warehouse_id")
        from genie_voice.databricks.client import get_workspace_client

        with self._conn() as conn, conn.cursor() as cur:
            self._ensure_card_reference_tables(cur)

        client = get_workspace_client(self.settings)
        wh = self.settings.databricks.sql_warehouse_id
        counts: dict[str, int] = {}

        for table, (pk, colspecs) in self._CARD_REFERENCE_SPECS.items():
            cols = [c for c, _ in colspecs]
            res = client.statement_execution.execute_statement(
                warehouse_id=wh,
                statement=f"SELECT {', '.join(cols)} FROM {self.settings.card_fqtn(table)}",
                wait_timeout="50s",
            )
            rows = (res.result.data_array if res.result else None) or []
            tbl = self._table(table)
            placeholders = ", ".join(f"CAST(%s AS {t})" for _, t in colspecs)
            updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in cols if c != pk)
            with self._conn() as conn, conn.cursor() as cur:
                for row in rows:
                    values = [(v if v not in ("", None) else None) for v in row]
                    cur.execute(
                        f"INSERT INTO {tbl} ({', '.join(cols)}) VALUES ({placeholders}) "
                        f"ON CONFLICT ({pk}) DO UPDATE SET {updates}",
                        tuple(values),
                    )
            counts[table] = len(rows)
        return counts

    # ---- hot-path read ---------------------------------------------------- #
    def get_cardholder_facts(self, customer_id: str) -> dict[str, Any]:
        """Fast facts for the live voice greeting: cardholder + current statement +
        per-category spending history (for the expense charts).

        Reads from the Lakebase serving cache (sub-ms Postgres). The itemized
        "why" is answered by Genie Agent mode, not from this cache.
        """
        if not self.enabled:
            return self._cardholder_facts_local(customer_id)
        with self._conn() as conn, conn.cursor() as cur:
            def rows(sql: str) -> list[dict[str, Any]]:
                cur.execute(sql, (customer_id,))
                cols = [d[0] for d in cur.description]
                return [dict(zip(cols, r)) for r in cur.fetchall()]

            cardholders_tbl = self._table("cardholders")
            statements_tbl = self._table("statements")
            spending_tbl = self._table("spending_by_category")
            rewards_tbl = self._table("rewards_ledger")
            ch_rows = rows(f"SELECT * FROM {cardholders_tbl} WHERE customer_id = %s")
            cardholder = ch_rows[0] if ch_rows else None
            stmts = rows(
                f"SELECT * FROM {statements_tbl} WHERE customer_id = %s "
                f"ORDER BY cycle DESC LIMIT 12"
            )
            spending = rows(
                f"SELECT * FROM {spending_tbl} WHERE customer_id = %s "
                f"ORDER BY cycle, category"
            )
            rewards = rows(
                f"SELECT * FROM {rewards_tbl} WHERE customer_id = %s "
                f"ORDER BY cycle DESC, category"
            )
        return self._shape_cardholder_facts(customer_id, cardholder, stmts, spending, rewards)

    def _cardholder_facts_local(self, customer_id: str) -> dict[str, Any]:
        from genie_voice.datagen.card.build_card import build_card_dataset

        ds = build_card_dataset(self.settings)
        cardholder = next((c for c in ds.cardholders if c.get("customer_id") == customer_id), None)
        stmts = sorted(
            [s for s in ds.statements if s.get("customer_id") == customer_id],
            key=lambda s: str(s.get("cycle")), reverse=True,
        )
        spending = sorted(
            [s for s in ds.spending_by_category if s.get("customer_id") == customer_id],
            key=lambda s: (str(s.get("cycle")), str(s.get("category"))),
        )
        rewards = sorted(
            [r for r in ds.rewards_ledger if r.get("customer_id") == customer_id],
            key=lambda r: (str(r.get("cycle")), str(r.get("category"))), reverse=True,
        )
        return self._shape_cardholder_facts(customer_id, cardholder, stmts, spending, rewards)

    @staticmethod
    def _shape_cardholder_facts(
        customer_id: str, cardholder: dict[str, Any] | None,
        statements: list[dict[str, Any]], spending: list[dict[str, Any]] | None = None,
        rewards: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        latest = statements[0] if statements else None
        summary: dict[str, Any] = {
            "status": cardholder.get("status") if cardholder else None,
            "points_balance": cardholder.get("points_balance") if cardholder else None,
            "credit_limit": _to_float(cardholder.get("credit_limit")) if cardholder else None,
        }
        if latest:
            this_expenses = (
                _to_float(latest.get("purchases"))
                + _to_float(latest.get("fees"))
                + _to_float(latest.get("interest"))
            )
            prior_stmts = statements[1:] if len(statements) > 1 else []
            if prior_stmts:
                prior_expenses = [
                    _to_float(s.get("purchases")) + _to_float(s.get("fees")) + _to_float(s.get("interest"))
                    for s in prior_stmts
                ]
                avg_monthly_expenses = round(sum(prior_expenses) / len(prior_expenses), 2)
            else:
                avg_monthly_expenses = 0.0
            summary.update({
                "current_cycle": latest.get("cycle"),
                "this_month_expenses": round(this_expenses, 2),
                "avg_monthly_expenses": avg_monthly_expenses,
                "expense_change": round(this_expenses - avg_monthly_expenses, 2),
                "new_balance": _to_float(latest.get("new_balance")),
                "prev_balance": _to_float(latest.get("prev_balance")),
                "min_payment": _to_float(latest.get("min_payment")),
                "due_date": latest.get("due_date"),
                "paid_in_full": latest.get("paid_in_full"),
            })

        # Rewards leakage summary, scoped to the current cycle (drives the
        # rewards-optimizer waterfall in the UI without waiting on Agent mode).
        current_cycle = latest.get("cycle") if latest else None
        rewards = rewards or []
        cur_rewards = [r for r in rewards if str(r.get("cycle")) == str(current_cycle)] or rewards
        if cur_rewards:
            earned = sum(int(r.get("points_earned") or 0) for r in cur_rewards)
            possible = sum(int(r.get("points_possible") or 0) for r in cur_rewards)
            reversed_pts = sum(int(r.get("reversed_points") or 0) for r in cur_rewards)
            expired_pts = sum(int(r.get("expired_points") or 0) for r in cur_rewards)
            summary["rewards"] = {
                "cycle": current_cycle,
                "points_earned": earned,
                "points_possible": possible,
                "reversed_points": reversed_pts,
                "expired_points": expired_pts,
                "points_gap": max(possible - earned, 0) + reversed_pts + expired_pts,
            }
        return {
            "customer_id": customer_id,
            "found": cardholder is not None,
            "cardholder": cardholder,
            "latest_statement": latest,
            "recent_statements": statements,
            "spending_by_category": spending or [],
            "rewards_ledger": cur_rewards,
            "summary": summary,
        }


def main() -> None:
    settings = get_settings()
    if not settings.card_issuer.enabled:
        print("card_issuer.enabled is false; skipping.")
        return
    serving = CardLakebaseServing(settings)
    if not serving.enabled:
        print("Lakebase disabled; card fast facts are served from datagen in local mode.")
        return
    print("Snapshotting card reference tables (Delta -> Lakebase) ...")
    counts = serving.snapshot_card_reference_tables()
    print("  ok: snapshotted card reference tables -> Lakebase: "
          + ", ".join(f"{t}={n}" for t, n in sorted(counts.items())))
    print("Done.")


if __name__ == "__main__":
    main()
