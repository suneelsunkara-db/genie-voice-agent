"""Data quality gate for Genie-ready UC analytics tables."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from genie_voice.config import Settings, get_settings
from genie_voice.databricks.client import get_workspace_client
from genie_voice.datagen.schema import (
    MODEL,
    REFERENCE_TABLES,
    T_GOLD,
)


@dataclass(frozen=True)
class Check:
    name: str
    sql: str
    expected: int = 0


def _q(name: str) -> str:
    return "`" + name.replace("`", "``") + "`"


def _fqtn(settings: Settings, table: str) -> str:
    return ".".join(
        [
            _q(settings.databricks.catalog),
            _q(settings.databricks.schema_name),
            _q(table),
        ]
    )


def _history_fqtn(settings: Settings, source_table: str) -> str:
    return _fqtn(
        settings,
        f"{settings.lakebase.cdf_history_prefix}{source_table}{settings.lakebase.cdf_history_suffix}",
    )


def _scalar(client, warehouse_id: str, sql: str) -> int:
    res = client.statement_execution.execute_statement(
        warehouse_id=warehouse_id,
        statement=sql,
        wait_timeout="30s",
    )
    rows = (res.result.data_array if res.result else None) or []
    if not rows:
        return 0
    return int(rows[0][0] or 0)


def _pk_checks(settings: Settings, tables: list[str]) -> list[Check]:
    checks: list[Check] = []
    for table in tables:
        spec = MODEL[table]
        if not spec.primary_key:
            continue
        fq = _fqtn(settings, table)
        null_pred = " OR ".join(f"{_q(col)} IS NULL" for col in spec.primary_key)
        key_cols = ", ".join(_q(col) for col in spec.primary_key)
        checks.append(
            Check(
                f"{table}: primary key columns are non-null",
                f"SELECT count(*) FROM {fq} WHERE {null_pred}",
            )
        )
        checks.append(
            Check(
                f"{table}: primary key is unique",
                f"""
                SELECT count(*) FROM (
                  SELECT {key_cols}, count(*) AS n
                  FROM {fq}
                  GROUP BY {key_cols}
                  HAVING count(*) > 1
                ) dup
                """,
            )
        )
    return checks


def _fk_checks(settings: Settings, tables: list[str]) -> list[Check]:
    checks: list[Check] = []
    for table in tables:
        child = _fqtn(settings, table)
        for fk in MODEL[table].foreign_keys:
            parent = _fqtn(settings, fk.ref_table)
            child_col = _q(fk.column)
            parent_col = _q(fk.ref_column)
            checks.append(
                Check(
                    f"{table}.{fk.column} references {fk.ref_table}.{fk.ref_column}",
                    f"""
                    SELECT count(*)
                    FROM {child} c
                    LEFT JOIN {parent} p ON c.{child_col} = p.{parent_col}
                    WHERE c.{child_col} IS NOT NULL AND p.{parent_col} IS NULL
                    """,
                )
            )
    return checks


def _constraint_metadata_checks(settings: Settings, tables: list[str]) -> list[Check]:
    checks: list[Check] = []
    cat = _q(settings.databricks.catalog)
    schema = settings.databricks.schema_name.replace("'", "''")
    constraints = f"{cat}.information_schema.table_constraints"
    for table in tables:
        spec = MODEL[table]
        table_name = table.replace("'", "''")
        if spec.primary_key:
            checks.append(
                Check(
                    f"{table}: UC primary key metadata exists",
                    f"""
                    SELECT CASE WHEN count(*) > 0 THEN 0 ELSE 1 END
                    FROM {constraints}
                    WHERE table_schema = '{schema}'
                      AND table_name = '{table_name}'
                      AND constraint_name = 'pk_{table}'
                      AND constraint_type = 'PRIMARY KEY'
                    """,
                )
            )
        for fk in spec.foreign_keys:
            checks.append(
                Check(
                    f"{table}.{fk.column}: UC foreign key metadata exists",
                    f"""
                    SELECT CASE WHEN count(*) > 0 THEN 0 ELSE 1 END
                    FROM {constraints}
                    WHERE table_schema = '{schema}'
                      AND table_name = '{table_name}'
                      AND constraint_name = 'fk_{table}_{fk.column}'
                      AND constraint_type = 'FOREIGN KEY'
                    """,
                )
            )
    return checks


def _business_checks(settings: Settings) -> list[Check]:
    facts_history = _history_fqtn(settings, "call_facts")
    facts = f"""
        (
          SELECT *
          FROM (
            SELECT *,
                   row_number() OVER (PARTITION BY call_id ORDER BY _sort_by DESC) AS _rn
            FROM {facts_history}
            WHERE _pg_change_type IN ('insert', 'update_postimage', 'delete')
          )
          WHERE _rn = 1 AND _pg_change_type <> 'delete'
        )
    """
    utterance_history = _history_fqtn(settings, "live_call_utterances")
    gold = _fqtn(settings, T_GOLD)
    invoices = _fqtn(settings, "invoices")
    payments = _fqtn(settings, "payments")
    customers = _fqtn(settings, "customers")

    return [
        Check(
            "call_facts has at least one call",
            f"SELECT CASE WHEN count(*) > 0 THEN 0 ELSE 1 END FROM {facts}",
        ),
        Check(
            "call_facts history current call_id is non-null",
            f"SELECT count(*) FROM {facts} WHERE call_id IS NULL",
        ),
        Check(
            "call_facts history current call_id is unique",
            f"""
            SELECT count(*) FROM (
              SELECT call_id, count(*) AS n
              FROM {facts}
              GROUP BY call_id
              HAVING count(*) > 1
            ) dup
            """,
        ),
        Check(
            "every call has at least one utterance",
            f"""
            SELECT count(*)
            FROM {facts} f
            LEFT JOIN (
              SELECT call_id, count(*) AS turns
              FROM (
                SELECT *,
                       row_number() OVER (PARTITION BY utterance_id ORDER BY _sort_by DESC) AS _rn
                FROM {utterance_history}
                WHERE _pg_change_type IN ('insert', 'update_postimage', 'delete')
              )
              WHERE _rn = 1 AND _pg_change_type <> 'delete'
              GROUP BY call_id
            ) s ON f.call_id = s.call_id
            WHERE coalesce(s.turns, 0) = 0
            """,
        ),
        Check(
            "utterance turn_index is unique per call",
            f"""
            SELECT count(*) FROM (
              SELECT call_id, turn_index, count(*) AS n
              FROM (
                SELECT *,
                       row_number() OVER (PARTITION BY utterance_id ORDER BY _sort_by DESC) AS _rn
                FROM {utterance_history}
                WHERE _pg_change_type IN ('insert', 'update_postimage', 'delete')
              )
              WHERE _rn = 1 AND _pg_change_type <> 'delete'
              GROUP BY call_id, turn_index
              HAVING count(*) > 1
            ) dup
            """,
        ),
        Check(
            "gold exists for every call",
            f"""
            SELECT count(*)
            FROM {facts} f
            LEFT JOIN {gold} g ON f.call_id = g.call_id
            WHERE g.call_id IS NULL
            """,
        ),
        Check(
            "gold required insight fields are populated",
            f"""
            SELECT count(*)
            FROM {gold}
            WHERE primary_intent IS NULL
               OR sentiment_label IS NULL
               OR next_best_action IS NULL
               OR summary IS NULL
            """,
        ),
        Check(
            "gold insight controlled vocabularies are valid",
            f"""
            SELECT count(*)
            FROM {gold}
            WHERE primary_intent NOT IN (
                'billing_dispute', 'late_fee', 'payment_arrangement', 'refund',
                'autopay_issue', 'plan_inquiry', 'cancellation_risk', 'billing_inquiry'
            )
               OR sentiment_label NOT IN ('negative', 'neutral', 'positive')
               OR disposition NOT IN ('resolved', 'follow_up', 'escalated')
               OR resolution_status NOT IN ('resolved', 'open')
               OR next_best_action NOT IN (
                'escalate_retention_offer', 'offer_fee_waiver', 'process_refund',
                'set_up_payment_plan', 'offer_plan_upgrade', 'continue'
               )
               OR exists(
                    all_intents,
                    x -> NOT array_contains(
                        array(
                            'billing_dispute', 'late_fee', 'payment_arrangement', 'refund',
                            'autopay_issue', 'plan_inquiry', 'cancellation_risk', 'billing_inquiry'
                        ),
                        x
                    )
               )
            """,
        ),
        Check(
            "call_facts CSAT range is valid",
            f"SELECT count(*) FROM {facts} WHERE csat IS NOT NULL AND (csat < 1 OR csat > 5)",
        ),
        Check(
            "call_facts history references valid customers and agents",
            f"""
            SELECT count(*)
            FROM {facts} f
            LEFT JOIN {customers} c ON f.customer_id = c.customer_id
            LEFT JOIN {_fqtn(settings, "agents")} a ON f.agent_id = a.agent_id
            WHERE c.customer_id IS NULL
               OR (f.agent_id IS NOT NULL AND a.agent_id IS NULL)
            """,
        ),
        Check(
            "reference controlled vocabularies are valid",
            f"""
            SELECT (
              SELECT count(*) FROM {customers}
              WHERE status NOT IN ('active', 'at_risk', 'churned')
                 OR segment NOT IN ('consumer', 'smb', 'enterprise')
                 OR region NOT IN ('NA', 'EMEA', 'APAC')
                 OR plan NOT IN ('basic', 'pro', 'premium')
            ) + (
              SELECT count(*) FROM {invoices}
              WHERE status NOT IN ('paid', 'open', 'overdue', 'disputed', 'refunded')
            ) + (
              SELECT count(*) FROM {payments}
              WHERE status NOT IN ('succeeded', 'declined', 'refunded')
                 OR method NOT IN ('card', 'bank_transfer', 'autopay')
            )
            """,
        ),
        Check(
            "mentioned invoices belong to the call customer",
            f"""
            SELECT count(*)
            FROM {gold} g
            JOIN {invoices} i ON g.mentioned_invoice_id = i.invoice_id
            WHERE g.mentioned_invoice_id IS NOT NULL
              AND g.customer_id <> i.customer_id
            """,
        ),
        Check(
            "payments belong to valid customers and invoices",
            f"""
            SELECT count(*)
            FROM {payments} p
            LEFT JOIN {customers} c ON p.customer_id = c.customer_id
            LEFT JOIN {invoices} i ON p.invoice_id = i.invoice_id
            WHERE c.customer_id IS NULL
               OR i.invoice_id IS NULL
               OR p.customer_id <> i.customer_id
            """,
        ),
    ]


def build_checks(settings: Settings) -> list[Check]:
    genie_tables = [*REFERENCE_TABLES, T_GOLD]
    return [
        *_constraint_metadata_checks(settings, genie_tables),
        *_pk_checks(settings, genie_tables),
        *_fk_checks(settings, genie_tables),
        *_business_checks(settings),
    ]


def run_data_quality(settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or get_settings()
    if not settings.databricks.sql_warehouse_id:
        raise RuntimeError("databricks.sql_warehouse_id is required for data quality checks.")

    client = get_workspace_client(settings)
    failures: list[dict[str, Any]] = []
    print("Running data quality checks for Genie-ready UC tables ...")
    for check in build_checks(settings):
        observed = _scalar(client, settings.databricks.sql_warehouse_id, check.sql)
        status = "ok" if observed == check.expected else "FAIL"
        print(f"  {status}: {check.name} (observed={observed}, expected={check.expected})")
        if observed != check.expected:
            failures.append(
                {"name": check.name, "observed": observed, "expected": check.expected}
            )

    if failures:
        raise RuntimeError(f"Data quality check failed: {failures}")
    print("Data quality checks passed.")
    return {"checks": len(build_checks(settings)), "failures": 0}


if __name__ == "__main__":
    run_data_quality()
