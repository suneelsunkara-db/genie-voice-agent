"""Unity Catalog writes via the SQL Statement Execution API (SQL warehouse)."""
from __future__ import annotations

from typing import Any

from genie_voice.config import Settings, get_settings


def _sql_str(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def execute_sql(settings: Settings, statement: str, *, wait_timeout: str = "30s") -> None:
    wh = settings.databricks.sql_warehouse_id
    if not wh:
        raise RuntimeError("databricks.sql_warehouse_id is required for UC SQL writes.")
    from genie_voice.databricks.client import get_workspace_client

    client = get_workspace_client(settings)
    client.statement_execution.execute_statement(
        warehouse_id=wh,
        statement=statement,
        wait_timeout=wait_timeout,
    )


def warehouse_configured(settings: Settings | None = None) -> bool:
    settings = settings or get_settings()
    return bool(settings.databricks.sql_warehouse_id)


def ensure_billing_adjustments_table(settings: Settings | None = None) -> None:
    """Idempotent UC table for live-assist billing adjustments (audit + Genie)."""
    from genie_voice.datagen.schema import MODEL, T_BILLING_ADJUSTMENTS

    settings = settings or get_settings()
    execute_sql(settings, MODEL[T_BILLING_ADJUSTMENTS].render_ddl(settings.fqtn))


def apply_billing_resolution_uc(
    settings: Settings,
    adjustment: dict[str, Any],
) -> dict[str, Any]:
    """Persist adjustment audit row and apply invoice mutation in UC."""
    invoices = settings.fqtn("invoices")
    adjustments = settings.fqtn("billing_adjustments")
    customer_id = str(adjustment["customer_id"])
    invoice_id = str(adjustment["invoice_id"])
    adjustment_id = str(adjustment["adjustment_id"])
    call_id = str(adjustment["call_id"])

    insert_adj = f"""
        MERGE INTO {adjustments} AS t
        USING (
          SELECT
            {_sql_str(adjustment_id)} AS adjustment_id,
            {_sql_str(call_id)} AS call_id,
            {_sql_str(customer_id)} AS customer_id,
            {_sql_str(invoice_id)} AS invoice_id,
            {bool(adjustment.get('waiver_applied'))} AS waiver_applied,
            {bool(adjustment.get('payment_plan_applied'))} AS payment_plan_applied,
            {float(adjustment['amount_before']):.2f} AS amount_before,
            {float(adjustment['late_fee_before']):.2f} AS late_fee_before,
            {_sql_str(str(adjustment['status_before']))} AS status_before,
            {float(adjustment['amount_after']):.2f} AS amount_after,
            {float(adjustment['late_fee_after']):.2f} AS late_fee_after,
            {_sql_str(str(adjustment['status_after']))} AS status_after,
            current_timestamp() AS applied_at,
            CAST(NULL AS TIMESTAMP) AS reverted_at
        ) AS s
        ON t.adjustment_id = s.adjustment_id
        WHEN MATCHED THEN UPDATE SET
          amount_after = s.amount_after,
          late_fee_after = s.late_fee_after,
          status_after = s.status_after,
          applied_at = s.applied_at,
          reverted_at = NULL
        WHEN NOT MATCHED THEN INSERT (
          adjustment_id, call_id, customer_id, invoice_id,
          waiver_applied, payment_plan_applied,
          amount_before, late_fee_before, status_before,
          amount_after, late_fee_after, status_after,
          applied_at, reverted_at
        ) VALUES (
          s.adjustment_id, s.call_id, s.customer_id, s.invoice_id,
          s.waiver_applied, s.payment_plan_applied,
          s.amount_before, s.late_fee_before, s.status_before,
          s.amount_after, s.late_fee_after, s.status_after,
          s.applied_at, s.reverted_at
        )
    """

    update_inv = f"""
        UPDATE {invoices}
        SET amount = {float(adjustment['amount_after']):.2f},
            late_fee = {float(adjustment['late_fee_after']):.2f},
            status = {_sql_str(str(adjustment['status_after']))}
        WHERE customer_id = {_sql_str(customer_id)}
          AND invoice_id = {_sql_str(invoice_id)}
    """

    execute_sql(settings, insert_adj)
    execute_sql(settings, update_inv)
    return {"ok": True, "adjustment_id": adjustment_id, "invoice_id": invoice_id}


def revert_billing_resolution_uc(
    settings: Settings,
    adjustment: dict[str, Any],
) -> dict[str, Any]:
    """Restore invoice values and mark the UC adjustment row reverted."""
    invoices = settings.fqtn("invoices")
    adjustments = settings.fqtn("billing_adjustments")
    customer_id = str(adjustment["customer_id"])
    invoice_id = str(adjustment["invoice_id"])
    adjustment_id = str(adjustment.get("adjustment_id") or f"{adjustment.get('call_id')}-{invoice_id}")

    update_inv = f"""
        UPDATE {invoices}
        SET amount = {float(adjustment['amount_before']):.2f},
            late_fee = {float(adjustment['late_fee_before']):.2f},
            status = {_sql_str(str(adjustment['status_before']))}
        WHERE customer_id = {_sql_str(customer_id)}
          AND invoice_id = {_sql_str(invoice_id)}
    """
    mark_reverted = f"""
        UPDATE {adjustments}
        SET reverted_at = current_timestamp()
        WHERE adjustment_id = {_sql_str(adjustment_id)}
          AND reverted_at IS NULL
    """
    execute_sql(settings, update_inv)
    execute_sql(settings, mark_reverted)
    return {"ok": True, "adjustment_id": adjustment_id, "invoice_id": invoice_id}
