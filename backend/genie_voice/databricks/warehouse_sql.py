"""Unity Catalog writes via the SQL Statement Execution API (SQL warehouse)."""
from __future__ import annotations

from typing import Any

from genie_voice.config import Settings, get_settings


def _opt_float(value: Any) -> str | None:
    """Bindable string for a nullable DOUBLE column; None binds SQL NULL."""
    try:
        return str(float(value)) if value is not None else None
    except (TypeError, ValueError):
        return None


def _params(values: dict[str, str | None]) -> list[Any]:
    """Build named StatementParameterListItem rows (all bound as STRING, then CAST
    in SQL). Binding values as parameters - never string-interpolating them -
    removes injection risk for the live-assist write path. A None value binds SQL
    NULL, which is how nullable numeric columns stay empty instead of zeroed."""
    from databricks.sdk.service.sql import StatementParameterListItem

    return [
        StatementParameterListItem(name=name, value=value, type="STRING")
        for name, value in values.items()
    ]


def execute_sql(
    settings: Settings,
    statement: str,
    *,
    parameters: list[Any] | None = None,
    wait_timeout: str = "30s",
) -> None:
    wh = settings.databricks.sql_warehouse_id
    if not wh:
        raise RuntimeError("databricks.sql_warehouse_id is required for UC SQL writes.")
    from genie_voice.databricks.client import get_workspace_client

    client = get_workspace_client(settings)
    client.statement_execution.execute_statement(
        warehouse_id=wh,
        statement=statement,
        parameters=parameters or None,
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


def ensure_voice_traces_table(settings: Settings | None = None) -> None:
    """Idempotent governed UC Delta table: the retained system-of-record for
    voice observability traces (STT → LLM iterations + tool calls → TTS).

    Lakebase Postgres backs the live UI; this Delta table is the durable,
    SQL-queryable copy eval workflows read from. ``spans``/``trace`` are stored as
    JSON strings (parse with ``from_json`` at query time)."""
    settings = settings or get_settings()
    tbl = settings.fqtn("voice_traces")
    execute_sql(
        settings,
        f"""
        CREATE TABLE IF NOT EXISTS {tbl} (
            trace_id                    STRING,
            session_id                  STRING,
            turn_id                     INT,
            call_id                     STRING,
            customer_id                 STRING,
            capability                  STRING,
            language                    STRING,
            detected_language           STRING,
            status                      STRING,
            input_transcript            STRING,
            output_text                 STRING,
            tool_names                  STRING,
            apply_billing_action_called BOOLEAN,
            lookup_account_count        INT,
            llm_iterations              INT,
            ttft_ms                     DOUBLE,
            answer_ttft_ms              DOUBLE,
            tts_first_ms                DOUBLE,
            server_ttfb_ms              DOUBLE,
            server_gen_ms               DOUBLE,
            total_ms                    DOUBLE,
            guard_roster                STRING,
            trace                       STRING,
            created_at                  TIMESTAMP
        ) USING DELTA
        """,
    )
    # Columns added after the table shipped. An existing table needs them
    # backfilled into its schema or every mirrored insert would fail — silently,
    # since the mirror is best-effort (lakebase.py `_mirror_trace_to_uc`).
    execute_sql(
        settings,
        f"""
        ALTER TABLE {tbl} ADD COLUMNS IF NOT EXISTS (
            ttft_ms        DOUBLE COMMENT 'Time to any audio; a latency filler ends the silence early',
            answer_ttft_ms DOUBLE COMMENT 'Time until the caller heard the actual reply',
            tts_first_ms   DOUBLE COMMENT 'TTS-local time to the answer first chunk',
            server_ttfb_ms DOUBLE COMMENT 'TTS endpoint time to its own first chunk',
            server_gen_ms  DOUBLE COMMENT 'TTS endpoint full synthesis time',
            guard_roster   STRING COMMENT 'JSON array: every guardrail check on the turn, incl. passed/delegated'
        )
        """,
    )


def insert_voice_trace_uc(settings: Settings, trace: dict[str, Any]) -> dict[str, Any]:
    """Append one turn trace to the governed UC Delta table.

    Governed write boundary: every value is bound as a named STRING parameter and
    cast to its typed column in SQL (never string-interpolated); the table
    identifier comes from trusted config only.
    """
    import json

    tbl = settings.fqtn("voice_traces")
    stmt = f"""
        INSERT INTO {tbl} (
          trace_id, session_id, turn_id, call_id, customer_id, capability,
          language, detected_language, status, input_transcript, output_text,
          tool_names, apply_billing_action_called, lookup_account_count,
          llm_iterations, ttft_ms, answer_ttft_ms, tts_first_ms, server_ttfb_ms,
          server_gen_ms, total_ms, guard_roster, trace, created_at
        ) VALUES (
          :trace_id, :session_id, CAST(:turn_id AS INT), :call_id, :customer_id, :capability,
          :language, :detected_language, :status, :input_transcript, :output_text,
          :tool_names, CAST(:apply_billing_action_called AS BOOLEAN), CAST(:lookup_account_count AS INT),
          CAST(:llm_iterations AS INT), CAST(:ttft_ms AS DOUBLE),
          CAST(:answer_ttft_ms AS DOUBLE), CAST(:tts_first_ms AS DOUBLE),
          CAST(:server_ttfb_ms AS DOUBLE), CAST(:server_gen_ms AS DOUBLE),
          CAST(:total_ms AS DOUBLE), :guard_roster, :trace, current_timestamp()
        )
    """
    params = _params(
        {
            "trace_id": str(trace.get("trace_id") or ""),
            "session_id": str(trace.get("session_id") or ""),
            "turn_id": str(int(trace.get("turn_id") or 0)),
            "call_id": str(trace.get("call_id") or ""),
            "customer_id": str(trace.get("customer_id") or ""),
            "capability": str(trace.get("capability") or ""),
            "language": str(trace.get("language") or ""),
            "detected_language": str(trace.get("detected_language") or ""),
            "status": str(trace.get("status") or ""),
            "input_transcript": str(trace.get("input_transcript") or ""),
            "output_text": str(trace.get("output_text") or ""),
            "tool_names": json.dumps(trace.get("tool_names") or []),
            "apply_billing_action_called": str(bool(trace.get("apply_billing_action_called"))).lower(),
            "lookup_account_count": str(int(trace.get("lookup_account_count") or 0)),
            "llm_iterations": str(int(trace.get("llm_iterations") or 0)),
            # Left NULL rather than zeroed when a trace never spoke (e.g. an
            # Agent-Mode deep dive), so latency aggregates ignore it instead of
            # being dragged down by a fake 0ms.
            "ttft_ms": _opt_float(trace.get("ttft_ms")),
            "answer_ttft_ms": _opt_float(trace.get("answer_ttft_ms")),
            "tts_first_ms": _opt_float(trace.get("tts_first_ms")),
            "server_ttfb_ms": _opt_float(trace.get("server_ttfb_ms")),
            "server_gen_ms": _opt_float(trace.get("server_gen_ms")),
            "total_ms": str(float(trace.get("total_ms") or 0.0)),
            "guard_roster": json.dumps(trace.get("guard_roster") or []),
            "trace": json.dumps(trace),
        }
    )
    execute_sql(settings, stmt, parameters=params)
    return {"ok": True, "trace_id": trace.get("trace_id")}


def apply_billing_resolution_uc(
    settings: Settings,
    adjustment: dict[str, Any],
) -> dict[str, Any]:
    """Persist adjustment audit row and apply invoice mutation in UC.

    Governed write boundary: all caller-supplied values are bound as named
    parameters (never string-interpolated) and cast to typed columns in SQL. Table
    identifiers come from trusted config (settings.fqtn) only.
    """
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
            :adjustment_id AS adjustment_id,
            :call_id AS call_id,
            :customer_id AS customer_id,
            :invoice_id AS invoice_id,
            CAST(:waiver_applied AS BOOLEAN) AS waiver_applied,
            CAST(:payment_plan_applied AS BOOLEAN) AS payment_plan_applied,
            CAST(:amount_before AS DECIMAL(10,2)) AS amount_before,
            CAST(:late_fee_before AS DECIMAL(10,2)) AS late_fee_before,
            :status_before AS status_before,
            CAST(:amount_after AS DECIMAL(10,2)) AS amount_after,
            CAST(:late_fee_after AS DECIMAL(10,2)) AS late_fee_after,
            :status_after AS status_after,
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
    merge_params = _params(
        {
            "adjustment_id": adjustment_id,
            "call_id": call_id,
            "customer_id": customer_id,
            "invoice_id": invoice_id,
            "waiver_applied": str(bool(adjustment.get("waiver_applied"))).lower(),
            "payment_plan_applied": str(bool(adjustment.get("payment_plan_applied"))).lower(),
            "amount_before": f"{float(adjustment['amount_before']):.2f}",
            "late_fee_before": f"{float(adjustment['late_fee_before']):.2f}",
            "status_before": str(adjustment["status_before"]),
            "amount_after": f"{float(adjustment['amount_after']):.2f}",
            "late_fee_after": f"{float(adjustment['late_fee_after']):.2f}",
            "status_after": str(adjustment["status_after"]),
        }
    )

    update_inv = f"""
        UPDATE {invoices}
        SET amount = CAST(:amount_after AS DECIMAL(10,2)),
            late_fee = CAST(:late_fee_after AS DECIMAL(10,2)),
            status = :status_after
        WHERE customer_id = :customer_id
          AND invoice_id = :invoice_id
    """
    update_params = _params(
        {
            "amount_after": f"{float(adjustment['amount_after']):.2f}",
            "late_fee_after": f"{float(adjustment['late_fee_after']):.2f}",
            "status_after": str(adjustment["status_after"]),
            "customer_id": customer_id,
            "invoice_id": invoice_id,
        }
    )

    execute_sql(settings, insert_adj, parameters=merge_params)
    execute_sql(settings, update_inv, parameters=update_params)
    return {"ok": True, "adjustment_id": adjustment_id, "invoice_id": invoice_id}


def revert_billing_resolution_uc(
    settings: Settings,
    adjustment: dict[str, Any],
) -> dict[str, Any]:
    """Restore invoice values and mark the UC adjustment row reverted.

    Same governed-write rules as apply: values bound as parameters, identifiers
    from trusted config only.
    """
    invoices = settings.fqtn("invoices")
    adjustments = settings.fqtn("billing_adjustments")
    customer_id = str(adjustment["customer_id"])
    invoice_id = str(adjustment["invoice_id"])
    adjustment_id = str(adjustment.get("adjustment_id") or f"{adjustment.get('call_id')}-{invoice_id}")

    update_inv = f"""
        UPDATE {invoices}
        SET amount = CAST(:amount_before AS DECIMAL(10,2)),
            late_fee = CAST(:late_fee_before AS DECIMAL(10,2)),
            status = :status_before
        WHERE customer_id = :customer_id
          AND invoice_id = :invoice_id
    """
    update_params = _params(
        {
            "amount_before": f"{float(adjustment['amount_before']):.2f}",
            "late_fee_before": f"{float(adjustment['late_fee_before']):.2f}",
            "status_before": str(adjustment["status_before"]),
            "customer_id": customer_id,
            "invoice_id": invoice_id,
        }
    )

    mark_reverted = f"""
        UPDATE {adjustments}
        SET reverted_at = current_timestamp()
        WHERE adjustment_id = :adjustment_id
          AND reverted_at IS NULL
    """
    execute_sql(settings, update_inv, parameters=update_params)
    execute_sql(settings, mark_reverted, parameters=_params({"adjustment_id": adjustment_id}))
    return {"ok": True, "adjustment_id": adjustment_id, "invoice_id": invoice_id}
