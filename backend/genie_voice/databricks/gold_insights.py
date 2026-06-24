"""Refresh final UC gold_call_insights as a regular Delta table."""
from __future__ import annotations

import json

from genie_voice.config import Settings, get_settings
from genie_voice.databricks.client import get_workspace_client
from genie_voice.datagen.schema import MODEL, T_GOLD
from genie_voice.enrich.fm import CALL_INSTRUCTION, SYSTEM_PROMPT, call_json_schema


def _q(name: str) -> str:
    return "`" + name.replace("`", "``") + "`"


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


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


def _exec(client, warehouse_id: str, statement: str) -> None:
    client.statement_execution.execute_statement(
        warehouse_id=warehouse_id,
        statement=statement,
        wait_timeout="50s",
    )


def refresh_gold_insights(settings: Settings | None = None) -> dict[str, str]:
    settings = settings or get_settings()
    wh = settings.databricks.sql_warehouse_id
    if not wh:
        raise RuntimeError("databricks.sql_warehouse_id is required for gold refresh.")

    client = get_workspace_client(settings)
    target = _fqtn(settings, T_GOLD)
    facts_history = _history_fqtn(settings, "call_facts")
    utterance_history = _history_fqtn(settings, "live_call_utterances")
    endpoint = settings.enrichment.model_endpoint.replace("'", "''")
    prompt = (f"{SYSTEM_PROMPT}\n{CALL_INSTRUCTION}\n\nTRANSCRIPT:\n").replace("'", "''")
    response_schema = json.dumps(call_json_schema()).replace("'", "''")
    struct = (
        "STRUCT<primary_intent:STRING, all_intents:ARRAY<STRING>, "
        "sentiment_score:DOUBLE, sentiment_label:STRING, disposition:STRING, "
        "resolution_status:STRING, next_best_action:STRING, "
        "mentioned_invoice_id:STRING, mentioned_amount:DOUBLE, summary:STRING>"
    )

    print("Refreshing gold_call_insights as final UC Delta table ...")
    _exec(
        client,
        wh,
        f"""
        CREATE OR REPLACE TABLE {target}
        AS
        SELECT
          CAST(call_id AS STRING) AS call_id,
          CAST(customer_id AS STRING) AS customer_id,
          CAST(o.primary_intent AS STRING) AS primary_intent,
          CAST(o.all_intents AS ARRAY<STRING>) AS all_intents,
          CAST(o.sentiment_score AS DOUBLE) AS sentiment_score,
          CAST(o.sentiment_label AS STRING) AS sentiment_label,
          CAST(o.disposition AS STRING) AS disposition,
          CAST(o.resolution_status AS STRING) AS resolution_status,
          CAST(o.next_best_action AS STRING) AS next_best_action,
          CAST(o.mentioned_invoice_id AS STRING) AS mentioned_invoice_id,
          CAST(o.mentioned_amount AS DECIMAL(10,2)) AS mentioned_amount,
          CAST(o.summary AS STRING) AS summary
        FROM (
          SELECT call_id, customer_id,
                 from_json(
                   ai_query('{endpoint}', concat('{prompt}', transcript),
                            responseFormat => '{response_schema}'),
                   '{struct}') AS o
          FROM (
            SELECT s.call_id, f.customer_id,
                   array_join(
                     transform(
                       array_sort(collect_list(struct(
                         s.start_sec AS start_sec,
                         concat(s.speaker_role, ': ', s.text) AS line))),
                       x -> x.line), '\n') AS transcript
            FROM (
              SELECT *
              FROM (
                SELECT *,
                       row_number() OVER (PARTITION BY utterance_id ORDER BY _sort_by DESC) AS _rn
                FROM {utterance_history}
                WHERE _pg_change_type IN ('insert', 'update_postimage', 'delete')
              )
              WHERE _rn = 1 AND _pg_change_type <> 'delete'
            ) s
            JOIN (
              SELECT *
              FROM (
                SELECT *,
                       row_number() OVER (PARTITION BY call_id ORDER BY _sort_by DESC) AS _rn
                FROM {facts_history}
                WHERE _pg_change_type IN ('insert', 'update_postimage', 'delete')
              )
              WHERE _rn = 1 AND _pg_change_type <> 'delete'
            ) f ON s.call_id = f.call_id
            GROUP BY s.call_id, f.customer_id
          )
        )
        """,
    )

    spec = MODEL[T_GOLD]
    if spec.properties:
        props = ", ".join(
            f"{_sql_string(k)} = {_sql_string(v)}" for k, v in sorted(spec.properties.items())
        )
        _exec(client, wh, f"ALTER TABLE {target} SET TBLPROPERTIES ({props})")
    _exec(client, wh, f"COMMENT ON TABLE {target} IS {_sql_string(spec.comment)}")
    for col in spec.columns:
        _exec(
            client,
            wh,
            f"ALTER TABLE {target} ALTER COLUMN {_q(col.name)} COMMENT {_sql_string(col.comment)}",
        )
    print(f"  ok: {T_GOLD} <- {utterance_history} + {facts_history}")
    return {T_GOLD: target}


if __name__ == "__main__":
    refresh_gold_insights()
