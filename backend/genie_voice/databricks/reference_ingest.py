"""Batch-ingest reference/customer/billing files from raw_batch_data into UC."""
from __future__ import annotations

from genie_voice.config import Settings, get_settings
from genie_voice.databricks.client import get_workspace_client
from genie_voice.datagen.schema import MODEL, REFERENCE_TABLES


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


def _exec(client, warehouse_id: str, statement: str) -> None:
    client.statement_execution.execute_statement(
        warehouse_id=warehouse_id,
        statement=statement,
        wait_timeout="50s",
    )


def ingest_reference_tables(settings: Settings | None = None) -> dict[str, str]:
    """Overwrite UC reference tables from deterministic raw_batch_data JSON files."""
    settings = settings or get_settings()
    wh = settings.databricks.sql_warehouse_id
    if not wh:
        raise RuntimeError("databricks.sql_warehouse_id is required for reference ingest.")

    client = get_workspace_client(settings)
    out: dict[str, str] = {}
    print("Ingesting reference tables from raw_batch_data into UC Delta ...")
    for table in REFERENCE_TABLES:
        spec = MODEL[table]
        fq = _fqtn(settings, table)
        path = settings.reference_table_path(table)
        casts = ", ".join(f"CAST({c.name} AS {c.type}) AS {c.name}" for c in spec.columns)
        _exec(
            client,
            wh,
            f"""
            CREATE OR REPLACE TABLE {fq}
            AS SELECT {casts}
            FROM read_files({_sql_string(path)}, format => {_sql_string(settings.pipeline.source_format)})
            """,
        )
        if spec.properties:
            props = ", ".join(
                f"{_sql_string(k)} = {_sql_string(v)}" for k, v in sorted(spec.properties.items())
            )
            _exec(client, wh, f"ALTER TABLE {fq} SET TBLPROPERTIES ({props})")
        _exec(client, wh, f"COMMENT ON TABLE {fq} IS {_sql_string(spec.comment)}")
        for col in spec.columns:
            _exec(
                client,
                wh,
                f"ALTER TABLE {fq} ALTER COLUMN {_q(col.name)} COMMENT {_sql_string(col.comment)}",
            )
        out[table] = path
        print(f"  ok: {table} <- {path}")
    return out


if __name__ == "__main__":
    ingest_reference_tables()
