"""Batch-ingest the card-issuer reference files from the card volume into UC Delta.

Card-issuer analogue of ``reference_ingest.py`` (Path B: UC Delta is the source
of truth for read-only account data). Reuses the same ``read_files`` +
CREATE OR REPLACE pattern so the flow matches billing support exactly, but reads
from the card volume and writes into the card schema.
"""
from __future__ import annotations

from genie_voice.config import Settings, get_settings
from genie_voice.databricks.client import get_workspace_client
from genie_voice.databricks.reference_ingest import _exec, _q, _sql_string
from genie_voice.datagen.card.schema_card import CARD_MODEL, CARD_REFERENCE_TABLES


def ingest_card_reference_tables(settings: Settings | None = None) -> dict[str, str]:
    """Overwrite card UC reference tables from the deterministic card volume JSON."""
    settings = settings or get_settings()
    if not settings.card_issuer.enabled:
        print("card_issuer.enabled is false; skipping card reference ingest.")
        return {}
    wh = settings.databricks.sql_warehouse_id
    if not wh:
        raise RuntimeError("databricks.sql_warehouse_id is required for card reference ingest.")

    client = get_workspace_client(settings)
    out: dict[str, str] = {}
    print("Ingesting card reference tables from the card volume into UC Delta ...")
    for table in CARD_REFERENCE_TABLES:
        spec = CARD_MODEL[table]
        fq = settings.card_fqtn(table)
        path = settings.card_reference_table_path(table)
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
    ingest_card_reference_tables()
