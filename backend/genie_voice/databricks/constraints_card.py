"""Apply UC informational PK/FK metadata to the card tables after ingest.

Card-issuer analogue of ``constraints.py``. CREATE OR REPLACE (in the ingest
step) drops constraints, so this re-adds them: all primary keys first, then
foreign keys (parents already exist), as NOT ENFORCED informational metadata so
Genie can infer joins for multi-step agent-mode reasoning.
"""
from __future__ import annotations

from genie_voice.config import Settings, get_settings
from genie_voice.databricks.client import get_workspace_client
from genie_voice.databricks.constraints import _exec, _q
from genie_voice.datagen.card.schema_card import CARD_ALL_TABLES, CARD_MODEL


def apply_card_constraints(settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    if not settings.card_issuer.enabled:
        print("card_issuer.enabled is false; skipping card constraints.")
        return
    wh = settings.databricks.sql_warehouse_id
    if not wh:
        raise RuntimeError("databricks.sql_warehouse_id is required to add card UC constraints.")

    client = get_workspace_client(settings)
    print("Applying card UC informational PK/FK constraints for Genie ...")

    for table in CARD_ALL_TABLES:
        spec = CARD_MODEL[table]
        if not spec.primary_key:
            continue
        fq = settings.card_fqtn(table)
        cols = ", ".join(_q(col) for col in spec.primary_key)
        for col in spec.primary_key:
            _exec(client, wh, f"ALTER TABLE {fq} ALTER COLUMN {_q(col)} SET NOT NULL",
                  f"set not null {table}.{col}")
        _exec(client, wh, f"ALTER TABLE {fq} DROP CONSTRAINT pk_{table}",
              f"drop existing primary key {table}", required=False)
        _exec(client, wh, f"ALTER TABLE {fq} ADD CONSTRAINT pk_{table} PRIMARY KEY ({cols}) NOT ENFORCED",
              f"primary key {table}")

    for table in CARD_ALL_TABLES:
        spec = CARD_MODEL[table]
        for fk in spec.foreign_keys:
            fq = settings.card_fqtn(table)
            _exec(client, wh, f"ALTER TABLE {fq} DROP CONSTRAINT fk_{table}_{fk.column}",
                  f"drop existing foreign key {table}.{fk.column}", required=False)
            _exec(
                client, wh,
                f"ALTER TABLE {fq} ADD CONSTRAINT fk_{table}_{fk.column} "
                f"FOREIGN KEY ({_q(fk.column)}) "
                f"REFERENCES {settings.card_fqtn(fk.ref_table)}({_q(fk.ref_column)}) NOT ENFORCED",
                f"foreign key {table}.{fk.column}",
            )


if __name__ == "__main__":
    apply_card_constraints()
