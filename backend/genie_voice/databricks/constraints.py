"""Apply UC informational PK/FK metadata after table refresh tasks complete."""
from __future__ import annotations

from genie_voice.config import Settings, get_settings
from genie_voice.databricks.client import get_workspace_client
from genie_voice.datagen.schema import ALL_TABLES, MODEL


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


def _exec(client, warehouse_id: str, statement: str, label: str, *, required: bool = True) -> bool:
    try:
        client.statement_execution.execute_statement(
            warehouse_id=warehouse_id,
            statement=statement,
            wait_timeout="30s",
        )
        print(f"  ok: {label}")
        return True
    except Exception as exc:  # noqa: BLE001
        if required:
            raise RuntimeError(f"Failed to apply UC constraint step '{label}': {exc}") from exc
        print(f"  skip: {label} ({exc})")
        return False


def apply_constraints(settings: Settings | None = None) -> None:
    """Add Genie-visible relationship metadata in dependency order.

    Refresh tasks create the physical Delta tables first. This task then adds
    informational constraints: all primary keys first, then foreign keys. That
    avoids the child-before-parent issue that broke `payments`.
    """
    settings = settings or get_settings()
    wh = settings.databricks.sql_warehouse_id
    if not wh:
        raise RuntimeError("databricks.sql_warehouse_id is required to add UC constraints.")

    client = get_workspace_client(settings)
    print("Applying UC informational PK/FK constraints for Genie ...")

    for table in ALL_TABLES:
        spec = MODEL[table]
        if not spec.primary_key:
            continue
        fq = _fqtn(settings, table)
        cols = ", ".join(_q(col) for col in spec.primary_key)
        for col in spec.primary_key:
            _exec(
                client,
                wh,
                f"ALTER TABLE {fq} ALTER COLUMN {_q(col)} SET NOT NULL",
                f"set not null {table}.{col}",
            )
        _exec(
            client,
            wh,
            f"ALTER TABLE {fq} DROP CONSTRAINT pk_{table}",
            f"drop existing primary key {table}",
            required=False,
        )
        _exec(
            client,
            wh,
            f"ALTER TABLE {fq} ADD CONSTRAINT pk_{table} "
            f"PRIMARY KEY ({cols}) NOT ENFORCED",
            f"primary key {table}",
        )

    for table in ALL_TABLES:
        spec = MODEL[table]
        for fk in spec.foreign_keys:
            fq = _fqtn(settings, table)
            _exec(
                client,
                wh,
                f"ALTER TABLE {fq} DROP CONSTRAINT fk_{table}_{fk.column}",
                f"drop existing foreign key {table}.{fk.column}",
                required=False,
            )
            _exec(
                client,
                wh,
                f"ALTER TABLE {fq} "
                f"ADD CONSTRAINT fk_{table}_{fk.column} "
                f"FOREIGN KEY ({_q(fk.column)}) "
                f"REFERENCES {_fqtn(settings, fk.ref_table)}({_q(fk.ref_column)}) "
                f"NOT ENFORCED",
                f"foreign key {table}.{fk.column}",
            )


if __name__ == "__main__":
    apply_constraints()
