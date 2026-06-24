"""Bootstrap + permission setup for Unity Catalog objects.

Idempotent and safe to run repeatedly (called by local-deploy.sh). Uses the SQL
Statement Execution API via the SDK against the configured SQL warehouse.

What it does:
  1. Verifies who we're authenticated as (U2M user by default).
  2. Creates the schema + raw landing Volumes inside an EXISTING catalog
     (never CREATE CATALOG unless databricks.create_catalog=true).
  3. Grants the running identity the privileges the app needs (USE CATALOG +
     ALL PRIVILEGES on the schema, covering tables/volume read+write). Grants are
     applied defensively: a failure (e.g. you already own the object) is logged,
     not fatal.
  4. Optionally creates typed modeled tables for legacy/manual runs. The normal
     deploy skips this because job tasks own the current datasets.
"""
from __future__ import annotations

import os

from genie_voice.config import Settings, get_settings
from genie_voice.databricks.client import current_user, get_workspace_client
from genie_voice.datagen.schema import ALL_TABLES, MODEL


def _exec(client, warehouse_id: str, statement: str) -> None:
    client.statement_execution.execute_statement(
        warehouse_id=warehouse_id, statement=statement, wait_timeout="30s"
    )


def _try(client, wh: str, statement: str, label: str) -> None:
    try:
        _exec(client, wh, statement)
        print(f"  ok: {label}")
    except Exception as exc:  # noqa: BLE001
        print(f"  skip: {label} ({exc})")


def ensure_tables(client, wh: str, settings: Settings) -> None:
    """Create the typed modeled tables (ALL_TABLES = reference + derived) with
    comments + informational PK/FK constraints. Parents are created first so
    FOREIGN KEY ... REFERENCES resolves. Idempotent (CREATE TABLE IF NOT EXISTS)."""
    for table in ALL_TABLES:
        _try(client, wh, MODEL[table].render_ddl(settings.fqtn), f"ensure table {table}")


def bootstrap(settings: Settings | None = None) -> dict[str, str]:
    settings = settings or get_settings()
    client = get_workspace_client(settings)
    wh = settings.databricks.sql_warehouse_id
    if not wh:
        raise RuntimeError("databricks.sql_warehouse_id is required to bootstrap UC objects.")

    cat = settings.databricks.catalog
    sch = settings.databricks.schema_name
    batch_vol = settings.volume.batch_name
    streaming_vol = settings.volume.streaming_name

    me = current_user(client)
    principal = settings.databricks.run_as or me
    print(f"Authenticated as: {me or '(unknown)'}; granting to: {principal or '(none)'}")

    # 1. Catalog (only if explicitly allowed).
    if settings.databricks.create_catalog:
        _try(client, wh, f"CREATE CATALOG IF NOT EXISTS {cat}", f"create catalog {cat}")
    else:
        print(f"  using existing catalog: {cat} (create_catalog=false)")

    # 2. Schema + Volume.
    _try(client, wh, f"CREATE SCHEMA IF NOT EXISTS {cat}.{sch}", f"create schema {cat}.{sch}")
    _try(client, wh, f"CREATE VOLUME IF NOT EXISTS {cat}.{sch}.{batch_vol}", f"create volume {batch_vol}")
    _try(
        client,
        wh,
        f"CREATE VOLUME IF NOT EXISTS {cat}.{sch}.{streaming_vol}",
        f"create volume {streaming_vol}",
    )

    # 3. Privileges for the running identity (idempotent / defensive).
    if principal:
        p = f"`{principal}`"
        _try(client, wh, f"GRANT USE CATALOG ON CATALOG {cat} TO {p}", "use catalog")
        _try(client, wh, f"GRANT ALL PRIVILEGES ON SCHEMA {cat}.{sch} TO {p}", "all privileges on schema")

    # 4. Legacy typed table DDL. The task-based deploy owns the current tables,
    #    so local-deploy skips this by default.
    if os.environ.get("GENIE_SKIP_TABLE_BOOTSTRAP", "false").lower() in ("1", "true", "yes"):
        print("  skipping table/view bootstrap (orchestration tasks own datasets)")
    else:
        ensure_tables(client, wh, settings)

    return {
        "authenticated_as": me,
        "principal": principal,
        "catalog": cat,
        "schema": sch,
        "batch_volume": batch_vol,
        "streaming_volume": streaming_vol,
        "reference_path": settings.reference_path,
        "raw_stt_path": settings.raw_stt_path,
    }


if __name__ == "__main__":
    info = bootstrap()
    print("bootstrap complete:", info)
