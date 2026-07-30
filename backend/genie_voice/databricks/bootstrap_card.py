"""Bootstrap the credit-card issuer UC objects.

A card-issuer analogue of ``bootstrap.py`` that reuses the same low-level SQL
execution + grant helpers, but targets the card-issuer schema + volume so the
contact-center schema is never touched. Idempotent and safe to re-run.

CLI:  python -m genie_voice.databricks.bootstrap_card [--skip-tables]
"""
from __future__ import annotations

from genie_voice.config import Settings, get_settings
from genie_voice.databricks.bootstrap import _try
from genie_voice.databricks.client import current_user, get_workspace_client
from genie_voice.datagen.card.schema_card import CARD_ALL_TABLES, CARD_MODEL


def ensure_card_tables(client, wh: str, settings: Settings) -> None:
    """Create the typed card tables (comments + informational PK/FK). Parents are
    listed first in CARD_ALL_TABLES so FOREIGN KEY ... REFERENCES resolves."""
    for table in CARD_ALL_TABLES:
        _try(client, wh, CARD_MODEL[table].render_ddl(settings.card_fqtn), f"ensure card table {table}")


def bootstrap_card(settings: Settings | None = None, *, skip_tables: bool = False) -> dict[str, str]:
    settings = settings or get_settings()
    if not settings.card_issuer.enabled:
        print("card_issuer.enabled is false; skipping card-issuer bootstrap.")
        return {"enabled": "false"}

    client = get_workspace_client(settings)
    wh = settings.databricks.sql_warehouse_id
    if not wh:
        raise RuntimeError("databricks.sql_warehouse_id is required to bootstrap card-issuer UC objects.")

    cat = settings.databricks.catalog
    sch = settings.card_issuer.schema_name
    batch_vol = settings.card_issuer.batch_volume

    me = current_user(client)
    principal = settings.databricks.run_as or me
    print(f"Authenticated as: {me or '(unknown)'}; granting to: {principal or '(none)'}")

    # Reuse the EXISTING catalog (never CREATE CATALOG here).
    print(f"  using existing catalog: {cat}")
    _try(client, wh, f"CREATE SCHEMA IF NOT EXISTS {cat}.{sch}", f"create schema {cat}.{sch}")
    _try(client, wh, f"CREATE VOLUME IF NOT EXISTS {cat}.{sch}.{batch_vol}", f"create volume {batch_vol}")

    if principal:
        p = f"`{principal}`"
        _try(client, wh, f"GRANT USE CATALOG ON CATALOG {cat} TO {p}", "use catalog")
        _try(client, wh, f"GRANT ALL PRIVILEGES ON SCHEMA {cat}.{sch} TO {p}", "all privileges on schema")

    if skip_tables:
        print("  skipping card table bootstrap (--skip-tables)")
    else:
        ensure_card_tables(client, wh, settings)

    return {
        "authenticated_as": me,
        "principal": principal,
        "catalog": cat,
        "schema": sch,
        "batch_volume": batch_vol,
        "reference_path": settings.card_reference_path,
    }


if __name__ == "__main__":
    import sys

    info = bootstrap_card(skip_tables="--skip-tables" in sys.argv)
    print("card bootstrap complete:", info)
