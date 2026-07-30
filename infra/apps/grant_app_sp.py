"""Grant the Databricks App's service principal the access it needs at runtime.

Run AFTER the app (and therefore its service principal) exists. Idempotent - safe
to re-run. Invoked automatically by deploy_app.sh, or manually:

    PYTHONPATH=backend python infra/apps/grant_app_sp.py --sp-client-id <uuid>

It grants each independently (one failure doesn't block the rest):

  1. Unity Catalog - USE CATALOG/SCHEMA + SELECT/MODIFY on the demo schema and
     READ VOLUME on the raw landing volumes, via the SQL warehouse.
  2. Lakebase - an OAuth Postgres role for the SP + CONNECT/USAGE/CREATE/CRUD on
     the serving schema, connecting as the Lakebase instance owner (you).
  3. Genie - best-effort CAN_RUN on the space the app queries by name.
  4. Card issuer (when enabled) - the same UC/Lakebase/Genie grants for the
     credit-card domain's OWN schema + volume + Genie space (both Genie lanes).

The connecting/granting identity is YOUR user (run_as from config.local.yaml),
who must own the catalog/schema, the Lakebase instance, and the Genie space.
"""
from __future__ import annotations

import argparse
import re

from genie_voice.config import get_settings


def _log(msg: str) -> None:
    print(f"[grant-sp] {msg}")


def _pg(name: str) -> str:
    """Quote a Postgres identifier."""
    return '"' + name.replace('"', '""') + '"'


def grant_unity_catalog(settings, sp: str) -> None:
    from genie_voice.databricks.warehouse_sql import execute_sql

    catalog = settings.databricks.catalog
    schema = settings.databricks.schema_name
    fq_schema = f"{catalog}.{schema}"
    batch_vol = f"{catalog}.{schema}.{settings.volume.batch_name}"
    stream_vol = f"{catalog}.{schema}.{settings.volume.streaming_name}"
    p = f"`{sp}`"
    stmts = [
        f"GRANT USE CATALOG ON CATALOG {catalog} TO {p}",
        f"GRANT USE SCHEMA ON SCHEMA {fq_schema} TO {p}",
        f"GRANT SELECT ON SCHEMA {fq_schema} TO {p}",
        f"GRANT MODIFY ON SCHEMA {fq_schema} TO {p}",
        f"GRANT READ VOLUME ON VOLUME {batch_vol} TO {p}",
        f"GRANT READ VOLUME ON VOLUME {stream_vol} TO {p}",
    ]
    for s in stmts:
        try:
            execute_sql(settings, s)
            _log(f"UC ok: {s}")
        except Exception as exc:  # noqa: BLE001
            _log(f"UC WARN ({s}): {exc}")


def grant_lakebase(settings, sp: str) -> None:
    if not settings.lakebase.enabled:
        _log("Lakebase disabled in config; skipping.")
        return
    from genie_voice.serve.lakebase import LakebaseServing

    lb = settings.lakebase
    schema = lb.schema_name
    serving = LakebaseServing(settings)
    stmts = [
        "CREATE EXTENSION IF NOT EXISTS databricks_auth",
        f"SELECT databricks_create_role('{sp}', 'SERVICE_PRINCIPAL')",
        f'GRANT CONNECT ON DATABASE {_pg(lb.database)} TO "{sp}"',
        f'GRANT CREATE ON DATABASE {_pg(lb.database)} TO "{sp}"',
        f'GRANT USAGE, CREATE ON SCHEMA {_pg(schema)} TO "{sp}"',
        f'GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA {_pg(schema)} TO "{sp}"',
        f'GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA {_pg(schema)} TO "{sp}"',
        f'ALTER DEFAULT PRIVILEGES IN SCHEMA {_pg(schema)} '
        f'GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO "{sp}"',
    ]
    try:
        with serving._conn() as conn, conn.cursor() as cur:  # noqa: SLF001
            for s in stmts:
                try:
                    cur.execute(s)
                    _log(f"LB ok: {s[:80]}")
                except Exception as exc:  # noqa: BLE001
                    # e.g. role already exists -> keep applying the rest.
                    _log(f"LB warn ({s[:60]}...): {exc}")
    except Exception as exc:  # noqa: BLE001
        _log(f"LB WARN: could not connect to Lakebase as owner: {exc}")


def _grant_genie_space(settings, sp: str, name: str) -> None:
    from genie_voice.databricks.client import get_workspace_client
    from genie_voice.genie.space import find_space_id

    client = get_workspace_client(settings)
    try:
        sid = find_space_id(client, name)
    except Exception as exc:  # noqa: BLE001
        _log(f"Genie WARN: could not list spaces: {exc}")
        return
    if not sid:
        _log(f"Genie: space '{name}' not found yet; create it, then rerun (or grant CAN_RUN in UI).")
        return
    try:
        client.api_client.do(
            "PATCH",
            f"/api/2.0/permissions/genie/{sid}",
            body={
                "access_control_list": [
                    {"service_principal_name": sp, "permission_level": "CAN_RUN"}
                ]
            },
        )
        _log(f"Genie ok: CAN_RUN on '{name}' ({sid})")
    except Exception as exc:  # noqa: BLE001
        _log(f"Genie WARN: grant failed ({exc}). Grant CAN_RUN on '{name}' to {sp} in the UI.")


def grant_genie(settings, sp: str) -> None:
    _grant_genie_space(settings, sp, settings.databricks.genie_space_name)


def grant_card_issuer(settings, sp: str) -> None:
    """Grant the card-issuer domain to the app SP (UC schema/volume, Lakebase
    serving schema, Genie space). Mirrors the telco grants; card lives in its OWN
    schema under the SAME catalog + Lakebase instance, so only the schema/volume/
    space names differ. No-op when card_issuer is disabled."""
    if not getattr(settings, "card_issuer", None) or not settings.card_issuer.enabled:
        _log("card_issuer disabled in config; skipping card grants.")
        return

    from genie_voice.databricks.warehouse_sql import execute_sql

    catalog = settings.databricks.catalog
    card_schema = settings.card_issuer.schema_name
    fq_schema = f"{catalog}.{card_schema}"
    batch_vol = f"{catalog}.{card_schema}.{settings.card_issuer.batch_volume}"
    p = f"`{sp}`"
    for s in [
        f"GRANT USE CATALOG ON CATALOG {catalog} TO {p}",
        f"GRANT USE SCHEMA ON SCHEMA {fq_schema} TO {p}",
        f"GRANT SELECT ON SCHEMA {fq_schema} TO {p}",
        f"GRANT READ VOLUME ON VOLUME {batch_vol} TO {p}",
    ]:
        try:
            execute_sql(settings, s)
            _log(f"card UC ok: {s}")
        except Exception as exc:  # noqa: BLE001
            _log(f"card UC WARN ({s}): {exc}")

    # Lakebase serving schema for the card fast-facts cache (same instance/db as
    # telco; the OAuth role was already created by grant_lakebase).
    if settings.lakebase.enabled:
        from genie_voice.serve.lakebase import LakebaseServing

        serving = LakebaseServing(settings)
        stmts = [
            f'GRANT USAGE, CREATE ON SCHEMA {_pg(card_schema)} TO "{sp}"',
            f'GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA {_pg(card_schema)} TO "{sp}"',
            f'GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA {_pg(card_schema)} TO "{sp}"',
            f'ALTER DEFAULT PRIVILEGES IN SCHEMA {_pg(card_schema)} '
            f'GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO "{sp}"',
        ]
        try:
            with serving._conn() as conn, conn.cursor() as cur:  # noqa: SLF001
                for s in stmts:
                    try:
                        cur.execute(s)
                        _log(f"card LB ok: {s[:80]}")
                    except Exception as exc:  # noqa: BLE001
                        _log(f"card LB warn ({s[:60]}...): {exc}")
        except Exception as exc:  # noqa: BLE001
            _log(f"card LB WARN: could not connect to Lakebase as owner: {exc}")

    # Genie space (both the Conversation + Agent-mode lanes read this space).
    _grant_genie_space(settings, sp, settings.card_issuer.genie_space_name)


def main() -> None:
    ap = argparse.ArgumentParser(description="Grant the app service principal its runtime access.")
    ap.add_argument("--sp-client-id", required=True, help="App service principal application (client) id")
    args = ap.parse_args()
    sp = args.sp_client_id.strip()
    if not re.fullmatch(r"[0-9a-fA-F-]{36}", sp):
        raise SystemExit(f"--sp-client-id does not look like a UUID: {sp!r}")

    settings = get_settings()
    _log(f"granting app service principal: {sp}")
    grant_unity_catalog(settings, sp)
    grant_lakebase(settings, sp)
    grant_genie(settings, sp)
    grant_card_issuer(settings, sp)
    _log("done")


if __name__ == "__main__":
    main()
