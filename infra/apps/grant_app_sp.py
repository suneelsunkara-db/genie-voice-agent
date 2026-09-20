"""Grant the Databricks App's service principal the access it needs at runtime.

Run AFTER the app (and therefore its service principal) exists. Idempotent - safe
to re-run. Invoked automatically by deploy_app.sh, or manually:

    PYTHONPATH=backend python infra/apps/grant_app_sp.py --sp-client-id <uuid>

It applies each required grant idempotently and exits non-zero on any missing
runtime permission:

  1. Unity Catalog - USE CATALOG/SCHEMA + SELECT/MODIFY on the demo schema and
     READ VOLUME on the raw landing volumes, via the SQL warehouse.
  2. Lakebase - an OAuth Postgres role for the SP + CONNECT/USAGE/CREATE/CRUD on
     the serving schema, connecting as the Lakebase instance owner (you).
  3. Genie - best-effort CAN_RUN on the space the app queries by name.
  4. Card issuer (when enabled) - the same UC/Lakebase/Genie grants for the
     credit-card domain's OWN schema + volume + Genie space (both Genie lanes).

  5. Unity Catalog model services (Option B FM chat) - USE CATALOG/SCHEMA on
     ``system`` / ``system.ai`` plus EXECUTE on each configured model service.

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
    failures: list[str] = []
    for s in stmts:
        try:
            execute_sql(settings, s)
            _log(f"UC ok: {s}")
        except Exception as exc:  # noqa: BLE001
            _log(f"UC WARN ({s}): {exc}")
            failures.append(s)
    if failures:
        raise RuntimeError("required UC grants failed:\n- " + "\n- ".join(failures))


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
    failures: list[str] = []
    try:
        # The connection is autocommit, so each statement is its own transaction;
        # a failing grant cannot poison the ones that follow (no SAVEPOINT needed,
        # and SAVEPOINT would itself error with NoActiveSqlTransaction here).
        with serving._conn() as conn, conn.cursor() as cur:  # noqa: SLF001
            for s in stmts:
                try:
                    cur.execute(s)
                    _log(f"LB ok: {s[:80]}")
                except Exception as exc:  # noqa: BLE001
                    # The role helper may report an existing role. That is the
                    # only expected idempotency error; all grants remain required.
                    if "databricks_create_role" in s and "already exists" in str(exc).lower():
                        _log(f"LB ok: role already exists for {sp}")
                    else:
                        _log(f"LB WARN ({s[:60]}...): {exc}")
                        failures.append(s)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"could not grant Lakebase access as owner: {exc}") from exc
    if failures:
        raise RuntimeError("required Lakebase grants failed:\n- " + "\n- ".join(failures))


def _grant_genie_space(settings, sp: str, name: str) -> None:
    from genie_voice.databricks.client import get_workspace_client
    from genie_voice.genie.space import find_space_id

    client = get_workspace_client(settings)
    try:
        sid = find_space_id(client, name)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"could not list Genie spaces while resolving '{name}': {exc}") from exc
    if not sid:
        raise RuntimeError(f"Genie space '{name}' was not found")
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
        raise RuntimeError(f"could not grant CAN_RUN on Genie space '{name}': {exc}") from exc


def grant_genie(settings, sp: str) -> None:
    _grant_genie_space(settings, sp, settings.databricks.genie_space_name)


def _genie_space_names(settings) -> list[str]:
    names = [settings.databricks.genie_space_name]
    card = getattr(settings, "card_issuer", None)
    if card is not None and getattr(card, "enabled", False):
        names.append(card.genie_space_name)
    return [n for n in names if n]


def grant_genie_run_principals(
    settings, users: list[str], groups: list[str]
) -> None:
    """Grant CAN_RUN on every app Genie space to end users / groups.

    Without this, app viewers can sign in but Genie NL->SQL fails for them (the
    space is owner-only). Runs as the space owner (the deployer). Fail-closed.
    """
    from genie_voice.databricks.client import get_workspace_client
    from genie_voice.genie.space import find_space_id

    acl = [{"user_name": u, "permission_level": "CAN_RUN"} for u in users]
    acl += [{"group_name": g, "permission_level": "CAN_RUN"} for g in groups]
    if not acl:
        return
    client = get_workspace_client(settings)
    failures: list[str] = []
    for name in _genie_space_names(settings):
        try:
            sid = find_space_id(client, name)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"resolve '{name}': {exc}")
            continue
        if not sid:
            failures.append(f"Genie space '{name}' was not found")
            continue
        try:
            client.api_client.do(
                "PATCH",
                f"/api/2.0/permissions/genie/{sid}",
                body={"access_control_list": acl},
            )
            _log(f"Genie ok: CAN_RUN for {len(acl)} principal(s) on '{name}' ({sid})")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"grant on '{name}': {exc}")
    if failures:
        raise RuntimeError("Genie viewer CAN_RUN grants failed: " + "; ".join(failures))


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
    failures: list[str] = []
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
            failures.append(s)
    if failures:
        raise RuntimeError("required card UC grants failed:\n- " + "\n- ".join(failures))

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
        failures = []
        try:
            # LakebaseServing connections use autocommit. Each grant is already
            # isolated, and SAVEPOINT is invalid without an explicit transaction.
            with serving._conn() as conn, conn.cursor() as cur:  # noqa: SLF001
                for s in stmts:
                    try:
                        cur.execute(s)
                        _log(f"card LB ok: {s[:80]}")
                    except Exception as exc:  # noqa: BLE001
                        _log(f"card LB warn ({s[:60]}...): {exc}")
                        failures.append(s)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"could not grant card Lakebase access: {exc}") from exc
        if failures:
            raise RuntimeError(
                "required card Lakebase grants failed:\n- " + "\n- ".join(failures)
            )

    # Genie space (both the Conversation + Agent-mode lanes read this space).
    _grant_genie_space(settings, sp, settings.card_issuer.genie_space_name)


def grant_model_services(settings, sp: str, names: list[str]) -> None:
    """Grant EXECUTE on Unity Catalog model services used for FM chat."""
    from genie_voice.databricks.ai_gateway import is_unity_model_service, model_service_id
    from genie_voice.databricks.client import get_workspace_client
    from genie_voice.databricks.warehouse_sql import execute_sql

    fqns = [model_service_id(n) for n in names if is_unity_model_service(n)]
    if not fqns:
        _log("no Unity model services to grant; skipping.")
        return

    failures: list[str] = []
    catalogs: set[str] = set()
    schemas: set[tuple[str, str]] = set()
    for fqn in fqns:
        catalog, schema, _leaf = fqn.split(".", 2)
        catalogs.add(catalog)
        schemas.add((catalog, schema))

    p = f"`{sp}`"
    for catalog in sorted(catalogs):
        stmt = f"GRANT USE CATALOG ON CATALOG `{catalog}` TO {p}"
        try:
            execute_sql(settings, stmt)
            _log(f"UC model-service ok: {stmt}")
        except Exception as exc:  # noqa: BLE001
            _log(f"UC model-service WARN ({stmt}): {exc}")
            failures.append(stmt)
    for catalog, schema in sorted(schemas):
        stmt = f"GRANT USE SCHEMA ON SCHEMA `{catalog}`.`{schema}` TO {p}"
        try:
            execute_sql(settings, stmt)
            _log(f"UC model-service ok: {stmt}")
        except Exception as exc:  # noqa: BLE001
            _log(f"UC model-service WARN ({stmt}): {exc}")
            failures.append(stmt)

    client = get_workspace_client(settings)
    for fqn in fqns:
        try:
            # MODEL SERVICE is not accepted by the SQL GRANT grammar in this
            # workspace. The public UC permissions API uses the singular
            # securable type `model_service`.
            client.api_client.do(
                "PATCH",
                f"/api/2.1/unity-catalog/permissions/model_service/{fqn}",
                body={"changes": [{"principal": sp, "add": ["EXECUTE"]}]},
            )
            _log(f"UC model-service EXECUTE ok: {fqn} -> {sp}")
        except Exception as exc:  # noqa: BLE001
            _log(f"UC model-service WARN ({fqn}): {exc}")
            failures.append(f"GRANT EXECUTE ON MODEL SERVICE {fqn}")

    if failures:
        raise RuntimeError(
            "required model-service grants failed:\n- " + "\n- ".join(failures)
        )


def grant_registered_models(settings, sp: str, names: list[str]) -> None:
    """Grant UC model execution/visibility using the supported SQL securable.

    Unity Catalog registered models are exposed as FUNCTION securables to SQL
    GRANT in this workspace. The permissions API's ``registered_model`` type
    returns ``REGISTERED_MODEL is not enabled`` and must not be used here.
    """
    if not names:
        return
    from genie_voice.databricks.client import get_workspace_client

    client = get_workspace_client(settings)
    failures: list[str] = []
    for name in names:
        fqn = name if name.count(".") == 2 else (
            f"{settings.databricks.catalog}.{settings.databricks.schema_name}.{name}"
        )
        parts = fqn.split(".")
        quoted = ".".join(f"`{part.replace('`', '``')}`" for part in parts)
        try:
            result = client.statement_execution.execute_statement(
                warehouse_id=settings.databricks.sql_warehouse_id,
                statement=f"GRANT EXECUTE ON FUNCTION {quoted} TO `{sp}`",
                wait_timeout="30s",
            )
            state = getattr(getattr(result, "status", None), "state", None)
            state = getattr(state, "value", state)
            if str(state) != "SUCCEEDED":
                error = getattr(getattr(result, "status", None), "error", None)
                raise RuntimeError(getattr(error, "message", None) or str(error or state))
            _log(f"UC model ok: EXECUTE on {fqn}")
        except Exception as exc:  # noqa: BLE001
            _log(f"UC model WARN ({fqn}): {exc}")
            failures.append(f"{fqn}: {exc}")
    if failures:
        raise RuntimeError(
            "required registered voice-model grants failed:\n- " + "\n- ".join(failures)
        )


def grant_pipeline_job(settings, sp: str) -> None:
    """Grant the app CAN_VIEW so status/readiness can inspect the deployed job."""
    from genie_voice.databricks.client import get_workspace_client

    client = get_workspace_client(settings)
    name = settings.pipeline.orchestration_job_name
    job = next(iter(client.jobs.list(name=name, limit=2)), None)
    if job is None or job.job_id is None:
        raise RuntimeError(f"orchestration job '{name}' was not found")
    try:
        client.api_client.do(
            "PATCH",
            f"/api/2.0/permissions/jobs/{job.job_id}",
            body={
                "access_control_list": [
                    {"service_principal_name": sp, "permission_level": "CAN_VIEW"}
                ]
            },
        )
        _log(f"job ok: CAN_VIEW on '{name}' ({job.job_id})")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"could not grant CAN_VIEW on job '{name}': {exc}") from exc


def main() -> None:
    ap = argparse.ArgumentParser(description="Grant the app service principal its runtime access.")
    ap.add_argument("--sp-client-id", required=True, help="App service principal application (client) id")
    ap.add_argument(
        "--model-services",
        default="",
        help="comma-separated Unity Catalog model service FQNs (FM chat)",
    )
    ap.add_argument(
        "--run-users",
        default="",
        help="comma-separated user emails to grant CAN_RUN on every app Genie space",
    )
    ap.add_argument(
        "--run-groups",
        default="",
        help="comma-separated group names to grant CAN_RUN on every app Genie space",
    )
    ap.add_argument(
        "--registered-models",
        default="",
        help="comma-separated UC registered voice models (short names or FQNs)",
    )
    args = ap.parse_args()
    sp = args.sp_client_id.strip()
    if not re.fullmatch(r"[0-9a-fA-F-]{36}", sp):
        raise SystemExit(f"--sp-client-id does not look like a UUID: {sp!r}")

    settings = get_settings()
    names = [n.strip() for n in (args.model_services or "").split(",") if n.strip()]
    run_users = [u.strip() for u in (args.run_users or "").split(",") if u.strip()]
    run_groups = [g.strip() for g in (args.run_groups or "").split(",") if g.strip()]
    registered_models = [
        n.strip() for n in (args.registered_models or "").split(",") if n.strip()
    ]
    _log(f"granting app service principal: {sp}")
    grant_unity_catalog(settings, sp)
    grant_lakebase(settings, sp)
    grant_genie(settings, sp)
    grant_card_issuer(settings, sp)
    grant_model_services(settings, sp, names)
    grant_registered_models(settings, sp, registered_models)
    grant_pipeline_job(settings, sp)
    grant_genie_run_principals(settings, run_users, run_groups)
    _log("done")


if __name__ == "__main__":
    main()
