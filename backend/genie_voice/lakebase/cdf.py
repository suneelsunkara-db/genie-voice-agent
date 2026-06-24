"""Gate orchestration until Lakebase CDF is running and visible in UC.

Lakebase CDF itself is started in the Lakebase UI. This module only performs the
documented programmatic checks around that UI-managed feed:
  - source tables exist and use REPLICA IDENTITY FULL
  - wal2delta reports each required table as STREAMING or SNAPSHOTTING
  - UC has non-empty lb_<table>_history Delta tables
"""
from __future__ import annotations

import time
import json
from typing import Any

from genie_voice.config import Settings, get_settings
from genie_voice.databricks.client import get_workspace_client
from genie_voice.serve import LakebaseServing

_PG_BASE = "/api/2.0/postgres"


def _quote_sql(name: str) -> str:
    return "`" + name.replace("`", "``") + "`"


def _pg_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _expected_tables(s: Settings) -> list[str]:
    return s.lakebase.cdf_required_tables or list(s.lakebase.sync_tables)


def _history_table(s: Settings, table: str) -> str:
    return f"{s.lakebase.cdf_history_prefix}{table}{s.lakebase.cdf_history_suffix}"


def _marker_path(settings: Settings) -> str:
    return f"{settings.checkpoint_path}/cdf_markers/latest_call_ingest.json"


def _resolve_lakebase_branch(api, instance: str) -> dict[str, Any]:
    projects = api.do("GET", f"{_PG_BASE}/projects").get("projects", []) or []
    project = next(
        (
            p for p in projects
            if instance in (p.get("project_id"), (p.get("status") or {}).get("display_name"))
            or p.get("project_id") == instance.replace("_", "-")
        ),
        None,
    )
    if not project:
        raise RuntimeError(f"Lakebase project '{instance}' not found via {_PG_BASE}/projects")

    project_id = project["project_id"]
    branches = api.do(
        "GET", f"{_PG_BASE}/projects/{project_id}/branches"
    ).get("branches", []) or []
    branch = next(
        (b for b in branches if (b.get("status") or {}).get("default")),
        branches[0] if branches else None,
    )
    if not branch:
        raise RuntimeError(f"Lakebase project '{project_id}' has no branches")

    state = (branch.get("status") or {}).get("current_state")
    if state and state != "READY":
        raise RuntimeError(f"Lakebase branch '{branch.get('name')}' is not ready: {state}")
    return {"project": project, "branch": branch}


def _replica_identity_ok(settings: Settings, tables: list[str]) -> None:
    lb = LakebaseServing(settings)
    schema = settings.lakebase.schema_name
    with lb._conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.relname, c.relreplident
            FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = %s AND c.relname = ANY(%s)
            """,
            (schema, tables),
        )
        found = {name: repl for name, repl in cur.fetchall()}

    missing = [t for t in tables if t not in found]
    not_full = [t for t in tables if found.get(t) != "f"]
    if missing:
        raise RuntimeError(
            "Lakebase CDF source tables are missing: " + ", ".join(sorted(missing))
        )
    if not_full:
        raise RuntimeError(
            "Lakebase CDF requires REPLICA IDENTITY FULL for: "
            + ", ".join(sorted(not_full))
        )


def _wal2delta_status(settings: Settings, tables: list[str]) -> dict[str, dict[str, Any]]:
    """Return CDF status from wal2delta.tables.

    The docs call this out as the Postgres-side way to inspect feed state. If
    this view is absent/unreadable, CDF has not been started or is not available
    to this role, so the orchestration must not continue to Genie.
    """
    lb = LakebaseServing(settings)
    with lb._conn() as conn, conn.cursor() as cur:
        try:
            cur.execute(
                """
                SELECT table_oid::regclass::text, status, committed_lsn, last_write_time
                FROM wal2delta.tables
                """
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Lakebase CDF status is not readable from wal2delta.tables. "
                "Per Databricks docs, start CDF in the Lakebase UI first and "
                "ensure this Postgres role can inspect wal2delta.tables."
            ) from exc
        rows = cur.fetchall()

    status: dict[str, dict[str, Any]] = {}
    suffixes = tuple(f".{t}" for t in tables)
    for table_ref, state, committed_lsn, last_write_time in rows:
        if table_ref in tables or str(table_ref).endswith(suffixes):
            status[str(table_ref).split(".")[-1]] = {
                "status": str(state).upper(),
                "committed_lsn": committed_lsn,
                "last_write_time": last_write_time,
            }
    return status


def _cdf_running(settings: Settings, tables: list[str]) -> dict[str, dict[str, Any]]:
    status = _wal2delta_status(settings, tables)
    missing = [t for t in tables if t not in status]
    inactive = [
        f"{t}={status[t]['status']}"
        for t in tables
        if t in status and status[t]["status"] not in {"STREAMING", "SNAPSHOTTING"}
    ]
    if missing or inactive:
        raise RuntimeError(
            "Lakebase CDF is not running for all required tables. "
            f"Missing from wal2delta.tables: {missing or '-'}; "
            f"inactive statuses: {inactive or '-'}."
        )
    return status


def _latest_ingest_started_at(settings: Settings) -> str | None:
    try:
        client = get_workspace_client(settings)
        resp = client.files.download(_marker_path(settings))
        contents = getattr(resp, "contents", resp)
        data = contents.read() if hasattr(contents, "read") else contents
        text = data.decode() if isinstance(data, bytes) else str(data)
        marker = json.loads(text)
        return marker.get("started_at_utc")
    except Exception:
        return None


def _uc_history_status(
    settings: Settings, tables: list[str], min_updated_at: str | None
) -> dict[str, dict[str, int | str | None]]:
    client = get_workspace_client(settings)
    wh = settings.databricks.sql_warehouse_id
    if not wh:
        raise RuntimeError("databricks.sql_warehouse_id is required for UC CDF verification.")

    catalog = _quote_sql(settings.databricks.catalog)
    schema = _quote_sql(settings.databricks.schema_name)
    out: dict[str, dict[str, int | str | None]] = {}
    for source in tables:
        hist = _quote_sql(_history_table(settings, source))
        fqtn = f"{catalog}.{schema}.{hist}"
        try:
            freshness_expr = (
                f"sum(CASE WHEN CAST(updated_at AS TIMESTAMP) >= TIMESTAMP '{min_updated_at}' "
                "THEN 1 ELSE 0 END) AS fresh_rows, "
                "CAST(max(CAST(updated_at AS TIMESTAMP)) AS STRING) AS max_updated_at"
                if min_updated_at
                else "count(*) AS fresh_rows, CAST(max(CAST(updated_at AS TIMESTAMP)) AS STRING) AS max_updated_at"
            )
            res = client.statement_execution.execute_statement(
                warehouse_id=wh,
                statement=f"SELECT count(*) AS total_rows, {freshness_expr} FROM {fqtn}",
                wait_timeout="30s",
            )
            rows = (res.result.data_array if res.result else None) or []
            if not rows:
                out[source] = {"total_rows": 0, "fresh_rows": 0, "max_updated_at": None}
            else:
                out[source] = {
                    "total_rows": int(rows[0][0] or 0),
                    "fresh_rows": int(rows[0][1] or 0),
                    "max_updated_at": rows[0][2],
                }
        except Exception:
            out[source] = {"total_rows": None, "fresh_rows": None, "max_updated_at": None}
    return out


def wait_for_lakebase_cdf(settings: Settings | None = None) -> dict[str, dict[str, int | str | None]]:
    s = settings or get_settings()
    if not s.lakebase.enabled:
        print("lakebase.enabled=false -> skipping Lakebase CDF check.")
        return {}
    if not s.lakebase.cdf_required:
        print("lakebase.cdf_required=false -> skipping Lakebase CDF check.")
        return {}

    client = get_workspace_client(s)
    resolved = _resolve_lakebase_branch(client.api_client, s.lakebase.instance)
    branch_name = resolved["branch"].get("name") or resolved["branch"].get("branch_id")
    tables = _expected_tables(s)
    min_updated_at = _latest_ingest_started_at(s)
    print(f"Lakebase branch ready: {branch_name}")
    print(
        "CDF must already be started in the Lakebase UI for "
        f"{s.lakebase.database}.{s.lakebase.schema_name} -> "
        f"{s.databricks.catalog}.{s.databricks.schema_name}."
    )
    print(f"Checking REPLICA IDENTITY FULL for {s.lakebase.schema_name}: {', '.join(tables)}")
    if min_updated_at:
        print(f"Waiting for UC CDF history at or after latest ingest: {min_updated_at} UTC")
    else:
        print("No latest ingest marker found; falling back to non-empty UC history check.")
    _replica_identity_ok(s, tables)

    deadline = time.time() + s.lakebase.cdf_wait_timeout_seconds
    last_counts: dict[str, dict[str, int | str | None]] = {}
    last_status: dict[str, dict[str, Any]] = {}
    while time.time() < deadline:
        try:
            last_status = _cdf_running(s, tables)
        except RuntimeError as exc:
            print(f"Waiting for Lakebase CDF status: {exc}")
            time.sleep(s.lakebase.cdf_poll_seconds)
            continue
        print(
            "wal2delta status: "
            + ", ".join(
                f"{table}={meta['status']} lsn={meta['committed_lsn'] or '-'}"
                for table, meta in sorted(last_status.items())
            )
        )
        last_counts = _uc_history_status(s, tables, min_updated_at)
        missing = [t for t, meta in last_counts.items() if meta["total_rows"] is None]
        empty = [t for t, meta in last_counts.items() if meta["total_rows"] == 0]
        stale = [
            t
            for t, meta in last_counts.items()
            if min_updated_at and (meta["fresh_rows"] or 0) == 0
        ]
        if not missing and not empty and not stale:
            for table, meta in sorted(last_counts.items()):
                print(
                    f"  CDF ready: {_history_table(s, table)} "
                    f"(rows={meta['total_rows']}, fresh={meta['fresh_rows']}, "
                    f"max_updated_at={meta['max_updated_at']})"
                )
            return last_counts
        print(
            "Waiting for Lakebase CDF -> UC history tables; "
            f"missing={missing or '-'} empty={empty or '-'} stale={stale or '-'} "
            f"expected={[ _history_table(s, t) for t in tables ]}"
        )
        time.sleep(s.lakebase.cdf_poll_seconds)

    raise RuntimeError(
        "Lakebase CDF did not publish all expected UC history tables before timeout. "
        f"Last wal2delta status: {last_status}. Last counts: {last_counts}. "
        "Per Databricks docs, CDF must be started in "
        "the Lakebase UI; choose database "
        f"'{s.lakebase.database}', source schema '{s.lakebase.schema_name}', "
        f"destination '{s.databricks.catalog}.{s.databricks.schema_name}'."
    )


if __name__ == "__main__":
    wait_for_lakebase_cdf()
