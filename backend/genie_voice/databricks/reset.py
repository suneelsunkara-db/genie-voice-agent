"""Clean rebuild helper for local-deploy.sh.

This is intentionally explicit and opt-in. It removes the demo's generated
tables, views, and Lakebase serving copies. UC Volume data is preserved by
default so reruns can reuse the landed raw_batch_data/raw_streaming_data files.
"""
from __future__ import annotations

import argparse

from genie_voice.config import Settings, get_settings
from genie_voice.databricks.client import current_user, get_workspace_client

_PG_BASE = "/api/2.0/postgres"
_LEGACY_JOB_NAMES = [
    "Genie Voice - Lakeflow Refresh",
    "Genie Voice - Voice Stream (bronze+silver)",
    "Genie Voice - Batch (reference+gold)",
]
_LEGACY_PIPELINE_NAMES = ["Genie Voice - Lakeflow Pipeline"]


def _exec(client, warehouse_id: str, statement: str) -> None:
    client.statement_execution.execute_statement(
        warehouse_id=warehouse_id, statement=statement, wait_timeout="30s"
    )


def _try(label: str, fn) -> None:
    try:
        fn()
        print(f"  ok: {label}")
    except Exception as exc:  # noqa: BLE001
        print(f"  skip: {label} ({exc})")


def _quote(name: str) -> str:
    return f"`{name.replace('`', '``')}`"


def _fqtn(settings: Settings, table: str) -> str:
    return ".".join(
        [
            _quote(settings.databricks.catalog),
            _quote(settings.databricks.schema_name),
            _quote(table),
        ]
    )


def _pg_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _schema_objects(client, wh: str, settings: Settings) -> list[tuple[str, str]]:
    stmt = f"""
        SELECT table_name, table_type
        FROM {_quote(settings.databricks.catalog)}.information_schema.tables
        WHERE table_schema = '{settings.databricks.schema_name.replace("'", "''")}'
        ORDER BY table_name
    """
    res = client.statement_execution.execute_statement(
        warehouse_id=wh, statement=stmt, wait_timeout="30s"
    )
    rows = (res.result.data_array if res.result else None) or []
    return [(str(r[0]), str(r[1]).upper()) for r in rows]


def _drop_uc_objects(settings: Settings) -> None:
    client = get_workspace_client(settings)
    wh = settings.databricks.sql_warehouse_id
    if not wh:
        raise RuntimeError("databricks.sql_warehouse_id is required to reset UC tables.")

    print(f"Authenticated as: {current_user(client)}")
    print(f"Dropping UC objects in {settings.databricks.catalog}.{settings.databricks.schema_name} ...")

    objects = _schema_objects(client, wh, settings)
    if not objects:
        print("  no UC tables/views found")
        return

    # Views first, then materialized views, then tables/streaming tables. This is
    # deliberately dynamic so stale objects from old deploy shapes are removed.
    views = [(n, t) for n, t in objects if t == "VIEW"]
    materialized = [(n, t) for n, t in objects if "MATERIALIZED" in t]
    tables = [(n, t) for n, t in objects if (n, t) not in views and (n, t) not in materialized]

    for name, _ in views:
        _try(
            f"drop view {name}",
            lambda name=name: _exec(client, wh, f"DROP VIEW IF EXISTS {_fqtn(settings, name)}"),
        )
    for name, _ in materialized:
        _try(
            f"drop materialized view {name}",
            lambda name=name: _exec(client, wh, f"DROP MATERIALIZED VIEW IF EXISTS {_fqtn(settings, name)}"),
        )
    for name, typ in tables:
        _try(
            f"drop table {name} ({typ})",
            lambda name=name: _exec(client, wh, f"DROP TABLE IF EXISTS {_fqtn(settings, name)}"),
        )


def _drop_lakeflow_pipeline(settings: Settings) -> None:
    client = get_workspace_client(settings)
    job_names = [settings.pipeline.orchestration_job_name, *_LEGACY_JOB_NAMES]
    print("Deleting orchestration/legacy jobs if they exist ...")
    for job_name in dict.fromkeys(job_names):
        for job in client.jobs.list(name=job_name):
            if job.job_id is None:
                continue
            _try(
                f"delete job {job.job_id} ({job_name})",
                lambda job_id=job.job_id: client.jobs.delete(job_id),
            )

    names = set([settings.pipeline.analytics_pipeline_name, *_LEGACY_PIPELINE_NAMES])
    print("Stopping/deleting analytics/legacy pipelines if they exist ...")
    for pipeline in client.pipelines.list_pipelines():
        if getattr(pipeline, "name", None) not in names:
            continue
        pipeline_id = pipeline.pipeline_id
        if not pipeline_id:
            continue
        _try(
            f"stop pipeline {pipeline_id}",
            lambda pipeline_id=pipeline_id: client.pipelines.stop_and_wait(pipeline_id),
        )
        _try(
            f"delete pipeline {pipeline_id}",
            lambda pipeline_id=pipeline_id: client.pipelines.delete(
                pipeline_id, cascade=True, force=True
            ),
        )


def _clear_volume_data(settings: Settings) -> None:
    client = get_workspace_client(settings)
    paths = [
        settings.raw_stt_path,
        settings.call_facts_path,
        settings.reference_path,
        settings.resolve_volume_path(settings.volume.audio_path),
        settings.resolve_volume_path(settings.volume.transcript_path),
        settings.checkpoint_path,
    ]
    print("Clearing UC Volume landing/checkpoint directories ...")
    for path in paths:
        _try(
            f"delete directory {path}",
            lambda path=path: _delete_path(client, path),
        )


def _delete_path(client, path: str) -> None:
    """Recursively delete a UC Volume directory or file via Files API."""
    normalized = path.rstrip("/")
    try:
        children = list(client.files.list_directory_contents(normalized))
    except Exception:
        client.files.delete(normalized)
        return

    for child in children:
        child_path = getattr(child, "path", None) or str(child)
        if child_path.endswith("/"):
            _delete_path(client, child_path.rstrip("/"))
        else:
            client.files.delete(child_path)
    client.files.delete_directory(normalized)


def _drop_lakebase_tables(settings: Settings) -> None:
    if not settings.lakebase.enabled:
        print("lakebase.enabled=false -> skipping Lakebase reset.")
        return

    client = get_workspace_client(settings)
    api = client.api_client
    print("Dropping Lakebase managed synced-table objects ...")
    for source in settings.lakebase.sync_tables:
        target = f"synced_tables/{settings.lakebase_synced_fqtn(source)}"
        _try(
            f"delete synced table {target}",
            lambda target=target: api.do("DELETE", f"{_PG_BASE}/{target}"),
        )

    print("Dropping Lakebase Postgres serving tables ...")
    try:
        from genie_voice.serve import LakebaseServing

        lb = LakebaseServing(settings)
        tables = [
            settings.lakebase.serving_table,
            settings.lakebase.live_utterances_table,
            *settings.lakebase.sync_tables,
            *(settings.lakebase_synced_table_name(t) for t in settings.lakebase.sync_tables),
        ]
        schema = _pg_ident(settings.lakebase.schema_name)
        with lb._conn() as conn, conn.cursor() as cur:  # noqa: SLF001 - local reset helper
            for table in dict.fromkeys(tables):
                cur.execute(f"DROP TABLE IF EXISTS {schema}.{_pg_ident(table)}")
                print(f"  ok: drop Lakebase table {settings.lakebase.schema_name}.{table}")
    except Exception as exc:  # noqa: BLE001
        print(f"  skip: Lakebase table reset ({exc})")


def reset(settings: Settings | None = None, *, clear_volumes: bool = False) -> None:
    settings = settings or get_settings()
    _drop_lakeflow_pipeline(settings)
    _drop_lakebase_tables(settings)
    _drop_uc_objects(settings)
    if clear_volumes:
        _clear_volume_data(settings)
    else:
        print("Keeping UC Volume data (clear_volumes=false).")
    print("Reset complete. Run bootstrap + datagen + pipeline next.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--keep-volumes",
        action="store_true",
        help="deprecated no-op; Volume contents are kept by default",
    )
    ap.add_argument(
        "--clear-volumes",
        action="store_true",
        help="also delete raw_batch_data/raw_streaming_data landing and checkpoint contents",
    )
    args = ap.parse_args()
    reset(clear_volumes=args.clear_volumes and not args.keep_volumes)
