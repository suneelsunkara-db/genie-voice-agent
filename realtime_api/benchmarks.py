"""Read multilingual voice benchmark scores from the Delta result tables.

The benchmark Databricks job writes one row per (run_id, dataset, language) to
``{catalog}.{schema}.benchmark_runs`` (and per-sample rows to
``benchmark_samples``). This module serves the latest run straight from Delta via
the SQL Statement Execution API — no ``summary.json`` intermediate — so the UI
always reflects the source of truth.
"""
from __future__ import annotations

import json
from typing import Any

from .benchmark_references import OUR_SYSTEM_ID, OUR_SYSTEM_LABEL, reference_rows
from .config import (
    config_dir_from_env,
    databricks_profile,
    delta_catalog,
    delta_schema,
    sql_warehouse_id,
)

_RUNS_TABLE = "benchmark_runs"


def _loads(value: Any) -> Any:
    if value in (None, ""):
        return {}
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        return {}


def _to_int(value: Any) -> int | None:
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return None


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _rows(result: Any) -> list[dict[str, Any]]:
    manifest = getattr(result, "manifest", None)
    schema = getattr(manifest, "schema", None)
    columns = [c.name for c in (getattr(schema, "columns", None) or [])]
    data = getattr(getattr(result, "result", None), "data_array", None) or []
    return [dict(zip(columns, row)) for row in data]


def load_benchmarks(run_id: str | None = None) -> dict[str, Any]:
    """Return the latest (or a specific) benchmark run from Delta as a UI payload.

    Never raises: on any misconfiguration or query error it returns
    ``{"available": False, "message": ...}`` so the endpoint degrades gracefully.
    """
    catalog = delta_catalog()
    schema = delta_schema()
    warehouse = sql_warehouse_id()
    if not (catalog and schema and warehouse):
        return {
            "available": False,
            "message": (
                "Benchmark Delta source not configured — need databricks.catalog, "
                "databricks.schema and databricks.sql_warehouse_id."
            ),
        }

    table = f"{catalog}.{schema}.{_RUNS_TABLE}"
    try:
        from databricks.sdk import WorkspaceClient
        from databricks.sdk.service.sql import StatementParameterListItem

        client = WorkspaceClient(profile=databricks_profile() or None)

        if run_id:
            result = client.statement_execution.execute_statement(
                warehouse_id=warehouse,
                statement=(
                    "SELECT run_id, dataset, language, evaluator, samples, errors, primary_score, "
                    "scores, latency_ms, issues_count, issues_by_kind, wall_seconds, "
                    "timestamp, status "
                    f"FROM {table} WHERE run_id = :run_id ORDER BY dataset, language"  # noqa: S608
                ),
                parameters=[StatementParameterListItem(name="run_id", value=run_id, type="STRING")],
                wait_timeout="50s",
            )
        else:
            result = client.statement_execution.execute_statement(
                warehouse_id=warehouse,
                statement=(
                    "SELECT run_id, dataset, language, evaluator, samples, errors, primary_score, "
                    "scores, latency_ms, issues_count, issues_by_kind, wall_seconds, "
                    "timestamp, status FROM ("
                    "  SELECT *, row_number() OVER ("
                    "    PARTITION BY dataset, language "
                    "    ORDER BY timestamp DESC, run_id DESC"
                    "  ) AS rn "
                    f"  FROM {table} WHERE status = 'complete'"  # noqa: S608
                    ") WHERE rn = 1 ORDER BY dataset, language"
                ),
                wait_timeout="50s",
            )
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "message": f"Benchmark Delta query failed: {exc}"}

    runs: list[dict[str, Any]] = []
    for row in _rows(result):
        scores = _loads(row.get("scores"))
        runs.append(
            {
                "system": OUR_SYSTEM_ID,
                "system_label": OUR_SYSTEM_LABEL,
                "source": "measured",
                "run_id": row.get("run_id"),
                "dataset": row.get("dataset"),
                "language": row.get("language"),
                "evaluator": row.get("evaluator"),
                "samples": _to_int(row.get("samples")),
                "errors": _to_int(row.get("errors")),
                "primary_score": _to_float(row.get("primary_score")),
                "primary_metric": scores.get("primary_metric"),
                "scores": scores,
                "latency_ms": _loads(row.get("latency_ms")),
                "issues": {
                    "count": _to_int(row.get("issues_count")),
                    "by_kind": _loads(row.get("issues_by_kind")),
                },
                "wall_seconds": _to_float(row.get("wall_seconds")),
                "status": row.get("status"),
                "timestamp": row.get("timestamp"),
            }
        )

    if not runs:
        return {"available": False, "message": f"No benchmark rows found for run_id {run_id}."}

    baselines = reference_rows()
    run_ids = sorted({str(r.get("run_id")) for r in runs if r.get("run_id")})
    return {
        "available": True,
        "source": "delta",
        "table": table,
        "run_id": run_id or (run_ids[0] if len(run_ids) == 1 else None),
        "run_ids": run_ids,
        "generated_at": max((r.get("timestamp") or "") for r in runs) or None,
        "datasets": sorted({r["dataset"] for r in runs if r.get("dataset")}),
        "languages": sorted({r["language"] for r in runs if r.get("language")}),
        "runs": runs,
        "our_system": {"id": OUR_SYSTEM_ID, "label": OUR_SYSTEM_LABEL},
        "baselines": baselines,
    }
