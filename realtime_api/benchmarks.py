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


# Latency key that carries the perceived time-to-first-token/audio (TTFT): the
# end-to-end client time from turn start until the first response audio chunk
# (STT + LLM + TTS transport). This is what the UI surfaces as "TTFT".
_TTFT_LATENCY_KEY = "client_ttfa_ms"


def _has_ttft(latency: Any) -> bool:
    """True when a run carries a usable TTFT (client time-to-first-audio) number."""
    if not isinstance(latency, dict):
        return False
    stat = latency.get(_TTFT_LATENCY_KEY)
    if isinstance(stat, (int, float)):
        return True
    if isinstance(stat, dict):
        return any(isinstance(stat.get(k), (int, float)) for k in ("p50", "p95", "p99", "mean"))
    return False


def _unit_metrics_ready(run: dict[str, Any]) -> bool:
    """True when a single (dataset, language) unit finished with its metrics.

    A unit is metric-ready when it has an accuracy score and a TTFT number. A
    unit where *every* sample errored (``errors >= samples``) has no audio and
    therefore no TTFT by nature — that is a reliability signal, not an
    unfinished run, so it still counts as "ready" (its TTFT renders as "—").
    """
    if run.get("primary_score") is None:
        return False
    samples = run.get("samples") or 0
    errors = run.get("errors") or 0
    if samples > 0 and errors >= samples:
        return True
    return _has_ttft(run.get("latency_ms"))


def _run_ts(runs: list[dict[str, Any]]) -> str:
    return max((r.get("timestamp") or "") for r in runs) if runs else ""


def select_ready_runs(runs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Pick, per dataset, the newest *fully ready* benchmark run.

    "Fully ready" means a single ``run_id`` that (a) swept the full language
    breadth ever measured for that dataset and (b) has accuracy + TTFT on every
    unit that produced results. A partial or in-flight run (fewer languages, or
    a language still missing its TTFT) is NOT promoted — the previous fully
    ready run keeps showing. If no run is fully ready yet, we fall back to the
    latest row per language so the page is never blank.

    Returns ``(selected_runs, meta)`` where ``meta`` records, per dataset, the
    chosen ``run_id`` and whether it was fully ready.
    """
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        ds = run.get("dataset")
        if ds:
            by_dataset.setdefault(ds, []).append(run)

    selected: list[dict[str, Any]] = []
    meta: dict[str, Any] = {}
    for dataset, ds_runs in by_dataset.items():
        by_run: dict[str, list[dict[str, Any]]] = {}
        for run in ds_runs:
            by_run.setdefault(str(run.get("run_id")), []).append(run)

        # Runs whose every covered unit reported its metrics.
        complete = {
            rid: rs for rid, rs in by_run.items() if all(_unit_metrics_ready(r) for r in rs)
        }
        if complete:
            widest = max(len({r.get("language") for r in rs}) for rs in complete.values())
            ready = {
                rid: rs
                for rid, rs in complete.items()
                if len({r.get("language") for r in rs}) >= widest
            }
            best_rid = max(ready, key=lambda rid: (_run_ts(ready[rid]), rid))
            chosen = ready[best_rid]
            selected.extend(chosen)
            meta[dataset] = {
                "run_id": best_rid,
                "ready": True,
                "languages": sorted({r.get("language") for r in chosen if r.get("language")}),
            }
            continue

        # Fallback: newest row per language (never leaves the page blank).
        latest: dict[str, dict[str, Any]] = {}
        for run in ds_runs:
            lang = str(run.get("language"))
            cur = latest.get(lang)
            if cur is None or (run.get("timestamp") or "") > (cur.get("timestamp") or ""):
                latest[lang] = run
        selected.extend(latest.values())
        meta[dataset] = {
            "run_id": None,
            "ready": False,
            "languages": sorted(k for k in latest if k),
        }
    return selected, meta


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
            # Pull every completed unit and choose the newest fully-ready run per
            # dataset in Python (see select_ready_runs). The runs table holds one
            # row per (run_id, dataset, language), so this stays small.
            result = client.statement_execution.execute_statement(
                warehouse_id=warehouse,
                statement=(
                    "SELECT run_id, dataset, language, evaluator, samples, errors, primary_score, "
                    "scores, latency_ms, issues_count, issues_by_kind, wall_seconds, "
                    "timestamp, status "
                    f"FROM {table} WHERE status = 'complete' "  # noqa: S608
                    "ORDER BY dataset, language, timestamp DESC"
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

    # An explicit run_id is served verbatim; the default view gates on readiness
    # so a half-finished sweep never replaces the last fully-ready run.
    selection_meta: dict[str, Any] = {}
    if not run_id:
        runs, selection_meta = select_ready_runs(runs)

    baselines = reference_rows()
    run_ids = sorted({str(r.get("run_id")) for r in runs if r.get("run_id")})
    all_ready = bool(selection_meta) and all(m.get("ready") for m in selection_meta.values())
    return {
        "available": True,
        "source": "delta",
        "table": table,
        "run_id": run_id or (run_ids[0] if len(run_ids) == 1 else None),
        "run_ids": run_ids,
        "all_metrics_ready": True if run_id else all_ready,
        "run_selection": selection_meta,
        "generated_at": max((r.get("timestamp") or "") for r in runs) or None,
        "datasets": sorted({r["dataset"] for r in runs if r.get("dataset")}),
        "languages": sorted({r["language"] for r in runs if r.get("language")}),
        "runs": runs,
        "our_system": {"id": OUR_SYSTEM_ID, "label": OUR_SYSTEM_LABEL},
        "baselines": baselines,
    }
