"""Pipeline observability endpoint for the UI flow tracker.

Reports per-stage status so the UI can visualize data moving through the flow:
capture -> volume -> Lakebase -> CDF sync -> gold -> serving, including:
  - the processing MODE of each step (real-time / streaming / batch / storage),
  - WHERE it runs and its approximate latency,
  - how many rows each stage holds (so a business user sees progress), and
  - whether the orchestration job is currently RUNNING or idle.

The expensive parts (medallion row counts + Jobs API run-state) are cached for a
few seconds so the UI can poll frequently for live `call_state` without hammering
the warehouse or the Jobs API on every tick.
"""
from __future__ import annotations

import os
import threading
import time

from fastapi import APIRouter

from genie_voice.config import get_settings

from ..deps import serving

router = APIRouter(prefix="/status", tags=["status"])

# TTL cache for the heavy bits (counts + job run-state). call_state is cheap and
# stays live every poll; this only throttles the warehouse / Jobs API lookups.
# The refresh runs in a background thread so a cold SQL warehouse / slow Jobs API
# never blocks the `/status` response (and therefore never blocks the live call
# list that rides in the same payload).
_META_TTL_S = 8.0
_meta_lock = threading.Lock()
_meta_cache: dict[str, object] = {"ts": 0.0, "value": None, "refreshing": False}


@router.get("")
def status() -> dict:
    s = get_settings()
    states = serving().list_call_states()
    meta = _pipeline_meta(s)
    counts = meta["counts"]
    jobs = meta["jobs"]

    orchestration_running = bool(jobs.get("orchestration", {}).get("running"))

    def step_status(count: int | None, running: bool) -> str:
        if running:
            return "running"
        if count and count > 0:
            return "done"
        return "idle"

    call_history = f"{s.lakebase.cdf_history_prefix}call_facts{s.lakebase.cdf_history_suffix}"
    facts_n = _as_int(counts.get(call_history))
    gold_n = _as_int(counts.get(s.medallion.gold_call_insights))

    cap = "Capture (synthetic)" if s.deployment == "local" else "Capture (live)"
    stages = [
        {"key": "capture", "label": cap, "provider": s.providers.stt.active,
         "mode": "real-time", "where": "Host app (producer)", "latency": "continuous",
         "status": "running" if orchestration_running else "done", "count": len(states)},
        {"key": "volume", "label": f"UC Volumes ({s.volume.batch_name}, {s.volume.streaming_name})",
         "path": s.raw_stt_path,
         "mode": "landing", "where": f"Batch: {s.reference_path} / Streaming: {s.raw_stt_path}",
         "latency": "storage",
         "status": "done"},
        {"key": "lakebase", "label": f"Lakebase call tables ({s.lakebase.schema_name})", "job": s.pipeline.orchestration_job_name,
         "mode": "serving", "where": "Lakebase Postgres call_facts/live_call_utterances", "latency": "sub-second",
         "status": step_status(len(states), False), "count": len(states)},
        {"key": "call_facts_history", "label": call_history, "job": s.pipeline.orchestration_job_name,
         "mode": "CDF history", "where": "Lakebase call_facts -> UC history table", "latency": "~seconds",
         "status": step_status(facts_n, orchestration_running), "count": facts_n},
        {"key": "gold", "label": s.medallion.gold_call_insights, "job": s.pipeline.orchestration_job_name,
         "model": s.enrichment.model_endpoint,
         "mode": "batch", "where": "Serverless orchestration task", "latency": "~1–5 min/run",
         "status": step_status(gold_n, orchestration_running), "count": gold_n},
        {"key": "serving", "label": f"Lakebase ({s.lakebase.instance})", "calls": len(states),
         "mode": "real-time", "where": f"Lakebase Postgres {s.lakebase.schema_name}.*", "latency": "sub-second serving",
         "status": step_status(len(states), False), "count": len(states)},
    ]

    return {
        "deployment": s.deployment,
        "mode": s.mode,
        "stt_provider": s.providers.stt.active,
        "enrichment": {"model_endpoint": s.enrichment.model_endpoint},
        "jobs": jobs,
        "stages": stages,
        "counts": counts,
        "call_states": states,
    }


def _as_int(v) -> int | None:
    try:
        return int(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _pipeline_meta(s) -> dict:
    """Non-blocking {counts, jobs}.

    Returns the last cached value immediately and refreshes it in the background
    when stale. On the very first call (nothing cached yet) it returns a light
    "warming up" placeholder instead of paying the warehouse/Jobs cold start
    inline, so `/status` stays fast and the call list renders right away.
    """
    now = time.monotonic()
    value = _meta_cache["value"]
    is_fresh = value is not None and now - float(_meta_cache["ts"]) < _META_TTL_S
    if not is_fresh:
        _kick_meta_refresh(s)
    if value is not None:
        return value  # type: ignore[return-value]
    return {
        "counts": {"note": "warming up"},
        "jobs": {"orchestration": {"name": s.pipeline.orchestration_job_name, "available": None}},
    }


def _kick_meta_refresh(s) -> None:
    """Start one background refresh of the heavy meta if none is already running."""
    with _meta_lock:
        if _meta_cache["refreshing"]:
            return
        _meta_cache["refreshing"] = True
    threading.Thread(target=_refresh_meta, args=(s,), daemon=True, name="status-meta").start()


def _refresh_meta(s) -> None:
    """Recompute counts + job run-state (warehouse + Jobs API) and update the cache."""
    try:
        value = {"counts": _medallion_counts(s), "jobs": _job_states(s)}
        _meta_cache["value"] = value
        _meta_cache["ts"] = time.monotonic()
    except Exception:  # noqa: BLE001
        pass
    finally:
        _meta_cache["refreshing"] = False


def warm_meta(s) -> None:
    """Synchronously populate the meta cache. Used by API-startup warming so the
    first real `/status` poll already has counts + job state available."""
    _refresh_meta(s)


def _medallion_counts(s) -> dict:
    # Online-ness is independent of the STT mock/live mode: the normal U2M demo
    # runs mode=mock but writes real Delta tables. Offline is signaled by the
    # local volume dir; otherwise we just need a warehouse to query.
    if os.environ.get("GENIE_LOCAL_VOLUME_DIR"):
        return _offline_counts()
    if not s.databricks.sql_warehouse_id:
        return {"note": "Set databricks.sql_warehouse_id for medallion counts"}
    try:
        from genie_voice.databricks.client import get_workspace_client

        client = get_workspace_client(s)
        wh = s.databricks.sql_warehouse_id
        out: dict = {}
        for tbl in [
            f"{s.lakebase.cdf_history_prefix}call_facts{s.lakebase.cdf_history_suffix}",
            s.medallion.gold_call_insights,
        ]:
            r = client.statement_execution.execute_statement(
                warehouse_id=wh,
                statement=f"SELECT count(*) FROM {s.fqtn(tbl)}",
                wait_timeout="30s",
            )
            val = r.result.data_array[0][0] if r.result and r.result.data_array else 0
            out[tbl] = int(val)
        return out
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def _offline_counts() -> dict:
    """Row counts from the local JSON exports written by the offline pipeline."""
    import glob
    import json

    base = os.environ.get("GENIE_LOCAL_VOLUME_DIR", "")
    out: dict = {}
    for fname in glob.glob(os.path.join(base, "tables", "*.json")):
        tbl = os.path.splitext(os.path.basename(fname))[0]
        try:
            with open(fname) as fh:
                rows = json.load(fh)
            out[tbl] = len(rows) if isinstance(rows, list) else 0
        except Exception:  # noqa: BLE001
            continue
    return out or {"note": "offline mode - no local table exports found yet"}


def _job_states(s) -> dict:
    """Best-effort current run-state of the STREAM and BATCH serverless jobs.

    Offline (no warehouse / local volume) we cannot reach the Jobs API, so we
    report `unknown` and let row counts convey progress instead.
    """
    if os.environ.get("GENIE_LOCAL_VOLUME_DIR") or not s.databricks.sql_warehouse_id:
        return {
            "orchestration": {"name": s.pipeline.orchestration_job_name, "available": False},
        }
    try:
        from genie_voice.databricks.client import get_workspace_client

        client = get_workspace_client(s)
        return {
            "orchestration": _job_state(client, s.pipeline.orchestration_job_name),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "orchestration": {"name": s.pipeline.orchestration_job_name, "available": False, "error": str(exc)},
        }


_RUNNING_STATES = {"PENDING", "RUNNING", "TERMINATING", "QUEUED", "BLOCKED"}


def _job_state(client, name: str) -> dict:
    """Latest run lifecycle/result for a job looked up by name."""
    info: dict = {"name": name, "available": True, "running": False}
    try:
        job = next(iter(client.jobs.list(name=name, limit=1)), None)
        if job is None:
            return {"name": name, "available": False, "deployed": False}
        info["deployed"] = True
        run = next(iter(client.jobs.list_runs(job_id=job.job_id, limit=1)), None)
        if run is None:
            info["last_result"] = None
            return info
        st = run.state
        life = getattr(getattr(st, "life_cycle_state", None), "value", None) or str(getattr(st, "life_cycle_state", ""))
        res = getattr(getattr(st, "result_state", None), "value", None) or (
            str(st.result_state) if getattr(st, "result_state", None) else None
        )
        info["life_cycle_state"] = life
        info["last_result"] = res
        info["running"] = life in _RUNNING_STATES
        return info
    except Exception as exc:  # noqa: BLE001
        return {"name": name, "available": False, "error": str(exc)}
