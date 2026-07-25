"""Voice observability endpoints: per-turn LLM/tool traces.

Reads the ``voice_traces`` Lakebase table populated (off the hot path) by the
realtime voice pipeline. Powers the in-app tracing view — an end-to-end look at
each turn's STT → LLM iterations (with the full messages the model saw) → tool
calls (arguments + results) → TTS.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..deps import serving

router = APIRouter(prefix="/traces", tags=["traces"])


@router.get("")
def list_traces(limit: int = 100, session_id: str | None = None, call_id: str | None = None) -> dict:
    limit = max(1, min(int(limit), 500))
    rows = serving().list_voice_traces(limit=limit, session_id=session_id, call_id=call_id)
    return {"traces": rows, "count": len(rows)}


@router.get("/sessions")
def list_sessions(limit: int = 200) -> dict:
    """Group recent traces into sessions (one row per call/WS connection).

    Rollups are objective, language-agnostic tool-call facts so the view can flag
    e.g. a session that looked up the account repeatedly but never applied a
    billing action — without any text/keyword heuristics.
    """
    limit = max(1, min(int(limit), 500))
    rows = serving().list_voice_traces(limit=limit)
    sessions: dict[str, dict] = {}
    for row in rows:
        sid = row.get("session_id") or "(none)"
        sess = sessions.setdefault(
            sid,
            {
                "session_id": sid,
                "call_id": row.get("call_id"),
                "customer_id": row.get("customer_id"),
                "turns": 0,
                "languages": set(),
                "apply_billing_action_called": False,
                "lookup_account_total": 0,
                "statuses": {},
                "latest": row.get("started_at") or row.get("created_at"),
            },
        )
        sess["turns"] += 1
        if row.get("language"):
            sess["languages"].add(row["language"])
        sess["apply_billing_action_called"] = sess["apply_billing_action_called"] or bool(
            row.get("apply_billing_action_called")
        )
        sess["lookup_account_total"] += int(row.get("lookup_account_count") or 0)
        status = str(row.get("status") or "ok")
        sess["statuses"][status] = sess["statuses"].get(status, 0) + 1
    out = []
    for sess in sessions.values():
        sess["languages"] = sorted(sess["languages"])
        out.append(sess)
    return {"sessions": out, "count": len(out)}


@router.get("/{trace_id}")
def get_trace(trace_id: str) -> dict:
    trace = serving().get_voice_trace(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail=f"No trace {trace_id}")
    return trace
