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


@router.get("/guardrails")
def guardrail_rollup(limit: int = 200) -> dict:
    """Aggregate the per-turn guardrail ledger for the Guardrails view.

    Only ``surface == "guardrail"`` rows are counted: turn-integrity mechanics
    (empty transcript, stale turn) are real checks but not guardrails, and folding
    them into "N checks ran, none fired" would overstate the claim. The filter is
    the entry's own declared surface, never a name denylist here — a mechanic added
    later must not need remembering in two places.

    Coverage matters as much as incidents. ``passed`` and ``delegated`` are what
    make "23 checks ran on this turn, none fired" a statement about the system
    rather than about an empty table.
    """
    limit = max(1, min(int(limit), 500))
    rows = serving().list_voice_traces(limit=limit)

    totals: dict[str, int] = {}
    guards: dict[str, dict] = {}
    by_language: dict[str, dict[str, int]] = {}
    recent_fired: list[dict] = []
    turns_with_roster = 0

    for row in rows:
        roster = [e for e in (row.get("guard_roster") or []) if e.get("surface", "guardrail") == "guardrail"]
        if roster:
            turns_with_roster += 1
        language = str(row.get("language") or "unknown")
        for entry in roster:
            outcome = str(entry.get("outcome") or "unknown")
            guard_id = str(entry.get("guard_id") or "unknown")
            totals[outcome] = totals.get(outcome, 0) + 1
            guard = guards.setdefault(
                guard_id,
                {
                    "guard_id": guard_id,
                    "seam": entry.get("seam"),
                    "stage": entry.get("stage"),
                    "owner": entry.get("owner"),
                    "runs": 0,
                    "outcomes": {},
                    "last_reason": None,
                },
            )
            guard["runs"] += 1
            guard["outcomes"][outcome] = guard["outcomes"].get(outcome, 0) + 1
            if entry.get("reason") and guard["last_reason"] is None:
                guard["last_reason"] = entry["reason"]
            lang = by_language.setdefault(language, {})
            lang[outcome] = lang.get(outcome, 0) + 1
            if outcome == "fired" and len(recent_fired) < 50:
                recent_fired.append(
                    {
                        "trace_id": row.get("trace_id"),
                        "session_id": row.get("session_id"),
                        "turn_id": row.get("turn_id"),
                        "language": row.get("language"),
                        "created_at": row.get("created_at") or row.get("started_at"),
                        "guard_id": guard_id,
                        "stage": entry.get("stage"),
                        "reason": entry.get("reason"),
                    }
                )

    checks = sum(totals.values())
    return {
        "turns": len(rows),
        "turns_with_roster": turns_with_roster,
        "checks": checks,
        "checks_per_turn": round(checks / turns_with_roster, 2) if turns_with_roster else 0.0,
        "totals": totals,
        "guards": sorted(guards.values(), key=lambda g: (-g["runs"], g["guard_id"])),
        "by_language": by_language,
        "recent_fired": recent_fired,
    }


@router.get("/{trace_id}")
def get_trace(trace_id: str) -> dict:
    trace = serving().get_voice_trace(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail=f"No trace {trace_id}")
    return trace
