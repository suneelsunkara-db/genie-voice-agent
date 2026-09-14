"""Voice observability endpoints: per-turn LLM/tool traces.

Reads the ``voice_traces`` Lakebase table populated (off the hot path) by the
realtime voice pipeline. Powers the in-app tracing view — an end-to-end look at
each turn's STT → LLM iterations (with the full messages the model saw) → tool
calls (arguments + results) → TTS.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from genie_voice.config import get_settings

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
                    "enforcer": entry.get("enforcer"),
                    "phase": entry.get("phase"),
                    "resource": entry.get("resource"),
                    "policy_version": entry.get("policy_version"),
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
                        "phase": entry.get("phase"),
                        "resource": entry.get("resource"),
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


def _statement_rows(result) -> list[dict]:
    columns = [
        column.name
        for column in (((result.manifest or {}).get("schema") or {}).get("columns") or [])
    ] if isinstance(result.manifest, dict) else [
        column.name for column in (result.manifest.schema.columns if result.manifest else [])
    ]
    data = result.result.data_array if result.result and result.result.data_array else []
    return [dict(zip(columns, row, strict=False)) for row in data]


@router.get("/gateway")
def gateway_insights() -> dict:
    """Live configuration and seven-day traffic for app-owned model services.

    Attachment writes remain UI-only during the Beta. The public read API now
    exposes attached policies, so this endpoint verifies each handler, phase,
    and rank against the manifest instead of relying on a synthetic probe.
    """
    settings = get_settings()
    gateway = settings.ai_gateway
    from genie_voice.guardrails import get_policy_manifest

    manifest = get_policy_manifest()
    if not gateway.enabled:
        return {
            "enabled": False,
            "services": [],
            "policy_manifest": {
                "version": manifest.version,
                "catalog": manifest.catalog(),
                "bundles": {
                    key: value.model_dump() for key, value in manifest.bundles.items()
                },
                "assignments": manifest.assignments.model_dump(),
            },
        }

    from genie_voice.databricks.client import get_workspace_client

    client = get_workspace_client(settings)
    services: list[dict] = []
    for key, configured in gateway.model_services.items():
        bundle_id = manifest.assignments.model_services[key]
        gateway_policy_ids = manifest.bundles[bundle_id].policies
        required_policies = [
            manifest.gateway_policies[policy_id].model_dump()
            | {"policy_id": policy_id}
            for policy_id in gateway_policy_ids
        ]
        item = {
            "key": key,
            "service": configured.service,
            "destination": configured.destination,
            "roles": configured.roles,
            "policy_bundle": bundle_id,
            "required_policies": required_policies,
            "policy_deployment_state": "external_action_required",
            "rate_limits": [],
            "inference_table": None,
            "traffic_7d": None,
        }
        try:
            model_service = client.api_client.do(
                "GET",
                f"/api/2.1/unity-catalog/model-services/{configured.service}",
            )
            config = model_service.get("config") or {}
            item["rate_limits"] = config.get("rate_limits") or []
            table = (config.get("inference_table") or {}).get("table")
            item["inference_table"] = str(table or "").removeprefix("tables/") or None
            policies, fully_configured = manifest.gateway_deployment(
                key, config.get("service_policies") or []
            )
            item["required_policies"] = policies
            item["policy_deployment_state"] = (
                "configured" if fully_configured else "external_action_required"
            )
        except Exception as exc:  # noqa: BLE001
            item["configuration_error"] = str(exc)[:500]
            services.append(item)
            continue

        if item["inference_table"] and settings.databricks.sql_warehouse_id:
            table_name = str(item["inference_table"]).replace("`", "``")
            try:
                result = client.statement_execution.execute_statement(
                    warehouse_id=settings.databricks.sql_warehouse_id,
                    statement=f"""
                        SELECT
                          count(*) AS requests,
                          count_if(status_code >= 400) AS errors,
                          round(avg(latency_ms), 1) AS avg_latency_ms,
                          round(percentile_approx(latency_ms, 0.95), 1) AS p95_latency_ms,
                          max(event_time) AS last_event_time
                        FROM `{table_name.replace('.', '`.`')}`
                        WHERE event_time >= current_timestamp() - INTERVAL 7 DAYS
                    """,
                    wait_timeout="30s",
                )
                rows = _statement_rows(result)
                item["traffic_7d"] = rows[0] if rows else None
            except Exception as exc:  # noqa: BLE001
                # The table is created only after first traffic and logs can lag.
                item["traffic_note"] = str(exc)[:300]
        services.append(item)

    return {
        "enabled": True,
        "services": services,
        "policy_manifest": {
            "version": manifest.version,
            "catalog": manifest.catalog(),
            "bundles": {
                key: value.model_dump() for key, value in manifest.bundles.items()
            },
            "assignments": manifest.assignments.model_dump(),
        },
        "policy_attachment": {
            "mode": "ui_only_beta",
            "observable_via_public_api": True,
            "note": (
                "Attachment writes remain UI-only. Deployment and this page verify the "
                "public read state against the manifest; DENY decisions are recorded in traces."
            ),
        },
    }


@router.get("/{trace_id}")
def get_trace(trace_id: str) -> dict:
    trace = serving().get_voice_trace(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail=f"No trace {trace_id}")
    return trace
