"""Voice observability endpoints: per-turn LLM/tool traces.

Reads the ``voice_traces`` Lakebase table populated (off the hot path) by the
realtime voice pipeline. Powers the in-app tracing view — an end-to-end look at
each turn's STT → LLM iterations (with the full messages the model saw) → tool
calls (arguments + results) → TTS.
"""
from __future__ import annotations

import json

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
    standalone = serving().list_guard_events(limit=limit)
    conversation_rows = [
        row for row in rows if row.get("traffic_class") == "conversation"
    ]
    legacy_rows = [row for row in rows if not row.get("traffic_class")]

    totals: dict[str, int] = {}
    guards: dict[str, dict] = {}
    by_language: dict[str, dict[str, int]] = {}
    recent_fired: list[dict] = []
    turns_with_roster = 0
    turn_checks = 0

    for row in conversation_rows:
        roster = [e for e in (row.get("guard_roster") or []) if e.get("surface", "guardrail") == "guardrail"]
        if roster:
            turns_with_roster += 1
            turn_checks += len(roster)
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
                        "context": "turn",
                    }
                )

    checks = sum(totals.values())
    return {
        "turns": len(conversation_rows),
        "legacy_unclassified_turns": len(legacy_rows),
        "standalone_events": len(standalone),
        "standalone_recent": standalone[:50],
        "turns_with_roster": turns_with_roster,
        "checks": checks,
        "checks_per_turn": (
            round(turn_checks / turns_with_roster, 2)
            if turns_with_roster
            else 0.0
        ),
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


def _string_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if item]
    if isinstance(value, str) and value:
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(item) for item in parsed if item]
        except json.JSONDecodeError:
            return []
    return []


def _page_coverage() -> list[dict]:
    return [
        {"surface": "home", "profile": "concierge", "traffic_class": "conversation", "models": ["stt", "navigation", "realtime_llm", "tts"], "managed_services": []},
        {"surface": "telco", "profile": "billing", "traffic_class": "conversation", "models": ["stt", "navigation", "realtime_llm", "tts"], "managed_services": ["genie_space"]},
        {"surface": "card", "profile": "card", "traffic_class": "conversation", "models": ["stt", "navigation", "realtime_llm", "conversion", "tts"], "managed_services": ["genie_space", "genie_agent_mode"]},
        {"surface": "knowledge", "profile": "knowledge", "traffic_class": "conversation", "models": ["stt", "navigation", "realtime_llm", "conversion", "tts"], "managed_services": ["genie_one"]},
        {"surface": "setup", "profile": None, "traffic_class": "readiness_probe", "models": ["stt", "tts", "policy_probe"], "managed_services": ["genie_configuration_checks"]},
        {"surface": "voice-benchmarks", "profile": None, "traffic_class": "read_only", "models": [], "managed_services": ["benchmark_delta"]},
        {"surface": "asr-benchmark", "profile": None, "traffic_class": "read_only", "models": [], "managed_services": ["benchmark_artifacts"]},
        {"surface": "traces", "profile": None, "traffic_class": "read_only", "models": [], "managed_services": ["voice_traces"]},
        {"surface": "guardrails", "profile": None, "traffic_class": "read_only", "models": [], "managed_services": ["voice_traces", "gateway_inference_tables"]},
        {"surface": "realtime-test", "profile": "selected", "traffic_class": "diagnostic", "models": ["stt", "navigation", "realtime_llm", "tts"], "managed_services": []},
        {"surface": "mcp", "profile": "selected", "traffic_class": "mcp", "models": ["stt", "navigation", "realtime_llm", "conversion", "tts"], "managed_services": ["profile_dependent"]},
    ]


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * percentile)
    return round(ordered[index], 1)


def _speech_endpoint_insights(
    conversation_traces: dict[str, dict], realtime
) -> list[dict]:
    """Aggregate custom Model Serving calls without claiming Gateway telemetry."""
    stats: dict[tuple[str, str], dict] = {}
    for trace_id, row in conversation_traces.items():
        for call in row.get("model_calls") or []:
            if str(call.get("transport") or "") != "model_serving":
                continue
            endpoint = str(call.get("endpoint") or "")
            role = str(call.get("model_role") or "unknown")
            key = (endpoint, role)
            stat = stats.setdefault(
                key,
                {
                    "endpoint": endpoint,
                    "model_role": role,
                    "requests": 0,
                    "errors": 0,
                    "_durations": [],
                    "_trace_ids": set(),
                    "_surfaces": set(),
                    "_profiles": set(),
                    "_ttfb": [],
                    "_generation": [],
                },
            )
            stat["requests"] += 1
            if call.get("status") != "ok":
                stat["errors"] += 1
            duration = call.get("duration_ms")
            if isinstance(duration, (int, float)):
                stat["_durations"].append(float(duration))
            first_call_for_trace = trace_id not in stat["_trace_ids"]
            stat["_trace_ids"].add(trace_id)
            if row.get("surface"):
                stat["_surfaces"].add(str(row["surface"]))
            if row.get("profile"):
                stat["_profiles"].add(str(row["profile"]))
            if role == "tts" and first_call_for_trace:
                ttfb = row.get("server_ttfb_ms") or row.get("tts_first_ms")
                generation = row.get("server_gen_ms")
                if isinstance(ttfb, (int, float)):
                    stat["_ttfb"].append(float(ttfb))
                if isinstance(generation, (int, float)):
                    stat["_generation"].append(float(generation))

    output: list[dict] = []
    for (endpoint, role), stat in stats.items():
        model_name = (
            "Qwen/Qwen3-ASR-1.7B"
            if endpoint == realtime.stt_endpoint and role == "stt"
            else "openbmb/VoxCPM2"
            if endpoint == realtime.tts_endpoint and role == "tts"
            else "Unclassified Model Serving call"
        )
        durations = stat.pop("_durations")
        trace_ids = stat.pop("_trace_ids")
        surfaces = stat.pop("_surfaces")
        profiles = stat.pop("_profiles")
        ttfb = stat.pop("_ttfb")
        generation = stat.pop("_generation")
        output.append(
            {
                **stat,
                "model_name": model_name,
                "trace_count": len(trace_ids),
                "surfaces": sorted(surfaces),
                "profiles": sorted(profiles),
                "avg_latency_ms": round(sum(durations) / len(durations), 1)
                if durations
                else None,
                "p95_latency_ms": _percentile(durations, 0.95),
                "avg_ttfb_ms": round(sum(ttfb) / len(ttfb), 1) if ttfb else None,
                "p95_ttfb_ms": _percentile(ttfb, 0.95),
                "avg_generation_ms": round(sum(generation) / len(generation), 1)
                if generation
                else None,
                "provenance_status": "verified" if trace_ids else "unavailable",
                "telemetry_source": "voice_trace_model_calls",
                "gateway_telemetry": False,
            }
        )
    return sorted(output, key=lambda item: (item["model_role"], item["endpoint"]))


def _attach_voice_gateway_config(client, speech: list[dict]) -> None:
    """Attach endpoint configuration without conflating it with Gateway request logs."""
    for item in speech:
        try:
            endpoint = client.api_client.do(
                "GET", f"/api/2.0/serving-endpoints/{item['endpoint']}"
            )
            gateway = endpoint.get("ai_gateway") or {}
            inference = gateway.get("inference_table_config") or {}
            item.update(
                {
                    "endpoint_task": endpoint.get("task"),
                    "ai_gateway_configured": bool(gateway),
                    "inference_table": inference,
                    "gateway_telemetry": bool(inference.get("enabled")),
                    "supported_gateway_features": ["inference_tables"],
                    "unsupported_gateway_features": [
                        "rate_limits",
                        "usage_tracking",
                        "fallback",
                        "chat_guardrails",
                    ],
                }
            )
        except Exception as exc:  # noqa: BLE001
            item.update(
                {
                    "ai_gateway_configured": False,
                    "gateway_configuration_error": str(exc),
                    "supported_gateway_features": ["inference_tables"],
                }
            )


def _model_inventory(settings, realtime, services: list[dict], speech: list[dict]) -> list[dict]:
    gateway_models = [
        {
            "id": item["key"],
            "name": item["destination"],
            "plane": "unity_ai_gateway",
            "roles": item["roles"],
            "resource": item["service"],
            "telemetry": "gateway_inference_table",
        }
        for item in services
    ]
    return [
        *gateway_models,
        {
            "id": "stt",
            "name": "Qwen/Qwen3-ASR-1.7B",
            "plane": "model_serving",
            "roles": ["stt"],
            "resource": realtime.stt_endpoint,
            "telemetry": "voice_trace_model_calls",
        },
        {
            "id": "tts",
            "name": "openbmb/VoxCPM2",
            "plane": "model_serving",
            "roles": ["tts"],
            "resource": realtime.tts_endpoint,
            "telemetry": "voice_trace_model_calls",
        },
        {
            "id": "gateway_evaluator",
            "name": "system.ai.gpt-5-2",
            "plane": "gateway_policy_evaluator",
            "roles": ["unsafe_content", "jailbreak", "hallucination_observer"],
            "resource": "attached_service_policies",
            "telemetry": "configuration_only",
        },
        {
            "id": "silero_vad",
            "name": "Silero VAD",
            "plane": "local_onnx",
            "roles": ["speech_detection"],
            "resource": "realtime_api/models",
            "telemetry": "local_runtime_only",
        },
        {
            "id": "smart_turn_v3",
            "name": "Smart Turn v3",
            "plane": "local_onnx",
            "roles": ["turn_completeness"],
            "resource": "realtime_api/models",
            "telemetry": "local_runtime_only",
        },
        {
            "id": "genie_space",
            "name": "Genie Space Conversation API",
            "plane": "managed_genie",
            "roles": ["telco_analytics", "card_analytics"],
            "resource": settings.databricks.genie_space_name,
            "telemetry": "tool_spans",
        },
        {
            "id": "genie_agent_mode",
            "name": "Genie Agent Mode",
            "plane": "managed_genie",
            "roles": ["card_deep_dive"],
            "resource": settings.card_issuer.genie_space_name,
            "telemetry": "tool_spans",
        },
        {
            "id": "genie_one",
            "name": "Genie One MCP",
            "plane": "managed_genie",
            "roles": ["workspace_knowledge"],
            "resource": "/api/2.0/mcp/genie",
            "telemetry": "tool_spans",
        },
        {
            "id": "batch_qwen",
            "name": settings.enrichment.batch_model_endpoint,
            "plane": "sql_ai_query",
            "roles": ["batch_gold_insights"],
            "resource": "pipeline_job",
            "telemetry": "warehouse_history",
        },
    ]


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

    from realtime_api.config import RealtimeSettings

    client = get_workspace_client(settings)
    realtime = RealtimeSettings.resolve()
    try:
        me = client.current_user.me()
        requesters = sorted({
            str(value)
            for value in (getattr(me, "id", None), getattr(me, "user_name", None))
            if value
        })
    except Exception:  # noqa: BLE001
        requesters = []
    requester_sql = ", ".join(
        f"'{value.replace(chr(39), chr(39) * 2)}'" for value in requesters
    ) or "''"
    trace_rows = serving().list_voice_traces(limit=5000)
    conversation_traces = {
        str(row.get("trace_id")): row
        for row in trace_rows
        if row.get("traffic_class") == "conversation" and row.get("trace_id")
    }
    speech = _speech_endpoint_insights(conversation_traces, realtime)
    _attach_voice_gateway_config(client, speech)
    expected_gateway_calls: dict[str, int] = {}
    for row in conversation_traces.values():
        for call in row.get("model_calls") or []:
            endpoint = str(call.get("endpoint") or "")
            transport = str(call.get("transport") or "")
            if transport == "unity_ai_gateway":
                expected_gateway_calls[endpoint] = expected_gateway_calls.get(endpoint, 0) + 1
    services: list[dict] = []
    recent_events: list[dict] = []
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
            "traffic_all_7d": None,
            "provenance": {"status": "unavailable", "reason": "inference table unavailable"},
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
            item["attached_policies"] = config.get("service_policies") or []
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
                        WITH base AS (
                          SELECT *,
                            CAST(coalesce(
                              get_json_object(response, '$.usage.prompt_tokens'),
                              get_json_object(response, '$.usage.input_tokens'), '0'
                            ) AS BIGINT) AS input_tokens,
                            CAST(coalesce(
                              get_json_object(response, '$.usage.completion_tokens'),
                              get_json_object(response, '$.usage.output_tokens'), '0'
                            ) AS BIGINT) AS output_tokens
                          FROM `{table_name.replace('.', '`.`')}`
                          WHERE event_time >= current_timestamp() - INTERVAL 7 DAYS
                        ),
                        conversation AS (
                          SELECT * FROM base
                          WHERE request_tags['app'] = 'genie-voice-agent'
                            AND request_tags['traffic_class'] = 'conversation'
                            AND requester IN ({requester_sql})
                        ),
                        per_minute AS (
                          SELECT date_trunc('minute', event_time) AS minute,
                            count(*) AS rpm,
                            sum(input_tokens + output_tokens) AS tpm
                          FROM conversation GROUP BY 1
                        )
                        SELECT
                          (SELECT count(*) FROM base) AS all_requests,
                          (SELECT count_if(status_code >= 400) FROM base) AS all_errors,
                          (SELECT count_if(request_tags['traffic_class'] IS NULL) FROM base)
                            AS unclassified_requests,
                          count(*) AS requests,
                          count_if(status_code >= 400) AS errors,
                          count_if(status_code = 429) AS rate_limited,
                          sum(input_tokens) AS input_tokens,
                          sum(output_tokens) AS output_tokens,
                          sum(input_tokens + output_tokens) AS total_tokens,
                          round(avg(input_tokens), 1) AS avg_input_tokens,
                          round(avg(output_tokens), 1) AS avg_output_tokens,
                          round(avg(latency_ms), 1) AS avg_latency_ms,
                          round(percentile_approx(latency_ms, 0.95), 1) AS p95_latency_ms,
                          round(avg(time_to_first_byte_ms), 1) AS avg_ttft_ms,
                          round(percentile_approx(time_to_first_byte_ms, 0.95), 1)
                            AS p95_ttft_ms,
                          coalesce((SELECT max(rpm) FROM per_minute), 0) AS peak_rpm,
                          coalesce((SELECT max(tpm) FROM per_minute), 0) AS peak_tpm,
                          count_if(status_code = 200 AND
                            get_json_object(response, '$.databricks_service_policy') IS NOT NULL)
                            AS policy_envelopes,
                          count_if(
                            request_tags['trace_id'] IS NULL OR
                            request_tags['session_id'] IS NULL OR
                            request_tags['turn_id'] IS NULL OR
                            request_tags['profile'] IS NULL OR
                            request_tags['surface'] IS NULL OR
                            request_tags['model_role'] IS NULL
                          ) AS incomplete_provenance,
                          collect_set(request_tags['trace_id']) AS trace_ids,
                          max(event_time) AS last_event_time
                        FROM conversation
                    """,
                    wait_timeout="30s",
                )
                rows = _statement_rows(result)
                metrics = rows[0] if rows else {}
                trace_ids = set(_string_list(metrics.pop("trace_ids", [])))
                matched = trace_ids & set(conversation_traces)
                unmatched = trace_ids - set(conversation_traces)
                requests = int(metrics.get("requests") or 0)
                expected = expected_gateway_calls.get(str(configured.service), 0)
                incomplete = int(metrics.get("incomplete_provenance") or 0)
                status = (
                    "verified"
                    if requests > 0 and not incomplete and not unmatched and requests >= expected
                    else "partial"
                    if requests > 0 or expected > 0
                    else "unavailable"
                )
                item["traffic_all_7d"] = {
                    "requests": metrics.pop("all_requests", 0),
                    "errors": metrics.pop("all_errors", 0),
                    "unclassified_requests": metrics.pop("unclassified_requests", 0),
                }
                item["traffic_7d"] = metrics
                item["provenance"] = {
                    "status": status,
                    "requesters": requesters,
                    "expected_trace_calls": expected,
                    "tagged_requests": requests,
                    "matched_trace_ids": len(matched),
                    "unmatched_trace_ids": sorted(unmatched),
                    "incomplete_requests": incomplete,
                    "filters": {
                        "app": "genie-voice-agent",
                        "traffic_class": "conversation",
                        "requesters": requesters,
                    },
                }
                recent = client.statement_execution.execute_statement(
                    warehouse_id=settings.databricks.sql_warehouse_id,
                    statement=f"""
                        SELECT event_time, request_id, invocation_id, status_code,
                          latency_ms, time_to_first_byte_ms, destination_model,
                          request_tags['trace_id'] AS trace_id,
                          request_tags['session_id'] AS session_id,
                          request_tags['turn_id'] AS turn_id,
                          request_tags['profile'] AS profile,
                          request_tags['surface'] AS surface,
                          request_tags['model_role'] AS model_role
                        FROM `{table_name.replace('.', '`.`')}`
                        WHERE event_time >= current_timestamp() - INTERVAL 7 DAYS
                          AND request_tags['app'] = 'genie-voice-agent'
                          AND request_tags['traffic_class'] = 'conversation'
                          AND requester IN ({requester_sql})
                        ORDER BY event_time DESC LIMIT 25
                    """,
                    wait_timeout="30s",
                )
                for event in _statement_rows(recent):
                    event["service_key"] = key
                    event["trace_matched"] = str(event.get("trace_id") or "") in conversation_traces
                    recent_events.append(event)
            except Exception as exc:  # noqa: BLE001
                # The table is created only after first traffic and logs can lag.
                item["traffic_note"] = str(exc)[:300]
        services.append(item)

    statuses = [str(item.get("provenance", {}).get("status")) for item in services]
    overall_status = (
        "verified"
        if statuses and all(status == "verified" for status in statuses)
        else "unavailable"
        if statuses and all(status == "unavailable" for status in statuses)
        else "partial"
    )
    return {
        "enabled": True,
        "services": services,
        "provenance": {
            "status": overall_status,
            "window": "7d",
            "conversation_trace_count": len(conversation_traces),
            "legacy_unclassified_trace_count": sum(
                1 for row in trace_rows if not row.get("traffic_class")
            ),
            "requesters": requesters,
        },
        "recent_events": sorted(
            recent_events, key=lambda event: str(event.get("event_time") or ""), reverse=True
        )[:50],
        "speech_endpoints": speech,
        "model_inventory": _model_inventory(settings, realtime, services, speech),
        "page_coverage": _page_coverage(),
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
