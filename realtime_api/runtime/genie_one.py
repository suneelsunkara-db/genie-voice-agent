"""OBO-only adapter for the managed Genie One MCP server."""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Callable

from .identity import SessionPrincipal, require_obo_token

WORKSPACE_QUERY_SPEC: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "workspace_query",
        "description": "Ask the user's governed Databricks workspace a data question.",
        "parameters": {
            "type": "object",
            "properties": {"question": {"type": "string"}},
            "required": ["question"],
        },
    },
}

_MAX_QUERY_RESULTS = 6
_MAX_RESULT_ROWS = 200

# Statuses that mean Genie is done with this response, one way or another.
TERMINAL_STATUSES = frozenset({"completed", "failed", "incomplete", "cancelled"})

_MAX_PROGRESS_STEPS = 12
# Genie reports progress as free-form steps whose field names are not part of any
# contract we own. Those strings may contain chain-of-thought, SQL, table names,
# query results, and other implementation details. They are classification input
# only: neither the wire contract nor TTS may receive the source text.
_STEP_LABEL_KEYS = ("title", "label", "name", "step", "summary", "description", "text")
_STEP_STATUS_KEYS = ("status", "state")
_DONE_STATUSES = frozenset({"completed", "complete", "done", "success", "succeeded", "finished"})
_ACTIVE_STATUSES = frozenset({"running", "in_progress", "active", "executing", "started", "pending"})

# Ordered, finite business vocabulary used by both the timeline and TTS narration.
# Keep labels free of product names and technical implementation terminology.
_BUSINESS_STAGES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "Preparing your answer",
        ("format", "summar", "prepar", "final answer", "respond"),
    ),
    (
        "Calculating the amount",
        ("cost", "spend", "price", "dollar", "billing", "charge", "sku"),
    ),
    (
        "Checking the results",
        ("query result", "returned", "rows", "validat", "verify", "check result"),
    ),
    (
        "Matching the requested item",
        ("model name", "potential match", "matching", "resolve", "identify", "distinct"),
    ),
    (
        "Analyzing usage",
        ("usage", "token", "consumption", "request count", "aggregate", "sum(", "count("),
    ),
    (
        "Finding the relevant information",
        (
            "sql",
            "query",
            "table",
            "dashboard",
            "catalog",
            "schema",
            "namespace",
            "source",
            "retriev",
            "search",
            "load",
        ),
    ),
)
_DEFAULT_BUSINESS_STAGE = "Understanding your question"


def _raw_step_text(item: Any) -> str:
    if isinstance(item, str):
        return item.strip()
    if not isinstance(item, dict):
        return ""
    for key in _STEP_LABEL_KEYS:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _business_stage(raw_text: str) -> str:
    """Classify an upstream implementation step into safe business language."""
    lowered = raw_text.casefold()
    for label, signals in _BUSINESS_STAGES:
        if any(signal in lowered for signal in signals):
            return label
    return _DEFAULT_BUSINESS_STAGE


def _step_status(item: Any) -> str:
    if not isinstance(item, dict):
        return ""
    for key in _STEP_STATUS_KEYS:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    return ""


def normalize_progress_steps(value: Any) -> list[dict[str, str]]:
    """Convert Genie's internal trace into business-safe ``{label, status}`` steps.

    Raw labels are never returned. Repeated internal operations collapse into one
    business stage, keeping a long analytical trace useful without exposing SQL,
    chain-of-thought, table names, query results, or Databricks terminology.
    """
    items = [value] if isinstance(value, str) else value
    if not isinstance(items, list):
        return []
    steps: list[dict[str, str]] = []
    for item in items[:_MAX_PROGRESS_STEPS]:
        raw_text = _raw_step_text(item)
        if not raw_text:
            continue
        label = _business_stage(raw_text)
        raw = _step_status(item)
        if raw in _DONE_STATUSES:
            status = "done"
        elif raw in _ACTIVE_STATUSES:
            status = "active"
        else:
            status = ""
        existing = next((step for step in steps if step["label"] == label), None)
        if existing is not None:
            # The latest occurrence is authoritative when upstream revisits a stage.
            existing["status"] = status
        else:
            steps.append({"label": label, "status": status})
    if steps and not any(step["status"] for step in steps):
        for step in steps[:-1]:
            step["status"] = "done"
        steps[-1]["status"] = "active"
    return [{"label": s["label"], "status": s["status"] or "pending"} for s in steps]


def current_progress_step(steps: list[dict[str, str]]) -> str:
    """The one step worth saying out loud: the active one, else the latest."""
    for step in reversed(steps):
        if step.get("status") == "active":
            return step.get("label", "")
    return steps[-1].get("label", "") if steps else ""


def _payload(result: Any) -> dict[str, Any]:
    structured = getattr(result, "structuredContent", None) or getattr(
        result, "structured_content", None
    )
    if isinstance(structured, dict):
        return structured
    for item in getattr(result, "content", None) or []:
        text = getattr(item, "text", None)
        if not text:
            continue
        try:
            value = json.loads(text)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            return {"answer": str(text)}
    return {}


async def _attach_query_results(
    client: Any,
    payload: dict[str, Any],
    *,
    conversation_id: str,
    response_id: str,
) -> dict[str, Any]:
    """Fetch typed rows for Genie's completed analytical query items.

    ``genie_poll_response`` returns narrative + item ids, not the result rows.
    Visualizations must use ``genie_get_query_result`` rather than parsing tables
    out of Markdown. Results are bounded before entering the WebSocket contract.
    """
    items = payload.get("query_items")
    if not isinstance(items, list) or not items:
        return payload
    results: list[dict[str, Any]] = []
    for item in items[:_MAX_QUERY_RESULTS]:
        if not isinstance(item, dict) or not item.get("item_id"):
            continue
        item_id = str(item["item_id"])
        try:
            result = _payload(
                await client.call_tool(
                    "genie_get_query_result",
                    {
                        "conversation_id": conversation_id,
                        "response_id": response_id,
                        "item_id": item_id,
                    },
                )
            )
        except Exception:  # noqa: BLE001
            continue
        raw_columns = result.get("columns") or []
        columns = [
            {
                "name": str(column.get("name") or ""),
                "type_text": str(column.get("type_text") or ""),
            }
            for column in raw_columns
            if isinstance(column, dict) and column.get("name")
        ]
        rows = result.get("rows") if isinstance(result.get("rows"), list) else []
        if not columns or not rows:
            continue
        results.append(
            {
                "item_id": item_id,
                "sql": item.get("sql"),
                "columns": columns,
                "rows": rows[:_MAX_RESULT_ROWS],
                "total_row_count": result.get("total_row_count", len(rows)),
                "truncated": bool(result.get("truncated"))
                or len(rows) > _MAX_RESULT_ROWS,
            }
        )
    if results:
        payload = dict(payload)
        payload["query_results"] = results
    return payload


async def _query(
    question: str,
    principal: SessionPrincipal,
    host: str,
    *,
    timeout_s: float,
    conversation_id: str | None = None,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    # Imported lazily: standalone STT/TTS deployments do not need MCP client code.
    import httpx
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    url = f"{host.rstrip('/')}/api/2.0/mcp/genie"
    # OBO travels on the HTTP client, which is where this transport takes auth.
    # Passing headers to streamable_http_client instead raises a TypeError that the
    # caller can only report as a generic failure, so keep the two in one place.
    async with httpx.AsyncClient(
        headers={"Authorization": f"Bearer {principal.access_token}"},
        timeout=httpx.Timeout(timeout_s, connect=30.0),
        follow_redirects=True,
    ) as http_client, streamable_http_client(url, http_client=http_client) as streams:
        read_stream, write_stream = streams[0], streams[1]
        async with ClientSession(read_stream, write_stream) as client:
            await client.initialize()
            # Continuing a conversation is what makes a follow-up ("the second one",
            # "Unity Catalog") mean anything: Genie resolves it against what it just
            # said. Omitting the id starts a fresh conversation with no memory of the
            # clarifying question it asked.
            ask_args: dict[str, Any] = {"question": question}
            if conversation_id:
                ask_args["conversation_id"] = conversation_id
            started = _payload(await client.call_tool("genie_ask", ask_args))
            if on_progress is not None:
                on_progress(dict(started))
            conversation_id = started.get("conversation_id") or conversation_id
            response_id = started.get("response_id")
            if not conversation_id or not response_id:
                return started
            deadline = time.monotonic() + timeout_s
            latest = started
            progress_fingerprint = ""
            while time.monotonic() < deadline:
                status = str(latest.get("status") or "").lower()
                if status in TERMINAL_STATUSES:
                    if status == "completed":
                        latest = await _attach_query_results(
                            client,
                            latest,
                            conversation_id=str(conversation_id),
                            response_id=str(response_id),
                        )
                    return latest
                await asyncio.sleep(1.0)
                latest = _payload(
                    await client.call_tool(
                        "genie_poll_response",
                        {
                            "conversation_id": conversation_id,
                            "response_id": response_id,
                        },
                    )
                )
                progress_value = {
                    "status": latest.get("status"),
                    "progress_steps": latest.get("progress_steps"),
                    "narration_instruction": latest.get("narration_instruction"),
                    "conversation_id": latest.get("conversation_id") or conversation_id,
                    "response_id": latest.get("response_id") or response_id,
                }
                fingerprint = json.dumps(progress_value, sort_keys=True, default=str)
                if on_progress is not None and fingerprint != progress_fingerprint:
                    progress_fingerprint = fingerprint
                    on_progress(progress_value)
            try:
                await client.call_tool(
                    "genie_cancel_response",
                    {
                        "conversation_id": conversation_id,
                        "response_id": response_id,
                    },
                )
            except Exception:  # noqa: BLE001
                pass
            # Carry the conversation id even on timeout: the caller can still follow
            # up in the same conversation instead of losing the thread.
            return {
                "error": "Genie One timed out",
                "timeout": True,
                "conversation_id": conversation_id,
            }


def run_workspace_query(
    question: str,
    *,
    principal: SessionPrincipal | None,
    host: str,
    session_id: str,
    turn_id: int,
    timeout_s: float = 180.0,
    conversation_id: str | None = None,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
) -> str:
    """Ask Genie One one question, optionally continuing ``conversation_id``.

    The returned JSON always carries the conversation id Genie used, so the caller
    can persist it for the next turn.
    """
    denied = require_obo_token(
        principal,
        session_id=session_id,
        turn_id=turn_id,
    )
    if denied is not None:
        return json.dumps(
            {
                "denied": True,
                "error": denied.message,
                "error_evidence": denied.as_dict(),
            }
        )
    try:
        result = asyncio.run(
            _query(
                question,
                principal,  # type: ignore[arg-type]
                host,
                timeout_s=timeout_s,
                conversation_id=conversation_id,
                on_progress=on_progress,
            )
        )
        return json.dumps(result, default=str)
    except Exception as exc:  # noqa: BLE001
        return json.dumps(
            {"error": f"Genie One failed: {exc}", "conversation_id": conversation_id}
        )
