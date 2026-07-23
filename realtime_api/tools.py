"""Contact-center tool definitions for the realtime voice LLM.

Each tool is a thin wrapper around the existing backend capabilities:
  - lookup_account: wraps LakebaseServing.get_account_facts
  - get_current_time: simple UTC/timezone time helper

Tools accept JSON arguments from the LLM tool call and return a JSON-serialized
string result. They share a ToolContext that carries per-session dependencies
(the Lakebase client, call_id, customer_id) so they can be resolved without
global state.
"""
from __future__ import annotations

import datetime
import json
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ToolContext:
    """Per-session state available to tool implementations."""
    customer_id: str | None = None
    call_id: str | None = None
    _detected_language: str | None = field(default=None, repr=False)
    _account_cache: dict[str, Any] | None = field(default=None, repr=False)


# Registry: name -> (spec_dict, executor_fn)
_TOOL_REGISTRY: dict[str, tuple[dict[str, Any], Callable[[dict[str, Any], ToolContext], str]]] = {}


def _register(spec: dict[str, Any], fn: Callable[[dict[str, Any], ToolContext], str]) -> None:
    name = spec["function"]["name"]
    _TOOL_REGISTRY[name] = (spec, fn)


# ---- lookup_account -------------------------------------------------------- #

_LOOKUP_ACCOUNT_SPEC = {
    "type": "function",
    "function": {
        "name": "lookup_account",
        "description": (
            "Look up the customer's account facts: invoices (amounts, due dates, "
            "status), payment history, overdue balances, autopay status, and "
            "account-level risk. Use this before discussing anything billing-related."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "customer_id": {
                    "type": "string",
                    "description": (
                        "The customer ID to look up. If omitted, uses the "
                        "customer associated with the current call session."
                    ),
                },
            },
        },
    },
}


def _run_lookup_account(arguments: dict[str, Any], ctx: ToolContext) -> str:
    customer_id = arguments.get("customer_id") or ctx.customer_id
    if not customer_id:
        return json.dumps({"error": "No customer_id available. Ask the caller for their account number."})

    if ctx._account_cache and ctx._account_cache.get("customer_id") == customer_id:
        return json.dumps(ctx._account_cache)

    try:
        from api.app.deps import serving
        facts = serving().get_account_facts(customer_id)
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": f"Account lookup failed: {exc}"})

    ctx._account_cache = facts
    return json.dumps(facts, default=str)


_register(_LOOKUP_ACCOUNT_SPEC, _run_lookup_account)


# ---- get_current_time ------------------------------------------------------ #

_GET_CURRENT_TIME_SPEC = {
    "type": "function",
    "function": {
        "name": "get_current_time",
        "description": "Get the current date and time, optionally for a specific IANA timezone.",
        "parameters": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "IANA timezone name, e.g. 'Asia/Bangkok'. Defaults to UTC.",
                },
            },
        },
    },
}


def _run_get_current_time(arguments: dict[str, Any], _ctx: ToolContext) -> str:
    tz_name = str(arguments.get("timezone") or "UTC")
    try:
        from zoneinfo import ZoneInfo
        now = datetime.datetime.now(ZoneInfo(tz_name))
    except Exception:  # noqa: BLE001
        tz_name = "UTC"
        now = datetime.datetime.now(datetime.timezone.utc)
    return json.dumps({
        "timezone": tz_name,
        "iso": now.isoformat(timespec="seconds"),
        "spoken": now.strftime("%A, %B %d, %Y at %I:%M %p"),
    })


_register(_GET_CURRENT_TIME_SPEC, _run_get_current_time)


# ---- ask_genie ------------------------------------------------------------- #

_ASK_GENIE_SPEC = {
    "type": "function",
    "function": {
        "name": "ask_genie",
        "description": (
            "Ask Databricks Genie a natural-language question about the customer's "
            "billing data. Use for queries like 'how many overdue invoices does this "
            "customer have?', 'what is the total outstanding amount?', or 'show me "
            "recent payment history'. Returns a text answer and optionally rows/columns."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The natural-language question to ask Genie.",
                },
            },
            "required": ["question"],
        },
    },
}


def _run_ask_genie(arguments: dict[str, Any], ctx: ToolContext) -> str:
    question = arguments.get("question", "").strip()
    if not question:
        return json.dumps({"error": "question is required"})
    try:
        from api.app.deps import genie
        result = genie().ask(question, language=ctx._detected_language)
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": f"Genie query failed: {exc}"})
    safe = {
        "answer": result.get("answer") or result.get("description"),
        "rows": result.get("rows"),
        "columns": result.get("columns"),
    }
    return json.dumps(safe, default=str)


_register(_ASK_GENIE_SPEC, _run_ask_genie)


# ---- apply_billing_action -------------------------------------------------- #

_APPLY_BILLING_SPEC = {
    "type": "function",
    "function": {
        "name": "apply_billing_action",
        "description": (
            "Apply a billing resolution action for the customer: waive late fees "
            "or set up a payment plan on their primary overdue invoice. Only call "
            "this when the customer explicitly requests it and you've confirmed the "
            "action with them."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["waive_late_fee", "payment_plan"],
                    "description": "The billing action to apply.",
                },
            },
            "required": ["action"],
        },
    },
}


def _run_apply_billing_action(arguments: dict[str, Any], ctx: ToolContext) -> str:
    action = arguments.get("action", "")
    if action not in ("waive_late_fee", "payment_plan"):
        return json.dumps({"error": "action must be 'waive_late_fee' or 'payment_plan'"})

    customer_id = ctx.customer_id
    call_id = ctx.call_id
    if not customer_id or not call_id:
        return json.dumps({"error": "No customer/call context. Cannot apply billing action."})

    try:
        from api.app.deps import serving
        svc = serving()

        # Reuse cached account facts from lookup_account (avoids redundant Lakebase read)
        account = ctx._account_cache
        if not account or account.get("customer_id") != customer_id:
            account = svc.get_account_facts(customer_id)
        if not account.get("found"):
            return json.dumps({"error": f"No account found for {customer_id}"})

        resolution = {
            "actions": {
                "waiver_applied": action == "waive_late_fee",
                "payment_plan_applied": action == "payment_plan",
            }
        }
        result = svc.apply_billing_resolution(call_id, customer_id, resolution, account)
        if not result.get("applied"):
            return json.dumps({"applied": False, "reason": result.get("reason", "unknown")})

        adjustment = result.get("adjustment", {})
        return json.dumps({
            "applied": True,
            "invoice_id": adjustment.get("invoice_id"),
            "amount_before": adjustment.get("amount_before"),
            "amount_after": adjustment.get("amount_after"),
            "late_fee_before": adjustment.get("late_fee_before"),
            "late_fee_after": adjustment.get("late_fee_after"),
            "status_after": adjustment.get("status_after"),
        })
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": f"Billing action failed: {exc}"})


_register(_APPLY_BILLING_SPEC, _run_apply_billing_action)


# ---- Public API ------------------------------------------------------------ #

def tools_spec() -> list[dict[str, Any]]:
    """OpenAI-format tool definitions for the LLM."""
    return [spec for spec, _ in _TOOL_REGISTRY.values()]


def run_tool(name: str, arguments: dict[str, Any], ctx: ToolContext) -> str:
    """Execute a tool by name and return its JSON result string."""
    entry = _TOOL_REGISTRY.get(name)
    if not entry:
        return json.dumps({"error": f"unknown tool: {name}"})
    _, fn = entry
    return fn(arguments, ctx)
