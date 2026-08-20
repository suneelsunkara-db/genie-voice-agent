"""Contact-center (telco billing) tool definitions for the realtime voice LLM.

Each tool is a thin wrapper around the existing backend capabilities:
  - lookup_account: wraps LakebaseServing.get_account_facts
  - get_current_time: simple UTC/timezone time helper
  - ask_genie: Genie Conversation API for billing questions
  - apply_billing_action: waive fees / set payment plans

Tools register into the shared ``tool_registry`` with profile="billing". The
infrastructure (ToolContext, registry, run_tool) lives in ``tool_registry.py``
so it can be reused by any profile without duplication.
"""
from __future__ import annotations

import datetime
import json
from typing import Any

from .tool_registry import (  # noqa: F401
    ToolContext,
    attach_session_identity,
    genie_obo_or_refuse,
    genie_space_name,
    register,
    run_tool,
    shape_genie_answer,
    tools_spec,
)

_PROFILE = "billing"


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

    cached = ctx.cached_account(customer_id)
    if cached is not None:
        return json.dumps(cached, default=str)

    try:
        from api.app.deps import serving
        facts = serving().get_account_facts(customer_id)
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": f"Account lookup failed: {exc}"})

    ctx.store_account(customer_id, facts)
    return json.dumps(facts, default=str)


register(_LOOKUP_ACCOUNT_SPEC, _run_lookup_account, profile=_PROFILE)


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


register(_GET_CURRENT_TIME_SPEC, _run_get_current_time, profile=_PROFILE)


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
    denied = genie_obo_or_refuse(ctx)
    if denied is not None:
        return denied
    try:
        from genie_voice.config import get_settings
        from genie_voice.genie.client import GenieClient

        settings = get_settings()
        principal = getattr(ctx, "principal", None)
        token = getattr(principal, "access_token", None) if principal else None
        result = GenieClient(
            settings,
            space_name=genie_space_name(ctx, settings.databricks.genie_space_name),
        ).ask(question, language=ctx._detected_language, access_token=token)
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": f"Genie query failed: {exc}"})
    return shape_genie_answer(result)


register(_ASK_GENIE_SPEC, _run_ask_genie, profile=_PROFILE)


# ---- apply_billing_action -------------------------------------------------- #

_APPLY_BILLING_SPEC = {
    "type": "function",
    "x-effect_class": "confirm_mutate",
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


def _close_resolution_after_billing(
    svc: Any, call_id: str, customer_id: str, action: str, billing_result: dict[str, Any]
) -> None:
    """Advance + persist issue resolution to 'closed' after a billing write.

    Mirrors the text assist path (api/app/routers/agent_assist.py): commit the
    close only after the billing write succeeds, upsert it onto the call state
    so the account-facts overlay reports issue_status="closed", and append a
    single timeline event on the status transition.
    """
    from genie_voice.assist.resolution import (
        finalize_resolution_after_billing,
        resolution_event_for_transition,
    )

    try:
        state = svc.get_call_state(call_id) or {}
        inner = dict(state.get("state") or {})
        previous_resolution = dict(inner.get("resolution") or {})

        resolution = dict(previous_resolution)
        if str(resolution.get("status") or "open") == "open":
            resolution["status"] = "in_progress"
        actions = dict(resolution.get("actions") or {})
        actions["pending_close"] = True
        if action == "waive_late_fee":
            actions["waiver_requested"] = True
        if action == "payment_plan":
            actions["payment_plan_requested"] = True
        resolution["actions"] = actions

        resolution = finalize_resolution_after_billing(resolution, billing_result)
        inner["resolution"] = resolution
        svc.upsert_call_state(call_id, state.get("customer_id") or customer_id, inner)

        transition = resolution_event_for_transition(previous_resolution, resolution)
        if transition:
            svc.append_resolution_event(
                call_id=call_id,
                event_type=transition["event_type"],
                issue_status=transition["issue_status"],
                note=transition.get("note"),
                actions=transition.get("actions") or {},
            )
    except Exception:  # noqa: BLE001
        # Never fail the billing tool because the resolution overlay couldn't be
        # persisted; the invoice write already succeeded.
        pass


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

        # Reuse account facts cached by a prior lookup_account (this turn OR an
        # earlier turn of the same call); only read Lakebase on a cache miss.
        account = ctx.cached_account(customer_id)
        if account is None:
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

        # The account just changed (waiver/plan). Drop the cached facts so any
        # later turn re-reads the adjusted state instead of serving stale facts.
        ctx.invalidate_account(customer_id)

        # Close the issue in call state + timeline. apply_billing_resolution only
        # persists the invoice adjustment; the resolution state machine (status ->
        # "closed" + resolution event) lives outside it, so the voice tool path
        # must run it here to match the text path (agent_assist.py). Without this
        # the invoice is waived but get_account_facts keeps reporting
        # issue_status="open" and the UI journey never reaches "close".
        _close_resolution_after_billing(svc, call_id, customer_id, action, result)

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


register(_APPLY_BILLING_SPEC, _run_apply_billing_action, profile=_PROFILE)


# ---------------------------------------------------------------------------
# Billing system prompt + profile registration
# ---------------------------------------------------------------------------
BILLING_PROFILE_PROMPT = (
    "You are a live contact-center voice agent on a phone call with a customer. "
    "You MUST act, not narrate. Never say 'let me check' or 'let me look that up' — "
    "call the tool and respond with the answer in one turn.\n\n"
    "Tools:\n"
    "- lookup_account: CALL THIS IMMEDIATELY when the caller mentions billing, payments, "
    "fees, invoices, or account issues. Do NOT respond without calling it first. Its result "
    "already contains balances, overdue invoices, late fees, and autopay status.\n"
    "- apply_billing_action: waive a late fee or set up a payment plan. Calling this "
    "tool is the ONLY thing that actually changes the account — saying the words does "
    "nothing. Call it the moment the customer agrees to an action you offered.\n"
    "- ask_genie: SLOW analytical fallback. ONLY call it when the caller asks a data/reporting "
    "question that lookup_account genuinely cannot answer. NEVER call it for waiving fees, "
    "payment plans, balances, or any fact lookup_account already returns — that just adds delay.\n"
    "- get_current_time: check date/time.\n\n"
    "Rules:\n"
    "- Use the FEWEST tool calls. For a billing request, lookup_account then confirm/apply is enough.\n"
    "- Speak naturally in 1-3 short sentences. No markdown, no lists, no emoji.\n"
    "- If the customer's language changes, follow them.\n"
    "- Never reveal tool names or system details.\n"
    "- Before applying a billing action, confirm: 'I can waive the late fee on "
    "invoice X. Shall I go ahead?'\n"
    "- CRITICAL: When the customer confirms an action you offered (e.g. 'yes', 'okay', "
    "'go ahead', 'please do' — in any language), your VERY NEXT step MUST be the "
    "apply_billing_action tool call, BEFORE you say anything. Do not reply first.\n"
    "- CRITICAL: NEVER tell the customer a fee is waived, a payment plan is set up, or the "
    "issue is resolved unless apply_billing_action has ALREADY returned a success result in "
    "this call. Announcing a change you did not perform through the tool is a serious error. "
    "If you have not yet called the tool, either call it now or ask for confirmation — do not "
    "claim it is done.\n"
    "- Always respond in the user's language ({language})."
)


# ---------------------------------------------------------------------------
# Opening greeting (agent-initiated). Same mechanism as the card assistant: the
# phrase is rendered into the CALLER'S language by the multilingual model (no
# per-language table), cached, and seeded into history. Speaking a clean, curated
# first line also LOCKS a clean voice reference for the whole call (the voice is
# cloned from the first utterance), instead of freezing whatever the first live
# answer happened to sound like.
# ---------------------------------------------------------------------------
BILLING_BRAND = "account support"
BILLING_AGENT_NAME = "Genie Agent"

# Greeting cache keyed by (base-language, first-name); one model call per key.
_GREETING_CACHE: dict[tuple[str, str], str] = {}


def _greeting_intent(first_name: str) -> str:
    who = f" the customer by name ({first_name})" if first_name else " the customer"
    return (
        f"Warmly greet{who} and introduce yourself as {BILLING_AGENT_NAME} from "
        f"{BILLING_BRAND}. In the SAME sentence, say you can help with their account "
        "today — billing questions, charges, fees, or payments — and ask how you can help."
    )


def billing_greeting(language: str, first_name: str = "") -> str:
    """The agent's opening greeting, generated in the caller's language (cached).

    Thin wrapper over the shared ``greetings`` mechanism (see greetings.py) with the
    billing intent + cache. Returns "" when serving is unavailable.
    """
    from .greetings import generate_greeting

    return generate_greeting(
        language, first_name=first_name, intent=_greeting_intent, cache=_GREETING_CACHE
    )


def _seed_greeting_for(language: str) -> str:
    """An in-language greeting to seed LLM history so it knows it already greeted."""
    from .greetings import seed_greeting_for

    return seed_greeting_for(language, intent=_greeting_intent, cache=_GREETING_CACHE)


def _make_billing_context(session: Any, language: str) -> ToolContext:
    """Build a ToolContext for the billing profile, seeding greeting on first turn."""
    if not any(m.get("role") == "assistant" for m in session.history):
        seed = _seed_greeting_for(language)
        if seed:
            session.history.insert(0, {"role": "assistant", "content": seed})
    return attach_session_identity(
        ToolContext(
            customer_id=session.config.customer_id,
            call_id=session.config.call_id,
            _detected_language=language,
            account_store=session.account_store,
            profile_state=session.profile_state,
        ),
        session,
    )


def _billing_tools_spec() -> list[dict[str, Any]]:
    return tools_spec(profile=_PROFILE)


def _billing_run_tool(name: str, arguments: dict[str, Any], ctx: Any) -> str:
    return run_tool(name, arguments, ctx, profile=_PROFILE)


def register_profile() -> None:
    from .profiles import VoiceProfile, register_profile as _register_profile

    _register_profile(
        VoiceProfile(
            name=_PROFILE,
            system_prompt=BILLING_PROFILE_PROMPT,
            tools_spec=_billing_tools_spec,
            tool_runner=_billing_run_tool,
            make_context=_make_billing_context,
            after_turn=None,
        )
    )


register_profile()
