"""Credit-card assistant tools for the realtime voice LLM (card profile).

Registers into the SAME shared ``tool_registry`` as the telco tools, scoped to
profile="card". Uses the same ``ToolContext`` — card-specific state lives in
``ctx.profile_state`` (keys: use_case, selected_use_case, facts_cache).

Two-lane Genie design:

  FAST lane (answer in the same turn):
    - card_account_facts : Lakebase fast-facts (balance position, points, status).
    - ask_card_genie     : Genie Conversation API on the card space for a quick,
                           specific fact (synchronous; returns rows + a sentence).

  DEEP lane (async, surfaced when ready):
    - start_deep_dive    : SIGNAL that the caller asked "why". Returns IMMEDIATELY
                           with the question; the UI runs the Genie Agent-Mode
                           investigation via the app backend's SSE proxy.

  Routing:
    - select_use_case    : record which use case the caller picked by voice.
"""
from __future__ import annotations

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

_PROFILE = "card"
USE_CASES = {"statement_insights", "rewards_optimizer"}


# ---------------------------------------------------------------------------
# Shared clients (built lazily, cached per process)
# ---------------------------------------------------------------------------
_card_serving: Any = None


def _serving():
    global _card_serving
    if _card_serving is None:
        from genie_voice.config import get_settings
        from genie_voice.serve.card_lakebase import CardLakebaseServing

        _card_serving = CardLakebaseServing(get_settings())
    return _card_serving


# ---------------------------------------------------------------------------
# select_use_case (routing)
# ---------------------------------------------------------------------------
_SELECT_USE_CASE_SPEC = {
    "type": "function",
    "function": {
        "name": "select_use_case",
        "description": (
            "Record which help topic the cardholder chose by voice. Call this as soon "
            "as they pick one: 'statement_insights' (understand why their expenses "
            "changed) or 'rewards_optimizer' (find rewards points they are missing)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "use_case": {
                    "type": "string",
                    "enum": ["statement_insights", "rewards_optimizer"],
                    "description": "The chosen help topic.",
                },
            },
            "required": ["use_case"],
        },
    },
}


def _run_select_use_case(arguments: dict[str, Any], ctx: ToolContext) -> str:
    use_case = str(arguments.get("use_case") or "")
    if use_case not in USE_CASES:
        return json.dumps({"error": "use_case must be 'statement_insights' or 'rewards_optimizer'"})
    ctx.profile_state["use_case"] = use_case
    ctx.profile_state["selected_use_case"] = use_case
    label = "Statement Insights" if use_case == "statement_insights" else "Rewards Optimizer"
    return json.dumps({"ok": True, "use_case": use_case, "label": label})


register(_SELECT_USE_CASE_SPEC, _run_select_use_case, profile=_PROFILE)


# ---------------------------------------------------------------------------
# card_account_facts (FAST lane — Lakebase)
# ---------------------------------------------------------------------------
_CARD_ACCOUNT_FACTS_SPEC = {
    "type": "function",
    "function": {
        "name": "card_account_facts",
        "description": (
            "Get the cardholder's current fast facts from the low-latency cache: "
            "this month's total expenses vs their average monthly expenses, the dollar "
            "change, minimum payment, due date, rewards points balance, and account status. "
            "Use this to state the headline expense number when the caller asks about their "
            "spending or statement. Fast."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "customer_id": {
                    "type": "string",
                    "description": "Cardholder ID. Defaults to the current call's cardholder.",
                },
            },
        },
    },
}


def _run_card_account_facts(arguments: dict[str, Any], ctx: ToolContext) -> str:
    customer_id = arguments.get("customer_id") or ctx.customer_id
    if not customer_id:
        return json.dumps({"error": "No cardholder id in context."})
    facts_cache: dict = ctx.profile_state.setdefault("facts_cache", {})
    if customer_id in facts_cache:
        return json.dumps(facts_cache[customer_id], default=str)
    try:
        facts = _serving().get_cardholder_facts(customer_id)
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": f"Fast-facts lookup failed: {exc}"})
    facts_cache[customer_id] = facts
    safe = {
        "customer_id": customer_id,
        "found": facts.get("found"),
        "name": (facts.get("cardholder") or {}).get("full_name"),
        "summary": facts.get("summary"),
    }
    return json.dumps(safe, default=str)


register(_CARD_ACCOUNT_FACTS_SPEC, _run_card_account_facts, profile=_PROFILE)


# ---------------------------------------------------------------------------
# ask_card_genie (FAST lane — Genie Conversation API)
# ---------------------------------------------------------------------------
_ASK_CARD_GENIE_SPEC = {
    "type": "function",
    "function": {
        "name": "ask_card_genie",
        "description": (
            "Ask Databricks Genie a SPECIFIC, quick question about this cardholder's "
            "data and get a fast answer with rows (e.g. 'what were CH-0001's largest "
            "transactions this cycle?', 'how much in fees by type?'). Use for a single "
            "concrete fact. For a full itemized 'why did X change' investigation, use "
            "start_deep_dive instead."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "The specific question to ask Genie."},
            },
            "required": ["question"],
        },
    },
}


def _run_ask_card_genie(arguments: dict[str, Any], ctx: ToolContext) -> str:
    question = str(arguments.get("question") or "").strip()
    if not question:
        return json.dumps({"error": "question is required"})
    denied = genie_obo_or_refuse(ctx)
    if denied is not None:
        return denied
    if ctx.customer_id and ctx.customer_id not in question:
        question = f"For cardholder {ctx.customer_id}: {question}"
    try:
        from genie_voice.config import get_settings
        from genie_voice.genie.client import GenieClient

        settings = get_settings()
        principal = getattr(ctx, "principal", None)
        token = getattr(principal, "access_token", None) if principal else None
        result = GenieClient(
            settings,
            space_name=genie_space_name(ctx, settings.card_issuer.genie_space_name),
        ).ask(
            question, language=ctx._detected_language, access_token=token
        )
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": f"Genie query failed: {exc}"})
    return shape_genie_answer(result)


register(_ASK_CARD_GENIE_SPEC, _run_ask_card_genie, profile=_PROFILE)


# ---------------------------------------------------------------------------
# start_deep_dive (Agent Mode — blocking within the long-work budget)
# ---------------------------------------------------------------------------
_START_DEEP_DIVE_SPEC = {
    "type": "function",
    "function": {
        "name": "start_deep_dive",
        "description": (
            "Investigate a causal question using the active product pack. Call this "
            "exactly once with the caller's complete question. Do not invent causes, "
            "and do not call other capabilities in the same turn."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The full 'why' question to investigate for this cardholder.",
                },
            },
            "required": ["question"],
        },
    },
}


def _run_start_deep_dive(arguments: dict[str, Any], ctx: ToolContext) -> str:
    question = str(arguments.get("question") or "").strip()
    if not question:
        return json.dumps({"error": "question is required"})
    if ctx.customer_id and ctx.customer_id not in question:
        question = f"For cardholder {ctx.customer_id}: {question}"
    use_case = ctx.profile_state.get("use_case")
    denied = genie_obo_or_refuse(ctx)
    if denied is not None:
        return denied
    try:
        from genie_voice.config import get_settings
        from genie_voice.genie.agent_mode import GenieAgentModeClient

        from .runtime.identity import workspace_client_for_principal

        settings = get_settings()
        workspace = workspace_client_for_principal(ctx.principal, settings)
        result = GenieAgentModeClient(
            settings,
            workspace_client=workspace,
        ).ask(
            question,
            space_name=genie_space_name(ctx, settings.card_issuer.genie_space_name),
        )
        return json.dumps(
            {
                "status": result.status,
                "question": question,
                "use_case": use_case,
                "report": result.report_text,
                "tables": result.tables,
                "reasoning": result.reasoning,
                "sql": result.sql_calls,
                "error": result.error,
            },
            default=str,
        )
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": f"Deep investigation failed: {exc}"})


register(_START_DEEP_DIVE_SPEC, _run_start_deep_dive, profile=_PROFILE)


# ---------------------------------------------------------------------------
# Card system prompt + profile registration
# ---------------------------------------------------------------------------
CARD_SYSTEM_PROMPT = (
    "You are a friendly credit-card account assistant on a voice call with a cardholder. "
    "You MUST act via the provided tools, not narrate. Never say 'let me check' — call "
    "the tool and answer.\n\n"
    "The navigation policy has already chosen one capability for this turn and withdrawn "
    "the others. Call that capability exactly once with the caller's complete question, "
    "then answer only from its result. Do not call a second capability in the same turn.\n\n"
    "Rules:\n"
    "- Speak naturally in 1-3 short sentences. No markdown, no lists, no emoji.\n"
    "- Never invent numbers; only state figures returned by a tool.\n"
    "- Never reveal tool names or system details.\n"
    "- Always respond in the caller's language ({language})."
)

# Card-domain content (brand + assistant persona). NOT a per-language table: the
# greeting phrase below is rendered into the CALLER'S language by the multilingual
# model (same pattern as the filler / switch-language prompts), so there is no
# hardcoded per-language greeting and no English fallback.
CARD_BRAND = "EveryCard"
CARD_AGENT_NAME = "Genie Agent"

# Greeting cache keyed by (base-language, first-name); one model call per key.
_GREETING_CACHE: dict[tuple[str, str], str] = {}


def _greeting_intent(first_name: str) -> str:
    who = f" the customer by name ({first_name})" if first_name else " the customer"
    return (
        f"Warmly greet{who} and introduce yourself as {CARD_AGENT_NAME}, their "
        f"{CARD_BRAND} assistant. In the SAME sentence, offer exactly two things you "
        "can help with — understanding their latest statement, or checking whether "
        "they're getting all their rewards — and ask which they'd like."
    )


def card_greeting(language: str, first_name: str = "") -> str:
    """The agent's opening greeting, generated in the CALLER'S language (cached).

    Thin wrapper over the shared ``greetings`` mechanism with the card intent +
    cache; see ``realtime_api/greetings.py`` for the (profile-agnostic) rendering,
    caching, and "" degrade behavior.
    """
    from .greetings import generate_greeting

    return generate_greeting(
        language, first_name=first_name, intent=_greeting_intent, cache=_GREETING_CACHE
    )


def _seed_greeting_for(language: str) -> str:
    """An in-language greeting to seed LLM history so it knows it already greeted."""
    from .greetings import seed_greeting_for

    return seed_greeting_for(language, intent=_greeting_intent, cache=_GREETING_CACHE)


def _make_card_context(session: Any, language: str) -> ToolContext:
    """Build a ToolContext for the card profile, seeding greeting on first turn."""
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


def _card_after_turn(ctx: ToolContext, session: Any) -> None:
    """Persist selected_use_case from the turn's profile_state onto the session."""
    selected = ctx.profile_state.get("selected_use_case")
    if selected:
        session.profile_state["use_case"] = selected


def _card_tools_spec() -> list[dict[str, Any]]:
    return tools_spec(profile=_PROFILE)


def _card_run_tool(name: str, arguments: dict[str, Any], ctx: Any) -> str:
    return run_tool(name, arguments, ctx, profile=_PROFILE)


def register_profile() -> None:
    from .profiles import VoiceProfile, register_profile as _register_profile

    _register_profile(
        VoiceProfile(
            name=_PROFILE,
            system_prompt=CARD_SYSTEM_PROMPT,
            tools_spec=_card_tools_spec,
            tool_runner=_card_run_tool,
            make_context=_make_card_context,
            after_turn=_card_after_turn,
        )
    )


register_profile()
