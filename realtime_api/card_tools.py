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
_card_genie: Any = None


def _serving():
    global _card_serving
    if _card_serving is None:
        from genie_voice.config import get_settings
        from genie_voice.serve.card_lakebase import CardLakebaseServing

        _card_serving = CardLakebaseServing(get_settings())
    return _card_serving


def _card_genie_client():
    """A Genie Conversation-API client pinned to the CARD space (resolved by name).

    The space is passed to the client constructor by NAME (not by poking the
    private ``_space_id``), so the client's own resolve/stale-space-retry logic
    targets the card space rather than the default telco space. Cached per process.
    """
    global _card_genie
    if _card_genie is None:
        from genie_voice.config import get_settings
        from genie_voice.genie.client import GenieClient

        settings = get_settings()
        _card_genie = GenieClient(settings, space_name=settings.card_issuer.genie_space_name)
    return _card_genie


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
    if ctx.customer_id and ctx.customer_id not in question:
        question = f"For cardholder {ctx.customer_id}: {question}"
    try:
        result = _card_genie_client().ask(question, language=ctx._detected_language)
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": f"Genie query failed: {exc}"})
    return shape_genie_answer(result)


register(_ASK_CARD_GENIE_SPEC, _run_ask_card_genie, profile=_PROFILE)


# ---------------------------------------------------------------------------
# start_deep_dive (DEEP lane — Genie Agent mode, async)
# ---------------------------------------------------------------------------
_START_DEEP_DIVE_SPEC = {
    "type": "function",
    "function": {
        "name": "start_deep_dive",
        "description": (
            "SIGNAL that the caller wants a DEEP, multi-step investigation (the "
            "itemized 'why') — e.g. why their expenses spiked this cycle, what "
            "categories drove the increase, or where rewards points are being lost. "
            "This returns immediately and runs OUT OF BAND (the app kicks off Genie "
            "Agent mode, which takes up to a minute). Call it once the caller asks "
            "'why'. Do NOT wait for it — tell the caller you're pulling the full "
            "breakdown and answer their immediate question using card_account_facts "
            "or ask_card_genie in the meantime."
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
    return json.dumps({
        "started": True,
        "question": question,
        "use_case": use_case,
        "note": "Deep investigation requested; the breakdown will follow shortly.",
    })


register(_START_DEEP_DIVE_SPEC, _run_start_deep_dive, profile=_PROFILE)


# ---------------------------------------------------------------------------
# Card system prompt + profile registration
# ---------------------------------------------------------------------------
CARD_SYSTEM_PROMPT = (
    "You are a friendly credit-card account assistant on a voice call with a cardholder. "
    "You MUST act via tools, not narrate. Never say 'let me check' — call the tool and answer.\n\n"
    "Conversation shape:\n"
    "- You already greeted the caller and offered two topics. WAIT for them to choose; do NOT "
    "volunteer spending details, fees, or amounts before they ask.\n"
    "- When they pick a topic, call select_use_case immediately.\n"
    "- statement_insights: they want to understand why their expenses changed.\n"
    "- rewards_optimizer: they want to know about missed rewards.\n\n"
    "CRITICAL — 'why' questions:\n"
    "- ANY time the caller asks 'why' (e.g. 'why did my expenses go up', 'why am I spending "
    "more', 'why is it higher', 'what caused the spike', 'break it down', 'why am I losing "
    "points'), you MUST call start_deep_dive(question) IMMEDIATELY. Do NOT answer the 'why' "
    "yourself — you do not have the data to explain it. Only start_deep_dive can produce the "
    "itemized breakdown.\n"
    "- In the SAME turn, also call card_account_facts() so you can state the headline number "
    "(e.g. 'Your expenses this month are $X, up from your usual $Y') while the deep dive runs.\n"
    "- After calling both tools, tell the caller: 'I'm pulling the full breakdown now — it'll "
    "be ready in a moment.' Do NOT fabricate reasons or percentages.\n\n"
    "CRITICAL — rewards questions:\n"
    "- The SAME rule applies to rewards. If the caller asks whether they are missing rewards, "
    "'am I getting all my points', 'am I leaving points on the table', 'am I missing any "
    "rewards', 'how do I earn more', or why their points are lower, you MUST call "
    "start_deep_dive(question) IMMEDIATELY to produce the itemized rewards-leakage breakdown — "
    "you cannot answer it yourself. In the SAME turn call card_account_facts() for the headline "
    "points gap, then say you're pulling the full rewards breakdown.\n\n"
    "Tools:\n"
    "- select_use_case(use_case): record their choice.\n"
    "- card_account_facts(): fast expenses / points / status for this cycle vs average.\n"
    "- ask_card_genie(question): single specific fact with data.\n"
    "- start_deep_dive(question): MUST call for any 'why' / 'how come' / 'break it down' question. "
    "Returns immediately; the investigation runs in the background.\n\n"
    "Rules:\n"
    "- Speak naturally in 1-3 short sentences. No markdown, no lists, no emoji.\n"
    "- Only discuss financial details AFTER the caller asks about them.\n"
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
    return ToolContext(
        customer_id=session.config.customer_id,
        call_id=session.config.call_id,
        _detected_language=language,
        account_store=session.account_store,
        profile_state=session.profile_state,
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
