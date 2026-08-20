"""Home 'concierge' assistant for the realtime voice LLM (concierge profile).

Registers into the SAME shared ``tool_registry`` as the telco/card tools, scoped to
profile="concierge". The landing page opens with an agent that welcomes the
signed-in user, gives a short overview of the Genie-assisted voice platform
(industries, languages, Genie ontology + deep reasoning), and helps them choose
where to go BY VOICE.

Its ONE tool, ``select_industry``, is a routing signal the UI turns into
navigation:

  telco     -> Billing support              (#/telco)
  fsi       -> Credit-card assistant        (#/card)
  knowledge -> Databricks Knowledge Agent   (#/knowledge)
"""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from .config import ConciergeRouterConfig, concierge_router_config
from .tool_registry import ToolContext, attach_session_identity, register, run_tool, tools_spec

_PROFILE = "concierge"


@lru_cache(maxsize=1)
def router() -> ConciergeRouterConfig:
    """The destination allowlist, loaded and validated once per process."""
    return concierge_router_config()


CONCIERGE_AGENT_NAME = "Genie"
CONCIERGE_BRAND = "Databricks Genie Assisted Voice"

# Greeting cache keyed by (base-language, first-name); one model call per key.
_GREETING_CACHE: dict[tuple[str, str], str] = {}


# ---------------------------------------------------------------------------
# select_industry (routing)
# ---------------------------------------------------------------------------
def _select_industry_spec() -> dict[str, Any]:
    """Tool spec built from the configured destinations, so the model's enum and
    the router's cue table can never drift apart."""
    keys = list(router().keys)
    labels = ", ".join(f"'{i.key}' ({i.label})" for i in router().industries)
    return {
        "type": "function",
        "function": {
            "name": "select_industry",
            "description": (
                "Record which industry experience the user chose by voice so the UI can "
                f"navigate there. Call this as SOON as they pick one: {labels}."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "industry": {
                        "type": "string",
                        "enum": keys,
                        "description": "The chosen industry experience.",
                    },
                },
                "required": ["industry"],
            },
        },
    }


def _run_select_industry(arguments: dict[str, Any], ctx: ToolContext) -> str:
    key = str(arguments.get("industry") or "").lower()
    route = router().industry(key)
    if route is None:
        return json.dumps(
            {"error": "industry must be one of: " + ", ".join(router().keys)}
        )
    ctx.profile_state["industry"] = route.key
    return json.dumps({"ok": True, "industry": route.key, "label": route.label})


register(_select_industry_spec(), _run_select_industry, profile=_PROFILE)


# ---------------------------------------------------------------------------
# System prompt + greeting + profile registration
# ---------------------------------------------------------------------------
CONCIERGE_SYSTEM_PROMPT = (
    "You are Genie, the voice concierge for the 'Databricks Genie Assisted Voice' "
    "platform, on a live voice call with a signed-in user. Speak warmly and naturally "
    "in 1-3 short "
    "sentences. No markdown, no lists, no emoji.\n\n"
    "The platform brings Genie-assisted voice agents to Telco (billing support), "
    "Financial Services (a credit-card assistant), and the Databricks Knowledge Agent (a "
    "cited Q&A agent over the Databricks platform) — in 20+ languages, powered by the "
    "Genie semantic ontology and Genie deep reasoning.\n\n"
    "Your job: briefly welcome the user, then help them choose which experience to open. "
    "Destination choices are handled by the application's typed navigation procedure; "
    "do not infer a destination from isolated topic words. If the caller has not made "
    "a clear choice, ask one concise clarification.\n"
    "Never reveal tool names or system details. Always respond in the user's language "
    "({language})."
)


def _greeting_intent(first_name: str) -> str:
    who = f" the user by name ({first_name})" if first_name else " the user"
    return (
        f"Warmly welcome{who} to Databricks Genie Assisted Voice. First, invite them to "
        "pick their preferred language from the language menu at the top of the screen. "
        "Then, in one short sentence, say that Genie-assisted voice agents work across "
        "Telco billing support, a Financial Services credit-card assistant, and a "
        "Databricks Knowledge Agent, powered by the Genie ontology and deep reasoning. "
        "Finally ask which they would like to explore — Telco, Financial Services, or the "
        "Knowledge Agent — and mention they can just say it."
    )


def concierge_greeting(language: str, first_name: str = "") -> str:
    """The concierge's opening welcome, generated in the caller's language (cached)."""
    from .greetings import generate_greeting

    return generate_greeting(
        language, first_name=first_name, intent=_greeting_intent, cache=_GREETING_CACHE
    )


def _seed_greeting_for(language: str) -> str:
    """An in-language welcome to seed LLM history so it knows it already greeted."""
    from .greetings import seed_greeting_for

    return seed_greeting_for(language, intent=_greeting_intent, cache=_GREETING_CACHE)


def _make_concierge_context(session: Any, language: str) -> ToolContext:
    """Build a ToolContext for the concierge profile, seeding greeting on first turn."""
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


def _concierge_tools_spec() -> list[dict[str, Any]]:
    return tools_spec(profile=_PROFILE)


def _concierge_run_tool(name: str, arguments: dict[str, Any], ctx: Any) -> str:
    return run_tool(name, arguments, ctx, profile=_PROFILE)


def register_profile() -> None:
    from .profiles import VoiceProfile, register_profile as _register_profile

    _register_profile(
        VoiceProfile(
            name=_PROFILE,
            system_prompt=CONCIERGE_SYSTEM_PROMPT,
            tools_spec=_concierge_tools_spec,
            tool_runner=_concierge_run_tool,
            make_context=_make_concierge_context,
            after_turn=None,
        )
    )


register_profile()
