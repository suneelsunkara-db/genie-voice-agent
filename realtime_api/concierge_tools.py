"""Home 'concierge' assistant for the realtime voice LLM (concierge profile).

Registers into the SAME shared ``tool_registry`` as the telco/card tools, scoped to
profile="concierge". The landing page opens with an agent that welcomes the
signed-in user, gives a short overview of the Genie-assisted voice platform
(industries, languages, Genie ontology + deep reasoning), and helps them choose
where to go BY VOICE.

Its ONE tool, ``select_industry``, is a routing signal the UI turns into
navigation:

  telco      -> Billing support        (#/telco)
  fsi        -> Credit-card assistant   (#/card)
  healthcare -> Healthcare assistant    (#/hls)
"""
from __future__ import annotations

import json
import re
from typing import Any

from .tool_registry import ToolContext, register, run_tool, tools_spec

_PROFILE = "concierge"
INDUSTRIES = {"telco", "fsi", "healthcare"}
_INDUSTRY_LABELS = {
    "telco": "Telco — Billing Support",
    "fsi": "Financial Services — Credit Card",
    "healthcare": "Healthcare",
}

CONCIERGE_AGENT_NAME = "Genie"
CONCIERGE_BRAND = "Databricks Genie Assisted Voice"

# Greeting cache keyed by (base-language, first-name); one model call per key.
_GREETING_CACHE: dict[tuple[str, str], str] = {}


# ---------------------------------------------------------------------------
# select_industry (routing)
# ---------------------------------------------------------------------------
_SELECT_INDUSTRY_SPEC = {
    "type": "function",
    "function": {
        "name": "select_industry",
        "description": (
            "Record which industry experience the user chose by voice so the UI can "
            "navigate there. Call this as SOON as they pick one: 'telco' (billing "
            "support), 'fsi' (credit-card assistant), or 'healthcare'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "industry": {
                    "type": "string",
                    "enum": ["telco", "fsi", "healthcare"],
                    "description": "The chosen industry experience.",
                },
            },
            "required": ["industry"],
        },
    },
}


def _run_select_industry(arguments: dict[str, Any], ctx: ToolContext) -> str:
    industry = str(arguments.get("industry") or "").lower()
    if industry not in INDUSTRIES:
        return json.dumps({"error": "industry must be 'telco', 'fsi', or 'healthcare'"})
    ctx.profile_state["industry"] = industry
    return json.dumps({"ok": True, "industry": industry, "label": _INDUSTRY_LABELS[industry]})


register(_SELECT_INDUSTRY_SPEC, _run_select_industry, profile=_PROFILE)


# ---------------------------------------------------------------------------
# Deterministic intent pre-router (framework seam)
# ---------------------------------------------------------------------------
# The concierge's whole job is to route by voice, and the home page is
# English-only (the language picker is disabled and defaults to English), so a
# small English cue table resolves the selection reliably WITHOUT waiting on the
# conversational LLM to emit the tool call. This is Tier 1; the LLM (Tier 2)
# still handles anything ambiguous or phrased as a question.
_INDUSTRY_CUES: dict[str, tuple[str, ...]] = {
    "telco": ("telco", "telecom", "telecoms", "billing", "phone bill", "phone", "mobile", "wireless", "cellular"),
    "fsi": ("fsi", "financial", "finance", "credit card", "credit-card", "card", "bank", "banking", "statement"),
    "healthcare": ("healthcare", "health", "clinical", "clinic", "patient", "medical", "claim", "claims", "coverage"),
}
# One warm confirmation instruction per industry, voiced in the caller's
# language (generated + cached by the pipeline). Instruction, not a literal line.
_CONFIRM_LABELS = {
    "telco": "Telco billing support",
    "fsi": "the credit-card assistant",
    "healthcare": "the healthcare assistant",
}
# Only route short, selection-style replies deterministically; longer utterances
# (questions, elaboration) defer to the LLM so we don't route on "what is
# healthcare?" — the confidence guardrail behind always-confirm-then-route.
_MAX_SELECTION_WORDS = 8


def _matches(cue: str, text: str) -> bool:
    if " " in cue or "-" in cue:
        return cue in text
    return re.search(rf"\b{re.escape(cue)}\b", text) is not None


def resolve_industry(transcript: str) -> str | None:
    """Confidently map a short spoken reply to an industry, else None.

    Returns an industry only when EXACTLY ONE matches (no match or a cross-domain
    tie is treated as ambiguous and left to the LLM).
    """
    text = (transcript or "").strip().lower()
    if not text or len(text.split()) > _MAX_SELECTION_WORDS:
        return None
    matched = {ind for ind, cues in _INDUSTRY_CUES.items() if any(_matches(c, text) for c in cues)}
    return next(iter(matched)) if len(matched) == 1 else None


def _resolve_concierge_intents(transcript: str, language: str) -> list[Any]:
    from .profiles import ResolvedIntent

    industry = resolve_industry(transcript)
    if industry is None:
        return []
    return [
        ResolvedIntent(
            name="select_industry",
            arguments={"industry": industry},
            confirm_intent=(
                f"In ONE short, warm sentence, tell the user you're taking them to "
                f"{_CONFIRM_LABELS[industry]} now. No lists, no emoji."
            ),
        )
    ]


# ---------------------------------------------------------------------------
# System prompt + greeting + profile registration
# ---------------------------------------------------------------------------
CONCIERGE_SYSTEM_PROMPT = (
    "You are Genie, the voice concierge for the 'Databricks Genie Assisted Voice' "
    "platform, on a live voice call with a signed-in user. Speak warmly and naturally "
    "in 1-3 short "
    "sentences. No markdown, no lists, no emoji.\n\n"
    "The platform brings Genie-assisted voice agents to multiple industries — Telco "
    "(billing support), Financial Services (a credit-card assistant), and Healthcare — "
    "in 20+ languages, powered by the Genie semantic ontology and Genie deep reasoning.\n\n"
    "Your job: briefly welcome the user, then help them choose which industry experience "
    "to open. The MOMENT they indicate a choice (e.g. 'telco', 'billing', 'phone bill', "
    "'card', 'credit card', 'bank', 'finance', 'health', 'clinical' — in ANY language), "
    "your VERY NEXT step MUST be the select_industry tool call, BEFORE you say anything "
    "else. Map natural phrases: billing/telecom/phone -> telco; card/credit card/bank/"
    "finance/FSI -> fsi; health/clinical/patient/medical -> healthcare. After the tool "
    "returns, confirm in one short sentence.\n"
    "Never reveal tool names or system details. Always respond in the user's language "
    "({language})."
)


def _greeting_intent(first_name: str) -> str:
    who = f" the user by name ({first_name})" if first_name else " the user"
    return (
        f"Warmly welcome{who} to Databricks Genie Assisted Voice. In two short sentences, say that "
        "Genie-assisted voice agents work across Telco billing support, a Financial "
        "Services credit-card assistant, and Healthcare, in more than twenty languages, "
        "powered by the Genie ontology and deep reasoning. Then ask which they would like "
        "to explore — Telco, Financial Services, or Healthcare."
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
    return ToolContext(
        customer_id=session.config.customer_id,
        call_id=session.config.call_id,
        _detected_language=language,
        account_store=session.account_store,
        profile_state=session.profile_state,
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
            resolve_intents=_resolve_concierge_intents,
        )
    )


register_profile()
