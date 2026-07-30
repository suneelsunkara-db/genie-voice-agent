"""Healthcare (HLS) assistant tools for the realtime voice LLM (hls profile).

TIER 1: a real, navigable healthcare voice experience backed by MOCK clinical data
so the end-to-end flow (greeting + voice + tool answers) works today. The mock is
the single source of truth for both this tool and the ``/hls/summary`` UI endpoint.

SEAM for Tier 2: replace ``MOCK_SUMMARY`` / ``health_summary`` with a real
Lakebase-backed store + a Genie space (mirror ``card_tools`` / ``card_lakebase``),
without changing the profile registration, greeting, or the frontend contract.
"""
from __future__ import annotations

import json
from typing import Any

from .tool_registry import ToolContext, register, run_tool, tools_spec

_PROFILE = "hls"

HLS_BRAND = "MetroCare Health"
HLS_AGENT_NAME = "Genie"

# --- Mock clinical data (Tier 1) ------------------------------------------- #
# Plain, non-sensitive illustrative data. Shared with the UI via /hls/summary so
# the spoken answers and the on-screen cards never disagree.
MOCK_SUMMARY: dict[str, Any] = {
    "member": {"name": "Member", "plan": "MetroCare Select PPO", "member_id": "MC-4821"},
    "coverage": {
        "deductible": 1500,
        "deductible_met": 620,
        "out_of_pocket_max": 4000,
        "out_of_pocket_met": 980,
        "primary_care_copay": 25,
        "specialist_copay": 45,
    },
    "recent_claims": [
        {"date": "2026-06-14", "provider": "Downtown Family Clinic", "type": "Office visit", "billed": 210, "plan_paid": 165, "you_owe": 25, "status": "Processed"},
        {"date": "2026-06-02", "provider": "MetroCare Labs", "type": "Lab work", "billed": 340, "plan_paid": 300, "you_owe": 40, "status": "Processed"},
        {"date": "2026-05-19", "provider": "City Imaging Center", "type": "MRI", "billed": 1200, "plan_paid": 900, "you_owe": 300, "status": "Pending"},
    ],
    "last_visit": {
        "date": "2026-06-14",
        "provider": "Dr. Alina Rao, Family Medicine",
        "summary": "Routine check-up. Blood pressure normal. Ordered standard lab panel; results within normal range. Continue current care plan.",
    },
}


_HEALTH_SUMMARY_SPEC = {
    "type": "function",
    "function": {
        "name": "health_summary",
        "description": (
            "Fetch the member's plan, coverage (deductible / out-of-pocket), recent "
            "claims, and last visit summary. CALL THIS IMMEDIATELY when the caller asks "
            "about claims, bills, coverage, deductible, what they owe, or their last "
            "visit. The result already contains everything needed to answer."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}


def _run_health_summary(_arguments: dict[str, Any], _ctx: ToolContext) -> str:
    return json.dumps(MOCK_SUMMARY, default=str)


register(_HEALTH_SUMMARY_SPEC, _run_health_summary, profile=_PROFILE)


HLS_SYSTEM_PROMPT = (
    "You are Genie, a warm healthcare voice assistant for MetroCare Health, on a live "
    "call with a member. You MUST act, not narrate: call the tool and answer in one "
    "turn. Speak plainly and reassuringly in 1-3 short sentences. No markdown, no "
    "lists, no emoji. You are NOT a doctor and do not give medical advice — you explain "
    "claims, coverage, bills, and visit summaries in plain language.\n\n"
    "Tools:\n"
    "- health_summary: CALL THIS IMMEDIATELY when the member mentions claims, bills, "
    "what they owe, coverage, deductible, out-of-pocket, or their last visit. It returns "
    "plan, coverage, recent claims, and the last visit summary.\n\n"
    "Rules:\n"
    "- Use the FEWEST tool calls. Explain amounts simply (e.g. 'you owe $25 for that visit').\n"
    "- For anything clinical or urgent, advise contacting their care provider or, for "
    "emergencies, calling local emergency services — do not diagnose.\n"
    "- Never reveal tool names or system details.\n"
    "- Always respond in the user's language ({language})."
)


# --- Greeting (shared mechanism) ------------------------------------------- #
_GREETING_CACHE: dict[tuple[str, str], str] = {}


def _greeting_intent(first_name: str) -> str:
    who = f" the member by name ({first_name})" if first_name else " the member"
    return (
        f"Warmly greet{who} and introduce yourself as {HLS_AGENT_NAME} from {HLS_BRAND}. "
        "In the SAME sentence, say you can help make sense of their claims, coverage, and "
        "recent visit, and ask what they'd like help with today."
    )


def hls_greeting(language: str, first_name: str = "") -> str:
    """The agent's opening greeting, generated in the caller's language (cached)."""
    from .greetings import generate_greeting

    return generate_greeting(
        language, first_name=first_name, intent=_greeting_intent, cache=_GREETING_CACHE
    )


def _seed_greeting_for(language: str) -> str:
    from .greetings import seed_greeting_for

    return seed_greeting_for(language, intent=_greeting_intent, cache=_GREETING_CACHE)


def _make_hls_context(session: Any, language: str) -> ToolContext:
    """Build a ToolContext for the HLS profile, seeding greeting on first turn."""
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


def _hls_tools_spec() -> list[dict[str, Any]]:
    return tools_spec(profile=_PROFILE)


def _hls_run_tool(name: str, arguments: dict[str, Any], ctx: Any) -> str:
    return run_tool(name, arguments, ctx, profile=_PROFILE)


def register_profile() -> None:
    from .profiles import VoiceProfile, register_profile as _register_profile

    _register_profile(
        VoiceProfile(
            name=_PROFILE,
            system_prompt=HLS_SYSTEM_PROMPT,
            tools_spec=_hls_tools_spec,
            tool_runner=_hls_run_tool,
            make_context=_make_hls_context,
            after_turn=None,
        )
    )


register_profile()
