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
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from .config import ConciergeRouterConfig, concierge_router_config
from .guardrails import report
from .languages import base_code
from .tool_registry import ToolContext, register, run_tool, tools_spec

if TYPE_CHECKING:
    from .guardrails import GuardLedger

_PROFILE = "concierge"


@lru_cache(maxsize=1)
def router() -> ConciergeRouterConfig:
    """The routing table, from config. Cached: the file is read once per process.

    Deliberately not module-level constants. These gate which assistant a caller
    reaches, and a validated config block makes a change to a cue or a threshold
    reviewable instead of a literal edit inside the router.
    """
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
# Deterministic intent pre-router (framework seam)
# ---------------------------------------------------------------------------
# The concierge's whole job is to route by voice, so a cue table resolves the
# selection reliably WITHOUT waiting on the conversational LLM to emit the tool
# call. This is Tier 1; the LLM (Tier 2) still handles anything ambiguous or
# phrased as a question. The table, the length limit and the languages it is
# authored for all come from realtime_voice.concierge_router in config.
def _matches(cue: str, text: str) -> bool:
    if " " in cue or "-" in cue:
        return cue in text
    return re.search(rf"\b{re.escape(cue)}\b", text) is not None


def _declined(ledger: "GuardLedger | None", guard_id: str, reason: str) -> None:
    """A decline IS an action: it stopped a deterministic route and handed the turn
    to the LLM. Recording it is the point — until now a decline left no trace at
    all, which is why "why didn't saying Telco route me?" had no answer."""
    report(ledger, guard_id, "fired", seam="decision", stage="routing", reason=reason)


def _held(ledger: "GuardLedger | None", guard_id: str, reason: str | None = None) -> None:
    report(ledger, guard_id, "passed", seam="decision", stage="routing", reason=reason)


def resolve_industry(
    transcript: str, ledger: "GuardLedger | None" = None, language: str | None = None
) -> str | None:
    """Confidently map a short spoken reply to an industry, else None.

    Returns an industry only when EXACTLY ONE matches (no match or a cross-domain
    tie is treated as ambiguous and left to the LLM). Reports each gate's outcome
    to ``ledger`` when one is supplied; reasons carry counts and industry names
    only, never the transcript, because the roster is persisted.
    """
    cfg = router()
    # The cue table is authored for specific languages (English today). Matching
    # it against a transcript in another language would route on coincidence, so
    # the router steps aside and lets the model handle the turn.
    if language and base_code(language) not in cfg.languages:
        report(
            ledger, "selection_language_scope", "not_evaluated",
            seam="decision", stage="routing",
            reason=f"cues authored for {list(cfg.languages)}, call is {base_code(language)}",
        )
        return None

    text = (transcript or "").strip().lower()
    words = len(text.split())
    if not text:
        report(
            ledger, "selection_length", "not_evaluated",
            seam="decision", stage="routing", reason="empty transcript",
        )
        return None
    if words > cfg.max_selection_words:
        _declined(
            ledger, "selection_length",
            f"{words} words > {cfg.max_selection_words}; deferred to LLM",
        )
        return None
    _held(ledger, "selection_length", f"{words} words")

    matched = {i.key for i in cfg.industries if any(_matches(c, text) for c in i.cues)}
    if not matched:
        _declined(ledger, "selection_allowlist", "no industry cue matched; deferred to LLM")
        return None
    _held(ledger, "selection_allowlist", f"matched {sorted(matched)}")

    if len(matched) > 1:
        _declined(
            ledger, "selection_ambiguity",
            f"{len(matched)} industries matched {sorted(matched)}; deferred to LLM",
        )
        return None
    _held(ledger, "selection_ambiguity")
    return next(iter(matched))


def _resolve_concierge_intents(
    transcript: str, language: str, ledger: "GuardLedger | None" = None
) -> list[Any]:
    from .profiles import ResolvedIntent

    industry = resolve_industry(transcript, ledger, language)
    if industry is None:
        return []
    route = router().industry(industry)
    assert route is not None  # resolve_industry only returns configured keys
    return [
        ResolvedIntent(
            name="select_industry",
            arguments={"industry": industry},
            confirm_intent=(
                f"In ONE short, warm sentence, tell the user you're taking them to "
                f"{route.confirm_label} now. No lists, no emoji."
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
        f"Warmly welcome{who} to Databricks Genie Assisted Voice. First, invite them to "
        "pick their preferred language from the language menu at the top of the screen. "
        "Then, in one short sentence, say that Genie-assisted voice agents work across "
        "Telco billing support, a Financial Services credit-card assistant, and Healthcare, "
        "powered by the Genie ontology and deep reasoning. Finally ask which they would "
        "like to explore — Telco, Financial Services, or Healthcare — and mention they can "
        "just say it."
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
