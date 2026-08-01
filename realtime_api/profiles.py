"""Pluggable assistant profiles for the tool-assist voice loop.

The realtime engine (STT → LLM+tools → TTS) is domain-agnostic. A *profile*
supplies the only things that differ between assistants: the system prompt, the
tool specs, the tool executor, and a per-turn tool context. The default (telco
contact-center) behavior is the built-in fallback used when no profile is
selected — see ``services.py``.

Profiles self-register on import (e.g. ``card_tools`` registers "card"), so the
core files (contracts/pipeline/session) never name a specific domain: they only
look one up by the ``profile`` value on ``session.start``. This keeps the voice
API free of any card- (or telco-) specific ``if`` checks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Protocol

if TYPE_CHECKING:
    from .guardrails import GuardLedger


@dataclass(frozen=True)
class ResolvedIntent:
    """A tool call a profile resolved deterministically from a transcript.

    Emitted by :attr:`VoiceProfile.resolve_intents` so a *deterministic* action
    (navigation, selection) never depends solely on the conversational LLM
    choosing to emit the tool call. The engine runs ``name``/``arguments`` through
    the profile's ``tool_runner``, surfaces the usual ``tool.called`` event, and —
    if ``confirm_intent`` is set — speaks a short in-language confirmation
    (generated + cached) BEFORE the UI acts on the result. ``confirm_intent`` is a
    natural-language instruction for the phrase model, not a literal line, so it
    renders correctly in every supported language.
    """
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    confirm_intent: str | None = None


class IntentResolver(Protocol):
    """A profile's deterministic pre-router.

    ``ledger`` is optional so the resolver stays callable (and unit-testable)
    without a live turn, but the pipeline always passes the turn's ledger: the
    router belongs to the DECISION seam, which keeps its own logic and reports its
    outcomes — including declines — to the shared roster.
    """

    def __call__(
        self, transcript: str, language: str, ledger: "GuardLedger | None" = None
    ) -> list["ResolvedIntent"]: ...


@dataclass(frozen=True)
class VoiceProfile:
    """Everything the tool-assist loop needs to run as a specific assistant."""
    name: str
    system_prompt: str
    tools_spec: Callable[[], list[dict[str, Any]]]
    tool_runner: Callable[[str, dict[str, Any], Any], str]
    # (session, language) -> tool context object passed to tools_spec's runner.
    make_context: Callable[[Any, str], Any]
    # Optional hook run after each turn to persist small state across turns
    # (ctx, session) -> None. State should live on ``session.profile_state``.
    after_turn: Callable[[Any, Any], None] | None = None
    # Optional deterministic intent pre-router (see IntentResolver). Runs BEFORE
    # the LLM each turn. Return a confident intent to short-circuit the LLM (run
    # tool + speak confirmation); return an empty list to defer to the LLM
    # (ambiguous / not a selection). Keeps the engine domain-agnostic — only the
    # profile knows what its intents are.
    resolve_intents: IntentResolver | None = None


_PROFILES: dict[str, VoiceProfile] = {}


def register_profile(profile: VoiceProfile) -> None:
    _PROFILES[profile.name] = profile


def get_profile(name: str | None) -> VoiceProfile | None:
    return _PROFILES.get(name) if name else None


def profile_names() -> tuple[str, ...]:
    return tuple(_PROFILES)
