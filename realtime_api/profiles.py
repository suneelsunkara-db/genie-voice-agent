"""Pluggable assistant profiles for the tool-assist voice loop.

The realtime engine (STT → LLM+tools → TTS) is domain-agnostic. A *profile*
supplies the system prompt, tool specs, tool executor, and per-turn context.
Capability IDs, owned tools, and selection signals live on the navigation
catalog, not in English cue tables inside the engine. Unset profile falls back
to the voice concierge — never an implicit billing pack.
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
    # Optional deterministic lane resolver: given the utterance, return
    # "workspace" when the question is outside this pack's own corpus and must
    # escalate to the governed workspace lane (Genie One, OBO-gated), "pack" when
    # this profile can cite the answer itself, or None when neither can. Keeps the
    # engine domain-agnostic — only the profile knows what its pack actually covers,
    # and only the engine knows whether a governed conversation is already open.
    resolve_lane: Callable[[str], str | None] | None = None


_PROFILES: dict[str, VoiceProfile] = {}


def register_profile(profile: VoiceProfile) -> None:
    _PROFILES[profile.name] = profile


def get_profile(name: str | None) -> VoiceProfile | None:
    return _PROFILES.get(name) if name else None


def profile_names() -> tuple[str, ...]:
    return tuple(_PROFILES)
