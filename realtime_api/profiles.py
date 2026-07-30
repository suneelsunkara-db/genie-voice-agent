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

from dataclasses import dataclass
from typing import Any, Callable


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


_PROFILES: dict[str, VoiceProfile] = {}


def register_profile(profile: VoiceProfile) -> None:
    _PROFILES[profile.name] = profile


def get_profile(name: str | None) -> VoiceProfile | None:
    return _PROFILES.get(name) if name else None


def profile_names() -> tuple[str, ...]:
    return tuple(_PROFILES)
