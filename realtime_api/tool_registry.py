"""Shared tool infrastructure for the realtime voice LLM.

All assistant profiles (telco billing, card issuer, etc.) register their tools
into this single registry. The core voice loop calls ``tools_spec(profile)`` and
``run_tool(name, args, ctx, profile)`` — it never imports a specific domain file.

Pattern:
  1. Define a tool spec (OpenAI function-calling schema) + an executor function.
  2. Call ``register(spec, fn, profile=...)`` at module level.
  3. The pipeline resolves the right tool set via ``tools_spec(profile)``.

ToolContext is the SINGLE shared context object passed to every tool executor,
regardless of profile. Profile-specific state lives in ``profile_state`` (a
dict keyed by anything the profile needs, e.g. ``use_case``, ``facts_cache``).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

# Type alias for tool executor functions
ToolExecutor = Callable[[dict[str, Any], "ToolContext"], str]


@dataclass
class ToolContext:
    """Per-session state available to ALL tool implementations."""

    customer_id: str | None = None
    call_id: str | None = None
    _detected_language: str | None = field(default=None, repr=False)
    # Session-scoped account cache (customer_id -> facts), shared across turns.
    # Lets a follow-up turn reuse facts from an earlier lookup without re-reading.
    account_store: dict[str, Any] | None = field(default=None, repr=False)
    # Profile-specific state (use_case, facts_cache, etc.). Profiles read/write
    # freely; the engine never inspects this dict.
    profile_state: dict[str, Any] = field(default_factory=dict)
    # OBO principal for Genie / workspace tools (fail-closed when token missing).
    principal: Any | None = field(default=None, repr=False)
    session_id: str | None = field(default=None, repr=False)
    turn_id: int | None = field(default=None, repr=False)
    # Optional override of the configured demo Genie space (Space / Agent Mode).
    space_name: str | None = field(default=None, repr=False)

    def cached_account(self, customer_id: str) -> dict[str, Any] | None:
        if self.account_store is None:
            return None
        return self.account_store.get(customer_id)

    def store_account(self, customer_id: str, facts: dict[str, Any]) -> None:
        if self.account_store is not None:
            self.account_store[customer_id] = facts

    def invalidate_account(self, customer_id: str) -> None:
        if self.account_store is not None:
            self.account_store.pop(customer_id, None)


def shape_genie_answer(result: dict[str, Any]) -> str:
    """Serialize a Genie Conversation-API result into the tool's JSON reply.

    Every ``ask_genie``-style tool (telco billing + card issuer + any future
    profile) returns the SAME safe shape — a spoken ``answer`` plus the ``rows`` /
    ``columns`` for the UI — so the projection lives here once instead of being
    re-implemented per profile. ``default=str`` keeps non-JSON cell types safe.
    """
    return json.dumps(
        {
            "answer": result.get("answer") or result.get("description"),
            "rows": result.get("rows"),
            "columns": result.get("columns"),
        },
        default=str,
    )


def attach_session_identity(ctx: "ToolContext", session: Any) -> "ToolContext":
    """Copy OBO principal + session ids onto a freshly built ToolContext."""
    ctx.principal = getattr(session, "principal", None)
    ctx.session_id = getattr(session, "session_id", None)
    ctx.turn_id = getattr(session, "turn_id", None)
    config = getattr(session, "config", None)
    ctx.space_name = getattr(config, "space_name", None) if config is not None else None
    return ctx


def genie_space_name(ctx: "ToolContext", default: str) -> str:
    """Space the caller named, else this app's configured demo space."""
    override = str(getattr(ctx, "space_name", None) or "").strip()
    return override or default


def genie_obo_or_refuse(ctx: "ToolContext") -> str | None:
    """Fail-closed Genie gate: return JSON refuse payload, or None if OBO ok."""
    from .runtime.identity import require_obo_token, refuse_text_for_obo

    err = require_obo_token(
        getattr(ctx, "principal", None),
        session_id=str(getattr(ctx, "session_id", None) or "unknown"),
        turn_id=getattr(ctx, "turn_id", None),
    )
    if err is None:
        return None
    return json.dumps(
        {
            "error": err.message,
            "error_evidence": err.as_dict(),
            "refuse": refuse_text_for_obo(err, language=getattr(ctx, "_detected_language", None) or "en"),
            "obo_deny": True,
        }
    )


# ---------------------------------------------------------------------------
# Profile-scoped registry
# ---------------------------------------------------------------------------
# Key: (profile, tool_name) -> (spec_dict, executor_fn)
# profile=None is an unscoped legacy registration, not a session default.
_REGISTRY: dict[tuple[str | None, str], tuple[dict[str, Any], ToolExecutor]] = {}


def register(spec: dict[str, Any], fn: ToolExecutor, *, profile: str | None = None) -> None:
    """Register a tool for a named profile (or an explicit unscoped caller)."""
    name = spec["function"]["name"]
    _REGISTRY[(profile, name)] = (spec, fn)


def tools_spec(profile: str | None = None) -> list[dict[str, Any]]:
    """Return OpenAI-format tool definitions for the given profile."""
    return [spec for (p, _), (spec, _) in _REGISTRY.items() if p == profile]


def run_tool(name: str, arguments: dict[str, Any], ctx: ToolContext, *, profile: str | None = None) -> str:
    """Execute a tool by name within the given profile's registry."""
    entry = _REGISTRY.get((profile, name))
    if not entry:
        return json.dumps({"error": f"unknown tool: {name}"})
    _, fn = entry
    return fn(arguments, ctx)


def tool_effect(name: str, *, profile: str | None = None) -> str:
    """Return the server-enforced effect class declared by a capability."""
    entry = _REGISTRY.get((profile, name))
    if not entry:
        return "read"
    spec, _ = entry
    effect = str(spec.get("x-effect_class") or "read")
    return effect if effect in {"read", "confirm_mutate", "mutate"} else "read"
