"""GoalFrame routing — scope / depth / effect → adapter matrix.

Intent produces a GoalFrame; the runtime maps (scope, depth) to one adapter.
The LLM never chooses among Genie Space / Agent Mode / Genie One as free tools.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

from .refuse import ErrorCode, ErrorEvidence


Scope = Literal["app", "pack", "workspace"]
Depth = Literal["fact", "investigate"]
Effect = Literal["read", "confirm_mutate", "mutate"]
Adapter = Literal[
    "self_know",
    "space_conversation",
    "agent_mode",
    "genie_one_mcp",
    "pack_facts",
    "refuse",
    "clarify",
]
BargeClass = Literal["amend", "new", "stop"]


class RoutePath(str, Enum):
    SELF_KNOW = "self_know"
    PACK_INTENT = "pack_intent"
    FAST_FACTS = "fast_facts"
    WORKSPACE = "workspace"
    REFUSE = "refuse"
    CLARIFY = "clarify"


@dataclass(frozen=True)
class GoalFrame:
    scope: Scope
    depth: Depth
    effect: Effect = "read"
    pack_id: str | None = None
    utterance: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def adapter(self) -> Adapter:
        return adapter_for(self)


# Investigate cues (why-in-pack → Agent Mode).
_INVESTIGATE_RE = re.compile(
    r"\b("
    r"why|how come|howcome|break(?:\s|-)?down|root cause|explain why|"
    r"what caused|what drove|investigate|dig into|analyze why"
    r")\b",
    re.I,
)
_FACT_RE = re.compile(
    r"\b("
    r"how much|how many|what is|what's|whats|when is|when was|"
    r"show me|lookup|look up|balance|total|count|metric|status"
    r")\b",
    re.I,
)
_STOP_RE = re.compile(r"\b(stop|cancel|never ?mind|forget it|quit)\b", re.I)
_AMEND_RE = re.compile(
    r"\b("
    r"actually|instead|i meant|change (?:it|that)|narrow|"
    r"for (?:last|this|the) |only |just |filter|exclude|include"
    r")\b",
    re.I,
)
# Workspace-only cues (not pack Lakebase).
_WORKSPACE_RE = re.compile(
    r"\b("
    r"workspace|cluster cost|job cost|unity catalog|uc table|"
    r"across (?:all )?packs|genie one|warehouse spend|dbsql"
    r")\b",
    re.I,
)
# Lakebase / pack fact cues that must NOT route to Genie One.
_LAKEBASE_RE = re.compile(
    r"\b("
    r"my (?:bill|invoice|balance|account|statement|rewards|points)|"
    r"overdue|late fee|payment plan|cardholder|this cycle"
    r")\b",
    re.I,
)
_SELF_KNOW_RE = re.compile(
    r"\b("
    r"what can you (?:do|help)|who are you|your (?:name|capabilities)|"
    r"help me navigate|what packs"
    r")\b",
    re.I,
)


def classify_depth(utterance: str) -> Depth | Literal["ambiguous"]:
    """Fact vs investigate. Ambiguous when neither cue fires clearly."""
    text = (utterance or "").strip()
    if not text:
        return "ambiguous"
    inv = bool(_INVESTIGATE_RE.search(text))
    fact = bool(_FACT_RE.search(text))
    if inv and not fact:
        return "investigate"
    if fact and not inv:
        return "fact"
    if inv and fact:
        # "why is my balance X" — investigate wins for why-cues.
        return "investigate"
    return "ambiguous"


def classify_barge_intent(
    utterance: str,
    *,
    active_goal: GoalFrame | None = None,
) -> BargeClass:
    """After barge-in STT: amend | new | stop. Unsure → new (safe default)."""
    text = (utterance or "").strip()
    if not text:
        return "stop"
    if _STOP_RE.search(text) and len(text.split()) <= 6:
        return "stop"
    if active_goal is not None and _AMEND_RE.search(text):
        # Same-pack topical continuity heuristic.
        return "amend"
    return "new"


def adapter_for(frame: GoalFrame) -> Adapter:
    """Matrix: pack+fact→Space Conversation; pack+investigate→Agent Mode; workspace→Genie One."""
    if frame.scope == "app":
        return "self_know"
    if frame.scope == "pack":
        if frame.depth == "investigate":
            return "agent_mode"
        return "space_conversation"
    if frame.scope == "workspace":
        return "genie_one_mcp"
    return "refuse"


@dataclass
class RouteDecision:
    path: RoutePath
    frame: GoalFrame | None
    adapter: Adapter
    reason: str = ""


def cascade_route(
    utterance: str,
    *,
    pack_id: str | None = None,
    pack_intent_matched: bool = False,
    has_fast_facts: bool = False,
    has_obo: bool = False,
    effect: Effect = "read",
) -> RouteDecision:
    """Deterministic cascade: self-know → pack intent → fast facts → workspace(OBO) → refuse.

    Misroute guards:
      - Lakebase-answerable (fast facts / pack facts) ↛ Genie One
      - pack fact ↛ workspace
      - why-in-pack → Agent Mode (investigate)
    """
    text = (utterance or "").strip()
    depth = classify_depth(text)

    if _SELF_KNOW_RE.search(text):
        frame = GoalFrame(scope="app", depth="fact", effect="read", utterance=text)
        return RouteDecision(RoutePath.SELF_KNOW, frame, "self_know", "app self-knowledge")

    if depth == "ambiguous" and pack_intent_matched:
        frame = GoalFrame(
            scope="pack",
            depth="fact",
            effect=effect,
            pack_id=pack_id,
            utterance=text,
            meta={"clarify_depth": True},
        )
        return RouteDecision(RoutePath.CLARIFY, frame, "clarify", "ambiguous depth in pack")

    if pack_intent_matched or (pack_id and not _WORKSPACE_RE.search(text)):
        # Pack-bound: never escalate to workspace Genie One for pack facts.
        d: Depth = "investigate" if depth == "investigate" else "fact"
        if depth == "ambiguous":
            d = "fact"  # pack-bound default: fact first
        # Lakebase / account cues stay on pack even if wording is fuzzy.
        if _LAKEBASE_RE.search(text) or has_fast_facts or pack_intent_matched:
            frame = GoalFrame(scope="pack", depth=d, effect=effect, pack_id=pack_id, utterance=text)
            adapter = adapter_for(frame)
            # Fast facts lane for read lookups when Lakebase can answer.
            if d == "fact" and has_fast_facts and not _INVESTIGATE_RE.search(text):
                return RouteDecision(
                    RoutePath.FAST_FACTS,
                    frame,
                    "pack_facts",
                    "lakebase/pack facts (not Genie One)",
                )
            path = RoutePath.PACK_INTENT
            return RouteDecision(path, frame, adapter, f"pack depth={d}")

    if has_fast_facts and not _WORKSPACE_RE.search(text):
        frame = GoalFrame(scope="pack", depth="fact", effect=effect, pack_id=pack_id, utterance=text)
        return RouteDecision(
            RoutePath.FAST_FACTS,
            frame,
            "pack_facts",
            "fast facts without workspace cue",
        )

    if _WORKSPACE_RE.search(text) or (not pack_id and not pack_intent_matched and not has_fast_facts):
        if not has_obo:
            return RouteDecision(
                RoutePath.REFUSE,
                None,
                "refuse",
                "workspace path requires OBO",
            )
        d2: Depth = "investigate" if depth == "investigate" else "fact"
        if depth == "ambiguous":
            return RouteDecision(
                RoutePath.CLARIFY,
                GoalFrame(scope="workspace", depth="fact", effect=effect, utterance=text),
                "clarify",
                "ambiguous workspace depth",
            )
        frame = GoalFrame(scope="workspace", depth=d2, effect=effect, utterance=text)
        return RouteDecision(RoutePath.WORKSPACE, frame, "genie_one_mcp", "workspace Genie One")

    return RouteDecision(
        RoutePath.REFUSE,
        None,
        "refuse",
        "no matching cascade step",
    )


def enforce_effect(
    effect: Effect,
    *,
    user_confirmed: bool = False,
    mutate_succeeded: bool = False,
) -> ErrorEvidence | None:
    """Light gateway enforcement of effect classes.

    - read: always ok
    - confirm_mutate: block execution until user_confirmed
    - mutate: speech of completion only after mutate_succeeded
    """
    if effect == "read":
        return None
    if effect == "confirm_mutate" and not user_confirmed:
        return ErrorEvidence(
            code=ErrorCode.AMBIGUOUS,
            message="confirm_mutate requires explicit user affirmation before execution",
            retryable=True,
        )
    if effect == "mutate" and not mutate_succeeded:
        return ErrorEvidence(
            code=ErrorCode.UNSUPPORTED,
            message="mutate completion speech blocked until success evidence",
            retryable=False,
        )
    return None
