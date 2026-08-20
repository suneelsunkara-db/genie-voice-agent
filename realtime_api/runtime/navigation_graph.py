"""Stateless LangGraph host for realtime capability navigation.

The graph sequences existing policy and typed navigation contracts.  It has no
checkpointer, memory, tools, model transport, or observability backend; the voice
turn remains owned by ``AgentRuntime``/``TurnTrace`` and Genie One remains the
workspace conversation owner.
"""
from __future__ import annotations

from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from .capabilities import NavigationDecision
from .goal_frame import RouteDecision
from .navigation import (
    NavigationClassifier,
    navigate_profile,
    route_for_navigation,
)


class NavigationState(TypedDict, total=False):
    utterance: str
    profile: str
    language: str
    classifier: NavigationClassifier
    has_obo: bool
    conversation_open: bool
    min_confidence: float
    context: str
    offer_open: bool
    decision: NavigationDecision
    route: RouteDecision


def _policy(
    state: NavigationState,
) -> Command[Literal["semantic", "dispatch"]]:
    """Resolve hard conversation ownership without spending a model call."""
    if state["profile"] == "knowledge" and state["conversation_open"]:
        decision = navigate_profile(
            state["utterance"],
            profile=state["profile"],
            language=state["language"],
            classifier=state["classifier"],
            has_obo=state["has_obo"],
            conversation_open=True,
            min_confidence=state["min_confidence"],
            context=state.get("context", ""),
            offer_open=bool(state.get("offer_open", False)),
        )
        return Command(goto="dispatch", update={"decision": decision})
    return Command(goto="semantic")


def _semantic(state: NavigationState) -> dict[str, Any]:
    """One forced structured classify call, including policy validation."""
    decision = navigate_profile(
        state["utterance"],
        profile=state["profile"],
        language=state["language"],
        classifier=state["classifier"],
        has_obo=state["has_obo"],
        conversation_open=False,
        min_confidence=state["min_confidence"],
        context=state.get("context", ""),
        offer_open=bool(state.get("offer_open", False)),
    )
    return {"decision": decision}


def _dispatch(state: NavigationState) -> dict[str, Any]:
    """Map the validated capability to the existing GoalFrame/adapter contract."""
    return {
        "route": route_for_navigation(
            state["decision"],
            utterance=state["utterance"],
            profile=state["profile"],
        )
    }


def _compile_navigation_graph():
    builder = StateGraph(NavigationState)
    builder.add_node("policy", _policy)
    builder.add_node("semantic", _semantic)
    builder.add_node("dispatch", _dispatch)
    builder.add_edge(START, "policy")
    builder.add_edge("semantic", "dispatch")
    builder.add_edge("dispatch", END)
    # Deliberately no checkpointer: a barge-in kills this voice turn.
    return builder.compile()


NAVIGATION_GRAPH = _compile_navigation_graph()
# Public compatibility alias retained for callers/tests from the Knowledge proof.
KNOWLEDGE_NAVIGATION_GRAPH = NAVIGATION_GRAPH


async def run_profile_navigation(
    utterance: str,
    *,
    profile: str,
    language: str,
    classifier: NavigationClassifier,
    has_obo: bool,
    conversation_open: bool,
    min_confidence: float,
    context: str = "",
    offer_open: bool = False,
) -> tuple[NavigationDecision, RouteDecision]:
    state = await NAVIGATION_GRAPH.ainvoke(
        {
            "utterance": utterance,
            "profile": profile,
            "language": language,
            "classifier": classifier,
            "has_obo": has_obo,
            "conversation_open": conversation_open,
            "min_confidence": min_confidence,
            "context": context,
            "offer_open": offer_open,
        }
    )
    return state["decision"], state["route"]


async def run_knowledge_navigation(
    utterance: str,
    *,
    language: str,
    classifier: NavigationClassifier,
    has_obo: bool,
    conversation_open: bool,
    min_confidence: float,
    context: str = "",
) -> tuple[NavigationDecision, RouteDecision]:
    return await run_profile_navigation(
        utterance,
        profile="knowledge",
        language=language,
        classifier=classifier,
        has_obo=has_obo,
        conversation_open=conversation_open,
        min_confidence=min_confidence,
        context=context,
    )
