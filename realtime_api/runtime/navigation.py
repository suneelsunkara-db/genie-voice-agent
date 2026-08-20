"""Policy-validated semantic navigation for the realtime agent runtime.

The semantic model may *propose* a language-neutral capability.  It cannot see
implementation tool names and cannot override conversation ownership, OBO, the
confidence gate, or profile allowlists.  This module is intentionally independent
of LangGraph: the functions are the node contracts a stateless graph will sequence
after shadow validation proves them.
"""
from __future__ import annotations

from typing import Any, Protocol

from pydantic import ValidationError

from .capabilities import (
    CapabilityDescriptor,
    CapabilityId,
    NavigationDecision,
    NavigationReason,
    capabilities_for_profile,
    capability_descriptor,
)
from .goal_frame import GoalFrame, RouteDecision, RoutePath


class NavigationClassifier(Protocol):
    def classify_navigation(
        self,
        utterance: str,
        *,
        language: str,
        capabilities: list[dict[str, Any]],
        context: str = "",
    ) -> dict[str, Any]: ...


def classifier_capabilities(profile: str) -> tuple[CapabilityDescriptor, ...]:
    """Model-visible catalog for one profile; implementation names stay hidden."""
    return capabilities_for_profile(profile)


def knowledge_classifier_capabilities() -> tuple[CapabilityDescriptor, ...]:
    """Backward-compatible name for the proven Knowledge catalog."""
    return classifier_capabilities("knowledge")


def navigate_profile(
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
) -> NavigationDecision:
    """Choose one capability, then enforce profile, confidence, and identity policy."""
    if not 0 <= min_confidence <= 1:
        raise ValueError("navigation min_confidence must be between 0 and 1")

    # Genie One owns follow-ups in an open Knowledge conversation. No other profile
    # may inherit that state, and no model call is needed to rediscover ownership.
    if profile == "knowledge" and conversation_open:
        if has_obo:
            return NavigationDecision(
                capability_id=CapabilityId.WORKSPACE_QUERY,
                confidence=1.0,
                depth="fact",
                reason=NavigationReason.OPEN_CONVERSATION,
            )
        return NavigationDecision(
            capability_id=CapabilityId.REFUSE,
            confidence=1.0,
            depth="fact",
            reason=NavigationReason.PERMISSION_REQUIRED,
        )

    catalog = classifier_capabilities(profile)
    raw = classifier.classify_navigation(
        utterance,
        language=language,
        capabilities=[item.classifier_view() for item in catalog],
        context=context,
    )
    try:
        proposed = NavigationDecision.model_validate(raw)
    except ValidationError:
        return NavigationDecision(
            capability_id=CapabilityId.CLARIFY,
            confidence=0.0,
            ambiguous=True,
            reason=NavigationReason.AMBIGUOUS,
        )

    allowed = {item.id for item in catalog}
    if proposed.capability_id not in allowed:
        return NavigationDecision(
            capability_id=CapabilityId.REFUSE,
            confidence=1.0,
            reason=NavigationReason.POLICY_OVERRIDE,
        )
    if proposed.ambiguous or proposed.confidence < min_confidence:
        return NavigationDecision(
            capability_id=CapabilityId.CLARIFY,
            confidence=proposed.confidence,
            ambiguous=True,
            depth=proposed.depth,
            reason=NavigationReason.LOW_CONFIDENCE,
        )
    descriptor = capability_descriptor(proposed.capability_id)
    if descriptor.requires_obo and not has_obo:
        return NavigationDecision(
            capability_id=CapabilityId.REFUSE,
            confidence=proposed.confidence,
            depth=descriptor.depth,
            reason=NavigationReason.PERMISSION_REQUIRED,
        )
    if proposed.capability_id == CapabilityId.BILLING_ACTION:
        # An account change needs an offer already open on this call *and*
        # an explicit confirmation. The classifier boolean is not enough.
        if not offer_open or not proposed.confirmed:
            return NavigationDecision(
                capability_id=CapabilityId.CLARIFY,
                confidence=proposed.confidence,
                ambiguous=True,
                depth="fact",
                reason=NavigationReason.CONFIRMATION_REQUIRED,
            )
    # Depth belongs to the selected capability contract, not to a second free
    # choice by the model.
    return proposed.model_copy(update={"depth": descriptor.depth})


def navigate_knowledge(
    utterance: str,
    *,
    language: str,
    classifier: NavigationClassifier,
    has_obo: bool,
    conversation_open: bool,
    min_confidence: float,
    context: str = "",
) -> NavigationDecision:
    """Choose and validate one Knowledge capability.

    Conversation ownership is a hard state transition and therefore never spends
    a classifier call.  All first-turn semantic proposals are parsed through the
    strict ``NavigationDecision`` schema and then policy-validated.
    """
    return navigate_profile(
        utterance,
        profile="knowledge",
        language=language,
        classifier=classifier,
        has_obo=has_obo,
        conversation_open=conversation_open,
        min_confidence=min_confidence,
        context=context,
        offer_open=False,
    )


def route_for_navigation(
    decision: NavigationDecision,
    *,
    utterance: str,
    profile: str,
) -> RouteDecision:
    """Project a validated capability onto the existing GoalFrame matrix."""
    descriptor = capability_descriptor(decision.capability_id)
    reason = f"capability={decision.capability_id.value};reason={decision.reason.value}"

    if decision.capability_id == CapabilityId.WORKSPACE_QUERY:
        frame = GoalFrame(
            scope="workspace",
            depth=decision.depth,
            effect=descriptor.effect,
            utterance=utterance,
            meta={"capability_id": decision.capability_id.value},
        )
        return RouteDecision(RoutePath.WORKSPACE, frame, descriptor.adapter, reason)
    if decision.capability_id in {
        CapabilityId.KNOWLEDGE_DOCS,
        CapabilityId.PACK_FACTS,
        CapabilityId.CARD_QUERY,
        CapabilityId.CARD_STATEMENT,
        CapabilityId.CARD_REWARDS,
        CapabilityId.BILLING_ANALYSIS,
        CapabilityId.BILLING_ACTION,
        CapabilityId.CURRENT_TIME,
    }:
        frame = GoalFrame(
            scope=descriptor.scope if descriptor.scope in {"app", "pack", "workspace"} else "pack",
            depth=descriptor.depth,
            effect=descriptor.effect,
            pack_id=profile,
            utterance=utterance,
            meta={"capability_id": decision.capability_id.value},
        )
        return RouteDecision(RoutePath.FAST_FACTS, frame, descriptor.adapter, reason)
    if decision.capability_id == CapabilityId.INVESTIGATE_AGENT_MODE:
        frame = GoalFrame(
            scope="pack",
            depth="investigate",
            effect=descriptor.effect,
            pack_id=profile,
            utterance=utterance,
            meta={"capability_id": decision.capability_id.value},
        )
        return RouteDecision(RoutePath.PACK_INTENT, frame, descriptor.adapter, reason)
    if descriptor.scope == "app":
        frame = GoalFrame(
            scope="app",
            depth="fact",
            effect="read",
            utterance=utterance,
            meta={"capability_id": decision.capability_id.value},
        )
        return RouteDecision(RoutePath.SELF_KNOW, frame, "self_know", reason)
    if decision.capability_id == CapabilityId.CLARIFY:
        return RouteDecision(RoutePath.CLARIFY, None, "clarify", reason)
    return RouteDecision(RoutePath.REFUSE, None, "refuse", reason)
