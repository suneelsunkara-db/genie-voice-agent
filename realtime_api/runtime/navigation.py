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


_PENDING_CLARIFICATION_KEY = "_pending_navigation_clarification"
_MAX_CONTINUATION_CHARS = 8_000
_MAX_CONTINUATION_TURNS = 4


def navigation_request_for_classification(
    utterance: str,
    *,
    profile_state: dict[str, object],
) -> str:
    """Return the complete pending request for semantic classification.

    A clarification answer is not an independent request.  The navigator must
    classify it together with the turns that prompted it; reconstruction after
    classification is too late and can trigger repeated or irrelevant follow-ups.
    """
    clean = utterance.strip()
    existing = profile_state.get(_PENDING_CLARIFICATION_KEY)
    turns = [
        str(item).strip()
        for item in existing
        if str(item).strip()
    ] if isinstance(existing, list) else []
    if clean and (not turns or turns[-1] != clean):
        turns.append(clean)
    if not turns:
        return clean
    turns = turns[-_MAX_CONTINUATION_TURNS:]
    while len("\n".join(turns)) > _MAX_CONTINUATION_CHARS and len(turns) > 1:
        turns.pop(0)
    return "\n".join(turns)


def resolve_navigation_continuation(
    decision: NavigationDecision,
    *,
    utterance: str,
    profile: str,
    profile_state: dict[str, object],
) -> tuple[NavigationDecision, str | None]:
    """Carry an incomplete request through its clarification turns.

    A clarification answer augments the request; it never replaces it. Home keeps
    its special destination-to-Knowledge handoff, while an already selected profile
    receives the same exact multi-turn request back for governed execution.
    """
    clean = utterance.strip()
    if decision.capability_id == CapabilityId.CLARIFY:
        existing = profile_state.get(_PENDING_CLARIFICATION_KEY)
        turns = [
            str(item).strip()
            for item in existing
            if str(item).strip()
        ] if isinstance(existing, list) else []
        if clean and (not turns or turns[-1] != clean):
            turns.append(clean)
        turns = turns[-_MAX_CONTINUATION_TURNS:]
        while len("\n".join(turns)) > _MAX_CONTINUATION_CHARS and len(turns) > 1:
            turns.pop(0)
        profile_state[_PENDING_CLARIFICATION_KEY] = turns
        return decision, None

    existing = profile_state.pop(_PENDING_CLARIFICATION_KEY, None)
    turns = [
        str(item).strip()
        for item in existing
        if str(item).strip()
    ] if isinstance(existing, list) else []
    if not turns:
        return decision, None

    if profile == "concierge" and decision.capability_id == CapabilityId.NAVIGATE_KNOWLEDGE:
        decision = decision.model_copy(
            update={
                "capability_id": CapabilityId.HANDOFF_KNOWLEDGE,
                "ambiguous": False,
                "reason": NavigationReason.CLARIFICATION_RESOLVED,
            }
        )
    if profile == "concierge" and decision.capability_id != CapabilityId.HANDOFF_KNOWLEDGE:
        return decision, None

    if clean and (not turns or turns[-1] != clean):
        turns.append(clean)
    handoff = "\n".join(turns)
    if len(handoff) > _MAX_CONTINUATION_CHARS:
        handoff = handoff[-_MAX_CONTINUATION_CHARS:]
    return decision, handoff


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
        if profile == "knowledge":
            return NavigationDecision(
                capability_id=(
                    CapabilityId.WORKSPACE_QUERY
                    if has_obo
                    else CapabilityId.REFUSE
                ),
                confidence=0.0,
                depth="fact",
                reason=(
                    NavigationReason.POLICY_OVERRIDE
                    if has_obo
                    else NavigationReason.PERMISSION_REQUIRED
                ),
            )
        return NavigationDecision(
            capability_id=CapabilityId.CLARIFY,
            confidence=0.0,
            ambiguous=True,
            reason=NavigationReason.AMBIGUOUS,
            request_state="ambiguous",
            clarification_focus="missing_details",
        )

    allowed = {item.id for item in catalog}
    if proposed.capability_id not in allowed:
        return NavigationDecision(
            capability_id=CapabilityId.REFUSE,
            confidence=1.0,
            reason=NavigationReason.POLICY_OVERRIDE,
        )
    # Home owns only the destination decision. Once it recognises a substantive
    # Databricks request, transfer the exact utterance to Knowledge even when the
    # request itself is incomplete; Genie One owns that downstream conversation.
    if profile == "concierge" and (
        proposed.capability_id == CapabilityId.HANDOFF_KNOWLEDGE
        or proposed.candidate_capability_id == CapabilityId.HANDOFF_KNOWLEDGE
    ):
        descriptor = capability_descriptor(CapabilityId.HANDOFF_KNOWLEDGE)
        return NavigationDecision(
            capability_id=CapabilityId.HANDOFF_KNOWLEDGE,
            confidence=proposed.confidence,
            depth=descriptor.depth,
            reason=NavigationReason.SEMANTIC_CLASSIFICATION,
        )
    # The Knowledge surface delegates both answering and clarification to the
    # same Genie One conversation. Do not interpose a local completeness gate:
    # a short or ambiguous workspace request is still useful upstream context,
    # and its next turn is joined with the returned conversation_id.
    if profile == "knowledge" and proposed.capability_id not in {
        CapabilityId.APP_SELF_KNOW,
        CapabilityId.REFUSE,
    }:
        if not has_obo:
            return NavigationDecision(
                capability_id=CapabilityId.REFUSE,
                confidence=proposed.confidence,
                depth="fact",
                reason=NavigationReason.PERMISSION_REQUIRED,
            )
        return NavigationDecision(
            capability_id=CapabilityId.WORKSPACE_QUERY,
            confidence=proposed.confidence,
            depth="fact",
            reason=NavigationReason.SEMANTIC_CLASSIFICATION,
        )
    # ``request_state`` and ``clarification_focus`` are one semantic contract.
    # A model can occasionally emit the internally contradictory combination
    # "complete" + a concrete missing-information focus.  Do not execute a
    # governed tool from that output: the focus is positive evidence that a
    # follow-up is required, while "none" is the only valid focus for a
    # complete request.
    if (
        proposed.request_state == "complete"
        and proposed.clarification_focus != "none"
        and proposed.capability_id
        not in {CapabilityId.APP_SELF_KNOW, CapabilityId.REFUSE}
    ):
        proposed = proposed.model_copy(
            update={
                "request_state": "incomplete",
                "ambiguous": True,
            }
        )
    if (
        proposed.capability_id not in {CapabilityId.APP_SELF_KNOW, CapabilityId.REFUSE}
        and proposed.request_state in {"incomplete", "ambiguous"}
    ):
        candidate = proposed.candidate_capability_id
        if candidate is None and proposed.capability_id != CapabilityId.CLARIFY:
            candidate = proposed.capability_id
        return NavigationDecision(
            capability_id=CapabilityId.CLARIFY,
            confidence=proposed.confidence,
            ambiguous=True,
            depth=proposed.depth,
            reason=(
                NavigationReason.INCOMPLETE
                if proposed.request_state == "incomplete"
                else NavigationReason.AMBIGUOUS
            ),
            request_state=proposed.request_state,
            clarification_focus=(
                proposed.clarification_focus
                if proposed.clarification_focus != "none"
                else "missing_details"
            ),
            candidate_capability_id=candidate,
        )
    if proposed.ambiguous or proposed.confidence < min_confidence:
        candidate = proposed.candidate_capability_id
        if candidate is None and proposed.capability_id not in {
            CapabilityId.CLARIFY,
            CapabilityId.REFUSE,
        }:
            candidate = proposed.capability_id
        return NavigationDecision(
            capability_id=CapabilityId.CLARIFY,
            confidence=proposed.confidence,
            ambiguous=True,
            depth=proposed.depth,
            reason=NavigationReason.LOW_CONFIDENCE,
            request_state=(
                proposed.request_state
                if proposed.request_state != "complete"
                else "ambiguous"
            ),
            clarification_focus=(
                proposed.clarification_focus
                if proposed.clarification_focus != "none"
                else "missing_details"
            ),
            candidate_capability_id=candidate,
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
        # A first request for a mutation is preparation, not confirmation. Route
        # it through a read-only capability that resolves the exact account item
        # and opens a concrete offer. This also safely corrects a classifier that
        # chose BILLING_ACTION instead of BILLING_ACTION_PREPARE.
        if not offer_open:
            preparation = capability_descriptor(CapabilityId.BILLING_ACTION_PREPARE)
            return NavigationDecision(
                capability_id=CapabilityId.BILLING_ACTION_PREPARE,
                confidence=proposed.confidence,
                depth=preparation.depth,
                reason=NavigationReason.ACTION_PREPARATION_REQUIRED,
            )
        # Even with an offer open, the classifier's explicit-confirmation boolean
        # must be true before policy exposes the mutation capability.
        if not proposed.confirmed:
            return NavigationDecision(
                capability_id=CapabilityId.CLARIFY,
                confidence=proposed.confidence,
                ambiguous=True,
                depth="fact",
                reason=NavigationReason.CONFIRMATION_REQUIRED,
            )
    # Depth belongs to the selected capability contract, not to a second free
    # choice by the model. A model-authored spoken reply is executable output, so
    # retain it only on the explicitly tool-free app-conversation capability.
    return proposed.model_copy(
        update={
            "depth": descriptor.depth,
            "spoken_reply": (
                proposed.spoken_reply
                if proposed.capability_id == CapabilityId.APP_SELF_KNOW
                else None
            ),
        }
    )


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

    if descriptor.scope == "workspace":
        frame = GoalFrame(
            scope="workspace",
            depth=decision.depth,
            effect=descriptor.effect,
            utterance=utterance,
            meta={"capability_id": decision.capability_id.value},
        )
        return RouteDecision(RoutePath.WORKSPACE, frame, descriptor.adapter, reason)
    if decision.capability_id in {
        CapabilityId.PACK_FACTS,
        CapabilityId.CARD_QUERY,
        CapabilityId.CARD_STATEMENT,
        CapabilityId.CARD_REWARDS,
        CapabilityId.BILLING_ANALYSIS,
        CapabilityId.BILLING_ACTION_PREPARE,
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
        if decision.reason == NavigationReason.CONFIRMATION_REQUIRED:
            return RouteDecision(RoutePath.CLARIFY, None, "confirm", reason)
        return RouteDecision(RoutePath.CLARIFY, None, "clarify", reason)
    return RouteDecision(RoutePath.REFUSE, None, "refuse", reason)
