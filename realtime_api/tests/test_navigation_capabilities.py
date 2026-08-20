from __future__ import annotations

import pytest
from pydantic import ValidationError

from realtime_api.runtime.capabilities import (
    CAPABILITIES,
    CapabilityId,
    NavigationDecision,
    NavigationReason,
    capabilities_for_profile,
    capability_descriptor,
)


def test_capability_ids_are_unique_and_described():
    ids = [item.id for item in CAPABILITIES]
    assert len(ids) == len(set(ids))
    assert all(item.purpose.strip() for item in CAPABILITIES)


def test_knowledge_catalog_exposes_outcomes_not_tool_names():
    catalog = capabilities_for_profile("knowledge")
    ids = {item.id for item in catalog}
    assert CapabilityId.KNOWLEDGE_DOCS in ids
    assert CapabilityId.WORKSPACE_QUERY in ids
    assert CapabilityId.CLARIFY in ids
    assert CapabilityId.REFUSE in ids
    assert CapabilityId.PACK_FACTS not in ids
    assert CapabilityId.INVESTIGATE_AGENT_MODE not in ids
    assert all(
        "knowledge_search" not in item.purpose
        and "workspace_query" not in item.purpose
        for item in catalog
    )


def test_workspace_capability_requires_obo():
    descriptor = capability_descriptor(CapabilityId.WORKSPACE_QUERY)
    assert descriptor.requires_obo is True
    assert descriptor.adapter == "genie_one_mcp"


def test_card_query_is_split_from_headline_facts():
    facts = capability_descriptor(CapabilityId.PACK_FACTS)
    query = capability_descriptor(CapabilityId.CARD_QUERY)
    assert "ask_card_genie" not in facts.tools
    assert query.tools == frozenset({"ask_card_genie"})
    assert query.requires_obo is True
    catalog = {item.id for item in capabilities_for_profile("card")}
    assert CapabilityId.CARD_QUERY in catalog
    assert CapabilityId.PACK_FACTS in catalog


def test_navigation_decision_is_strict_and_bounded():
    decision = NavigationDecision(
        capability_id=CapabilityId.WORKSPACE_QUERY,
        confidence=0.97,
        depth="fact",
        reason=NavigationReason.SEMANTIC_CLASSIFICATION,
    )
    assert decision.capability_id == CapabilityId.WORKSPACE_QUERY

    with pytest.raises(ValidationError):
        NavigationDecision(
            capability_id=CapabilityId.KNOWLEDGE_DOCS,
            confidence=1.1,
        )
    with pytest.raises(ValidationError):
        NavigationDecision(
            capability_id=CapabilityId.KNOWLEDGE_DOCS,
            confidence=0.9,
            implementation_tool="knowledge_search",
        )
