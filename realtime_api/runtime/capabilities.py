"""Language-neutral capability contracts for agent navigation.

The navigator classifies a caller's *speech act* onto one stable capability ID.
It never chooses an implementation name (``knowledge_search``, ``workspace_query``)
or receives a corpus keyword list.  Application policy validates the proposal and
only then maps the capability to the existing GoalFrame/adapter matrix.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .goal_frame import Adapter, Depth, Effect, Scope


class CapabilityId(StrEnum):
    """Stable, language-neutral actions understood by the navigation kernel."""

    APP_SELF_KNOW = "app.self_know"
    NAVIGATE_TELCO = "navigate.telco"
    NAVIGATE_FSI = "navigate.fsi"
    NAVIGATE_KNOWLEDGE = "navigate.knowledge"
    KNOWLEDGE_DOCS = "knowledge.docs"
    WORKSPACE_QUERY = "workspace.query"
    CARD_STATEMENT = "card.statement"
    CARD_REWARDS = "card.rewards"
    PACK_FACTS = "pack.facts"
    CARD_QUERY = "card.query"
    INVESTIGATE_AGENT_MODE = "investigate.agent_mode"
    BILLING_ANALYSIS = "billing.analysis"
    BILLING_ACTION_PREPARE = "billing.action.prepare"
    BILLING_ACTION = "billing.action"
    CURRENT_TIME = "utility.current_time"
    CLARIFY = "system.clarify"
    REFUSE = "system.refuse"


class NavigationReason(StrEnum):
    """Auditable reason codes; never localized prose."""

    OPEN_CONVERSATION = "open_conversation"
    SEMANTIC_CLASSIFICATION = "semantic_classification"
    LOW_CONFIDENCE = "low_confidence"
    AMBIGUOUS = "ambiguous"
    PERMISSION_REQUIRED = "permission_required"
    UNSUPPORTED = "unsupported"
    POLICY_OVERRIDE = "policy_override"
    ACTION_PREPARATION_REQUIRED = "action_preparation_required"
    CONFIRMATION_REQUIRED = "confirmation_required"


@dataclass(frozen=True)
class CapabilitySelection:
    """Engine-executed selection signal. The conversational LLM never sees this tool."""

    tool_name: str
    arguments: dict[str, Any]
    confirm_phrase: str


@dataclass(frozen=True)
class CapabilityDescriptor:
    """One capability the policy layer may expose to the semantic navigator."""

    id: CapabilityId
    purpose: str
    scope: Scope | Literal["system"]
    depth: Depth
    effect: Effect
    adapter: Adapter
    requires_obo: bool = False
    requires_tool: bool = False
    tools: frozenset[str] = field(default_factory=frozenset)
    selection: CapabilitySelection | None = None
    prompt: str = ""
    profiles: frozenset[str] = frozenset()

    def classifier_view(self) -> dict[str, str | bool]:
        """Minimal model-facing contract: no implementation/tool names."""
        return {
            "id": self.id.value,
            "purpose": self.purpose,
            "requires_obo": self.requires_obo,
        }


class NavigationDecision(BaseModel):
    """Strict output produced by policy or the multilingual semantic classifier."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    capability_id: CapabilityId
    confidence: float = Field(ge=0.0, le=1.0)
    ambiguous: bool = False
    confirmed: bool = False
    depth: Depth = "fact"
    reason: NavigationReason = NavigationReason.SEMANTIC_CLASSIFICATION


CAPABILITIES: tuple[CapabilityDescriptor, ...] = (
    CapabilityDescriptor(
        id=CapabilityId.APP_SELF_KNOW,
        purpose="Explain this application's own capabilities or help the caller navigate it.",
        scope="app",
        depth="fact",
        effect="read",
        adapter="self_know",
    ),
    CapabilityDescriptor(
        id=CapabilityId.NAVIGATE_TELCO,
        purpose=(
            "Open the Telco billing-support experience after the caller chooses it. "
            "This is a destination choice, not a billing question."
        ),
        scope="app",
        depth="fact",
        effect="read",
        adapter="self_know",
        selection=CapabilitySelection(
            tool_name="select_industry",
            arguments={"industry": "telco"},
            confirm_phrase="confirm.navigate_telco",
        ),
        profiles=frozenset({"concierge"}),
    ),
    CapabilityDescriptor(
        id=CapabilityId.NAVIGATE_FSI,
        purpose=(
            "Open the Financial Services credit-card experience after the caller "
            "chooses it. This is a destination choice, not a card question."
        ),
        scope="app",
        depth="fact",
        effect="read",
        adapter="self_know",
        selection=CapabilitySelection(
            tool_name="select_industry",
            arguments={"industry": "fsi"},
            confirm_phrase="confirm.navigate_fsi",
        ),
        profiles=frozenset({"concierge"}),
    ),
    CapabilityDescriptor(
        id=CapabilityId.NAVIGATE_KNOWLEDGE,
        purpose=(
            "Open the Databricks Knowledge experience after the caller chooses it. "
            "This is a destination choice, not a platform question."
        ),
        scope="app",
        depth="fact",
        effect="read",
        adapter="self_know",
        selection=CapabilitySelection(
            tool_name="select_industry",
            arguments={"industry": "knowledge"},
            confirm_phrase="confirm.navigate_knowledge",
        ),
        profiles=frozenset({"concierge"}),
    ),
    CapabilityDescriptor(
        id=CapabilityId.KNOWLEDGE_DOCS,
        purpose=(
            "Explain a Databricks product, feature, concept, documented behavior, "
            "or general how-to using governed documentation."
        ),
        scope="pack",
        depth="fact",
        effect="read",
        adapter="pack_facts",
        requires_tool=True,
        tools=frozenset({"knowledge_search"}),
        prompt=(
            "Call the provided documentation capability exactly once with the caller's "
            "question, then answer only from its result in {language}."
        ),
        profiles=frozenset({"knowledge"}),
    ),
    CapabilityDescriptor(
        id=CapabilityId.WORKSPACE_QUERY,
        purpose=(
            "Answer from the caller's own governed workspace: usage, spend, counts, "
            "trends, jobs, tables, dashboards, permissions, freshness, or other "
            "organization-specific data and analysis."
        ),
        scope="workspace",
        depth="fact",
        effect="read",
        adapter="genie_one_mcp",
        requires_obo=True,
        requires_tool=True,
        tools=frozenset({"workspace_query"}),
        profiles=frozenset({"knowledge"}),
    ),
    CapabilityDescriptor(
        id=CapabilityId.PACK_FACTS,
        purpose=(
            "Look up a headline read-only fact from the pack's low-latency store: "
            "current balance, overdue amount, points, due date, or account status. "
            "Not a causal investigation and not a broad analytical report."
        ),
        scope="pack",
        depth="fact",
        effect="read",
        adapter="pack_facts",
        requires_tool=True,
        tools=frozenset({"card_account_facts", "lookup_account"}),
        prompt=(
            "Answer the caller's concrete read-only question using the provided "
            "headline-fact capability. Call it once, state only returned facts, "
            "and respond in {language}."
        ),
        profiles=frozenset({"billing", "card"}),
    ),
    CapabilityDescriptor(
        id=CapabilityId.CARD_QUERY,
        purpose=(
            "Ask the cardholder's governed card space a specific factual question "
            "that is not a headline Lakebase fact and is not a causal investigation."
        ),
        scope="pack",
        depth="fact",
        effect="read",
        adapter="space_conversation",
        requires_obo=True,
        requires_tool=True,
        tools=frozenset({"ask_card_genie"}),
        prompt=(
            "Call the provided card-space capability exactly once with the complete "
            "question, then answer only from its result in {language}."
        ),
        profiles=frozenset({"card"}),
    ),
    CapabilityDescriptor(
        id=CapabilityId.INVESTIGATE_AGENT_MODE,
        purpose="Investigate causes, drivers, or explanations using the active product pack.",
        scope="pack",
        depth="investigate",
        effect="read",
        adapter="agent_mode",
        requires_obo=True,
        requires_tool=True,
        tools=frozenset({"start_deep_dive"}),
        prompt=(
            "Call the provided investigation capability exactly once with the caller's "
            "complete question. Do not invent causes. Respond in {language}."
        ),
        profiles=frozenset({"card"}),
    ),
    CapabilityDescriptor(
        id=CapabilityId.CARD_STATEMENT,
        purpose=(
            "Select the card Statement Insights topic when the caller is merely "
            "choosing that offered experience, without yet asking for a fact or cause."
        ),
        scope="pack",
        depth="fact",
        effect="read",
        adapter="pack_facts",
        selection=CapabilitySelection(
            tool_name="select_use_case",
            arguments={"use_case": "statement_insights"},
            confirm_phrase="confirm.card_statement",
        ),
        profiles=frozenset({"card"}),
    ),
    CapabilityDescriptor(
        id=CapabilityId.CARD_REWARDS,
        purpose=(
            "Select the card Rewards Optimizer topic when the caller is merely "
            "choosing that offered experience, without yet asking for a fact or analysis."
        ),
        scope="pack",
        depth="fact",
        effect="read",
        adapter="pack_facts",
        selection=CapabilitySelection(
            tool_name="select_use_case",
            arguments={"use_case": "rewards_optimizer"},
            confirm_phrase="confirm.card_rewards",
        ),
        profiles=frozenset({"card"}),
    ),
    CapabilityDescriptor(
        id=CapabilityId.BILLING_ANALYSIS,
        purpose=(
            "Run a read-only analytical or reporting question over governed billing "
            "data when a normal account lookup cannot answer it."
        ),
        scope="pack",
        depth="fact",
        effect="read",
        adapter="space_conversation",
        requires_obo=True,
        requires_tool=True,
        tools=frozenset({"ask_genie"}),
        prompt=(
            "Call the provided analytical capability exactly once with the complete "
            "question, then answer only from its result in {language}."
        ),
        profiles=frozenset({"billing"}),
    ),
    CapabilityDescriptor(
        id=CapabilityId.BILLING_ACTION_PREPARE,
        purpose=(
            "Prepare a requested billing resolution such as waiving a late fee or "
            "setting up a payment plan when no exact offer is open yet. Read the "
            "current caller's account, identify the concrete invoice and amount, "
            "and ask for confirmation. This capability never changes account state."
        ),
        scope="pack",
        depth="fact",
        effect="read",
        adapter="pack_facts",
        requires_tool=True,
        tools=frozenset({"prepare_billing_action"}),
        prompt=(
            "Call the provided preparation capability exactly once for the requested "
            "action. Present only the returned proposal and ask for explicit "
            "confirmation in {language}. Do not claim the action has been applied."
        ),
        profiles=frozenset({"billing"}),
    ),
    CapabilityDescriptor(
        id=CapabilityId.BILLING_ACTION,
        purpose=(
            "Execute an exact billing resolution that the agent already offered on "
            "this call, but only after the caller explicitly confirms that offer. "
            "A first request to waive a fee or start a plan belongs to "
            "billing.action.prepare instead."
        ),
        scope="pack",
        depth="fact",
        effect="confirm_mutate",
        adapter="pack_facts",
        requires_tool=True,
        tools=frozenset({"apply_billing_action"}),
        prompt=(
            "Call the provided account-changing capability before claiming any change "
            "succeeded. If its result is not successful, say so. Respond in {language}."
        ),
        profiles=frozenset({"billing"}),
    ),
    CapabilityDescriptor(
        id=CapabilityId.CURRENT_TIME,
        purpose="Answer a request for the current date or time in a timezone.",
        scope="app",
        depth="fact",
        effect="read",
        adapter="pack_facts",
        requires_tool=True,
        tools=frozenset({"get_current_time"}),
        prompt="Call the provided time capability once, then answer concisely in {language}.",
        profiles=frozenset({"billing"}),
    ),
    CapabilityDescriptor(
        id=CapabilityId.CLARIFY,
        purpose="Ask one concise clarification because more than one allowed capability fits.",
        scope="system",
        depth="fact",
        effect="read",
        adapter="clarify",
    ),
    CapabilityDescriptor(
        id=CapabilityId.REFUSE,
        purpose="Refuse because no allowed capability can answer the request reliably.",
        scope="system",
        depth="fact",
        effect="read",
        adapter="refuse",
    ),
)

_BY_ID = {item.id: item for item in CAPABILITIES}


def capability_descriptor(capability_id: CapabilityId) -> CapabilityDescriptor:
    return _BY_ID[capability_id]


def capabilities_for_profile(profile: str) -> tuple[CapabilityDescriptor, ...]:
    """Capabilities visible to one profile, plus shared app/system outcomes."""
    return tuple(
        item
        for item in CAPABILITIES
        if not item.profiles or profile in item.profiles
    )
