from __future__ import annotations

import asyncio
from typing import Any

import pytest

from realtime_api.pipelines.speech_llm_toolassist_speech import (
    _navigation_intents,
    _timeout_for_route,
    _tool_work_timeout_s,
    _tools_for_capability,
)
from realtime_api.runtime.capabilities import CapabilityId
from realtime_api.runtime.navigation import (
    classifier_capabilities,
    navigate_profile,
    route_for_navigation,
)
from realtime_api.runtime.navigation_graph import (
    NAVIGATION_GRAPH,
    run_profile_navigation,
)


class ScriptedClassifier:
    def __init__(
        self,
        capability: CapabilityId,
        confidence: float = 0.98,
        *,
        confirmed: bool = False,
    ) -> None:
        self.capability = capability
        self.confidence = confidence
        self.confirmed = confirmed
        self.calls: list[dict[str, Any]] = []

    def classify_navigation(
        self,
        utterance: str,
        *,
        language: str,
        capabilities: list[dict[str, Any]],
        context: str = "",
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "utterance": utterance,
                "language": language,
                "capabilities": capabilities,
                "context": context,
            }
        )
        return {
            "capability_id": self.capability.value,
            "confidence": self.confidence,
            "ambiguous": False,
            "confirmed": self.confirmed,
            "depth": "fact",
        }


@pytest.mark.parametrize(
    ("utterance", "language", "capability", "industry"),
    [
        ("Take me to telecom billing", "en-US", CapabilityId.NAVIGATE_TELCO, "telco"),
        ("Llévame a servicios financieros", "es-ES", CapabilityId.NAVIGATE_FSI, "fsi"),
        ("मुझे नॉलेज एजेंट पर ले जाएँ", "hi-IN", CapabilityId.NAVIGATE_KNOWLEDGE, "knowledge"),
        ("พาฉันไปฝ่ายบริการบิลโทรศัพท์", "th-TH", CapabilityId.NAVIGATE_TELCO, "telco"),
        ("带我去金融服务", "zh-CN", CapabilityId.NAVIGATE_FSI, "fsi"),
    ],
)
def test_concierge_destination_is_a_typed_selection(
    utterance: str,
    language: str,
    capability: CapabilityId,
    industry: str,
):
    classifier = ScriptedClassifier(capability)
    decision = navigate_profile(
        utterance,
        profile="concierge",
        language=language,
        classifier=classifier,
        has_obo=False,
        conversation_open=False,
        min_confidence=0.8,
    )
    intents = _navigation_intents(decision)
    assert len(intents) == 1
    assert intents[0].name == "select_industry"
    assert intents[0].arguments == {"industry": industry}
    assert all("select_industry" not in str(item) for item in classifier.calls[0]["capabilities"])


@pytest.mark.parametrize(
    ("utterance", "language"),
    [
        ("Why did my expenses increase this month?", "en-US"),
        ("¿Por qué aumentaron mis gastos este mes?", "es-ES"),
        ("इस महीने मेरे खर्च क्यों बढ़े?", "hi-IN"),
        ("ทำไมค่าใช้จ่ายของฉันเพิ่มขึ้นในเดือนนี้", "th-TH"),
        ("为什么我这个月的支出增加了？", "zh-CN"),
    ],
)
def test_card_investigation_withdraws_all_non_investigation_tools(
    utterance: str, language: str
):
    classifier = ScriptedClassifier(CapabilityId.INVESTIGATE_AGENT_MODE)
    decision = navigate_profile(
        utterance,
        profile="card",
        language=language,
        classifier=classifier,
        has_obo=True,
        conversation_open=False,
        min_confidence=0.8,
    )
    route = route_for_navigation(decision, utterance=utterance, profile="card")
    tools = [
        {"function": {"name": name}}
        for name in (
            "select_use_case",
            "card_account_facts",
            "ask_card_genie",
            "start_deep_dive",
        )
    ]
    selected = _tools_for_capability(tools, decision.capability_id)
    assert route.adapter == "agent_mode"
    assert [(item["function"]["name"]) for item in selected] == ["start_deep_dive"]


def test_card_topic_selection_is_not_misread_as_an_investigation():
    classifier = ScriptedClassifier(CapabilityId.CARD_REWARDS)
    decision = navigate_profile(
        "I'd like the rewards option",
        profile="card",
        language="en-US",
        classifier=classifier,
        has_obo=True,
        conversation_open=False,
        min_confidence=0.8,
    )
    intents = _navigation_intents(decision)
    assert intents[0].arguments == {"use_case": "rewards_optimizer"}


@pytest.mark.parametrize(
    ("capability", "expected_tool"),
    [
        (CapabilityId.PACK_FACTS, "lookup_account"),
        (CapabilityId.BILLING_ANALYSIS, "ask_genie"),
        (CapabilityId.BILLING_ACTION, "apply_billing_action"),
        (CapabilityId.CURRENT_TIME, "get_current_time"),
    ],
)
def test_billing_capability_exposes_only_its_owned_tool(
    capability: CapabilityId, expected_tool: str
):
    tools = [
        {"function": {"name": name}}
        for name in (
            "lookup_account",
            "ask_genie",
            "apply_billing_action",
            "get_current_time",
        )
    ]
    selected = _tools_for_capability(tools, capability)
    assert [item["function"]["name"] for item in selected] == [expected_tool]


def test_billing_analysis_requires_obo_and_policy_fails_closed():
    classifier = ScriptedClassifier(CapabilityId.BILLING_ANALYSIS)
    decision = navigate_profile(
        "Show the overdue invoice trend by month",
        profile="billing",
        language="en-US",
        classifier=classifier,
        has_obo=False,
        conversation_open=False,
        min_confidence=0.8,
    )
    assert decision.capability_id == CapabilityId.REFUSE


def test_confirmation_context_reaches_billing_classifier():
    classifier = ScriptedClassifier(CapabilityId.BILLING_ACTION, confirmed=True)
    decision = navigate_profile(
        "Yes, go ahead",
        profile="billing",
        language="en-US",
        classifier=classifier,
        has_obo=True,
        conversation_open=False,
        min_confidence=0.8,
        context="I can waive the late fee. Shall I go ahead?",
        offer_open=True,
    )
    assert decision.capability_id == CapabilityId.BILLING_ACTION
    assert classifier.calls[0]["context"].startswith("I can waive")


def test_billing_change_without_open_offer_is_forced_to_clarify():
    classifier = ScriptedClassifier(CapabilityId.BILLING_ACTION, confirmed=True)
    decision = navigate_profile(
        "Yes, go ahead",
        profile="billing",
        language="en-US",
        classifier=classifier,
        has_obo=True,
        conversation_open=False,
        min_confidence=0.8,
        context="I can waive the late fee. Shall I go ahead?",
        offer_open=False,
    )
    assert decision.capability_id == CapabilityId.CLARIFY
    assert decision.reason.value == "confirmation_required"


def test_billing_change_without_explicit_confirmation_is_forced_to_clarify():
    classifier = ScriptedClassifier(CapabilityId.BILLING_ACTION, confirmed=False)
    decision = navigate_profile(
        "Please waive my late fee",
        profile="billing",
        language="en-US",
        classifier=classifier,
        has_obo=True,
        conversation_open=False,
        min_confidence=0.8,
        context="I can waive the late fee. Shall I go ahead?",
        offer_open=True,
    )
    assert decision.capability_id == CapabilityId.CLARIFY
    assert decision.ambiguous is True


def test_profile_catalogs_do_not_expose_implementation_names():
    implementation_names = {
        "select_industry",
        "select_use_case",
        "lookup_account",
        "ask_genie",
        "apply_billing_action",
        "start_deep_dive",
    }
    for profile in ("concierge", "card", "billing"):
        catalog = classifier_capabilities(profile)
        rendered = str([item.classifier_view() for item in catalog])
        assert not any(name in rendered for name in implementation_names)


def test_shared_navigation_graph_is_stateless_for_all_profiles():
    classifier = ScriptedClassifier(CapabilityId.PACK_FACTS)
    decision, route = asyncio.run(
        run_profile_navigation(
            "What is my current balance?",
            profile="billing",
            language="en-US",
            classifier=classifier,
            has_obo=True,
            conversation_open=False,
            min_confidence=0.8,
        )
    )
    assert decision.capability_id == CapabilityId.PACK_FACTS
    assert route.adapter == "pack_facts"
    assert getattr(NAVIGATION_GRAPH, "checkpointer", None) is None


def test_genie_one_gets_the_long_work_budget():
    """Measured workspace answers land at 2-3.5 minutes.

    A shorter cap saves the caller nothing and turns those answers into timeouts,
    so Genie One shares the long-work budget and the caller is kept engaged by
    spoken progress instead.
    """
    assert _tool_work_timeout_s("genie_one_mcp", "knowledge") == 420.0
    assert _tool_work_timeout_s("agent_mode", "card") == 420.0
    assert _tool_work_timeout_s("pack_facts", "knowledge") == 50.0


def test_turn_budget_outlives_the_tool_budget_it_supervises():
    """The tool's graceful timeout needs room to be rendered and spoken.

    When the two budgets are equal the outer one fires first and cancels the turn,
    so the typed timeout (which carries the conversation id) never reaches the
    caller and they hear a hard failure instead of a sentence.
    """
    for adapter in ("genie_one_mcp", "agent_mode"):
        tool_s = _tool_work_timeout_s(adapter, "knowledge")
        assert _timeout_for_route(adapter, "knowledge") > tool_s


def test_pack_facts_withdraws_card_space_tool():
    tools = [
        {"function": {"name": name}}
        for name in ("card_account_facts", "ask_card_genie", "start_deep_dive")
    ]
    selected = _tools_for_capability(tools, CapabilityId.PACK_FACTS)
    assert [item["function"]["name"] for item in selected] == ["card_account_facts"]


def test_card_query_exposes_only_the_space_tool():
    classifier = ScriptedClassifier(CapabilityId.CARD_QUERY)
    decision = navigate_profile(
        "What were my largest transactions this cycle?",
        profile="card",
        language="en-US",
        classifier=classifier,
        has_obo=True,
        conversation_open=False,
        min_confidence=0.8,
    )
    route = route_for_navigation(decision, utterance="q", profile="card")
    tools = [
        {"function": {"name": name}}
        for name in ("card_account_facts", "ask_card_genie", "start_deep_dive")
    ]
    selected = _tools_for_capability(tools, decision.capability_id)
    assert route.adapter == "space_conversation"
    assert [item["function"]["name"] for item in selected] == ["ask_card_genie"]


def test_long_governed_answer_is_spoken_as_its_summary_not_row_by_row():
    """The FSI deep dive speaks the rendered summary, never the composed rows.

    Reading the composed claims aloud turns an Agent Mode report into minutes of
    "column: value" narration, which is what the summarizer exists to prevent.
    """
    from realtime_api.pipelines.speech_llm_toolassist_speech import _spoken_answer

    spoken = _spoken_answer(
        claims=[
            {"text": "category: Travel; delta: 1240.55; share: 0.41"},
            {"text": "category: Dining; delta: 880.10; share: 0.29"},
            {"text": "category: Fuel; delta: 410.00; share: 0.14"},
        ],
        tool_invocations=[{"name": "start_deep_dive"}],
        response_text="Here is a long unused model reply.",
        rendered_summary="Travel drove most of the increase this cycle.",
        requires_tool=True,
        runtime_error_code=None,
        refuse_text=None,
        language="en-US",
    )
    assert spoken == "Travel drove most of the increase this cycle."


def test_tool_evidence_without_a_summary_falls_back_to_cited_claims():
    from realtime_api.pipelines.speech_llm_toolassist_speech import _spoken_answer

    spoken = _spoken_answer(
        claims=[{"text": "balance: 412.90"}],
        tool_invocations=[{"name": "card_account_facts"}],
        response_text="Uncited model prose.",
        rendered_summary="",
        requires_tool=True,
        runtime_error_code=None,
        refuse_text=None,
        language="en-US",
    )
    assert spoken == "balance: 412.90"


def test_factual_capability_without_a_tool_call_is_refused():
    from realtime_api.pipelines.speech_llm_toolassist_speech import _spoken_answer
    from realtime_api.runtime.refuse import ErrorCode, refuse_speech

    spoken = _spoken_answer(
        claims=[],
        tool_invocations=[],
        response_text="I think you spent about 60 million tokens.",
        rendered_summary="",
        requires_tool=True,
        runtime_error_code=None,
        refuse_text=None,
        language="en-US",
    )
    assert spoken == refuse_speech(ErrorCode.NO_EVIDENCE, language="en-US")
