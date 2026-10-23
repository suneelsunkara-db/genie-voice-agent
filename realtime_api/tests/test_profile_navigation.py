from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from realtime_api.pipelines.speech_llm_toolassist_speech import (
    _finalize_billing_offer,
    _direct_terminal_call,
    _has_long_work,
    _navigation_intents,
    _public_agent_event,
    _timeout_for_route,
    _tool_work_timeout_s,
    _tools_and_prompt,
    _tools_for_capability,
)
from realtime_api.runtime.capabilities import (
    CapabilityId,
    NavigationDecision,
    NavigationReason,
)
from realtime_api.runtime.navigation import (
    classifier_capabilities,
    navigate_profile,
    resolve_navigation_continuation,
    route_for_navigation,
)
from realtime_api.runtime.navigation_graph import (
    NAVIGATION_GRAPH,
    run_profile_navigation,
)


def test_agent_mode_internal_english_report_is_not_sent_to_browser():
    internal = {
        "kind": "action.completed",
        "payload": {
            "name": "start_deep_dive",
            "result": {
                "status": "completed",
                "report": "Internal English report",
                "canonical_question": "Why did expenses increase?",
                "canonical_request": "Analyze expenses.",
                "reasoning": ["Writing final response"],
                "tables": [{"columns": ["amount"], "rows": [[3400]]}],
            },
        },
    }

    public = _public_agent_event(internal)

    assert public["payload"]["result"] == {
        "status": "completed",
        "tables": [{"columns": ["amount"], "rows": [[3400]]}],
    }
    assert internal["payload"]["result"]["report"] == "Internal English report"


class ScriptedClassifier:
    def __init__(
        self,
        capability: CapabilityId,
        confidence: float = 0.98,
        *,
        confirmed: bool = False,
        spoken_reply: str | None = None,
        request_state: str = "complete",
        clarification_focus: str = "none",
    ) -> None:
        self.capability = capability
        self.confidence = confidence
        self.confirmed = confirmed
        self.spoken_reply = spoken_reply
        self.request_state = request_state
        self.clarification_focus = clarification_focus
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
        decision = {
            "capability_id": self.capability.value,
            "confidence": self.confidence,
            "ambiguous": False,
            "confirmed": self.confirmed,
            "depth": "fact",
            "request_state": self.request_state,
            "clarification_focus": self.clarification_focus,
        }
        if self.spoken_reply is not None:
            decision["spoken_reply"] = self.spoken_reply
        return decision


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
        ("We are forecasting CPU and GPU usage for our top customers.", "en-US"),
        ("Estamos pronosticando el uso de CPU y GPU de nuestros principales clientes.", "es-ES"),
        ("हम अपने प्रमुख ग्राहकों के CPU और GPU उपयोग का पूर्वानुमान लगा रहे हैं।", "hi-IN"),
        ("เรากำลังคาดการณ์การใช้ CPU และ GPU ของลูกค้ารายใหญ่", "th-TH"),
        ("我们正在预测主要客户的 CPU 和 GPU 使用量。", "zh-CN"),
    ],
)
def test_concierge_knowledge_question_preserves_semantic_handoff(
    utterance: str,
    language: str,
):
    classifier = ScriptedClassifier(CapabilityId.HANDOFF_KNOWLEDGE)
    decision = navigate_profile(
        utterance,
        profile="concierge",
        language=language,
        classifier=classifier,
        has_obo=True,
        conversation_open=False,
        min_confidence=0.8,
    )
    intents = _navigation_intents(
        decision,
        utterance=utterance,
        language=language,
    )
    assert intents[0].name == "select_industry"
    assert intents[0].arguments == {
        "industry": "knowledge",
        "handoff_query": utterance,
        "handoff_language": language,
    }


def test_home_transfers_incomplete_databricks_request_without_local_clarification():
    utterance = "I have a question about forecasting usage for my customer."
    decision = navigate_profile(
        utterance,
        profile="concierge",
        language="en-US",
        classifier=ScriptedClassifier(
            CapabilityId.HANDOFF_KNOWLEDGE,
            request_state="incomplete",
            clarification_focus="data_scope",
        ),
        has_obo=True,
        conversation_open=False,
        min_confidence=0.8,
    )
    assert decision.capability_id == CapabilityId.HANDOFF_KNOWLEDGE
    intents = _navigation_intents(
        decision,
        utterance=utterance,
        language="en-US",
    )
    assert intents[0].arguments["handoff_query"] == utterance


def test_concierge_clarification_preserves_original_request_for_knowledge():
    state: dict[str, object] = {}
    original = "We have started an effort looking at top customers and forecasting."
    clarify = NavigationDecision(
        capability_id=CapabilityId.CLARIFY,
        confidence=0.7,
        ambiguous=True,
        reason=NavigationReason.LOW_CONFIDENCE,
    )
    first, handoff = resolve_navigation_continuation(
        clarify,
        utterance=original,
        profile="concierge",
        profile_state=state,
    )
    assert first == clarify
    assert handoff is None

    selection = NavigationDecision(
        capability_id=CapabilityId.NAVIGATE_KNOWLEDGE,
        confidence=0.95,
    )
    resolved, handoff = resolve_navigation_continuation(
        selection,
        utterance="General knowledge, forecasting insights.",
        profile="concierge",
        profile_state=state,
    )
    assert resolved.capability_id == CapabilityId.HANDOFF_KNOWLEDGE
    assert resolved.reason == NavigationReason.CLARIFICATION_RESOLVED
    assert handoff == (
        f"{original}\n"
        "General knowledge, forecasting insights."
    )
    intents = _navigation_intents(
        resolved,
        utterance=handoff,
        language="en-US",
    )
    assert intents[0].arguments["handoff_query"] == handoff
    assert not state


def test_non_knowledge_selection_clears_pending_clarification():
    state: dict[str, object] = {}
    clarify = NavigationDecision(
        capability_id=CapabilityId.CLARIFY,
        confidence=0.5,
        ambiguous=True,
    )
    resolve_navigation_continuation(
        clarify,
        utterance="I need help with an account.",
        profile="concierge",
        profile_state=state,
    )
    telco = NavigationDecision(
        capability_id=CapabilityId.NAVIGATE_TELCO,
        confidence=0.99,
    )
    resolved, handoff = resolve_navigation_continuation(
        telco,
        utterance="Telco.",
        profile="concierge",
        profile_state=state,
    )
    assert resolved == telco
    assert handoff is None
    assert not state


def test_incomplete_workspace_request_is_delegated_to_genie_one():
    decision = navigate_profile(
        "Estimate usage for my customer.",
        profile="knowledge",
        language="en-US",
        classifier=ScriptedClassifier(
            CapabilityId.WORKSPACE_QUERY,
            request_state="incomplete",
            clarification_focus="data_scope",
        ),
        has_obo=True,
        conversation_open=False,
        min_confidence=0.8,
    )
    assert decision.capability_id == CapabilityId.WORKSPACE_QUERY
    assert decision.reason == NavigationReason.SEMANTIC_CLASSIFICATION
    assert decision.request_state == "complete"
    assert decision.clarification_focus == "none"


def test_contradictory_genie_one_readiness_fields_do_not_trigger_local_clarification():
    decision = navigate_profile(
        "Estimate usage for my customer.",
        profile="knowledge",
        language="en-US",
        classifier=ScriptedClassifier(
            CapabilityId.WORKSPACE_QUERY,
            request_state="complete",
            clarification_focus="data_scope",
        ),
        has_obo=True,
        conversation_open=False,
        min_confidence=0.8,
    )
    assert decision.capability_id == CapabilityId.WORKSPACE_QUERY
    assert decision.reason == NavigationReason.SEMANTIC_CLASSIFICATION
    assert decision.request_state == "complete"
    assert decision.clarification_focus == "none"


def test_home_clarification_asks_about_intent_not_app_menu():
    tools, prompt = _tools_and_prompt(
        SimpleNamespace(name="concierge"),
        route_adapter="clarify",
        workspace_selected=False,
        conversation_joinable=False,
        clarification_focus="data_scope",
    )
    assert tools == []
    assert "content-aware question" in prompt
    assert "Do not propose categories, examples" in prompt
    assert "Do not ask the caller to choose Telco" in prompt
    assert "missing entity, population, data source" in prompt
    assert "do not suggest a source" in prompt


def test_home_outcome_clarification_forbids_invented_options():
    _, prompt = _tools_and_prompt(
        SimpleNamespace(name="concierge"),
        route_adapter="clarify",
        workspace_selected=False,
        conversation_joinable=False,
        clarification_focus="outcome",
    )
    assert "Reuse only subject nouns the caller actually said" in prompt
    assert "do not introduce a domain, activity, metric" in prompt
    assert "forecast" not in prompt.lower()


def test_policy_keeps_direct_reply_only_for_tool_free_app_conversation():
    reply = "Of course—go ahead."
    self_know = navigate_profile(
        "Can I ask you a question?",
        profile="concierge",
        language="en-US",
        classifier=ScriptedClassifier(
            CapabilityId.APP_SELF_KNOW,
            spoken_reply=reply,
        ),
        has_obo=True,
        conversation_open=False,
        min_confidence=0.8,
    )
    assert self_know.spoken_reply == reply

    knowledge = navigate_profile(
        "What did we spend?",
        profile="concierge",
        language="en-US",
        classifier=ScriptedClassifier(
            CapabilityId.HANDOFF_KNOWLEDGE,
            spoken_reply="An ungrounded answer",
        ),
        has_obo=True,
        conversation_open=False,
        min_confidence=0.8,
    )
    assert knowledge.spoken_reply is None


def test_engagement_is_reserved_for_long_running_routes():
    assert _has_long_work(adapter="self_know", workspace_selected=False) is False
    assert _has_long_work(adapter="clarify", workspace_selected=False) is False
    assert _has_long_work(adapter="billing", workspace_selected=False) is False
    assert _has_long_work(adapter="agent_mode", workspace_selected=False) is True
    assert _has_long_work(adapter="anything", workspace_selected=True) is True


def test_terminal_tool_receives_exact_request_from_its_schema():
    request = "Opaque caller request ENTITY-123."
    selected = _direct_terminal_call(
        [
            {
                "type": "function",
                "function": {
                    "name": "terminal_analysis",
                    "parameters": {
                        "type": "object",
                        "properties": {"request": {"type": "string"}},
                        "required": ["request"],
                    },
                },
            }
        ],
        request,
    )
    assert selected == ("terminal_analysis", {"request": request})


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
        (CapabilityId.BILLING_ACTION_PREPARE, "prepare_billing_action"),
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
            "prepare_billing_action",
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


def test_billing_change_without_open_offer_is_routed_to_read_only_preparation():
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
    assert decision.capability_id == CapabilityId.BILLING_ACTION_PREPARE
    assert decision.reason.value == "action_preparation_required"
    route = route_for_navigation(decision, utterance="Yes, go ahead", profile="billing")
    assert route.frame is not None
    assert route.frame.effect == "read"


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
    route = route_for_navigation(decision, utterance="Please waive my late fee", profile="billing")
    assert route.adapter == "confirm"


def test_billing_offer_opens_confirmation_only_after_audio():
    candidate = {
        "action": "waive_late_fee",
        "customer_id": "CUST-4028",
        "invoice_id": "INV-90114",
    }
    silent = SimpleNamespace(profile_state={"pending_billing_offer": candidate})
    _finalize_billing_offer(
        silent,
        capability_id=CapabilityId.BILLING_ACTION_PREPARE,
        tts_chunks=0,
    )
    assert "pending_confirm_mutate" not in silent.profile_state
    assert "pending_billing_offer" not in silent.profile_state

    spoken = SimpleNamespace(profile_state={"pending_billing_offer": candidate})
    _finalize_billing_offer(
        spoken,
        capability_id=CapabilityId.BILLING_ACTION_PREPARE,
        tts_chunks=1,
    )
    assert spoken.profile_state["pending_confirm_mutate"] == candidate
    assert "pending_billing_offer" not in spoken.profile_state


def test_profile_catalogs_do_not_expose_implementation_names():
    implementation_names = {
        "select_industry",
        "select_use_case",
        "lookup_account",
        "ask_genie",
        "prepare_billing_action",
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


def test_upstream_answer_render_is_preferred_for_speech():
    from realtime_api.pipelines.speech_llm_toolassist_speech import _spoken_answer

    spoken = _spoken_answer(
        response_text="Here is a long unused model reply.",
        rendered_summary="Travel drove most of the increase this cycle.",
        runtime_error_code=None,
        refuse_text=None,
        language="en-US",
    )
    assert spoken == "Travel drove most of the increase this cycle."


def test_natural_response_is_never_replaced_by_structured_claims():
    from realtime_api.pipelines.speech_llm_toolassist_speech import _spoken_answer

    spoken = _spoken_answer(
        response_text="Your current balance is $412.90.",
        rendered_summary="",
        runtime_error_code=None,
        refuse_text=None,
        language="en-US",
    )
    assert spoken == "Your current balance is $412.90."


def test_empty_answer_uses_the_upstream_refusal():
    from realtime_api.pipelines.speech_llm_toolassist_speech import _spoken_answer

    spoken = _spoken_answer(
        response_text="",
        rendered_summary="",
        runtime_error_code=None,
        refuse_text="Genie could not answer that question.",
        language="en-US",
    )
    assert spoken == "Genie could not answer that question."


def test_failed_evidence_render_never_falls_back_to_uncertified_model_text():
    from realtime_api.pipelines.speech_llm_toolassist_speech import _spoken_answer

    spoken = _spoken_answer(
        response_text="**$10,525. Broken fallback from the conversational model.",
        rendered_summary="",
        runtime_error_code=None,
        refuse_text="I found the result but could not safely read it aloud.",
        language="en-US",
        evidence_render_failed=True,
    )

    assert spoken == "I found the result but could not safely read it aloud."
