from __future__ import annotations

import asyncio
from typing import Any

from realtime_api.runtime.capabilities import (
    CapabilityId,
    NavigationReason,
)
from realtime_api.runtime.navigation import navigate_knowledge, route_for_navigation
from realtime_api.runtime.navigation_graph import (
    KNOWLEDGE_NAVIGATION_GRAPH,
    run_knowledge_navigation,
)
from realtime_api.services import DatabricksServing


class ScriptedClassifier:
    def __init__(self, decisions: dict[str, dict[str, Any]]) -> None:
        self.decisions = decisions
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
        return self.decisions[utterance]


def _decision(capability: CapabilityId, confidence: float = 0.98) -> dict[str, Any]:
    return {
        "capability_id": capability.value,
        "confidence": confidence,
        "ambiguous": False,
        "depth": "fact",
    }


def test_multilingual_owned_measurements_map_to_workspace_capability():
    samples = {
        "How much have we spent on the Qwen model in recent months?": "en-US",
        "¿Cuánto hemos gastado en el modelo Qwen en los últimos meses?": "es-ES",
        "पिछले कुछ महीनों में हमने Qwen मॉडल पर कितना खर्च किया?": "hi-IN",
        "ช่วงไม่กี่เดือนที่ผ่านมาเราใช้จ่ายกับโมเดล Qwen เท่าไร": "th-TH",
        "过去几个月我们在 Qwen 模型上花了多少钱？": "zh-CN",
    }
    classifier = ScriptedClassifier(
        {text: _decision(CapabilityId.WORKSPACE_QUERY) for text in samples}
    )

    for utterance, language in samples.items():
        decision = navigate_knowledge(
            utterance,
            language=language,
            classifier=classifier,
            has_obo=True,
            conversation_open=False,
            min_confidence=0.80,
        )
        route = route_for_navigation(
            decision,
            utterance=utterance,
            profile="knowledge",
        )
        assert decision.capability_id == CapabilityId.WORKSPACE_QUERY
        assert route.adapter == "genie_one_mcp"
        assert route.frame is not None and route.frame.scope == "workspace"

    assert [call["language"] for call in classifier.calls] == list(samples.values())
    assert all(
        "knowledge_search" not in str(call["capabilities"])
        and "workspace_query" not in str(call["capabilities"])
        for call in classifier.calls
    )


def test_documentation_explanation_maps_to_docs_pack():
    utterance = "What is Model Serving?"
    classifier = ScriptedClassifier(
        {utterance: _decision(CapabilityId.KNOWLEDGE_DOCS)}
    )
    decision = navigate_knowledge(
        utterance,
        language="en-US",
        classifier=classifier,
        has_obo=True,
        conversation_open=False,
        min_confidence=0.80,
    )
    route = route_for_navigation(decision, utterance=utterance, profile="knowledge")
    assert route.adapter == "pack_facts"
    assert route.frame is not None and route.frame.scope == "pack"


def test_open_workspace_conversation_is_policy_owned_without_model_call():
    classifier = ScriptedClassifier({})
    decision = navigate_knowledge(
        "Unity Catalog",
        language="en-US",
        classifier=classifier,
        has_obo=True,
        conversation_open=True,
        min_confidence=0.80,
    )
    assert decision.capability_id == CapabilityId.WORKSPACE_QUERY
    assert decision.reason == NavigationReason.OPEN_CONVERSATION
    assert classifier.calls == []


def test_workspace_proposal_without_obo_fails_closed():
    utterance = "Show my workspace spend"
    classifier = ScriptedClassifier(
        {utterance: _decision(CapabilityId.WORKSPACE_QUERY)}
    )
    decision = navigate_knowledge(
        utterance,
        language="en-US",
        classifier=classifier,
        has_obo=False,
        conversation_open=False,
        min_confidence=0.80,
    )
    assert decision.capability_id == CapabilityId.REFUSE
    assert decision.reason == NavigationReason.PERMISSION_REQUIRED


def test_low_confidence_or_malformed_proposal_clarifies():
    low = "Tell me about this"
    malformed = "Something else"
    classifier = ScriptedClassifier(
        {
            low: _decision(CapabilityId.KNOWLEDGE_DOCS, confidence=0.51),
            malformed: {"answer": "I think this is docs"},
        }
    )
    for utterance in (low, malformed):
        decision = navigate_knowledge(
            utterance,
            language="en-US",
            classifier=classifier,
            has_obo=True,
            conversation_open=False,
            min_confidence=0.80,
        )
        assert decision.capability_id == CapabilityId.CLARIFY
        assert decision.ambiguous is True


class CaptureClient:
    def __init__(self) -> None:
        self.inputs: dict[str, Any] | None = None

    def predict(self, *, endpoint: str, inputs: dict[str, Any]) -> dict[str, Any]:
        self.inputs = inputs
        return {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "select_navigation_capability",
                                    "arguments": (
                                        '{"capability_id":"workspace.query",'
                                        '"confidence":0.99,"ambiguous":false,'
                                        '"depth":"fact"}'
                                    ),
                                }
                            }
                        ]
                    }
                }
            ]
        }


def test_serving_classifier_forces_one_structured_navigation_call():
    client = CaptureClient()
    serving = DatabricksServing(
        client=client,
        stt_endpoint="stt",
        llm_endpoint="llm",
        tts_endpoint="tts",
    )
    result = serving.classify_navigation(
        "How much did we spend?",
        language="en-US",
        capabilities=[
            {
                "id": "knowledge.docs",
                "purpose": "Explain product documentation.",
                "requires_obo": False,
            },
            {
                "id": "workspace.query",
                "purpose": "Measure the caller's workspace.",
                "requires_obo": True,
            },
        ],
    )
    assert result["capability_id"] == "workspace.query"
    assert client.inputs is not None
    assert client.inputs["temperature"] == 0.0
    assert client.inputs["tool_choice"] == {
        "type": "function",
        "function": {"name": "select_navigation_capability"},
    }
    assert client.inputs["tools"][0]["function"]["parameters"]["additionalProperties"] is False


def test_stateless_graph_sequences_semantic_route_and_has_no_checkpointer():
    utterance = "How much have we spent on Qwen?"
    classifier = ScriptedClassifier(
        {utterance: _decision(CapabilityId.WORKSPACE_QUERY)}
    )
    decision, route = asyncio.run(
        run_knowledge_navigation(
            utterance,
            language="en-US",
            classifier=classifier,
            has_obo=True,
            conversation_open=False,
            min_confidence=0.80,
        )
    )
    assert decision.capability_id == CapabilityId.WORKSPACE_QUERY
    assert route.adapter == "genie_one_mcp"
    assert getattr(KNOWLEDGE_NAVIGATION_GRAPH, "checkpointer", None) is None


def test_stateless_graph_skips_semantic_model_for_open_conversation():
    classifier = ScriptedClassifier({})
    decision, route = asyncio.run(
        run_knowledge_navigation(
            "Unity Catalog",
            language="en-US",
            classifier=classifier,
            has_obo=True,
            conversation_open=True,
            min_confidence=0.80,
        )
    )
    assert decision.reason == NavigationReason.OPEN_CONVERSATION
    assert route.adapter == "genie_one_mcp"
    assert classifier.calls == []
