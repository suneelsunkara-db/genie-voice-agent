from realtime_api.contracts import SessionStart
from realtime_api.pipelines.speech_llm_toolassist_speech import turn_provenance
from realtime_api.tracing import TurnTrace


def test_every_voice_profile_has_a_conversation_surface() -> None:
    expected = {
        "concierge": "home",
        "billing": "telco",
        "card": "card",
        "knowledge": "knowledge",
    }
    for profile, surface in expected.items():
        config = SessionStart.from_event(
            {"type": "session.start", "profile": profile, "surface": surface}
        )
        assert turn_provenance(config) == (profile, surface, "conversation")


def test_non_conversation_ingress_is_classified() -> None:
    for surface, traffic_class in {
        "realtime-test": "diagnostic",
        "diagnostic": "diagnostic",
        "mcp": "mcp",
        "benchmark": "benchmark",
    }.items():
        config = SessionStart.from_event(
            {"type": "session.start", "profile": "billing", "surface": surface}
        )
        assert turn_provenance(config) == ("billing", surface, traffic_class)


def test_trace_persists_content_free_model_lineage() -> None:
    trace = TurnTrace(
        session_id="session-1",
        turn_id=3,
        capability="voice",
        profile="card",
        surface="card",
    )
    trace.record_model_call(
        {
            "endpoint": "system.ai.gpt-5-5",
            "transport": "unity_ai_gateway",
            "model_role": "conversion",
            "request_id": "request-1",
            "duration_ms": 14.2,
            "status": "ok",
            "prompt": "must not persist",
            "response": "must not persist",
        }
    )
    payload = trace.to_dict()
    assert payload["profile"] == "card"
    assert payload["surface"] == "card"
    assert payload["traffic_class"] == "conversation"
    assert payload["model_calls"] == [
        {
            "endpoint": "system.ai.gpt-5-5",
            "transport": "unity_ai_gateway",
            "model_role": "conversion",
            "request_id": "request-1",
            "duration_ms": 14.2,
            "status": "ok",
        }
    ]
