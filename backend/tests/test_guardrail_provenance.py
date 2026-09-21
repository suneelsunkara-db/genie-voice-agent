import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from api.app.routers import traces


class _Store:
    def list_voice_traces(self, **_kwargs):
        return [
            {
                "trace_id": "conversation",
                "traffic_class": "conversation",
                "language": "en-US",
                "guard_roster": [
                    {"guard_id": "speech_output_boundary", "outcome": "passed", "surface": "guardrail"}
                ],
            },
            {
                "trace_id": "probe",
                "traffic_class": "readiness_probe",
                "language": "en-US",
                "guard_roster": [
                    {"guard_id": "service_policy", "outcome": "fired", "surface": "guardrail"}
                ],
            },
            {
                "trace_id": "legacy",
                "language": "en-US",
                "guard_roster": [
                    {"guard_id": "legacy", "outcome": "fired", "surface": "guardrail"}
                ],
            },
        ]

    def list_guard_events(self, **_kwargs):
        return [{"event_id": "standalone", "guard_id": "service_policy", "outcome": "fired"}]


def test_guardrail_rollup_separates_conversation_probe_and_standalone(monkeypatch) -> None:
    monkeypatch.setattr(traces, "serving", lambda: _Store())
    payload = traces.guardrail_rollup()
    assert payload["turns"] == 1
    assert payload["checks"] == 1
    assert payload["totals"] == {"passed": 1}
    assert payload["legacy_unclassified_turns"] == 1
    assert payload["standalone_events"] == 1


def test_page_inventory_covers_every_routed_and_external_surface() -> None:
    surfaces = {row["surface"] for row in traces._page_coverage()}
    assert {
        "home",
        "telco",
        "card",
        "knowledge",
        "setup",
        "voice-benchmarks",
        "asr-benchmark",
        "traces",
        "guardrails",
        "realtime-test",
        "mcp",
    } <= surfaces


def test_speech_endpoint_metrics_are_trace_derived_and_gateway_free() -> None:
    realtime = SimpleNamespace(stt_endpoint="qwen-asr", tts_endpoint="voxcpm2")
    conversation_traces = {
        "trace-1": {
            "trace_id": "trace-1",
            "surface": "card",
            "profile": "card",
            "server_ttfb_ms": 180.0,
            "server_gen_ms": 900.0,
            "model_calls": [
                {
                    "endpoint": "qwen-asr",
                    "model_role": "stt",
                    "transport": "model_serving",
                    "duration_ms": 320.0,
                    "status": "ok",
                },
                {
                    "endpoint": "voxcpm2",
                    "model_role": "tts",
                    "transport": "model_serving",
                    "duration_ms": 1100.0,
                    "status": "ok",
                },
            ],
        },
        "trace-2": {
            "trace_id": "trace-2",
            "surface": "knowledge",
            "profile": "knowledge",
            "server_ttfb_ms": 220.0,
            "server_gen_ms": 1300.0,
            "model_calls": [
                {
                    "endpoint": "qwen-asr",
                    "model_role": "stt",
                    "transport": "model_serving",
                    "duration_ms": 480.0,
                    "status": "ok",
                },
                {
                    "endpoint": "voxcpm2",
                    "model_role": "tts",
                    "transport": "model_serving",
                    "duration_ms": 1500.0,
                    "status": "error",
                },
            ],
        },
    }

    insights = traces._speech_endpoint_insights(conversation_traces, realtime)
    by_role = {item["model_role"]: item for item in insights}
    assert by_role["stt"] == {
        "endpoint": "qwen-asr",
        "model_role": "stt",
        "requests": 2,
        "errors": 0,
        "model_name": "Qwen/Qwen3-ASR-1.7B",
        "trace_count": 2,
        "surfaces": ["card", "knowledge"],
        "profiles": ["card", "knowledge"],
        "avg_latency_ms": 400.0,
        "p95_latency_ms": 480.0,
        "avg_ttfb_ms": None,
        "p95_ttfb_ms": None,
        "avg_generation_ms": None,
        "provenance_status": "verified",
        "telemetry_source": "voice_trace_model_calls",
        "gateway_telemetry": False,
    }
    assert by_role["tts"]["model_name"] == "openbmb/VoxCPM2"
    assert by_role["tts"]["errors"] == 1
    assert by_role["tts"]["avg_ttfb_ms"] == 200.0
    assert by_role["tts"]["p95_ttfb_ms"] == 220.0
    assert by_role["tts"]["avg_generation_ms"] == 1100.0
