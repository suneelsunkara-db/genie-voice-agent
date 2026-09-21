import sys
from pathlib import Path

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
