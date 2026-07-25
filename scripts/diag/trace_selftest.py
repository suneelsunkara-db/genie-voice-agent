"""Self-test the trace capture → durable store → read-back path (no network).

Forces the offline/file-backed store (GENIE_LOCAL_VOLUME_DIR flips Lakebase off),
builds a representative turn trace, submits it through the background sink, then
reads it back exactly like the /traces API does.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

_d = tempfile.mkdtemp(prefix="trace_selftest_")
os.environ["GENIE_LOCAL_VOLUME_DIR"] = os.path.join(_d, "vol")
os.makedirs(os.environ["GENIE_LOCAL_VOLUME_DIR"], exist_ok=True)
os.environ["GENIE_TRACE_DIR"] = os.path.join(_d, "traces")

from realtime_api.tracing import TurnTrace, get_sink, submit_trace  # noqa: E402


def build_trace() -> TurnTrace:
    t = TurnTrace(
        session_id="sess-selftest-1",
        turn_id=2,
        capability="speech-llm-toolassist-speech",
        call_id="CALL-1",
        customer_id="CUST-DEMO",
    )
    t.language = "th-TH"
    t.detected_language = "th-TH"
    t.input_transcript = "ดำเนินการต่อ"
    with t.span("stt", "STT", input={"audio_bytes": 32000, "sample_rate_hz": 16000}) as s:
        s.set_output({"transcript": "ดำเนินการต่อ", "detected_language": "th-TH"}).set_attribute("stt_ms", 420)
    t.span("history", "GUARD", input={"messages": [
        {"role": "user", "content": "ขอยกเว้นค่าปรับ"},
        {"role": "assistant", "content": "ยกเว้นค่าปรับ INV-1002 ได้ค่ะ ให้ดำเนินการเลยไหมคะ"},
    ]}).set_output({"message_count": 2}).end()
    llm = t.span("llm.iteration.0", "LLM", input={
        "messages": [
            {"role": "system", "content": "You are a live contact-center voice agent..."},
            {"role": "user", "content": "ขอยกเว้นค่าปรับ"},
            {"role": "assistant", "content": "ให้ดำเนินการเลยไหมคะ"},
            {"role": "user", "content": "ดำเนินการต่อ"},
        ],
        "tools_available": ["lookup_account", "apply_billing_action", "ask_genie", "get_current_time"],
        "temperature": 0.4,
    })
    llm.set_output({"content": "", "tool_calls": [
        {"id": "call_1", "function": {"name": "lookup_account", "arguments": "{}"}},
    ]}).set_attribute("tool_call_count", 1).end()
    tool = t.span("tool.lookup_account", "TOOL", input={})
    tool.set_output({"found": True, "summary": {"overdue_invoice_count": 1}}).set_attribute(
        "tool_call_id", "call_1"
    ).end()
    llm2 = t.span("llm.iteration.1", "LLM", input={"messages": ["…"], "tools_available": []})
    llm2.set_output({"content": "ให้ดำเนินการเลยไหมคะ", "tool_calls": []}).set_attribute(
        "tool_call_count", 0
    ).end()
    with t.span("tts", "TTS", input={"text": "ให้ดำเนินการเลยไหมคะ", "language": "th-TH"}) as s:
        s.set_output({"chunks": 12})
    t.output_text = "ให้ดำเนินการเลยไหมคะ"
    return t


def main() -> None:
    t = build_trace()
    submit_trace(t)
    get_sink()._q.join()  # wait for the background writer to flush

    from api.app.deps import serving

    svc = serving()
    print(f"lakebase.enabled={svc.enabled} (expected False -> file store)")
    listed = svc.list_voice_traces(limit=10)
    print(f"list_voice_traces -> {len(listed)} row(s)")
    for row in listed:
        print(
            f"  turn#{row['turn_id']} status={row['status']} lang={row['language']} "
            f"tools={row['tool_names']} apply={row['apply_billing_action_called']} "
            f"lookups={row['lookup_account_count']} iters={row['llm_iterations']} ms={row['total_ms']}"
        )
    full = svc.get_voice_trace(t.trace_id)
    assert full is not None, "get_voice_trace returned None"
    assert full["apply_billing_action_called"] is False
    assert full["lookup_account_count"] == 1
    assert len(full["spans"]) == 6
    llm_span = next(s for s in full["spans"] if s["name"] == "llm.iteration.0")
    assert llm_span["input"]["messages"][-1]["content"] == "ดำเนินการต่อ"
    print(f"\nget_voice_trace -> {len(full['spans'])} spans, "
          f"first LLM input captured full messages ✓")
    print(f"trace file: {os.path.join(os.environ['GENIE_TRACE_DIR'], 'voice_traces.jsonl')}")
    print("\nSELFTEST PASSED")


if __name__ == "__main__":
    main()
