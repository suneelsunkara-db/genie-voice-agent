from __future__ import annotations

import pytest

from realtime_api.guardrails.boundaries import (
    SpeechBoundaryViolation,
    admit_speech_output,
    admit_transcript,
    enforce_speech_text,
    extract_inline_tool_calls,
)
from realtime_api.guardrails.ledger import GuardLedger


def test_inline_tool_call_is_recovered_and_removed() -> None:
    calls, text = extract_inline_tool_calls(
        'I will check. <tool_call>{"name":"lookup","arguments":{"id":"1"}}</tool_call>'
    )
    assert calls == [{"name": "lookup", "arguments": {"id": "1"}}]
    assert text == "I will check."


def test_speech_boundary_blocks_tool_only_output() -> None:
    with pytest.raises(SpeechBoundaryViolation):
        enforce_speech_text('<tool_call>{"name":"lookup","arguments":{}}</tool_call>')


def test_speech_boundary_preserves_normal_multilingual_text() -> None:
    assert enforce_speech_text("สวัสดีค่ะ") == ("สวัสดีค่ะ", False)


def test_transcript_admission_records_model_owned_decisions() -> None:
    ledger = GuardLedger()
    admitted = admit_transcript(
        "hello\x00",
        detected_language="en-US",
        pinned_language="auto",
        resource="qwen-asr",
        ledger=ledger,
    )
    assert admitted == "hello"
    assert [(entry.guard_id, entry.outcome) for entry in ledger.entries] == [
        ("language_id", "delegated"),
        ("no_speech_suppression", "passed"),
    ]
    assert all(entry.resource == "qwen-asr" for entry in ledger.entries)


def test_speech_admission_is_resource_bound_and_records_decision() -> None:
    ledger = GuardLedger()
    admission = admit_speech_output("hello", resource="voxcpm2", ledger=ledger)
    assert admission.text == "hello"
    assert admission.resource == "voxcpm2"
    assert ledger.entries[0].guard_id == "speech_output_boundary"
