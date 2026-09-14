"""Mandatory modality-boundary enforcement shared by every voice path."""
from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from .ledger import GuardLedger, report

_TOOL_CALL_TAG_RE = re.compile(r"</?\s*tool_call\s*>", re.IGNORECASE)


class SpeechBoundaryViolation(ValueError):
    """Text was not safe to hand to a speech synthesizer."""


class TranscriptBoundaryViolation(ValueError):
    """An STT result was malformed at the transcript trust boundary."""


@dataclass(frozen=True)
class SpeechAdmission:
    """Capability token proving text passed the shared pre-TTS boundary."""

    text: str
    resource: str
    changed: bool


def admit_transcript(
    text: str,
    *,
    detected_language: str | None,
    pinned_language: str | None,
    resource: str,
    ledger: GuardLedger | None = None,
) -> str:
    """Admit an STT result before trace, browser, history, model, or tool use."""
    if not isinstance(text, str):
        raise TranscriptBoundaryViolation("STT transcript must be text")
    # NUL is never valid user speech and can corrupt downstream persistence.
    admitted = text.replace("\x00", "")
    pinned = bool(pinned_language) and pinned_language != "auto"
    report(
        ledger,
        "language_id",
        "not_evaluated" if pinned else "delegated",
        stage="stt",
        owner="qwen",
        phase="post_stt",
        resource=resource,
        reason=(
            f"session pinned language={pinned_language}"
            if pinned
            else f"detected_language={detected_language or 'unknown'}"
        ),
    )
    report(
        ledger,
        "no_speech_suppression",
        "fired" if not admitted.strip() else "passed",
        stage="stt",
        owner="qwen",
        phase="post_stt",
        resource=resource,
        reason="empty transcript from STT" if not admitted.strip() else None,
    )
    return admitted


def _iter_json_objects(text: str) -> Iterator[tuple[str, Any]]:
    i, n = 0, len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        depth, j, in_str, escaped = 0, i, False, False
        while j < n:
            char = text[j]
            if in_str:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_str = False
            elif char == '"':
                in_str = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    raw = text[i : j + 1]
                    try:
                        yield raw, json.loads(raw)
                    except json.JSONDecodeError:
                        pass
                    break
            j += 1
        i = j + 1


def extract_inline_tool_calls(content: str) -> tuple[list[dict[str, Any]], str]:
    """Extract marked tool calls and return text with their representation removed."""
    if not content or "tool_call" not in content.lower():
        return [], content
    calls: list[dict[str, Any]] = []
    cleaned = content
    for raw, obj in _iter_json_objects(content):
        if isinstance(obj, dict) and isinstance(obj.get("name"), str):
            arguments = obj.get("arguments")
            calls.append(
                {
                    "name": obj["name"],
                    "arguments": arguments if isinstance(arguments, dict) else {},
                }
            )
            cleaned = cleaned.replace(raw, "")
    return calls, _TOOL_CALL_TAG_RE.sub("", cleaned).strip()


def sanitize_speech_text(text: str) -> tuple[str, bool]:
    """Return text safe for TTS and whether a tool representation was removed."""
    _, cleaned = extract_inline_tool_calls(text)
    if cleaned:
        return cleaned, cleaned != text
    if "tool_call" not in (text or "").lower():
        return text, False
    return "", True


def enforce_speech_text(text: str) -> tuple[str, bool]:
    """Fail closed when sanitation leaves no user-facing speech."""
    cleaned, changed = sanitize_speech_text(text)
    if not cleaned.strip():
        raise SpeechBoundaryViolation("speech output was empty or contained only tool markup")
    return cleaned, changed


def admit_speech_output(
    text: str,
    *,
    resource: str,
    ledger: GuardLedger | None = None,
) -> SpeechAdmission:
    """Admit exactly the text that may be displayed, committed, and synthesized."""
    try:
        cleaned, changed = enforce_speech_text(text)
    except SpeechBoundaryViolation:
        report(
            ledger,
            "speech_output_boundary",
            "fired",
            stage="pre_tts",
            phase="before_synthesis",
            resource=resource,
            reason="empty or tool-only speech blocked",
        )
        raise
    report(
        ledger,
        "speech_output_boundary",
        "fired" if changed else "passed",
        stage="pre_tts",
        phase="before_synthesis",
        resource=resource,
        reason="tool representation removed at shared TTS boundary" if changed else None,
    )
    return SpeechAdmission(text=cleaned, resource=resource, changed=changed)
