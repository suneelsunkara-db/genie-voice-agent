"""Generic async "deep-dive" lane: Genie Agent Mode → trace → spoken summary.

This is the framework home for the DEEP lane of the two-lane Genie design (the
FAST lane is an in-turn tool call; this is the async "why" investigation that can
take up to a minute). It is industry-agnostic — a profile just points its
``start_deep_dive`` signal at the same ``/deepdive`` SSE route, and this runner:

  1. streams Genie Agent Mode SSE progress (reasoning / SQL / report),
  2. records a ``TurnTrace`` linked to the originating voice call, and
  3. adds a SHORT spoken "why" summary (one shared-serving LLM call) in the
     caller's language, so the voice pinpoints the cause instead of reading the
     whole report.

No card- (or telco-) specific logic lives here; the only inputs are the question,
the caller's language, and the trace-linking ids.
"""
from __future__ import annotations

import json
import queue
from functools import lru_cache
from typing import Any

# Capability label for the deep-dive trace (distinguishes it from voice turns).
DEEPDIVE_CAPABILITY = "genie-agent-mode-deepdive"


@lru_cache(maxsize=1)
def _summary_knobs() -> tuple[float, int]:
    """Config-sourced summarizer sampling (temperature, max_tokens).

    The spoken "why" must be SHORT and factual → low temperature + capped tokens.
    Operator-tunable via realtime_voice.deep_dive.summary_{temperature,max_tokens};
    cached (config is static per process). Falls back to sane defaults on error.
    """
    try:
        from .config import RealtimeSettings

        s = RealtimeSettings.resolve()
        return float(s.deep_dive_summary_temperature), int(s.deep_dive_summary_max_tokens)
    except Exception:  # noqa: BLE001
        return 0.3, 220


def summarize_deepdive(question: str, report_text: str, language: str | None) -> str:
    """Turn the Agent-Mode report into a SHORT spoken 'why' via one LLM call.

    The report is long, structured markdown; reading it verbatim rushes and drags,
    and the headline alone doesn't name the cause. This yields 2-3 conversational
    sentences that pinpoint the single main driver + the key number + one next
    step, in the caller's language — spoken, not shown. Reuses the SHARED voice
    serving (public ``summarize``), never a second client. Returns "" on any
    failure so the client can fall back to its own heuristic summary.
    """
    text = (report_text or "").strip()
    if not text:
        return ""
    from .serving_factory import shared_serving

    lang = language or "en-US"
    system = (
        "You are a warm, concise voice assistant. You are given a detailed written "
        "analysis and must SPEAK a short summary to the customer: 2-3 sentences, in "
        f"the language identified by the BCP-47 code '{lang}'. Requirements: name the "
        "SINGLE main cause and the most important number, then give ONE concrete next "
        "step. Natural and conversational for text-to-speech. NO markdown, NO bullet "
        "points, NO headings, NO citation markers like [1], NO dollar-figure tables. "
        "Output ONLY the spoken sentences."
    )
    user = f"Customer asked: {question}\n\nAnalysis to summarize:\n{text[:6000]}"
    temperature, max_tokens = _summary_knobs()
    return shared_serving().summarize(
        system=system,
        user=user,
        temperature=temperature,
        max_tokens=max_tokens,
    )


# --------------------------------------------------------------------------- #
# Tracing helpers (pure) — fold streamed Agent-Mode events into a TurnTrace.
# --------------------------------------------------------------------------- #
def _norm_status(status: Any) -> str:
    """Map Agent-Mode status onto the trace status vocabulary."""
    s = str(status or "").lower()
    if s in ("completed", "ok", "success") or not s:
        return "ok"
    return "error" if s in ("failed", "error") else s


def new_deepdive_trace(
    question: str,
    *,
    call_id: str | None,
    session_id: str | None,
    customer_id: str | None,
) -> Any:
    """Build a TurnTrace for one Agent-Mode investigation (turn_id=0)."""
    from .tracing import TurnTrace

    trace = TurnTrace(
        session_id=session_id or "",
        turn_id=0,
        capability=DEEPDIVE_CAPABILITY,
        call_id=call_id,
        customer_id=customer_id,
    )
    trace.input_transcript = question
    return trace


def record_deepdive_event(trace: Any, ev: dict[str, Any]) -> None:
    """Fold one streamed Agent-Mode event into the trace as a finalized span.

    reasoning → LLM span (the business-language step), sql → TOOL span (the query
    as provenance), report → sets the trace output + a summary span, error → marks
    the trace failed. Best-effort: never raises.
    """
    kind = ev.get("kind")
    try:
        if kind == "reasoning":
            with trace.span("agent.reasoning", "LLM") as s:
                s.set_output(ev.get("text"))
        elif kind == "sql":
            with trace.span("genie.query", "TOOL", input=ev.get("sql")):
                pass
        elif kind == "report":
            trace.output_text = ev.get("report")
            trace.status = _norm_status(ev.get("status"))
            with trace.span("agent.report", "LLM") as s:
                s.set_attribute("tables", len(ev.get("tables") or []))
                s.set_attribute("sql_calls", len(ev.get("sql") or []))
                s.set_attribute("reasoning_steps", len(ev.get("reasoning") or []))
        elif kind == "error":
            trace.status = "error"
            trace.error = json.dumps(ev.get("error"), default=str)[:2000]
    except Exception:  # noqa: BLE001 - tracing must never break the stream
        pass


def deep_dive_read_timeout_s() -> float:
    """Config-sourced Agent-Mode read timeout (single source of truth)."""
    try:
        from .config import RealtimeSettings

        return float(RealtimeSettings.resolve().deep_dive_read_timeout_s)
    except Exception:  # noqa: BLE001 — fall back to the client default
        return 420.0


def run_deep_dive(
    question: str,
    sink: "queue.Queue[dict]",
    *,
    call_id: str | None = None,
    session_id: str | None = None,
    customer_id: str | None = None,
    use_case: str | None = None,
    language: str | None = None,
    genie_space_name: str | None = None,
    read_timeout_s: float | None = None,
) -> None:
    """Run the (blocking) Genie Agent Mode call, streaming SSE progress to ``sink``
    AND recording a TurnTrace for observability.

    ``genie_space_name`` selects WHICH Genie Agent (space) answers, so this lane
    is industry-agnostic — the card route passes the card space, a billing route
    would pass the billing space, and neither relies on an Agent-Mode default.
    The on-screen report is requested in the caller's ``language`` (Agent Mode
    frames its narrative accordingly) and a short spoken summary is added in the
    same language. A terminal ``{"kind": "done"}`` always follows so the SSE
    generator can stop.
    """
    timeout = read_timeout_s if read_timeout_s is not None else deep_dive_read_timeout_s()
    trace = None
    try:
        trace = new_deepdive_trace(
            question, call_id=call_id, session_id=session_id, customer_id=customer_id
        )
        if use_case:
            try:
                trace.span("deepdive", "GUARD", input={"use_case": use_case}).end()
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001 - tracing setup must not block the run
        trace = None

    def _on_event(ev: dict[str, Any]) -> None:
        # On the terminal report, add a spoken "why" summary (one LLM call) so the
        # voice pinpoints the cause instead of reading the whole report.
        if isinstance(ev, dict) and ev.get("kind") == "report":
            try:
                spoken = summarize_deepdive(question, ev.get("report") or "", language)
                if spoken:
                    ev = {**ev, "spoken_summary": spoken}
            except Exception:  # noqa: BLE001 - summary is best-effort; client has a fallback
                pass
        if trace is not None:
            record_deepdive_event(trace, ev)
        sink.put(ev)

    try:
        from genie_voice.genie.agent_mode import GenieAgentModeClient

        GenieAgentModeClient().ask(
            question,
            on_event=_on_event,
            language=language,
            space_name=genie_space_name,
            read_timeout_s=timeout,
        )
    except Exception as exc:  # noqa: BLE001
        if trace is not None:
            trace.status = "error"
            trace.error = str(exc)
        sink.put({"kind": "error", "error": {"message": str(exc)}})
    finally:
        if trace is not None:
            try:
                from .tracing import submit_trace

                submit_trace(trace)
            except Exception:  # noqa: BLE001
                pass
        sink.put({"kind": "done"})
