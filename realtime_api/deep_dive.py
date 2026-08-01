"""Generic async "deep-dive" lane: Genie Agent Mode → trace → spoken summary.

This is the framework home for the DEEP lane of the two-lane Genie design (the
FAST lane is an in-turn tool call; this is the async "why" investigation that can
take up to a minute). It is industry-agnostic — a profile just points its
``start_deep_dive`` signal at the same ``/deepdive`` SSE route, and this runner:

  1. streams Genie Agent Mode SSE progress (reasoning / SQL / report),
  2. records a ``TurnTrace`` linked to the originating voice call, and
  3. renders the finished report for the caller's language: a SHORT spoken "why"
     for the voice, then the on-screen report translated out of the agent's
     English (the agent is always asked in English because asking it to write in
     the caller's language kills the run for some languages — see
     ``GenieAgentModeClient.ask``). Both come from the SHARED voice serving, and
     the spoken line is released FIRST so the voice never waits on the
     translation.

No card- (or telco-) specific logic lives here; the only inputs are the question,
the caller's language, and the trace-linking ids.
"""
from __future__ import annotations

import json
import queue
from functools import lru_cache
from typing import Any, Callable

# Capability label for the deep-dive trace (distinguishes it from voice turns).
DEEPDIVE_CAPABILITY = "genie-agent-mode-deepdive"


@lru_cache(maxsize=1)
def _summary_knobs() -> tuple[float, int, int]:
    """Config-sourced summarizer sampling (temperature, summary/localize max_tokens).

    The spoken "why" must be SHORT and factual → low temperature + capped tokens;
    the report translation shares the temperature but needs room for a whole
    report. Operator-tunable via realtime_voice.deep_dive.*; cached (config is
    static per process). Falls back to sane defaults on error.
    """
    try:
        from .config import RealtimeSettings

        s = RealtimeSettings.resolve()
        return (
            float(s.deep_dive_summary_temperature),
            int(s.deep_dive_summary_max_tokens),
            int(s.deep_dive_localize_max_tokens),
        )
    except Exception:  # noqa: BLE001
        return 0.3, 220, 1800


@lru_cache(maxsize=1)
def _conversion_endpoint() -> str | None:
    """Config-sourced endpoint for the text-to-text conversions (summary + report
    translation). ``None`` → the shared serving falls back to its ``llm_endpoint``.
    Cached; config is static per process."""
    try:
        from .config import RealtimeSettings

        return RealtimeSettings.resolve().conversion_endpoint or None
    except Exception:  # noqa: BLE001
        return None


def is_english(language: str | None) -> bool:
    """True when a BCP-47 tag (or None) means "English" — i.e. nothing to translate."""
    return not language or str(language).split("-", 1)[0].strip().lower() == "en"


def language_name(language: str | None) -> str:
    """English NAME of a BCP-47 tag ('hi-IN' → 'Hindi') for use inside a prompt.

    A prompt that says "write in hi-IN" is a worse instruction than one that says
    "write in Hindi". Matches the shared catalog on the normalized tag rather than
    splitting the subtag, so nb-NO resolves to Norwegian instead of missing. Falls
    back to the raw tag, which is still better than dropping the instruction.
    """
    tag = str(language or "")
    try:
        from genie_voice.i18n import LANGUAGE_CATALOG, language_spec, normalize_language

        norm = normalize_language(tag)  # raises on unsupported → fall back below
        return next(
            (eng for (cat_tag, eng) in LANGUAGE_CATALOG.values() if cat_tag == norm),
            None,
        ) or language_spec(norm).english_name
    except Exception:  # noqa: BLE001
        return tag


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
    temperature, max_tokens, _ = _summary_knobs()
    return shared_serving().summarize(
        system=system,
        user=user,
        temperature=temperature,
        max_tokens=max_tokens,
        endpoint=_conversion_endpoint(),
    )


def localize_report(report_text: str, language: str | None) -> str:
    """Translate the agent's English report into the caller's language for the UI.

    Agent Mode answers in English (see ``GenieAgentModeClient.ask`` for why we no
    longer ask it to do otherwise), so the on-screen "why" is translated here — one
    call on the SAME shared voice serving as the spoken summary, no new client and
    no new endpoint. Markdown structure, figures and citation markers are preserved
    so the report keeps its provenance. Returns "" for English or on any failure,
    which leaves the caller showing the English original rather than nothing.
    """
    text = (report_text or "").strip()
    if not text or is_english(language):
        return ""
    from .serving_factory import shared_serving

    name = language_name(language)
    system = (
        f"You are a professional financial translator. Translate the user's report into {name}. "
        "Rules: keep the markdown structure exactly as-is (headings, bold, bullet lists, "
        "tables); keep every number, currency amount, date, merchant name and citation "
        "marker such as [[1]] unchanged; translate the surrounding prose and any table "
        "headers. Do not summarize, do not add commentary, do not answer the report. "
        f"Output ONLY the translated report in {name}."
    )
    temperature, _, max_tokens = _summary_knobs()
    try:
        return shared_serving().summarize(
            system=system,
            user=text[:8000],
            temperature=temperature,
            max_tokens=max_tokens,
            endpoint=_conversion_endpoint(),
        ).strip()
    except Exception:  # noqa: BLE001 — an English report beats a blank panel
        return ""


def stream_report_renderings(
    question: str,
    ev: dict[str, Any],
    language: str | None,
    emit: Callable[[dict[str, Any]], None],
) -> None:
    """Emit the finished report in two beats so the VOICE never waits on translation.

    The customer is already 30-60s into the investigation, so the moment the short
    spoken "why" exists we hand it over — with the agent's English report, which is
    what we have — and the caller starts hearing the answer. The translation runs
    concurrently and arrives as a second ``report_localized`` event that replaces the
    on-screen text in place. ``localization_pending`` tells the client a swap is
    coming so it can say so instead of appearing to change its mind.

    Both renderings are best-effort and independent: a failed translation leaves the
    English report standing, a failed summary leaves the report without a spoken
    line, and neither can block the other.
    """
    from concurrent.futures import Future, ThreadPoolExecutor

    def _settled(f: "Future[str]") -> str:
        try:
            return f.result() or ""
        except Exception:  # noqa: BLE001
            return ""

    english = str(ev.get("report") or "")
    pending = bool(english) and not is_english(language)
    # One worker: the translation runs in it while the summary runs on this thread.
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="deepdive-i18n") as pool:
        localized: "Future[str] | None" = (
            pool.submit(localize_report, english, language) if pending else None
        )
        try:
            spoken = summarize_deepdive(question, english, language)
        except Exception:  # noqa: BLE001 — the client falls back to reading nothing aloud
            spoken = ""

        first = {**ev, "report_language": "en", "localization_pending": pending}
        if spoken:
            first["spoken_summary"] = spoken
        emit(first)

        text = _settled(localized) if localized is not None else ""
        if text:
            emit({"kind": "report_localized", "report": text, "report_language": language})


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
        elif kind == "report_localized":
            # The customer read the translation, so that is what the trace shows;
            # the span records that a translation happened and into what.
            trace.output_text = ev.get("report")
            with trace.span("report.localize", "LLM") as s:
                s.set_attribute("language", ev.get("report_language"))
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
    The agent is asked in English; ``language`` decides what the caller then reads
    and hears, both rendered here. A terminal ``{"kind": "done"}`` always follows
    so the SSE generator can stop.
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

    def _put(ev: dict[str, Any]) -> None:
        if trace is not None:
            record_deepdive_event(trace, ev)
        sink.put(ev)

    def _on_event(ev: dict[str, Any]) -> None:
        # The terminal report becomes two events: one the caller can immediately
        # HEAR, then the translated text they READ (see stream_report_renderings).
        if isinstance(ev, dict) and ev.get("kind") == "report":
            try:
                stream_report_renderings(question, ev, language, _put)
                return
            except Exception:  # noqa: BLE001 - rendering is best-effort; the report still ships
                pass
        _put(ev)

    try:
        from genie_voice.genie.agent_mode import GenieAgentModeClient

        GenieAgentModeClient().ask(
            question,
            on_event=_on_event,
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
