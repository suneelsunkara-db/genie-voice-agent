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
from typing import Any, Callable, Iterator

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
    _, max_tokens, _ = _summary_knobs()
    # Temperature is intentionally omitted: the conversion endpoint (e.g. gpt-5-5)
    # accepts only its default and 400s on any explicit value; the model default is
    # fine for a short spoken summary.
    return shared_serving().summarize(
        system=system,
        user=user,
        max_tokens=max_tokens,
        endpoint=_conversion_endpoint(),
    )


def _localize_system(language: str | None) -> str:
    """The translator system prompt, shared by the one-shot and streaming paths so
    both produce an identically-constrained translation (only the delivery differs).
    """
    name = language_name(language)
    return (
        f"You are a professional financial translator. Translate the user's report into {name}. "
        "Rules: keep the markdown structure exactly as-is (headings, bold, bullet lists, "
        "tables); keep every number, currency amount, date, merchant name and citation "
        "marker such as [[1]] unchanged; translate the surrounding prose and any table "
        "headers. Do not summarize, do not add commentary, do not answer the report. "
        f"Output ONLY the translated report in {name}."
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

    _, _, max_tokens = _summary_knobs()
    try:
        return shared_serving().summarize(
            system=_localize_system(language),
            user=text[:8000],
            max_tokens=max_tokens,
            endpoint=_conversion_endpoint(),
        ).strip()
    except Exception:  # noqa: BLE001 — an English report beats a blank panel
        return ""


def localize_report_stream(report_text: str, language: str | None) -> "Iterator[str]":
    """Stream the report translation as text deltas so the on-screen "why" paints
    progressively in the caller's language — never a flash of English, never a wait
    for the whole thing.

    Prefers the serving's streaming completion; if the conversion endpoint does not
    support streaming (or the stream errors before any text), it falls back to the
    one-shot :func:`localize_report` and yields the whole translation as a single
    chunk. Yields nothing for English or when there is nothing to translate — the
    caller then keeps the English report standing.
    """
    text = (report_text or "").strip()
    if not text or is_english(language):
        return
    _, _, max_tokens = _summary_knobs()
    system = _localize_system(language)

    stream_fn = None
    try:
        from .serving_factory import shared_serving

        stream_fn = getattr(shared_serving(), "summarize_stream", None)
    except Exception:  # noqa: BLE001 — fall through to the one-shot below
        stream_fn = None

    if callable(stream_fn):
        produced = False
        try:
            # Temperature omitted on purpose (see summarize_deepdive): the conversion
            # endpoint only accepts its default.
            for piece in stream_fn(
                system=system,
                user=text[:8000],
                max_tokens=max_tokens,
                endpoint=_conversion_endpoint(),
            ):
                if piece:
                    produced = True
                    yield piece
        except Exception:  # noqa: BLE001 — a partial stream still beats a blank
            if produced:
                return
        if produced:
            return

    # No streaming (or it yielded nothing): one-shot translate and emit as one chunk.
    full = localize_report(text, language)
    if full:
        yield full


def stream_report_renderings(
    question: str,
    ev: dict[str, Any],
    language: str | None,
    emit: Callable[[dict[str, Any]], None],
) -> None:
    """Render the finished report for the caller: a spoken "why" plus the on-screen
    report, in the caller's language, without a flash of English and without making
    the customer wait.

    English (or empty) report: one ``report`` event carries the text as-is and the
    spoken line — nothing to translate, no swap promised.

    Non-English: the customer is already 30-60s in, so we release the short spoken
    "why" immediately (they HEAR the answer now, in their language), and the report
    panel opens EMPTY with ``localization_pending`` — never the English text. The
    translation is then STREAMED into the panel as ``report_localized_delta`` events
    so the caller reads their own language from the first token, and a terminal
    ``report_localized`` carries the full text and clears the pending flag. If the
    translation produces nothing (endpoint down / not granted), that terminal event
    falls back to the English report so the panel is readable rather than blank or
    stuck on "translating…".
    """
    english = str(ev.get("report") or "")
    pending = bool(english) and not is_english(language)

    # The spoken line is what keeps the conversation moving; compute it first so it
    # ships on the very first event. Best-effort — silence beats a wrong language.
    try:
        spoken = summarize_deepdive(question, english, language)
    except Exception:  # noqa: BLE001 — the client falls back to reading nothing aloud
        spoken = ""

    if not pending:
        first = {**ev, "report_language": "en", "localization_pending": False}
        if spoken:
            first["spoken_summary"] = spoken
        emit(first)
        return

    # Non-English: announce the report SHELL with an empty body (tables/SQL/reasoning
    # still ride along) so the client shows a localizing panel — not English.
    shell = {**ev, "report": "", "report_language": language, "localization_pending": True}
    if spoken:
        shell["spoken_summary"] = spoken
    emit(shell)

    chunks: list[str] = []
    for delta in localize_report_stream(english, language):
        if delta:
            chunks.append(delta)
            emit({"kind": "report_localized_delta", "delta": delta, "report_language": language})

    full = "".join(chunks).strip()
    # Terminal event: the full translation, or the English original as a fallback so
    # the pending flag always resolves and the panel is never left blank.
    emit(
        {
            "kind": "report_localized",
            "report": full or english,
            "report_language": language if full else "en",
        }
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
