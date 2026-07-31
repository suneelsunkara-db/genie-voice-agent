"""End-to-end turn tracing for the realtime voice pipeline.

Captures a Langfuse/MLflow-style span tree per voice turn (STT → LLM tool loop →
TTS, including every tool call's inputs/outputs and per-iteration LLM messages)
WITHOUT adding latency to the hot path:

  - Span capture is just in-memory dict appends, guarded by a lock so the LLM
    worker thread and a concurrently-streaming filler-TTS on the event loop can
    both record safely.
  - The completed trace is handed to a background daemon writer via a
    non-blocking queue (drops on overflow). The turn NEVER blocks on the DB.

Persistence is pluggable: the default sink writes to Lakebase (via the shared
``api.app.deps.serving()`` singleton) and can optionally mirror spans to MLflow
Tracing (Databricks Agent Framework) when enabled — both off the hot path.
"""
from __future__ import annotations

import logging
import os
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Callable

logger = logging.getLogger("realtime_voice")

# Cap very large tool results / message payloads so a single trace row stays
# reasonable. Full text is what makes the view useful, so this is generous.
_MAX_STR = 20_000


def _as_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _truncate(value: Any) -> Any:
    if isinstance(value, str) and len(value) > _MAX_STR:
        return value[:_MAX_STR] + f"…[+{len(value) - _MAX_STR} chars]"
    if isinstance(value, list):
        return [_truncate(v) for v in value]
    if isinstance(value, dict):
        return {k: _truncate(v) for k, v in value.items()}
    return value


@dataclass
class Span:
    name: str
    kind: str  # STT | LLM | TOOL | TTS | GUARD
    start_ms: float
    end_ms: float | None = None
    duration_ms: float | None = None
    status: str = "ok"
    input: Any = None
    output: Any = None
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "start_ms": round(self.start_ms, 2),
            "end_ms": round(self.end_ms, 2) if self.end_ms is not None else None,
            "duration_ms": round(self.duration_ms, 2) if self.duration_ms is not None else None,
            "status": self.status,
            "input": _truncate(self.input),
            "output": _truncate(self.output),
            "attributes": _truncate(self.attributes),
        }


class _SpanHandle:
    """Mutable handle returned by ``TurnTrace.span`` to set output/attrs/status."""

    def __init__(self, trace: "TurnTrace", span: Span) -> None:
        self._trace = trace
        self._span = span

    def set_output(self, output: Any) -> "_SpanHandle":
        self._span.output = output
        return self

    def set_attribute(self, key: str, value: Any) -> "_SpanHandle":
        self._span.attributes[key] = value
        return self

    def set_status(self, status: str) -> "_SpanHandle":
        self._span.status = status
        return self

    def end(self) -> None:
        self._trace._end_span(self._span)

    def __enter__(self) -> "_SpanHandle":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            self._span.status = "error"
            self._span.attributes.setdefault("error", repr(exc))
        self.end()


class TurnTrace:
    """Accumulates spans for a single voice turn. Thread-safe span additions."""

    def __init__(
        self,
        *,
        session_id: str,
        turn_id: int,
        capability: str,
        call_id: str | None = None,
        customer_id: str | None = None,
    ) -> None:
        self.trace_id = uuid.uuid4().hex
        self.session_id = session_id
        self.turn_id = turn_id
        self.capability = capability
        self.call_id = call_id
        self.customer_id = customer_id
        self.language: str | None = None
        self.detected_language: str | None = None
        self.input_transcript: str | None = None
        self.output_text: str | None = None
        self.status: str = "ok"
        self.error: str | None = None
        self.started_at = datetime.now(UTC).isoformat()
        self._t0 = time.perf_counter()
        # Two different latencies, because a filler clip makes them diverge and
        # reporting only the flattering one hides real delay:
        #   ttft_ms        - any audio at all, so a filler ends the dead air
        #   answer_ttft_ms - the actual reply, which is what the caller was waiting for
        # Equal when no filler played. The remaining fields describe the ANSWER's
        # synthesis (a filler is canned, so its timing says nothing about the turn).
        self.ttft_ms: float | None = None
        self.answer_ttft_ms: float | None = None
        self.tts_first_ms: float | None = None  # TTS-local, before our own lookahead
        self.server_ttfb_ms: float | None = None  # the endpoint's own time to chunk 1
        self.server_gen_ms: float | None = None  # full synthesis time on the endpoint
        self._spans: list[Span] = []
        self._lock = threading.Lock()

    def _now_ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000.0

    def note_audio(self, event: dict[str, Any], *, primary: bool = True) -> None:
        """Record first-audio timings from a ``response.audio`` event.

        Called by ``stream_tts`` for every audio event, so timings are captured for
        every path that speaks (answer, filler, router confirmation, language
        switch) instead of only the main LLM path. ``primary=False`` marks audio
        that merely covers latency (a filler): it ends the dead air but is not the
        reply, so it must not claim the answer's latency.
        """
        with self._lock:
            # One clock reading for both, so that with no filler in play the two
            # describe the same instant instead of drifting apart by a hair.
            now = self._now_ms()
            if self.ttft_ms is None:
                self.ttft_ms = now
            if not primary:
                return
            if self.answer_ttft_ms is None:
                self.answer_ttft_ms = now
            if self.tts_first_ms is None:
                self.tts_first_ms = _as_float(event.get("tts_first_ms"))
            # Server timings ride the LAST chunk of a synthesis, so they arrive
            # after the first audio rather than with it.
            if self.server_ttfb_ms is None:
                self.server_ttfb_ms = _as_float(event.get("server_ttfb_ms"))
            if self.server_gen_ms is None:
                self.server_gen_ms = _as_float(event.get("server_gen_ms"))

    def span(self, name: str, kind: str, *, input: Any = None) -> _SpanHandle:
        span = Span(name=name, kind=kind, start_ms=self._now_ms(), input=input)
        with self._lock:
            self._spans.append(span)
        return _SpanHandle(self, span)

    def _end_span(self, span: Span) -> None:
        span.end_ms = self._now_ms()
        span.duration_ms = span.end_ms - span.start_ms

    @property
    def tool_names(self) -> list[str]:
        return [s.name.split(".", 1)[-1] for s in self._spans if s.kind == "TOOL"]

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            spans = [s.to_dict() for s in self._spans]
        tool_names = [s.name.split(".", 1)[-1] for s in self._spans if s.kind == "TOOL"]
        llm_iterations = sum(1 for s in self._spans if s.kind == "LLM")
        return {
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "capability": self.capability,
            "call_id": self.call_id,
            "customer_id": self.customer_id,
            "language": self.language,
            "detected_language": self.detected_language,
            "input_transcript": _truncate(self.input_transcript),
            "output_text": _truncate(self.output_text),
            "status": self.status,
            "error": self.error,
            "tool_names": tool_names,
            # Objective, language-agnostic signals for the observability view.
            "tools_this_turn": tool_names,
            "apply_billing_action_called": "apply_billing_action" in tool_names,
            "lookup_account_count": tool_names.count("lookup_account"),
            "llm_iterations": llm_iterations,
            "started_at": self.started_at,
            # answer_ttft_ms is the latency to judge the turn on; ttft_ms only says
            # when the silence ended, and total_ms is the whole turn including
            # however long the agent then spoke for.
            "ttft_ms": round(self.ttft_ms, 2) if self.ttft_ms is not None else None,
            "answer_ttft_ms": (
                round(self.answer_ttft_ms, 2) if self.answer_ttft_ms is not None else None
            ),
            "tts_first_ms": round(self.tts_first_ms, 2) if self.tts_first_ms is not None else None,
            "server_ttfb_ms": self.server_ttfb_ms,
            "server_gen_ms": self.server_gen_ms,
            "total_ms": round(self._now_ms(), 2),
            "spans": spans,
        }


class TraceSink:
    """Background, non-blocking trace persistence.

    ``submit`` never blocks the caller: it enqueues and returns. A daemon thread
    drains the queue and runs ``persist`` (a slow, blocking DB write) off the hot
    path. If the queue is full (writer wedged), the trace is dropped with a log —
    observability must never degrade the live call.
    """

    def __init__(self, persist: Callable[[dict[str, Any]], None], *, maxsize: int = 512) -> None:
        self._persist = persist
        self._q: "queue.Queue[dict[str, Any] | None]" = queue.Queue(maxsize=maxsize)
        self._thread = threading.Thread(target=self._run, name="trace-writer", daemon=True)
        self._started = False
        self._start_lock = threading.Lock()

    def _ensure_started(self) -> None:
        if self._started:
            return
        with self._start_lock:
            if not self._started:
                self._thread.start()
                self._started = True

    def submit(self, trace: dict[str, Any]) -> None:
        self._ensure_started()
        try:
            self._q.put_nowait(trace)
        except queue.Full:
            logger.warning("trace sink full; dropping trace %s", trace.get("trace_id"))

    def _run(self) -> None:
        while True:
            trace = self._q.get()
            if trace is None:
                return
            try:
                self._persist(trace)
            except Exception:  # noqa: BLE001 - never let a bad write kill the writer
                logger.warning("trace persist failed", exc_info=True)
            finally:
                self._q.task_done()


# ---- default sink: Lakebase (+ optional MLflow) ---------------------------- #

def _persist_default(trace: dict[str, Any]) -> None:
    """Persist a trace to Lakebase, and optionally mirror to MLflow Tracing.

    Imports are lazy + guarded so the realtime package never hard-depends on the
    contact-center API or MLflow at import time.
    """
    try:
        from api.app.deps import serving

        serving().insert_voice_trace(trace)
    except Exception:  # noqa: BLE001
        logger.warning("lakebase trace write failed", exc_info=True)

    if os.getenv("GENIE_TRACE_MLFLOW", "").lower() in ("1", "true", "yes"):
        _mirror_to_mlflow(trace)


def _mirror_to_mlflow(trace: dict[str, Any]) -> None:
    """Optional: emit the same span tree to MLflow Tracing (Agent Framework).

    Off by default (GENIE_TRACE_MLFLOW). Runs on the writer thread, so it never
    adds latency to the voice turn. Fully guarded: absent/old MLflow is a no-op.
    """
    try:
        import mlflow  # type: ignore
    except Exception:  # noqa: BLE001
        return
    try:
        client = mlflow.tracking.MlflowClient()  # type: ignore[attr-defined]
        root = client.start_trace(
            name=f"voice_turn.{trace.get('capability')}",
            inputs={"transcript": trace.get("input_transcript"), "language": trace.get("language")},
            tags={
                "session_id": str(trace.get("session_id")),
                "turn_id": str(trace.get("turn_id")),
                "call_id": str(trace.get("call_id")),
                "status": str(trace.get("status")),
            },
        )
        request_id = root.request_id
        for span in trace.get("spans", []):
            child = client.start_span(
                request_id=request_id,
                name=span.get("name"),
                parent_id=root.span_id,
                inputs=span.get("input"),
                attributes={"kind": span.get("kind"), **(span.get("attributes") or {})},
            )
            client.end_span(request_id=request_id, span_id=child.span_id, outputs=span.get("output"))
        client.end_trace(request_id=request_id, outputs={"response": trace.get("output_text")})
    except Exception:  # noqa: BLE001
        logger.debug("mlflow trace mirror skipped", exc_info=True)


_SINK: TraceSink | None = None
_SINK_LOCK = threading.Lock()


def get_sink() -> TraceSink:
    global _SINK
    if _SINK is None:
        with _SINK_LOCK:
            if _SINK is None:
                _SINK = TraceSink(_persist_default)
    return _SINK


def submit_trace(trace: TurnTrace) -> None:
    """Fire-and-forget: enqueue a completed turn trace for background persistence."""
    try:
        get_sink().submit(trace.to_dict())
    except Exception:  # noqa: BLE001 - tracing must never break a turn
        logger.debug("submit_trace failed", exc_info=True)
