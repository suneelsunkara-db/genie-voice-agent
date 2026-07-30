"""Credit-card assistant endpoints.

The card voice UX runs a two-lane Genie design. The FAST lane (a quick fact) is
answered in-turn by the realtime voice tools (Genie Conversation API). The DEEP
lane — the itemized "why" — runs Genie AGENT MODE, which takes up to a minute, so
it must NOT sit on the voice hot path.

This router exposes:
  GET /card/profile/{customer_id} — full 360 profile + statement history for graphs
  GET /card/deepdive              — SSE proxy for Genie Agent Mode investigations

The deep-dive lane is also TRACED end-to-end (same TurnTrace machinery as the
voice turns) so an Agent-Mode investigation shows up in Trace Explorer alongside
the STT→LLM→TTS turns of the same call (linked by call_id / session_id).
"""
from __future__ import annotations

import asyncio
import json
import queue
import threading

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

router = APIRouter(prefix="/card", tags=["card"])


@router.get("/profile/{customer_id}")
async def profile(customer_id: str) -> dict:
    """Full cardholder profile + statement/spend/rewards history for the UI graphs.

    Returns { cardholder, recent_statements[], spending_by_category[],
    rewards_ledger[], summary } — enough to render the 360 strip, the expense
    trend and the spike / rewards-leakage waterfalls without waiting on the voice
    LLM or Agent Mode.
    """
    def _fetch():
        from genie_voice.config import get_settings
        from genie_voice.serve.card_lakebase import CardLakebaseServing

        svc = CardLakebaseServing(get_settings())
        return svc.get_cardholder_facts(customer_id)

    return await asyncio.to_thread(_fetch)


@router.get("/languages")
async def languages() -> dict:
    """End-to-end supported voice languages for the call-language picker.

    Single source of truth (same as the billing cockpit): the realtime voice
    loop's config-driven set — Qwen3-ASR ∩ VoxCPM2, ~24 languages. Each option
    carries its BCP-47 tag + English name; the client resolves native display
    labels via Intl.DisplayNames, so there is no hardcoded per-language list.
    Falls back to English-only if the realtime config can't be loaded.
    """
    def _fetch() -> dict:
        try:
            from realtime_api.config import RealtimeSettings
            from realtime_api.languages import language_payload

            rt = RealtimeSettings.resolve()
            payload = language_payload(rt.supported_languages)
            if payload.get("options"):
                return payload
        except Exception:  # noqa: BLE001 — never let a config hiccup break the picker
            pass
        english = [{"code": "en-US", "base": "en", "english_name": "English"}]
        return {"languages": ["en-US"], "default": "en-US", "options": english, "count": 1}

    return await asyncio.to_thread(_fetch)


@router.get("/greeting")
async def greeting(language: str = "en-US", name: str = "") -> dict:
    """The agent's opening greeting, generated in the caller's ``language``.

    One multilingual-model call renders the greeting for ANY supported language
    (cached per language+name) — the client fetches this instead of holding a
    hardcoded per-language table, and speaks the returned text through the same
    cloned voice. Returns ``{text}`` ("" if serving is unavailable, so the client
    can proceed to listening without speaking a fake English line).
    """
    def _gen() -> dict:
        try:
            from realtime_api.card_tools import card_greeting

            return {"text": card_greeting(language, name), "language": language}
        except Exception:  # noqa: BLE001 — never let a serving hiccup break the open
            return {"text": "", "language": language}

    return await asyncio.to_thread(_gen)


@router.get("/deepdive")
async def deepdive(
    request: Request,
    question: str,
    use_case: str | None = None,
    call_id: str | None = None,
    session_id: str | None = None,
    customer_id: str | None = None,
    language: str | None = None,
) -> StreamingResponse:
    """Stream a Genie Agent-Mode investigation as SSE.

    Query params:
      - question:    the full "why" question (already scoped to the cardholder).
      - use_case:    optional label, echoed back on each event for UI routing.
      - language:    caller's BCP-47 tag, so the spoken summary is in their language.
      - call_id / session_id / customer_id: link this deep dive's trace to the
        originating voice call in Trace Explorer.

    Emits ``data: {json}`` events with kinds: started | reasoning | sql | report |
    error | done. ``:`` heartbeat comments keep intermediaries from closing idle
    connections during the agent's long reasoning gaps.
    """
    async def event_stream():
        from genie_voice.config import get_settings
        from realtime_api.deep_dive import deep_dive_read_timeout_s, run_deep_dive

        timeout_s = deep_dive_read_timeout_s()
        # Pin the DEEP lane to the CARD Genie space by name (the lane itself is
        # industry-agnostic; the route names its own domain's space).
        card_space = get_settings().card_issuer.genie_space_name
        sink: "queue.Queue[dict]" = queue.Queue()
        threading.Thread(
            target=run_deep_dive,
            args=(question, sink),
            kwargs={
                "call_id": call_id,
                "session_id": session_id,
                "customer_id": customer_id,
                "use_case": use_case,
                "language": language,
                "genie_space_name": card_space,
                "read_timeout_s": timeout_s,
            },
            daemon=True,
        ).start()
        # Meta first: the client derives its stall watchdog from the SAME timeout
        # the server uses (+ buffer), so the two can never disagree.
        yield f"data: {json.dumps({'kind': 'meta', 'timeout_ms': int(timeout_s * 1000)})}\n\n"
        while True:
            if await request.is_disconnected():
                break
            try:
                item = await asyncio.to_thread(sink.get, True, 1.0)
            except queue.Empty:
                yield ": ping\n\n"
                continue
            if use_case is not None and isinstance(item, dict):
                item = {**item, "use_case": use_case}
            yield f"data: {json.dumps(item, default=str)}\n\n"
            if isinstance(item, dict) and item.get("kind") == "done":
                break

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
