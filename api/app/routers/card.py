"""Credit-card assistant endpoints.

The card voice UX runs a two-lane Genie design. The FAST lane (a quick fact) is
answered in-turn by the realtime voice tools (Genie Conversation API). The DEEP
lane — the itemized "why" — runs Genie AGENT MODE, which takes up to a minute, so
it must NOT sit on the voice hot path.

This router exposes profile, language, and greeting support endpoints. Agent Mode
investigations now run exclusively inside the live WebSocket AgentRuntime.
"""
from __future__ import annotations

import asyncio
from fastapi import APIRouter

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


