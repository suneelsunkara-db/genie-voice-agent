"""Framework-level supported-language catalog.

The set of languages the voice stack supports is a FRAMEWORK concern (the
intersection of the configured STT + TTS capabilities), not a per-industry one.
Every voice surface — home concierge, card, healthcare — reads this ONE endpoint
so the language bar is identical everywhere and always matches what the pipeline
can actually speak. (``/card/languages`` predates this and returns the same
payload for backward-compat.)
"""
from __future__ import annotations

import asyncio

from fastapi import APIRouter

router = APIRouter(tags=["languages"])


@router.get("/languages")
async def languages() -> dict:
    """Config-driven supported languages (STT ∩ TTS), with native-label options."""
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
