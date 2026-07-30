"""Home 'concierge' voice-assistant endpoints.

The landing page opens with an agent that welcomes the signed-in user and helps
them pick an industry experience by voice (the actual routing is driven by the
``select_industry`` realtime tool). This router only serves the opening greeting,
generated in the caller's language — same design as ``/card/greeting`` and
``/calls/greeting``.
"""
from __future__ import annotations

import asyncio

from fastapi import APIRouter

router = APIRouter(prefix="/concierge", tags=["concierge"])


@router.get("/greeting")
async def greeting(language: str = "en-US", name: str = "") -> dict:
    """The concierge's opening welcome, generated in the caller's ``language``.

    One multilingual-model call renders it for ANY supported language (cached per
    language+name). The client speaks the returned text through the session's
    cloned voice, which also locks a clean voice reference for the whole call.
    Returns ``{text}`` ("" if serving is unavailable, so the client can proceed to
    listening without speaking a fake English line).
    """
    def _gen() -> dict:
        try:
            from realtime_api.concierge_tools import concierge_greeting

            return {"text": concierge_greeting(language, name), "language": language}
        except Exception:  # noqa: BLE001 — never let a serving hiccup break the open
            return {"text": "", "language": language}

    return await asyncio.to_thread(_gen)
