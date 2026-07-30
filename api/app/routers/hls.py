"""Healthcare (HLS) voice-assistant endpoints.

TIER 1: the opening greeting (same design as ``/card/greeting``) and a mock
clinical ``/summary`` shared with the realtime ``health_summary`` tool, so the
spoken answers and the on-screen cards use ONE source of truth.

SEAM for Tier 2: back ``/summary`` (and the tool) with a real Lakebase/Genie
store; the route + frontend contract stay the same.
"""
from __future__ import annotations

import asyncio

from fastapi import APIRouter

router = APIRouter(prefix="/hls", tags=["hls"])


@router.get("/greeting")
async def greeting(language: str = "en-US", name: str = "") -> dict:
    """The HLS agent's opening greeting, generated in the caller's ``language``."""
    def _gen() -> dict:
        try:
            from realtime_api.hls_tools import hls_greeting

            return {"text": hls_greeting(language, name), "language": language}
        except Exception:  # noqa: BLE001 — never let a serving hiccup break the open
            return {"text": "", "language": language}

    return await asyncio.to_thread(_gen)


@router.get("/summary")
async def summary() -> dict:
    """The member's plan, coverage, recent claims, and last visit (Tier-1 mock)."""
    from realtime_api.hls_tools import MOCK_SUMMARY

    return MOCK_SUMMARY
