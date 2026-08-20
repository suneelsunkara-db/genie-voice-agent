"""Databricks Knowledge Agent endpoints.

TIER 1: the opening greeting (same design as ``/card/greeting``) and a
``/corpus`` listing shared with the realtime ``knowledge_search`` tool, so the
spoken answers and the on-screen topic cards use ONE source of truth.

SEAM for Tier 2: back the corpus (and the tool) with Databricks Vector Search
over real docs; the route + frontend contract stay the same.
"""
from __future__ import annotations

import asyncio

from fastapi import APIRouter

router = APIRouter(prefix="/knowledge", tags=["knowledge"])


@router.get("/greeting")
async def greeting(language: str = "en-US", name: str = "") -> dict:
    """The knowledge agent's opening greeting, generated in the caller's ``language``."""
    def _gen() -> dict:
        try:
            from realtime_api.knowledge_tools import knowledge_greeting

            return {"text": knowledge_greeting(language, name), "language": language}
        except Exception:  # noqa: BLE001 — never let a serving hiccup break the open
            return {"text": "", "language": language}

    return await asyncio.to_thread(_gen)


@router.get("/corpus")
async def corpus() -> dict:
    """The questions the page offers, grouped by category and tagged with their lane.

    ``categories`` is the stable display order. Each topic names the ``lane`` that
    answers it (``pack`` = the cited docs corpus, ``workspace`` = a live Genie One
    round-trip against the caller's governed workspace) and the ``source`` the answer
    comes from. Pack topics carry the cited ``answer``; workspace topics carry a
    ``preview`` of what asking will do, because their answer only exists once Genie
    One actually runs.
    """
    from realtime_api.knowledge_tools import KNOWLEDGE_CATEGORIES, knowledge_topics

    return {"categories": list(KNOWLEDGE_CATEGORIES), "topics": knowledge_topics()}
