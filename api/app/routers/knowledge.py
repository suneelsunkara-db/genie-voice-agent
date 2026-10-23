"""Databricks Knowledge Agent page metadata.

The greeting and ``/corpus`` endpoint publish localized starter questions.
Answers come from the managed Genie One conversation, not from static metadata.
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
async def corpus(language: str = "en-US") -> dict:
    """Caller-language questions grouped by localized category.

    ``categories`` keeps a stable display order. Every topic is a live Genie One
    workspace question and carries localized display text plus its canonical source
    question; there is no canned answer.
    """
    from realtime_api.knowledge_tools import knowledge_categories, knowledge_topics

    return {
        "categories": knowledge_categories(language),
        "topics": knowledge_topics(language),
        "language": language,
    }
