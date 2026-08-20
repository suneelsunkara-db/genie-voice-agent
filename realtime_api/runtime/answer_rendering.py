"""Shared long-answer rendering for voice + screen.

This is the WebSocket-runtime home of the proven FSI deep-dive rendering flow:

1. turn a long governed answer into a short spoken summary in the call language;
2. release that summary immediately so TTS can start; and
3. translate the full written answer concurrently, streaming deltas to the panel.

The source adapter remains responsible for deciding whether text is governed
evidence. This module only renders text that already passed that boundary.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Iterator


@lru_cache(maxsize=1)
def _render_knobs() -> tuple[int, int, str | None]:
    try:
        from ..config import RealtimeSettings

        settings = RealtimeSettings.resolve()
        return (
            int(settings.deep_dive_summary_max_tokens),
            int(settings.deep_dive_localize_max_tokens),
            settings.conversion_endpoint or None,
        )
    except Exception:  # noqa: BLE001
        return 220, 1800, None


def is_english(language: str | None) -> bool:
    return not language or str(language).split("-", 1)[0].strip().lower() == "en"


def language_name(language: str | None) -> str:
    tag = str(language or "")
    try:
        from genie_voice.i18n import LANGUAGE_CATALOG, language_spec, normalize_language

        normalized = normalize_language(tag)
        return next(
            (
                english_name
                for catalog_tag, english_name in LANGUAGE_CATALOG.values()
                if catalog_tag == normalized
            ),
            None,
        ) or language_spec(normalized).english_name
    except Exception:  # noqa: BLE001
        return tag


def summarize_for_voice(question: str, answer: str, language: str | None) -> str:
    """Return a short, translated summary suitable for both TTS and the panel."""
    text = (answer or "").strip()
    if not text:
        return ""

    from ..serving_factory import shared_serving

    lang = language or "en-US"
    summary_tokens, _, endpoint = _render_knobs()
    system = (
        "You render a governed analytical answer for a realtime voice assistant. "
        "Write 2-3 concise, natural sentences in the language identified by BCP-47 "
        f"code '{lang}'. Preserve the answer's important facts, limitations, permission "
        "boundaries, and most useful next step. Do not invent capabilities or numbers. "
        "Use no markdown, headings, bullets, or citation markers. Output only the "
        "sentences the agent should speak."
    )
    user = f"User asked: {question}\n\nGoverned answer:\n{text[:8000]}"
    return shared_serving().summarize(
        system=system,
        user=user,
        max_tokens=summary_tokens,
        endpoint=endpoint,
    ).strip()


def _translation_system(language: str | None) -> str:
    name = language_name(language)
    return (
        f"Translate the user's governed analytical answer into {name}. Preserve the "
        "markdown structure, numbers, currency amounts, dates, names, SQL, and citation "
        "markers exactly. Translate only the surrounding prose and table headers. Do "
        "not summarize, add commentary, or answer the report. Output only the translated "
        f"answer in {name}."
    )


def localize_answer_stream(answer: str, language: str | None) -> Iterator[str]:
    """Yield translated full-answer deltas; empty for English or empty answers.

    English is the source language of Genie One and Agent Mode reports. Skipping
    gpt-5-5 here is the same contract for Knowledge and the FSI deep-dive panel:
    paint the original report immediately, never pay for a same-language rewrite.
    """
    text = (answer or "").strip()
    if not text or is_english(language):
        return

    from ..serving_factory import shared_serving

    _, localize_tokens, endpoint = _render_knobs()
    serving = shared_serving()
    stream = getattr(serving, "summarize_stream", None)
    produced = False
    if callable(stream):
        try:
            for piece in stream(
                system=_translation_system(language),
                user=text[:8000],
                max_tokens=localize_tokens,
                endpoint=endpoint,
            ):
                if piece:
                    produced = True
                    yield piece
        except Exception:  # noqa: BLE001
            if produced:
                return
        if produced:
            return

    try:
        translated = serving.summarize(
            system=_translation_system(language),
            user=text[:8000],
            max_tokens=localize_tokens,
            endpoint=endpoint,
        ).strip()
    except Exception:  # noqa: BLE001
        translated = ""
    if translated:
        yield translated
