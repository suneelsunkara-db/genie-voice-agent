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
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from .evidence import Evidence


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


_MAX_REPORT_ROWS = 30
_MAX_REPORT_COLUMNS = 10


def table_as_markdown(
    columns: list[str],
    rows: list[list[object]],
    *,
    max_rows: int = _MAX_REPORT_ROWS,
) -> str:
    """A governed result table as markdown, so it can be summarized like prose.

    Genie answers a "how much / top N" question with numbers and, often, no
    narrative. That result is still the answer, so it needs the same rendering a
    narrative gets — reciting the rows is what the row claims are for, not what the
    caller asked. Bounded: a summary needs the shape and the leading rows, not every
    row, and the typed rows reach the screen through the evidence contract anyway.
    """
    header = [str(column) for column in columns[:_MAX_REPORT_COLUMNS]]
    if not header or not rows:
        return ""
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in rows[:max_rows]:
        cells = [
            "" if value is None else str(value).replace("|", "\\|")
            for value in list(row)[: len(header)]
        ]
        cells += [""] * (len(header) - len(cells))
        lines.append("| " + " | ".join(cells) + " |")
    if len(rows) > max_rows:
        lines.append("")
        lines.append(f"({len(rows)} rows in total; the first {max_rows} are shown.)")
    return "\n".join(lines)


def governed_answer_render(evidence: "Evidence") -> tuple[str, str]:
    """Split a governed result into (text to summarize for voice, panel report).

    A narrative answer is both: it is summarized for speech and painted as the
    written report. A result that arrived as rows only is summarized from its table
    but has no report — the panel renders the typed rows as a table and chart, so
    emitting the same rows again as markdown would just duplicate them.

    Returning ("", "") means there is nothing to render, and the turn falls back to
    the composed row claims as its spoken evidence.
    """
    prose = evidence.prose
    if prose is not None and prose.text.strip():
        text = prose.text.strip()
        return text, text
    table = evidence.table
    if table is not None and table.columns and table.rows:
        return table_as_markdown(list(table.columns), [list(row) for row in table.rows]), ""
    return "", ""


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
        "boundaries, and most useful next step. The governed answer may be a result "
        "table; if so, state what it shows and the few figures that answer the "
        "question, rounded for speech, and never read the table row by row. Use only "
        "facts present in the governed answer: do not invent capabilities or numbers. "
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
