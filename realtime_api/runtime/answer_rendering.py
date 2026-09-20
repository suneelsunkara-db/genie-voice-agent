"""Shared upstream-answer rendering for voice + screen.

This is the WebSocket-runtime home of the proven FSI deep-dive rendering flow:

1. turn a natural-language Genie answer into a short spoken answer;
2. release that summary immediately so TTS can start; and
3. translate the full written answer concurrently, streaming deltas to the panel.

Structured rows remain typed evidence for tables and charts. They are never
promoted into customer-facing ``column: value`` narration.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from .evidence import Evidence
    from ..tracing import TurnTrace


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


def same_language(left: str | None, right: str | None) -> bool:
    """Compare BCP-47 primary language tags."""
    primary = lambda value: str(value or "en").split("-", 1)[0].strip().lower()
    return primary(left) == primary(right)


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


def canonicalize_question_for_agent_mode(question: str, language: str | None) -> str:
    """Translate a caller's question to Agent Mode's canonical English input.

    Genie Agent Mode is materially more reliable when both its question and report
    language are English.  The voice runtime therefore owns an explicit boundary:
    caller language on the outside, canonical English on the Agent Mode side.
    Report localization remains the inverse boundary in ``localize_answer_stream``.

    Fail closed when translation produces no text. Sending the original non-English
    question would violate the Agent Mode contract and recreate a language-dependent
    failure that is much harder to diagnose.
    """
    text = (question or "").strip()
    if not text or is_english(language):
        return text

    from ..serving_factory import shared_serving

    _, localize_tokens, endpoint = _render_knobs()
    source = language_name(language)
    system = (
        f"Translate the user's analytical question from {source} into English. "
        "Preserve names, customer IDs, account IDs, dates, numbers, currency amounts, "
        "and quoted business terms exactly. Do not answer, summarize, explain, or add "
        "instructions. Output only the English question."
    )
    translated = shared_serving().summarize(
        system=system,
        user=text,
        max_tokens=min(localize_tokens, 512),
        endpoint=endpoint,
    ).strip()
    if not translated:
        raise ValueError("question translation returned no text")
    return translated


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


def upstream_answer_render(evidence: "Evidence") -> tuple[str, str]:
    """Return natural upstream prose for speech and the report panel.

    Genie One prose, Genie Space answers, and Agent Mode reports are authoritative
    natural-language answers from their upstream service. Tables remain structured
    UI evidence. If an upstream service returns rows without prose, the caller can
    use the conversational model's natural response instead of narrating columns.
    """
    prose = evidence.prose
    if prose is not None and prose.text.strip():
        text = prose.text.strip()
        return text, text
    display = (evidence.display_prose or "").strip()
    if display:
        return display, display
    return "", ""


def _extractive_voice_excerpt(text: str, *, max_sentences: int = 3) -> str:
    cleaned = re.sub(r"(?m)^\s{0,3}(?:#{1,6}\s*|[-*]\s+)", "", text).strip()
    sentences = [
        match.group(0).strip()
        for match in re.finditer(r"[^.!?。！？\n]+(?:[.!?。！？]+|$)", cleaned)
        if match.group(0).strip()
    ]
    return " ".join(sentences[:max_sentences]).strip()


def _numeric_tokens(text: str) -> set[str]:
    return set(re.findall(r"(?<!\w)[+-]?\d[\d,.:%/-]*(?!\w)", text))


def summarize_for_voice(
    question: str,
    answer: str,
    language: str | None,
    *,
    source_language: str | None = "en",
    trace: "TurnTrace | None" = None,
) -> str:
    """Return a short, translated summary suitable for both TTS and the panel."""
    text = (answer or "").strip()
    if not text:
        return ""
    excerpt = _extractive_voice_excerpt(text)
    if not excerpt or same_language(source_language, language):
        return excerpt

    from ..serving_factory import shared_serving

    lang = language or "en-US"
    summary_tokens, _, endpoint = _render_knobs()
    system = (
        "You translate an upstream analytical answer for a realtime voice assistant. "
        "Translate the supplied extract into the language identified by BCP-47 "
        f"code '{lang}'. Preserve every fact, limitation, name, date, identifier, "
        "and number exactly. Do not summarize, round, omit, infer, or add anything. "
        "Use no markdown, headings, bullets, or citation markers. Output only the "
        "sentences the agent should speak."
    )
    user = f"User asked: {question}\n\nGoverned answer extract:\n{excerpt}"
    serving = shared_serving()
    # Reasoning-capable conversion endpoints can occasionally spend a small token
    # ceiling internally and return an empty visible message. One bounded retry
    # with the same prompt and a 512-token ceiling is enough room for reasoning
    # while the prompt still constrains the visible answer to 2-3 sentences.
    budgets = list(dict.fromkeys((summary_tokens, max(summary_tokens, 512))))
    last_error: Exception | None = None
    for budget in budgets:
        try:
            summary = serving.summarize(
                system=system,
                user=user,
                max_tokens=budget,
                endpoint=endpoint,
                trace=trace,
            ).strip()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            continue
        if summary and _numeric_tokens(summary) <= _numeric_tokens(excerpt):
            return summary
    if last_error is not None:
        raise RuntimeError("voice summary generation failed") from last_error
    raise ValueError("voice summary generation returned no text")


def _translation_system(language: str | None) -> str:
    name = language_name(language)
    return (
        f"Translate the user's analytical answer into {name}. Preserve the "
        "markdown structure, numbers, currency amounts, dates, names, SQL, and citation "
        "markers exactly. Translate only the surrounding prose and table headers. Do "
        "not summarize, add commentary, or answer the report. Output only the translated "
        f"answer in {name}."
    )


def localize_answer_stream(
    answer: str,
    language: str | None,
    *,
    source_language: str | None = "en",
    trace: "TurnTrace | None" = None,
) -> Iterator[str]:
    """Yield translated full-answer deltas; empty for English or empty answers.

    English is the source language of Genie One and Agent Mode reports. Skipping
    gpt-5-5 here is the same contract for Knowledge and the FSI deep-dive panel:
    paint the original report immediately, never pay for a same-language rewrite.
    """
    text = (answer or "").strip()
    if not text or same_language(source_language, language):
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
                trace=trace,
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
            trace=trace,
        ).strip()
    except Exception:  # noqa: BLE001
        translated = ""
    if translated:
        yield translated
