"""Shared upstream-answer rendering for voice + screen.

This is the WebSocket-runtime home of the proven FSI deep-dive rendering flow:

1. turn a natural-language Genie answer into a short spoken answer;
2. release that summary immediately so TTS can start; and
3. translate the full written answer concurrently, streaming deltas to the panel.

Structured rows remain typed evidence for tables and charts. They are never
promoted into customer-facing ``column: value`` narration.
"""
from __future__ import annotations

import html
import re
import unicodedata
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


class SpeechRenderError(ValueError):
    """An upstream answer could not be certified for spoken output."""


_FENCE_RE = re.compile(r"(?ms)^\s*```.*?^\s*```\s*$")
_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]+\)")
_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_AUTOLINK_RE = re.compile(r"<(?:https?://|mailto:)[^>]+>", re.I)
_URL_RE = re.compile(r"https?://\S+", re.I)
_HTML_TAG_RE = re.compile(r"</?[A-Za-z][^>]*>")
_INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
_TABLE_RULE_RE = re.compile(r"^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*:?-{3,}:?\s*\|?\s*$")
_MARKDOWN_REMAINS_RE = re.compile(
    r"(?:\*\*|__|~~|```|`|\[[^\]]+\]\([^)]+\)|(?m:^\s{0,3}#{1,6}\s))"
)
_CITATION_MARKER_RE = re.compile(
    r"(?:\\\[\[\s*\d+(?:\s*[,;–-]\s*\d+)*\]\\\]"
    r"|\[\[\s*\d+(?:\s*[,;–-]\s*\d+)*\]\]"
    r"|\[\s*\d+(?:\s*[,;–-]\s*\d+)*\])"
)
_PLACEHOLDER_RE = re.compile(r"\[\[FACT_(\d+)]]")

# Atomic values that must survive localization byte-for-byte. The broad numeric
# branch deliberately includes Unicode decimal digits and common separators; the
# identifier branch protects model/endpoint/version names that contain a digit.
_FACT_ATOM_RE = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"(?:[$€£¥₹₩₽]\s*)?[+-]?\d(?:[\d\s,.'’٬٫:/-]*\d)?"
    r"(?:\s*(?:%|USD|EUR|GBP|JPY|INR|SGD|DBU(?:s)?|ms|s|min|hours?))?"
    r"(?![A-Za-z0-9_])"
    r"|(?<![\w-])(?=[A-Za-z][A-Za-z0-9_.:/-]*\d)"
    r"[A-Za-z](?:[A-Za-z0-9_.:/-]*[A-Za-z0-9])?(?![\w-])",
    re.I,
)
_SPEECH_END = frozenset(".!?。！？؟।…;؛")
_CLAUSE_END = frozenset(",;:—–，；،")
_MAX_SPEECH_CHARS = 640
_MAX_SEGMENT_FACTS = 3
_DENSE_SPEECH_CHARS = 210
_DENSE_SPEECH_FACTS = 5
_MAX_SUMMARY_FACTS = 4
_ABBREVIATION_RE = re.compile(
    r"\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs|etc|e\.g|i\.e|a\.m|p\.m|U\.S)\.",
    re.I,
)
_ENGLISH_DATE_RANGE_RE = re.compile(
    r"\(?\b("
    r"January|February|March|April|May|June|July|August|September|October|November|December"
    r")\s+(\d{4})\s*[–—-]\s*("
    r"January|February|March|April|May|June|July|August|September|October|November|December"
    r")\s+(\d{4})\)?",
    re.I,
)


def _strip_inline_markdown(text: str) -> str:
    """Convert one Markdown block to plain speech without interpreting its facts."""
    def replace_link(match: re.Match[str]) -> str:
        label = match.group(1).strip()
        if re.match(r"^(?:query result|view\b|source\b|citation\b|details?\b)", label, re.I):
            return ""
        return label

    value = html.unescape(text)
    value = _IMAGE_RE.sub(lambda match: match.group(1), value)
    value = _LINK_RE.sub(replace_link, value)
    value = _AUTOLINK_RE.sub("", value)
    value = _URL_RE.sub("", value)
    value = _CITATION_MARKER_RE.sub("", value)
    value = _INLINE_CODE_RE.sub(lambda match: match.group(1), value)
    value = _HTML_TAG_RE.sub("", value)
    value = re.sub(r"(\*\*|__|~~)", "", value)
    value = re.sub(r"(?<!\w)[*_](?=\S)|(?<=\S)[*_](?!\w)", "", value)
    # Emoji and pictographs are presentation, not words for VoxCPM to pronounce.
    value = "".join(
        char
        for char in value
        if not (
            0x1F000 <= ord(char) <= 0x1FAFF
            or 0x2600 <= ord(char) <= 0x27BF
        )
    )
    value = re.sub(r"\s+", " ", value).strip()
    return re.sub(r"\s+([,.;:!?。！？؟।…])", r"\1", value)


def _clarify_english_date_ranges(text: str) -> str:
    """Render compact written month ranges as unambiguous spoken English."""
    return _ENGLISH_DATE_RANGE_RE.sub(
        lambda match: (
            f"from {match.group(1)} {match.group(2)} "
            f"through {match.group(3)} {match.group(4)}"
        ),
        text,
    )


def _speech_blocks(markdown: str) -> tuple[list[str], list[str], list[str]]:
    """Return plain paragraphs, list items and headings from a Markdown answer."""
    without_fences = _FENCE_RE.sub("", markdown)
    paragraphs: list[str] = []
    list_items: list[str] = []
    headings: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if current:
            value = _strip_inline_markdown(" ".join(current))
            if value:
                paragraphs.append(value)
            current.clear()

    for raw_line in without_fences.splitlines():
        line = raw_line.strip()
        if not line:
            flush()
            continue
        if _TABLE_RULE_RE.match(line) or (
            line.count("|") >= 2 and line.startswith("|") and line.endswith("|")
        ):
            flush()
            continue
        if re.fullmatch(r"\[[^\]]+]\([^)]+\)", line):
            flush()
            continue
        heading = re.match(r"^#{1,6}\s+(.+)$", line)
        if heading:
            flush()
            value = _strip_inline_markdown(heading.group(1))
            if value:
                headings.append(value)
            continue
        item = re.match(r"^(?:[-*+]|\d+[.)])\s+(.+)$", line)
        if item:
            flush()
            value = _strip_inline_markdown(item.group(1))
            if value:
                list_items.append(value)
            continue
        line = re.sub(r"^>\s?", "", line)
        current.append(line)
    flush()
    return paragraphs, list_items, headings


def _protect_facts(text: str) -> tuple[str, list[str]]:
    atoms: list[str] = []

    def replace(match: re.Match[str]) -> str:
        value = match.group(0).strip()
        atoms.append(value)
        return f"[[FACT_{len(atoms) - 1}]]"

    return _FACT_ATOM_RE.sub(replace, text), atoms


def _restore_facts(text: str, atoms: list[str]) -> str:
    seen = [int(value) for value in _PLACEHOLDER_RE.findall(text)]
    if sorted(seen) != list(range(len(atoms))):
        raise SpeechRenderError("translated speech did not preserve every factual atom exactly once")
    restored = _PLACEHOLDER_RE.sub(lambda match: atoms[int(match.group(1))], text)
    if _PLACEHOLDER_RE.search(restored):
        raise SpeechRenderError("unresolved factual placeholder")
    return restored


def _restore_selected_facts(text: str, atoms: list[str]) -> str:
    """Restore a verified subset chosen for a short spoken summary."""
    seen = [int(value) for value in _PLACEHOLDER_RE.findall(text)]
    if not seen or len(seen) > _MAX_SUMMARY_FACTS:
        raise SpeechRenderError("spoken summary selected an invalid number of facts")
    if len(seen) != len(set(seen)) or any(index >= len(atoms) for index in seen):
        raise SpeechRenderError("spoken summary duplicated or invented a factual placeholder")
    restored = _PLACEHOLDER_RE.sub(lambda match: atoms[int(match.group(1))], text)
    if _PLACEHOLDER_RE.search(restored):
        raise SpeechRenderError("unresolved factual placeholder")
    return restored


def _restore_segment(segment: str, atoms: list[str]) -> str:
    indexes = sorted({int(value) for value in _PLACEHOLDER_RE.findall(segment)})
    segment_atoms = [atoms[index] for index in indexes]
    remap = {old: new for new, old in enumerate(indexes)}
    compact = _PLACEHOLDER_RE.sub(
        lambda match: f"[[FACT_{remap[int(match.group(1))]}]]",
        segment,
    )
    return _restore_facts(compact, segment_atoms)


def _sentence_units(text: str) -> list[str]:
    """Language-neutral punctuation units with facts and abbreviations protected."""
    protected, atoms = _protect_facts(text)
    abbreviations: list[str] = []

    def protect_abbreviation(match: re.Match[str]) -> str:
        abbreviations.append(match.group(0))
        return f"[[ABBR_{len(abbreviations) - 1}]]"

    protected = _ABBREVIATION_RE.sub(protect_abbreviation, protected)
    parts = [
        match.group(0).strip()
        for match in re.finditer(
            r"[^.!?。！？؟।…;؛\n]+(?:[.!?。！？؟।…;؛]+|\n|$)",
            protected,
        )
        if match.group(0).strip()
    ]
    output: list[str] = []
    for part in parts or [protected]:
        part = re.sub(
            r"\[\[ABBR_(\d+)]]",
            lambda match: abbreviations[int(match.group(1))],
            part,
        )
        output.append(_restore_segment(part, atoms))
    return output


def _bounded_speech_text(text: str, *, max_chars: int = _MAX_SPEECH_CHARS) -> str:
    """Bound plain speech without cutting through a protected factual atom."""
    protected, atoms = _protect_facts(text)
    if len(protected) <= max_chars:
        return _restore_facts(protected, atoms)
    boundary = max(
        (index + 1 for index, char in enumerate(protected[:max_chars]) if char in _SPEECH_END),
        default=0,
    )
    if boundary < max_chars // 3:
        boundary = protected.rfind(" ", 0, max_chars)
    if boundary <= 0:
        boundary = max_chars
        while boundary > 0 and unicodedata.combining(protected[boundary - 1]):
            boundary -= 1
    open_placeholder = protected.rfind("[[", 0, boundary)
    close_placeholder = protected.rfind("]]", 0, boundary)
    if open_placeholder > close_placeholder:
        boundary = open_placeholder
    bounded = protected[:boundary].rstrip(" ,:;-") + "…"
    used = sorted({int(value) for value in _PLACEHOLDER_RE.findall(bounded)})
    compact_atoms = [atoms[index] for index in used]
    remap = {old: new for new, old in enumerate(used)}
    bounded = _PLACEHOLDER_RE.sub(
        lambda match: f"[[FACT_{remap[int(match.group(1))]}]]",
        bounded,
    )
    return _restore_facts(bounded, compact_atoms)


def _extractive_voice_excerpt(text: str) -> str:
    """Select answer-bearing Markdown blocks without punctuation tokenization."""
    paragraphs, list_items, headings = _speech_blocks(text)
    candidates = paragraphs or list_items or headings
    if not candidates:
        raise SpeechRenderError("upstream answer contained no speakable prose")
    selected: list[str] = []
    for block in candidates:
        if selected and len(" ".join((*selected, block))) > _MAX_SPEECH_CHARS:
            break
        selected.append(block)
        if len(selected) >= 2:
            break
    units = _sentence_units(" ".join(selected or candidates[:1]))
    return _bounded_speech_text(" ".join(units[:3]))


def certify_speech_text(text: str) -> str:
    """Fail closed unless text is bounded plain prose suitable for TTS."""
    value = re.sub(r"\s+", " ", (text or "").strip())
    if not value:
        raise SpeechRenderError("speech text is empty")
    if len(value) > _MAX_SPEECH_CHARS + 1:
        raise SpeechRenderError("speech text exceeds the realtime voice budget")
    if _MARKDOWN_REMAINS_RE.search(value) or _URL_RE.search(value):
        raise SpeechRenderError("speech text still contains Markdown or a URL")
    if _CITATION_MARKER_RE.search(value):
        raise SpeechRenderError("speech text contains a citation marker")
    if _PLACEHOLDER_RE.search(value) or "[[FACT_" in value:
        raise SpeechRenderError("speech text contains unresolved factual placeholders")
    if any(unicodedata.category(char) == "Cc" for char in value):
        raise SpeechRenderError("speech text contains control characters")
    return value


def split_speech_segments(
    text: str,
    *,
    max_chars: int = 180,
    max_fact_atoms: int = _MAX_SEGMENT_FACTS,
) -> list[str]:
    """Create model-safe TTS segments without breaking factual atoms.

    VoxCPM becomes unstable on a single accounting sentence containing many
    amounts even when the sentence is grammatically valid. Sentence punctuation
    alone is therefore insufficient: dense units are also split at language-
    neutral clause punctuation after numbers, dates, IDs and model names have
    been protected.
    """
    value = certify_speech_text(text)
    units = _sentence_units(value)
    output: list[str] = []
    clause_chars = "".join(re.escape(char) for char in _CLAUSE_END)
    for unit in units:
        protected, atoms = _protect_facts(unit)
        fact_count = len(_PLACEHOLDER_RE.findall(protected))
        if len(unit) <= max_chars and fact_count <= max_fact_atoms:
            output.append(unit)
            continue
        clauses = [
            match.group(0).strip()
            for match in re.finditer(
                rf"[^{clause_chars}]+(?:[{clause_chars}]+|$)",
                protected,
            )
            if match.group(0).strip()
        ]
        current = ""
        protected_segments: list[str] = []
        for clause in clauses or [protected]:
            candidate = f"{current} {clause}".strip() if current else clause
            candidate_facts = len(_PLACEHOLDER_RE.findall(candidate))
            if current and (
                len(candidate) > max_chars or candidate_facts > max_fact_atoms
            ):
                protected_segments.append(current)
                current = clause
            else:
                current = candidate
        if current:
            protected_segments.append(current)
        output.extend(_restore_segment(segment, atoms) for segment in protected_segments)
    return output


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
    if is_english(source_language):
        excerpt = _clarify_english_date_ranges(excerpt)
    protected_excerpt, atoms = _protect_facts(excerpt)
    dense = any(
        len(unit) > _DENSE_SPEECH_CHARS
        or len(_protect_facts(unit)[1]) > _DENSE_SPEECH_FACTS
        for unit in _sentence_units(excerpt)
    )
    if same_language(source_language, language) and not dense:
        return certify_speech_text(excerpt)

    from ..serving_factory import shared_serving

    lang = language or "en-US"
    summary_tokens, _, endpoint = _render_knobs()
    if dense:
        system = (
            "Create a short spoken answer in the language identified by BCP-47 "
            f"code '{lang}' from the governed analytical extract. Tokens such as "
            "[[FACT_0]] are immutable factual placeholders. Select only the facts "
            "needed to answer the user's question: use at most four placeholders, "
            "never use one twice, and do not invent any number, date, ID, amount, "
            "name, or claim. Write two or three short sentences with at most two "
            "facts per sentence and no parenthetical asides. Use no markdown, "
            "headings, bullets, URLs, or citation markers. Output only what the "
            "assistant should speak."
        )
    else:
        system = (
            "You translate an upstream analytical answer for a realtime voice assistant. "
            "Translate the supplied extract into the language identified by BCP-47 "
            f"code '{lang}'. Tokens such as [[FACT_0]] are immutable factual placeholders: "
            "return every placeholder exactly once and do not translate, alter, duplicate, "
            "or omit one. Do not summarize, round, omit, infer, or add anything. "
            "Use no markdown, headings, bullets, or citation markers. Output only the "
            "sentences the agent should speak."
        )
    user = f"User asked: {question}\n\nGoverned answer extract:\n{protected_excerpt}"
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
        if not summary:
            continue
        without_placeholders = _PLACEHOLDER_RE.sub("", summary)
        if _FACT_ATOM_RE.search(without_placeholders):
            continue
        try:
            restored = (
                _restore_selected_facts(summary, atoms)
                if dense
                else _restore_facts(summary, atoms)
            )
            return certify_speech_text(restored)
        except SpeechRenderError:
            continue
    if last_error is not None:
        raise RuntimeError("voice summary generation failed") from last_error
    raise SpeechRenderError("voice summary generation returned no certified text")


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
