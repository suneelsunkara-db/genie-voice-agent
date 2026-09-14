"""Deterministic, committed speech used for fixed product moments.

Greetings, wait acknowledgements, progress narration, language prompts, and
navigation confirmations are application copy, not inference. Keeping them in a
reviewed multilingual catalog removes latency and prevents an LLM judge from
mistaking a deterministic UI transition for an unsupported model claim.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from string import Formatter

_CATALOG_PATH = Path(__file__).resolve().parent.parent / "phrases" / "runtime.json"

ENGLISH: dict[str, str] = {
    "greeting.concierge": (
        "Welcome. Choose your language, then say Telco, Financial "
        "Services, or Knowledge Agent."
    ),
    "greeting.concierge.named": (
        "Welcome back, {name}. Choose your language, then say Telco, Financial "
        "Services, or Knowledge Agent."
    ),
    "greeting.billing": "Hi, how can I help you today?",
    "greeting.billing.named": "Hi {name}, how can I help you today?",
    "greeting.card": "Hi, how can I help you today?",
    "greeting.card.named": "Hi {name}, how can I help you today?",
    "greeting.knowledge": "Hi, what would you like to know today?",
    "greeting.knowledge.named": "Hi {name}, what would you like to know today?",
    "filler.ack": "I'm looking into that now. Please hold on for a moment.",
    "filler.progress_1": "I'm still reviewing the relevant information.",
    "filler.progress_2": "The analysis is taking a little longer, and I'm still working on it.",
    "filler.progress_3": "I'm checking the results and will share the answer shortly.",
    "progress.understanding": "I'm making sure I understand your question.",
    "progress.finding_data": "I'm finding the relevant information.",
    "progress.running_analysis": "I'm running the analysis now.",
    "progress.checking_results": "I'm checking the results.",
    "progress.preparing_answer": "I'm preparing your answer.",
    "language.switch": (
        "I heard {detected_language}. Please switch the app language to "
        "{detected_language} so we can continue."
    ),
    "sensitive.blocked": (
        "For your security, don't say card numbers, government identifiers, "
        "passwords, or passcodes. Please use the secure form instead."
    ),
    "confirm.navigate_telco": "Opening Telco billing support.",
    "confirm.navigate_fsi": "Opening Financial Services.",
    "confirm.navigate_knowledge": "Opening the Knowledge Agent.",
    "confirm.card_statement": "Statement Insights is selected. What would you like to know?",
    "confirm.card_rewards": "Rewards Optimizer is selected. What would you like to know?",
}


def _fields(template: str) -> set[str]:
    return {
        str(field)
        for _, field, _, _ in Formatter().parse(template)
        if field is not None
    }


@lru_cache(maxsize=1)
def catalog() -> dict[str, dict[str, str]]:
    try:
        loaded = json.loads(_CATALOG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        loaded = {}
    result: dict[str, dict[str, str]] = {"en-US": dict(ENGLISH)}
    for tag, values in loaded.items():
        if not isinstance(values, dict):
            continue
        translated: dict[str, str] = {}
        for key, source in ENGLISH.items():
            candidate = str(values.get(key) or "").strip()
            if candidate and _fields(candidate) == _fields(source):
                translated[key] = candidate
        result[str(tag)] = translated
    return result


def phrase(key: str, *, language: str, **values: str) -> str:
    """Render a fixed phrase in ``language``, falling back to reviewed English."""
    tag = str(language or "").strip()
    base = tag.split("-", 1)[0].lower()
    phrases = catalog()
    localized = phrases.get(tag) or phrases.get(tag.lower())
    if localized is None and base:
        localized = next(
            (items for code, items in phrases.items() if code.split("-", 1)[0].lower() == base),
            None,
        )
    template = (localized or {}).get(key) or ENGLISH.get(key)
    if template is None:
        raise KeyError(f"unknown fixed speech phrase: {key}")
    required = _fields(template)
    missing = sorted(required - set(values))
    if missing:
        raise ValueError(f"phrase {key!r} is missing values: {missing}")
    return template.format(**values)
