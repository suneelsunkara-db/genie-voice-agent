"""Shared, profile-agnostic opening copy for agent-initiated calls.

Both the card and billing assistants open the call by SPEAKING first. The opening
line is loaded from reviewed committed translations, cached per
(base-language, first-name), and seeded into history so the model knows it already
greeted.

Only the phrase key differs per profile. The mechanism lives here once and reads
the committed runtime phrase catalog.
"""
from __future__ import annotations

from .runtime.phrases import phrase

# Keyed by (base-language, lowercased first-name).
GreetingCache = dict[tuple[str, str], str]


def generate_greeting(
    language: str,
    *,
    first_name: str,
    phrase_key: str,
    cache: GreetingCache,
) -> str:
    """Render reviewed opening copy in the caller's language."""
    from .languages import base_code

    key = (base_code(language) or "en", (first_name or "").strip().lower())
    if key in cache:
        return cache[key]
    name = (first_name or "").strip()
    selected_key = f"{phrase_key}.named" if name else phrase_key
    text = phrase(selected_key, language=language, name=name)
    if text:
        cache[key] = text
    return text


def seed_greeting_for(
    language: str,
    *,
    phrase_key: str,
    cache: GreetingCache,
) -> str:
    """A cached in-language greeting to seed LLM history (so it knows it greeted).

    Any cached greeting for the same base language is fine as context (the exact
    name is irrelevant to the model), so this reuses whatever the greeting endpoint
    already loaded. Falls back to the nameless committed variant.
    """
    from .languages import base_code

    base = base_code(language) or "en"
    for (cached_base, _name), text in cache.items():
        if cached_base == base and text:
            return text
    return generate_greeting(
        language, first_name="", phrase_key=phrase_key, cache=cache
    )
