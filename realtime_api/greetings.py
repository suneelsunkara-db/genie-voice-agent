"""Shared, profile-agnostic opening-greeting synthesis for agent-initiated calls.

Both the card and billing assistants open the call by SPEAKING first. The opening
line is generated in the caller's language by the multilingual model (one
``phrase`` call, the same pattern as fillers / switch-language prompts), cached per
(base-language, first-name), and seeded into the LLM history so the model knows it
already greeted.

Only the *intent* (brand + persona + what the agent offers) differs per profile;
the mechanism below is identical, so it lives here once instead of being copied
into every profile. A profile supplies an ``intent(first_name) -> str`` builder and
its own cache dict.
"""
from __future__ import annotations

from typing import Callable

# Keyed by (base-language, lowercased first-name); one model call per key.
GreetingCache = dict[tuple[str, str], str]

IntentBuilder = Callable[[str], str]


def generate_greeting(
    language: str,
    *,
    first_name: str,
    intent: IntentBuilder,
    cache: GreetingCache,
) -> str:
    """Render the opening greeting in the caller's ``language`` (cached).

    One multilingual ``phrase`` call renders ANY supported language, so there is no
    hardcoded per-language table and no English fallback. Returns "" when serving
    is unavailable, so callers degrade to just listening instead of speaking a fake
    English line.
    """
    from .languages import base_code

    key = (base_code(language) or "en", (first_name or "").strip().lower())
    if key in cache:
        return cache[key]
    try:
        from .serving_factory import shared_serving

        text = shared_serving().phrase(intent(first_name), language=language).strip()
    except Exception:  # noqa: BLE001 — no fake fallback; caller handles "".
        text = ""
    if text:
        cache[key] = text
    return text


def seed_greeting_for(
    language: str,
    *,
    intent: IntentBuilder,
    cache: GreetingCache,
) -> str:
    """A cached in-language greeting to seed LLM history (so it knows it greeted).

    Any cached greeting for the same base language is fine as context (the exact
    name is irrelevant to the model), so this reuses whatever the greeting endpoint
    already generated — no extra hot-path model call. Falls back to generating a
    nameless variant; "" on failure.
    """
    from .languages import base_code

    base = base_code(language) or "en"
    for (cached_base, _name), text in cache.items():
        if cached_base == base and text:
            return text
    return generate_greeting(language, first_name="", intent=intent, cache=cache)
