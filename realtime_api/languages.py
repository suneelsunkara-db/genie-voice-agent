"""Language options helper for the realtime voice loop.

Thin adapter over the one canonical reference table in ``genie_voice.i18n``
(``LANGUAGE_CATALOG``). The supported *set* is config-driven (STT ∩ TTS in
``realtime_voice.*``); this module only shapes it into UI picker options. Native
display names are derived on the client with ``Intl.DisplayNames`` — there are no
hand-maintained per-language name maps anywhere.
"""
from __future__ import annotations

from typing import Any

from genie_voice.i18n import DEFAULT_LANGUAGE as DEFAULT_TAG
from genie_voice.i18n import LANGUAGE_CATALOG as CATALOG


def base_code(language: str | None) -> str:
    """Primary ISO-639 subtag of a BCP-47 tag or base code ('en-US' -> 'en')."""
    return (language or "").split("-", 1)[0].lower()


def tag_for(base: str) -> str:
    """Default BCP-47 tag for a base code, or the input unchanged if unknown."""
    entry = CATALOG.get(base.lower())
    return entry[0] if entry else base


def english_name(language: str | None) -> str:
    """English name for a base code or BCP-47 tag ('vi' / 'vi-VN' -> 'Vietnamese')."""
    entry = CATALOG.get(canonical_base(language))
    return entry[1] if entry else str(language or "")


# Reverse lookup: English name (lowercased) -> ISO base. STT endpoints often
# report the *name* ('chinese') rather than a code ('zh'); this lets us map
# either form back to the canonical base so comparisons don't false-trigger.
_NAME_TO_BASE = {name.lower(): base for base, (_tag, name) in CATALOG.items()}


def canonical_base(language: str | None) -> str:
    """Best-effort ISO-639 base for any language token STT/UI might produce.

    Accepts BCP-47 tags ('zh-CN'), base codes ('zh'), or English names
    ('chinese', 'Mandarin' via i18n aliases) and returns the base ('zh'). Falls
    back to the raw primary subtag when unrecognized, so callers can still
    compare without raising.
    """
    raw = base_code(language)
    if raw in CATALOG:
        return raw
    token = str(language or "").strip().lower()
    if token in _NAME_TO_BASE:
        return _NAME_TO_BASE[token]
    try:  # last resort: reuse the backend catalog's rich alias table
        from genie_voice.i18n import normalize_language

        return base_code(normalize_language(language))
    except Exception:  # noqa: BLE001 — unknown token: keep the raw subtag
        return raw


def canonical_tag(language: str | None) -> str:
    """Canonical BCP-47 tag for any token ('chinese'/'zh' -> 'zh-CN').

    Unknown tokens are returned unchanged so downstream stages still get a value.
    """
    base = canonical_base(language)
    entry = CATALOG.get(base)
    return entry[0] if entry else str(language or "")


def language_payload(base_codes: tuple[str, ...] | list[str]) -> dict[str, Any]:
    """THE canonical supported-language payload for every UI picker.

    Single source of truth so the billing cockpit, the card page, and the
    standalone realtime API all report the identical set + shape:
    ``{languages: [tags], options: [{code, base, english_name}], default, count}``.
    Native display labels are resolved on the client via Intl.DisplayNames.
    """
    options = language_options(base_codes)
    return {
        "languages": [item["code"] for item in options],
        "options": options,
        "default": DEFAULT_TAG,
        "count": len(options),
    }


def language_options(base_codes: tuple[str, ...] | list[str]) -> list[dict[str, str]]:
    """Describe the config-supported languages for the UI picker.

    Returns ``[{code: <BCP-47 tag>, base: <iso>, english_name: <name>}]`` ordered
    as configured, with English guaranteed present and first (the picker default).
    Native labels are resolved on the client via Intl.DisplayNames.
    """
    seen: set[str] = set()
    options: list[dict[str, str]] = []
    for code in base_codes:
        base = base_code(code)
        if base in seen:
            continue
        entry = CATALOG.get(base)
        if not entry:
            continue
        seen.add(base)
        tag, name = entry
        options.append({"code": tag, "base": base, "english_name": name})
    options.sort(key=lambda item: (item["base"] != "en", item["english_name"]))
    if not any(item["base"] == "en" for item in options):
        options.insert(0, {"code": "en-US", "base": "en", "english_name": "English"})
    return options
