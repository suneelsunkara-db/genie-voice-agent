"""Interaction-language catalog for multilingual voice assist.

Single source of truth for the languages Genie Voice supports. The *set* is
config-driven — the end-to-end voice loop's STT ∩ TTS languages from the
``realtime_voice.*`` config block — so it always matches what the picker and the
voice APIs offer (~24 today). This module maps each base ISO-639 code to a
BCP-47 tag + English name and generates one ``LanguageSpec`` per language; native
display names are resolved on the client via ``Intl.DisplayNames``, so there are
no hand-maintained per-language name maps. The zh ASR-comparison variants remain
first-class codes for the benchmark UI.

The selected language controls speech recognition and customer-facing prose. It
does not change the business data contract: account facts remain the canonical
USD/INV-900xx dataset unless the schema is explicitly localized later.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import re

# A language code is any BCP-47 tag the config supports (~24) plus the zh
# ASR-comparison variants — no longer a closed union, so selection scales with
# the config rather than a hardcoded list.
LanguageCode = str

DEFAULT_LANGUAGE: LanguageCode = "en-US"

# Canonical reference table: base ISO-639 code -> (default BCP-47 tag, English
# name). Complete for every code that can appear in the Qwen3-ASR / VoxCPM2
# candidate lists, so the supported set (their intersection) is always fully
# described with no fallback. This is the ONE language reference table in the
# codebase — realtime_api imports it from here.
LANGUAGE_CATALOG: dict[str, tuple[str, str]] = {
    "ar": ("ar-SA", "Arabic"),
    "cs": ("cs-CZ", "Czech"),
    "da": ("da-DK", "Danish"),
    "de": ("de-DE", "German"),
    "el": ("el-GR", "Greek"),
    "en": ("en-US", "English"),
    "es": ("es-ES", "Spanish"),
    "fa": ("fa-IR", "Persian"),
    "fi": ("fi-FI", "Finnish"),
    "fil": ("fil-PH", "Filipino"),
    "fr": ("fr-FR", "French"),
    "he": ("he-IL", "Hebrew"),
    "hi": ("hi-IN", "Hindi"),
    "hu": ("hu-HU", "Hungarian"),
    "id": ("id-ID", "Indonesian"),
    "it": ("it-IT", "Italian"),
    "ja": ("ja-JP", "Japanese"),
    "km": ("km-KH", "Khmer"),
    "ko": ("ko-KR", "Korean"),
    "lo": ("lo-LA", "Lao"),
    "mk": ("mk-MK", "Macedonian"),
    "ms": ("ms-MY", "Malay"),
    "my": ("my-MM", "Burmese"),
    "nl": ("nl-NL", "Dutch"),
    "no": ("nb-NO", "Norwegian"),
    "pl": ("pl-PL", "Polish"),
    "pt": ("pt-PT", "Portuguese"),
    "ro": ("ro-RO", "Romanian"),
    "ru": ("ru-RU", "Russian"),
    "sv": ("sv-SE", "Swedish"),
    "sw": ("sw-KE", "Swahili"),
    "th": ("th-TH", "Thai"),
    "tr": ("tr-TR", "Turkish"),
    "vi": ("vi-VN", "Vietnamese"),
    "yue": ("yue-HK", "Cantonese"),
    "zh": ("zh-CN", "Chinese"),
}

_ZH_REPLY_INSTRUCTION = (
    "Reply in natural Simplified Chinese while preserving invoice IDs, customer IDs, and USD amounts exactly."
)


@dataclass(frozen=True)
class LanguageSpec:
    code: LanguageCode
    label: str
    english_name: str
    deepgram_language: str
    whisper_language: str
    reply_instruction: str


def _base(tag: str) -> str:
    return (tag or "").split("-", 1)[0].lower()


def _reply_instruction(english_name: str) -> str:
    return (
        f"Reply in natural {english_name} while preserving invoice IDs, "
        "customer IDs, and USD amounts exactly."
    )


def _make_spec(tag: str, english_name: str) -> LanguageSpec:
    return LanguageSpec(
        code=tag,
        label=english_name,
        english_name=english_name,
        deepgram_language=_base(tag),
        whisper_language=english_name.lower(),
        reply_instruction=_reply_instruction(english_name),
    )


# The zh ASR-comparison variants (same spoken language, different STT models) and
# the Qwen3 zh-CN label are kept distinct so the benchmark comparison stays clear.
_SPEC_OVERRIDES: dict[str, LanguageSpec] = {
    "zh-CN": LanguageSpec(
        "zh-CN", "Chinese (Qwen3)", "Mandarin Chinese (Qwen3)", "zh", "chinese", _ZH_REPLY_INSTRUCTION
    ),
    "zh-CN-sensevoice": LanguageSpec(
        "zh-CN-sensevoice", "Chinese (SenseVoice)", "Mandarin Chinese (SenseVoice)", "zh", "chinese", _ZH_REPLY_INSTRUCTION
    ),
    "zh-CN-paraformer": LanguageSpec(
        "zh-CN-paraformer", "Chinese (Paraformer)", "Mandarin Chinese (Paraformer 8k)", "zh", "chinese", _ZH_REPLY_INSTRUCTION
    ),
}

# Extra ASR-comparison codes not present in the config intersection (they share
# the zh-CN audio language) but surfaced in the benchmark UI.
_ZH_VARIANT_CODES: tuple[str, ...] = ("zh-CN-sensevoice", "zh-CN-paraformer")


def _config_supported_tags() -> list[str]:
    """End-to-end voice languages (STT ∩ TTS) as BCP-47 tags, from config.

    Reads the same ``realtime_voice`` block the voice APIs resolve from, so this
    catalog and the picker/voice loop never diverge. Degrades to English-only if
    the config can't be read (keeps imports safe in bare contexts) — that's an
    availability guard, not a per-language fallback.
    """
    try:
        from genie_voice.config.settings import _load_yaml

        rv = _load_yaml().get("realtime_voice") or {}
    except Exception:  # noqa: BLE001
        rv = {}

    def _langs(key: str) -> list[str]:
        for candidate in (rv.get(key) or {}).values():
            langs = candidate.get("supported_languages")
            if langs:
                return [str(x).lower() for x in langs]
        return []

    stt = set(_langs("stt_candidates"))
    tts = _langs("tts_candidates")
    bases = [c for c in tts if c in stt] if (stt and tts) else (tts or sorted(stt))
    tags = [LANGUAGE_CATALOG[b][0] for b in bases if b in LANGUAGE_CATALOG]
    if "en-US" not in tags:
        tags.insert(0, "en-US")
    return tags


def _build_specs() -> tuple[tuple[str, ...], dict[str, LanguageSpec]]:
    order: list[str] = []
    specs: dict[str, LanguageSpec] = {}
    for tag in _config_supported_tags():
        if tag in specs:
            continue
        name = LANGUAGE_CATALOG.get(_base(tag), (tag, tag))[1]
        specs[tag] = _SPEC_OVERRIDES.get(tag) or _make_spec(tag, name)
        order.append(tag)
    for tag in _ZH_VARIANT_CODES:
        if tag not in specs:
            specs[tag] = _SPEC_OVERRIDES[tag]
            order.append(tag)
    return tuple(order), specs


SUPPORTED_LANGUAGES, LANGUAGE_SPECS = _build_specs()


def _build_aliases() -> dict[str, str]:
    aliases: dict[str, str] = {}
    for tag, spec in LANGUAGE_SPECS.items():
        aliases[tag.lower()] = tag
        aliases.setdefault(_base(tag), tag)
        aliases.setdefault(spec.english_name.lower(), tag)
        base = _base(tag)
        if base in LANGUAGE_CATALOG:
            aliases.setdefault(LANGUAGE_CATALOG[base][1].lower(), tag)
    # Friendly overrides so the common names resolve to the primary code.
    aliases.update({
        "en": "en-US",
        "english": "en-US",
        "zh": "zh-CN",
        "zh-cn": "zh-CN",
        "chinese": "zh-CN",
        "mandarin": "zh-CN",
        "sensevoice": "zh-CN-sensevoice",
        "zh-cn-sensevoice": "zh-CN-sensevoice",
        "paraformer": "zh-CN-paraformer",
        "zh-cn-paraformer": "zh-CN-paraformer",
    })
    return aliases


_ALIASES = _build_aliases()


def normalize_language(language: str | None) -> LanguageCode:
    value = (language or DEFAULT_LANGUAGE).strip()
    if value in LANGUAGE_SPECS:
        return value
    low = value.lower()
    canonical = _ALIASES.get(low) or _ALIASES.get(_base(low))
    if canonical:
        return canonical
    supported = ", ".join(SUPPORTED_LANGUAGES)
    raise ValueError(f"Unsupported language {value!r}; supported: {supported}")


def is_chinese_language(language: str | None) -> bool:
    return normalize_language(language).startswith("zh-CN")


def content_language(language: str | None) -> LanguageCode:
    """Map zh-CN ASR variants to zh-CN for replies, Genie, and UI copy."""
    code = normalize_language(language)
    if code.startswith("zh-CN"):
        return "zh-CN"
    return code


def asr_model_language(language: str | None) -> str:
    """Language token sent to Databricks ASR models."""
    code = normalize_language(language)
    if code.startswith("zh-CN"):
        return "zh"
    return code


def is_zh_asr_compare_language(language: str | None) -> bool:
    return is_chinese_language(language)


def language_spec(language: str | None) -> LanguageSpec:
    return LANGUAGE_SPECS[normalize_language(language)]


def stt_options_for_language(settings: Any, language: str | None) -> dict[str, Any]:
    """Return active STT options overlaid with any per-language route.

    Config shape is intentionally additive and backward compatible:

    providers:
      stt:
        options:
          databricks:
            endpoint: voice_asr_en_finetuned_whisper_lora
            routes:
              th-TH:
                endpoint: voice_asr_th_base
    """
    options = dict(settings.providers.stt.active_options())
    routes = options.get("routes") or options.get("language_routes") or {}
    route = routes.get(normalize_language(language)) or {}
    if isinstance(route, dict):
        options.update(route)
    options["language"] = normalize_language(language)
    return options


def canonical_business_context_instruction(language: str | None) -> str:
    spec = language_spec(language)
    return (
        f"Interaction language: {spec.english_name} ({spec.code}). "
        "The source-of-truth business data is canonical English/US billing data: "
        "invoice IDs look like INV-90003, customer IDs look like CUST-4028, and "
        "money amounts are USD. If the customer speaks another language, extract "
        "canonical JSON fields using those table semantics. Do not infer baht, "
        "rupiah, yuan, or localized invoice formats unless they are explicitly "
        "present in the account facts."
    )


def localized_reply_opener(language: str | None, *, genie_insight: bool) -> str:
    language_code = content_language(language)
    if language_code == "th-TH":
        return "จากข้อมูลของ Genie " if genie_insight else "จากข้อมูลบัญชีของคุณ "
    if language_code == "id-ID":
        return "Berdasarkan insight Genie, " if genie_insight else "Berdasarkan akun Anda, "
    if language_code == "zh-CN":
        return "根据 Genie 洞察，" if genie_insight else "根据您的账户信息，"
    return "Based on Genie insights, " if genie_insight else "Based on your account, "


_INTERNAL_DISPLAY_TERM = (
    r"invoices\.|customer_id|overdue_invoice|overdue_invoices|"
    r"\bSQL\b|\bschema\b|\btable\b|\bcolumn\b|\bquery\b|"
    r"คอลัมน์|ตาราง|kolom|tabel|查询|查询结果|列|表"
)


def sanitize_generated_display_text(text: str | None) -> str:
    """Remove leaked schema/tool wording from customer-facing display prose."""
    value = (text or "").strip()
    if not value:
        return value
    # Drop parenthetical citations like "(based on invoices.status = ...)".
    value = re.sub(
        rf"\s*[\(\（][^\)\）]*(?:{_INTERNAL_DISPLAY_TERM})[^\)\）]*[\)\）]",
        "",
        value,
        flags=re.IGNORECASE,
    )
    # Drop common Chinese lead-ins that expose internal query filters.
    value = re.sub(
        rf"^根据\s*`?[^，,]*(?:{_INTERNAL_DISPLAY_TERM})[^，,]*[，,]\s*",
        "",
        value,
        flags=re.IGNORECASE,
    )
    # Remove backticked internal identifiers without touching visible IDs/amounts.
    value = re.sub(
        rf"`[^`]*(?:{_INTERNAL_DISPLAY_TERM})[^`]*`",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(r"[^。.!?！？]*(?:行汇总数据|row summary|rows? returned)[^。.!?！？]*[。.!?！？]?", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\bcustomer_id\b", "customer ID", value, flags=re.IGNORECASE)
    value = re.sub(r"\boverdue_invoices?\b", "overdue invoices", value, flags=re.IGNORECASE)
    value = re.sub(r"\s{2,}", " ", value)
    value = re.sub(r"\s+([.,!?;:，。])", r"\1", value)
    return value.strip()


_CANONICAL_BUSINESS_TOKEN_RE = re.compile(
    r"(?:INV|CUST|CALL)-[\w-]+|\$[\d,]+(?:\.\d{1,2})?|\bUSD\b",
    re.IGNORECASE,
)


def prose_for_language_check(text: str | None) -> str:
    """Strip canonical IDs/amounts so script-ratio checks focus on spoken prose."""
    value = (text or "").strip()
    if not value:
        return value
    return _CANONICAL_BUSINESS_TOKEN_RE.sub(" ", value)


def generated_text_language_check(text: str | None, language: str | None) -> dict[str, Any]:
    """Best-effort guardrail for customer-facing generated prose.

    This is intentionally a metadata check, not a translator. It catches obvious
    language misses for scripts where we can be reliable (Thai/Chinese) while
    preserving canonical IDs, SQL fragments, and USD amounts.
    """
    language_code = content_language(language)
    value = prose_for_language_check(text)
    if not value:
        return {"checked": False, "expected_language": language_code, "matches": False, "reason": "empty"}

    letters = re.findall(r"[A-Za-z\u0E00-\u0E7F\u4E00-\u9FFF]", value)
    if not letters:
        return {"checked": False, "expected_language": language_code, "matches": True, "reason": "no_letters"}

    thai = len(re.findall(r"[\u0E00-\u0E7F]", value))
    han = len(re.findall(r"[\u4E00-\u9FFF]", value))
    latin = len(re.findall(r"[A-Za-z]", value))
    total = max(1, len(letters))

    if language_code == "th-TH":
        matches = thai / total >= 0.20
        return {
            "checked": True,
            "expected_language": language_code,
            "matches": matches,
            "reason": "thai_script_ratio",
            "script_ratio": round(thai / total, 3),
        }
    if language_code == "zh-CN":
        matches = han / total >= 0.20
        return {
            "checked": True,
            "expected_language": language_code,
            "matches": matches,
            "reason": "han_script_ratio",
            "script_ratio": round(han / total, 3),
        }
    if language_code == "en-US":
        non_latin = thai + han
        matches = latin >= non_latin
        return {
            "checked": True,
            "expected_language": language_code,
            "matches": matches,
            "reason": "latin_script_dominant",
        }

    # Indonesian uses Latin script like English; script-only detection is not
    # reliable enough to reject output. Mark checked=false but carry metadata.
    return {
        "checked": False,
        "expected_language": language_code,
        "matches": True,
        "reason": "latin_language_not_script_distinguishable",
    }
