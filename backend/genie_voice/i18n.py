"""Interaction-language contract for multilingual voice assist.

The selected language controls speech recognition and customer-facing prose. It
does not change the current business data contract: account facts remain the
canonical USD/INV-900xx dataset unless the schema is explicitly localized later.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal
import re

LanguageCode = Literal["en-US", "th-TH", "id-ID", "zh-CN"]

DEFAULT_LANGUAGE: LanguageCode = "en-US"
SUPPORTED_LANGUAGES: tuple[LanguageCode, ...] = ("en-US", "th-TH", "id-ID", "zh-CN")


@dataclass(frozen=True)
class LanguageSpec:
    code: LanguageCode
    label: str
    english_name: str
    deepgram_language: str
    whisper_language: str
    reply_instruction: str


LANGUAGE_SPECS: dict[LanguageCode, LanguageSpec] = {
    "en-US": LanguageSpec(
        code="en-US",
        label="English",
        english_name="English",
        deepgram_language="en-US",
        whisper_language="english",
        reply_instruction="Use plain spoken English.",
    ),
    "th-TH": LanguageSpec(
        code="th-TH",
        label="Thai",
        english_name="Thai",
        deepgram_language="th",
        whisper_language="thai",
        reply_instruction="Reply in natural Thai while preserving invoice IDs, customer IDs, and USD amounts exactly.",
    ),
    "id-ID": LanguageSpec(
        code="id-ID",
        label="Indonesian",
        english_name="Indonesian",
        deepgram_language="id",
        whisper_language="indonesian",
        reply_instruction="Reply in natural Indonesian while preserving invoice IDs, customer IDs, and USD amounts exactly.",
    ),
    "zh-CN": LanguageSpec(
        code="zh-CN",
        label="Chinese",
        english_name="Mandarin Chinese",
        deepgram_language="zh",
        whisper_language="chinese",
        reply_instruction="Reply in natural Simplified Chinese while preserving invoice IDs, customer IDs, and USD amounts exactly.",
    ),
}


def normalize_language(language: str | None) -> LanguageCode:
    value = (language or DEFAULT_LANGUAGE).strip()
    aliases = {
        "en": "en-US",
        "english": "en-US",
        "th": "th-TH",
        "thai": "th-TH",
        "id": "id-ID",
        "indonesian": "id-ID",
        "zh": "zh-CN",
        "zh-cn": "zh-CN",
        "chinese": "zh-CN",
        "mandarin": "zh-CN",
    }
    canonical = aliases.get(value.lower(), value)
    if canonical not in SUPPORTED_LANGUAGES:
        supported = ", ".join(SUPPORTED_LANGUAGES)
        raise ValueError(f"Unsupported language {value!r}; supported: {supported}")
    return canonical  # type: ignore[return-value]


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
    language_code = normalize_language(language)
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


def generated_text_language_check(text: str | None, language: str | None) -> dict[str, Any]:
    """Best-effort guardrail for customer-facing generated prose.

    This is intentionally a metadata check, not a translator. It catches obvious
    language misses for scripts where we can be reliable (Thai/Chinese) while
    preserving canonical IDs, SQL fragments, and USD amounts.
    """
    language_code = normalize_language(language)
    value = (text or "").strip()
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
