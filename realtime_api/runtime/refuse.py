"""Typed ErrorEvidence codes and refuse-speech templates.

Speech after failure comes from these templates only — never model improvisation
after an ack when Evidence is empty or permission is denied.

The templates are translated OFFLINE into every supported language and committed
(``phrases/refusals.json``, written by ``scripts/i18n/translate_refusals.py``),
for the same reason the frontend message catalog is: this is the one line the
caller hears when a tool, a permission check, or the model itself just failed.
Generating it from the LLM at that moment would put the apology behind the thing
that is already broken, and add latency to the worst turn of the call.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path


class ErrorCode(str, Enum):
    NO_EVIDENCE = "no_evidence"
    PERMISSION = "permission"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    UNSUPPORTED = "unsupported"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class ErrorEvidence:
    code: ErrorCode
    message: str
    retryable: bool = False

    def as_dict(self) -> dict:
        return {
            "code": self.code.value,
            "message": self.message,
            "retryable": self.retryable,
        }


# The English source catalog. Every other language is a committed translation of
# exactly these strings, so a new code cannot ship without one.
_REFUSE_EN: dict[ErrorCode, str] = {
    ErrorCode.NO_EVIDENCE: (
        "I wasn't able to find data I can cite for that. "
        "Could you rephrase, or ask a more specific question?"
    ),
    ErrorCode.PERMISSION: (
        "I don't have your Databricks permission to look that up right now. "
        "Please sign in again, or ask an admin to grant access."
    ),
    ErrorCode.TIMEOUT: (
        "That lookup took too long and I had to stop. Please try again in a moment."
    ),
    ErrorCode.CANCELLED: "Okay — I've stopped that request.",
    ErrorCode.UNSUPPORTED: (
        "I can't help with that from this assistant. Try a different question or pack."
    ),
    ErrorCode.AMBIGUOUS: (
        "I want to make sure I get this right — could you clarify what you'd like me to look up?"
    ),
}


_CATALOG_PATH = Path(__file__).resolve().parent.parent / "phrases" / "refusals.json"


@lru_cache(maxsize=1)
def _catalog() -> dict[str, dict[str, str]]:
    """Committed offline translations, keyed by BCP-47 tag then error code."""
    try:
        loaded = json.loads(_CATALOG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {
        str(tag): {str(code): str(text) for code, text in phrases.items()}
        for tag, phrases in loaded.items()
        if isinstance(phrases, dict) and not str(tag).startswith("_")
    }


def _localized(code: ErrorCode, language: str | None) -> str | None:
    tag = str(language or "").strip()
    if not tag:
        return None
    catalog = _catalog()
    # Exact tag first, then the base language, so a caller on es-MX still gets
    # Spanish from the es-ES bundle rather than falling back to English.
    base = tag.split("-", 1)[0].lower()
    for key in (tag, tag.lower(), base):
        phrases = catalog.get(key)
        if phrases is None:
            phrases = next(
                (v for k, v in catalog.items() if k.split("-", 1)[0].lower() == base),
                None,
            )
        text = (phrases or {}).get(code.value, "").strip()
        if text:
            return text
    return None


def refuse_speech(code: ErrorCode | str, *, language: str = "en") -> str:
    """Return the refuse line for ``code``, in ``language`` when it is translated.

    Falls back to English rather than failing: a missing translation must still
    produce a spoken refusal, because silence after a failed lookup reads as the
    call having dropped.
    """
    if isinstance(code, str):
        try:
            code = ErrorCode(code)
        except ValueError:
            code = ErrorCode.UNSUPPORTED
    if code not in _REFUSE_EN:
        code = ErrorCode.UNSUPPORTED
    return _localized(code, language) or _REFUSE_EN[code]


def permission_refuse(*, detail: str = "missing user access token") -> ErrorEvidence:
    return ErrorEvidence(
        code=ErrorCode.PERMISSION,
        message=detail,
        retryable=False,
    )


def no_evidence_refuse(*, detail: str = "empty tabular result") -> ErrorEvidence:
    return ErrorEvidence(
        code=ErrorCode.NO_EVIDENCE,
        message=detail,
        retryable=True,
    )
