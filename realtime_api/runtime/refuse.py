"""Typed ErrorEvidence codes and refuse-speech templates.

Speech after failure comes from these templates only — never model improvisation
after an ack when Evidence is empty or permission is denied.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


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


# Short English refuse shells. Callers may localize via the existing phrase helper;
# literals (codes) stay English.
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


def refuse_speech(code: ErrorCode | str, *, language: str = "en") -> str:
    """Return the refuse template for ``code``. ``language`` reserved for i18n warm path."""
    _ = language  # templates are English shells; phrase warm can localize later
    if isinstance(code, str):
        try:
            code = ErrorCode(code)
        except ValueError:
            code = ErrorCode.UNSUPPORTED
    return _REFUSE_EN.get(code, _REFUSE_EN[ErrorCode.UNSUPPORTED])


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
