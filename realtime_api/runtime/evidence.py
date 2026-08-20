"""Evidence lane — cite-or-silence SpokenClaims.

Cite-or-silence is a rule about ATTRIBUTION, not about shape. Two things are
speakable: cells of a structured table (cited by field path), and a narrative
answer an authorized upstream service returned under its own message id (cited by
that id). Everything else — above all our own model's prose — is display-only and
can never become a spoken claim.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .refuse import ErrorCode, ErrorEvidence, refuse_speech


@dataclass(frozen=True)
class TableEvidence:
    columns: list[str]
    rows: list[list[Any]]
    sql: str | None = None
    as_of: str | None = None
    citations: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "columns": list(self.columns),
            "rows": [list(row) for row in self.rows],
            "sql": self.sql,
            "as_of": self.as_of,
            "citations": list(self.citations),
        }


@dataclass(frozen=True)
class ProseEvidence:
    """A narrative answer an authorized upstream service returned, with its cite.

    Deliberately distinct from ``display_prose``: this text was produced BY a
    governed system the caller is authorized against, and carries that system's own
    identifier, so it is attributable and therefore speakable. Only an adapter that
    knows its source's semantics may fill this in — text our own model wrote stays
    ``display_prose`` forever.

    ``citations`` is the gate: prose with no upstream id is not attributed prose.
    """

    text: str
    citations: list[str] = field(default_factory=list)
    as_of: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "citations": list(self.citations),
            "as_of": self.as_of,
        }


@dataclass
class Evidence:
    """Structured truth object.

    ``table`` and ``prose`` are speakable (each carries its own cite). Only
    ``display_prose`` is barred from speech.
    """

    source: str
    table: TableEvidence | None = None
    prose: ProseEvidence | None = None  # attributed upstream answer — speakable
    display_prose: str | None = None  # our own/unattributed text — UI only
    error: ErrorEvidence | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def has_tabular(self) -> bool:
        return bool(self.table and self.table.columns and self.table.rows)

    @property
    def has_attributed_prose(self) -> bool:
        return bool(self.prose and self.prose.text.strip() and self.prose.citations)

    @property
    def is_error(self) -> bool:
        return self.error is not None

    def as_dict(self) -> dict[str, Any]:
        """Wire-safe evidence. Display prose is explicitly labeled non-speakable."""
        return {
            "source": self.source,
            "table": self.table.as_dict() if self.table else None,
            "prose": self.prose.as_dict() if self.prose else None,
            "display_prose": self.display_prose,
            "error": self.error.as_dict() if self.error else None,
            "meta": dict(self.meta),
        }


_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_MD_BULLET = re.compile(r"^\s*[-*\u2022]\s+", re.M)
_MD_NOISE = re.compile(r"[*_`#>|]+")


def speakable_prose(text: str, *, max_chars: int = 600) -> str:
    """Markdown answer → something a TTS voice can read without artifacts.

    Drops markdown syntax and truncates at a sentence boundary. The full answer
    still reaches the screen; only what the voice reads aloud is trimmed, because a
    voice turn that recites a 3,000-character report cannot be interrupted usefully.
    """
    cleaned = _MD_LINK.sub(r"\1", text or "")
    cleaned = _MD_BULLET.sub("", cleaned)
    cleaned = _MD_NOISE.sub("", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    cut = cleaned[:max_chars]
    for stop in (". ", "! ", "? ", "; "):
        idx = cut.rfind(stop)
        if idx > max_chars * 0.5:
            return cut[: idx + 1].strip()
    return cut.rsplit(" ", 1)[0].strip() + "\u2026"


@dataclass(frozen=True)
class SpokenClaim:
    """A speakable sentence that MUST cite Evidence field paths."""

    text: str
    field_paths: tuple[str, ...]
    durable: bool = False

    def __post_init__(self) -> None:
        if not self.field_paths:
            raise ValueError("SpokenClaim requires at least one field_path cite")
        if not (self.text or "").strip():
            raise ValueError("SpokenClaim text must be non-empty")


class EvidenceComposer:
    """Cite-or-silence: reject any claim that does not cite Evidence.

    Table cells cite a field path; an attributed upstream answer cites that
    upstream's id. Unattributed prose is never converted into SpokenClaims, and
    evidence with neither shape refuses.
    """

    #: The single cite a prose claim carries: the upstream answer itself.
    PROSE_PATH = "prose.text"

    def __init__(self) -> None:
        self.citation_reject_count = 0
        self.evidence_empty_refuse = 0

    def spoken_claims_from_table(
        self,
        evidence: Evidence,
        *,
        max_rows: int = 3,
        max_columns: int = 6,
        language: str = "en",
    ) -> list[SpokenClaim] | ErrorEvidence:
        """Build claims from tabular cells only; never from ``display_prose``."""
        if evidence.error is not None:
            return evidence.error
        if not evidence.has_tabular or evidence.table is None:
            self.evidence_empty_refuse += 1
            return ErrorEvidence(
                code=ErrorCode.NO_EVIDENCE,
                message="No tabular result to cite",
                retryable=True,
            )

        table = evidence.table
        claims: list[SpokenClaim] = []
        for r_idx, row in enumerate(table.rows[:max_rows]):
            parts: list[str] = []
            paths: list[str] = []
            for c_idx, col in enumerate(table.columns[:max_columns]):
                if c_idx >= len(row):
                    break
                cell = row[c_idx]
                if cell is None or (isinstance(cell, str) and not cell.strip()):
                    continue
                path = f"table.rows[{r_idx}].{col}"
                label = str(col).replace("_", " ").replace(".", " ")
                parts.append(f"{label}: {cell}")
                paths.append(path)
            if not paths:
                continue
            text = "; ".join(parts)
            try:
                claims.append(SpokenClaim(text=text, field_paths=tuple(paths), durable=False))
            except ValueError:
                self.citation_reject_count += 1

        if not claims:
            self.evidence_empty_refuse += 1
            return ErrorEvidence(
                code=ErrorCode.NO_EVIDENCE,
                message="Tabular result had no speakable cells",
                retryable=True,
            )
        return claims

    def spoken_claims_from_prose(
        self,
        evidence: Evidence,
        *,
        max_chars: int = 600,
        language: str = "en",
    ) -> list[SpokenClaim] | ErrorEvidence:
        """Build ONE claim from prose an authorized upstream returned.

        This is what makes questions whose honest answer is a sentence — capability,
        policy, explanation — answerable without loosening attribution: the claim
        cites the upstream message it came from, so it is still traceable.
        """
        if evidence.error is not None:
            return evidence.error
        if not evidence.has_attributed_prose or evidence.prose is None:
            self.evidence_empty_refuse += 1
            return ErrorEvidence(
                code=ErrorCode.NO_EVIDENCE,
                message="No attributed upstream answer to cite",
                retryable=True,
            )
        text = speakable_prose(evidence.prose.text, max_chars=max_chars)
        if not text:
            self.evidence_empty_refuse += 1
            return ErrorEvidence(
                code=ErrorCode.NO_EVIDENCE,
                message="Upstream answer had no speakable text",
                retryable=True,
            )
        return [SpokenClaim(text=text, field_paths=(self.PROSE_PATH,), durable=False)]

    def validate_claim(self, claim: SpokenClaim, evidence: Evidence) -> bool:
        """Reject claims that lack cites or cite paths absent from Evidence."""
        if not claim.field_paths:
            self.citation_reject_count += 1
            return False
        # A prose claim cites the upstream answer itself, which only holds when that
        # answer actually arrived with the upstream's own id attached.
        if self.PROSE_PATH in claim.field_paths:
            if len(claim.field_paths) != 1 or not evidence.has_attributed_prose:
                self.citation_reject_count += 1
                return False
            return True
        if not evidence.has_tabular or evidence.table is None:
            self.citation_reject_count += 1
            return False
        cols = set(evidence.table.columns)
        for path in claim.field_paths:
            # Expected shape: table.rows[N].<column>
            marker = "]."
            col = path.split(marker, 1)[1] if path and marker in path else ""
            if col not in cols:
                self.citation_reject_count += 1
                return False
        return True

    def refuse_for(self, error: ErrorEvidence, *, language: str = "en") -> str:
        return refuse_speech(error.code, language=language)

    def compose_or_refuse(
        self,
        evidence: Evidence,
        *,
        language: str = "en",
    ) -> tuple[list[SpokenClaim], str | None]:
        """Return (claims, None) or ([], refuse_text). Never speaks display prose."""
        # Explicitly ignore display_prose — cite-or-silence.
        _ = evidence.display_prose
        # Numbers first: a table cites individual cells, which is a stronger claim
        # than a narrative. Attributed prose is the fallback, and a table-less,
        # prose-less result still lands on the tabular refusal path (which surfaces
        # any error, e.g. a permission denial, ahead of "no evidence").
        if not evidence.has_tabular and evidence.has_attributed_prose:
            result = self.spoken_claims_from_prose(evidence, language=language)
        else:
            result = self.spoken_claims_from_table(evidence, language=language)
        if isinstance(result, ErrorEvidence):
            return [], self.refuse_for(result, language=language)
        validated = [c for c in result if self.validate_claim(c, evidence)]
        if not validated:
            self.evidence_empty_refuse += 1
            err = ErrorEvidence(
                code=ErrorCode.NO_EVIDENCE,
                message="All candidate claims failed citation validation",
                retryable=True,
            )
            return [], self.refuse_for(err, language=language)
        return validated, None
