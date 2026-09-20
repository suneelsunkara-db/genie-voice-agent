"""Typed evidence and provenance metadata for tool results.

Evidence supports tables, citations, traces, and follow-up diagnostics. It does
not decide which natural-language answer the customer may see or hear; that answer
comes from Genie/Agent Mode or the conversational response rendering boundary.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .refuse import ErrorEvidence


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
    """A narrative answer carrying upstream provenance."""

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
    """Structured tool result plus optional upstream natural-language prose."""

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
        """Return the typed evidence in its wire-safe representation."""
        return {
            "source": self.source,
            "table": self.table.as_dict() if self.table else None,
            "prose": self.prose.as_dict() if self.prose else None,
            "display_prose": self.display_prose,
            "error": self.error.as_dict() if self.error else None,
            "meta": dict(self.meta),
        }
