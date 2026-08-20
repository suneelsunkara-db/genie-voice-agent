"""Map Genie payloads → Evidence.

Each adapter decides what its source may speak, because only it knows the source's
semantics:

  Genie Space / Agent Mode — numeric lanes. Their prose is model-written narration
    OVER query results, so a number could appear in the prose that no cell backs.
    Prose stays display-only; speech comes from the tables.
  Genie One — the governed answering service itself. Its answer arrives under a
    conversation/response id, and many legitimate questions ("what can you answer
    for me", "what are the limits") have no tabular answer at all. Its prose is
    therefore attributed and speakable, cited by that id, whether or not rows came
    back with it; when rows are present they are the cell-level cite for numbers.
"""
from __future__ import annotations

import re
from typing import Any

from .evidence import Evidence, ProseEvidence, TableEvidence
from .refuse import ErrorCode, ErrorEvidence, no_evidence_refuse


def _normalize_rows(rows: Any, columns: list[str]) -> list[list[Any]]:
    if not rows:
        return []
    out: list[list[Any]] = []
    for row in rows:
        if isinstance(row, dict):
            out.append([row.get(c) for c in columns])
        elif isinstance(row, (list, tuple)):
            out.append(list(row))
        else:
            out.append([row])
    return out


def evidence_from_genie_space(result: dict[str, Any] | None) -> Evidence:
    """Genie Conversation API result → Evidence.

    Uses ``columns`` + ``rows`` (same shape as ``GenieClient._query_result_rows``).
    ``answer`` / ``description`` become ``display_prose`` only.
    """
    if not result:
        return Evidence(
            source="genie_space",
            error=no_evidence_refuse(detail="empty Genie result"),
        )

    columns = [str(c) for c in (result.get("columns") or []) if c is not None]
    raw_rows = result.get("rows")
    rows = _normalize_rows(raw_rows, columns)
    prose = result.get("answer") or result.get("description")
    prose_s = str(prose).strip() if prose else None

    if not columns or not rows:
        return Evidence(
            source="genie_space",
            display_prose=prose_s,
            error=ErrorEvidence(
                code=ErrorCode.NO_EVIDENCE,
                message="Genie returned no tabular result",
                retryable=True,
            ),
            meta={"sql": result.get("sql"), "conversation_id": result.get("conversation_id")},
        )

    table = TableEvidence(
        columns=columns,
        rows=rows,
        sql=str(result["sql"]) if result.get("sql") else None,
        citations=[f"genie_space:{result.get('message_id') or 'unknown'}"],
    )
    return Evidence(
        source="genie_space",
        table=table,
        display_prose=prose_s,
        meta={"conversation_id": result.get("conversation_id")},
    )


_PROSE_KEYS = (
    # `final_answer` is what the managed Genie One MCP server actually returns; the
    # rest are the shapes seen from the Conversation API and older MCP builds.
    "final_answer",
    "answer",
    "response",
    "text",
    "content",
    "message",
    "summary",
    "report",
    "description",
)


def _prose_from_genie_one(result: dict[str, Any]) -> str | None:
    """First non-empty answer text, across the shapes the MCP server may return.

    Deliberately tolerant: the managed server has returned the answer as a string, a
    nested object, and a list of content parts, and an adapter that only knew one of
    those would silently downgrade a good answer into a refusal.
    """
    for key in _PROSE_KEYS:
        value = result.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, dict):
            for nested in _PROSE_KEYS:
                inner = value.get(nested)
                if isinstance(inner, str) and inner.strip():
                    return inner.strip()
        if isinstance(value, list):
            parts: list[str] = []
            for item in value:
                if isinstance(item, str) and item.strip():
                    parts.append(item.strip())
                elif isinstance(item, dict):
                    for nested in _PROSE_KEYS:
                        inner = item.get(nested)
                        if isinstance(inner, str) and inner.strip():
                            parts.append(inner.strip())
                            break
            if parts:
                return "\n".join(parts)
    return None


def _statement_table(payload: dict[str, Any]) -> tuple[list[str], list[list[Any]]]:
    """Columns + rows out of a SQL-statement-shaped result block."""
    manifest = payload.get("manifest")
    schema = manifest.get("schema") if isinstance(manifest, dict) else None
    raw_columns = (schema or {}).get("columns") if isinstance(schema, dict) else None
    columns = [
        str(c["name"])
        for c in (raw_columns or [])
        if isinstance(c, dict) and c.get("name")
    ]
    if not columns:
        columns = [str(c) for c in (payload.get("columns") or []) if c is not None]
    data = payload.get("data_array") or payload.get("data") or payload.get("rows") or []
    return columns, _normalize_rows(data, columns)


_MEASURE_TYPE = re.compile(
    r"^\s*(tinyint|smallint|int|integer|bigint|long|float|double|real|decimal|numeric)\b",
    re.I,
)


def _is_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    try:
        float(str(value).replace(",", "").replace("$", "").strip())
    except (TypeError, ValueError):
        return False
    return True


def _has_measure(raw_columns: Any, rows: list[list[Any]]) -> bool:
    """Whether a result carries a quantity, by declared type or, absent one, by value.

    Genie reaches an answer in steps: it resolves how an entity is spelled in the
    data, then aggregates. Every step comes back as a completed query result, so
    "which result answers the question" cannot be "the first one" — it has to be the
    one that carries a measure.
    """
    for index, column in enumerate(raw_columns or []):
        type_text = str(column.get("type_text") or "") if isinstance(column, dict) else ""
        if type_text:
            if _MEASURE_TYPE.match(type_text):
                return True
            continue
        values = [row[index] for row in rows if index < len(row) and row[index] is not None]
        if values and all(_is_number(value) for value in values):
            return True
    return False


def _genie_one_table(result: dict[str, Any]) -> tuple[list[str], list[list[Any]], str | None]:
    def _names(raw: Any) -> list[str]:
        return [
            str(c.get("name") if isinstance(c, dict) else c)
            for c in (raw or [])
            if (isinstance(c, dict) and c.get("name"))
            or (not isinstance(c, dict) and c is not None)
        ]

    columns = _names(result.get("columns"))
    rows = _normalize_rows(result.get("rows"), columns)
    if columns and rows:
        return columns, rows, None
    query_results = result.get("query_results")
    if isinstance(query_results, list):
        fallback: tuple[list[str], list[list[Any]], str | None] | None = None
        for query_result in query_results:
            if not isinstance(query_result, dict):
                continue
            raw_columns = query_result.get("columns")
            columns = _names(raw_columns)
            rows = _normalize_rows(query_result.get("rows"), columns)
            if not columns or not rows:
                continue
            sql = query_result.get("sql")
            candidate = (columns, rows, str(sql) if sql else None)
            if _has_measure(raw_columns, rows):
                return candidate
            if fallback is None:
                fallback = candidate
        if fallback is not None:
            return fallback
    for key in ("query_result", "statement_response", "result", "table"):
        block = result.get(key)
        if isinstance(block, dict):
            columns, rows = _statement_table(block)
            if columns and rows:
                return columns, rows, None
    return [], [], None


def evidence_from_genie_one(result: dict[str, Any] | None) -> Evidence:
    """Genie One MCP result → Evidence, with its answer attributed to the response.

    A table, when Genie One returns one, is the speakable evidence. Otherwise its
    answer text is attributed prose cited by the conversation/response id, so
    descriptive questions get a real answer instead of a refusal.
    """
    if not result:
        return Evidence(
            source="genie_one",
            error=no_evidence_refuse(detail="empty Genie One result"),
        )

    status = str(result.get("status") or "").lower()
    cite_id = (
        result.get("message_id")
        or result.get("response_id")
        or result.get("conversation_id")
    )
    meta = {
        "conversation_id": result.get("conversation_id"),
        "response_id": result.get("response_id"),
        "status": status or None,
        # Genie One's own thread URL: the caller can open the exact conversation the
        # spoken answer came from, which is stronger provenance than an id.
        "deep_link": result.get("deep_link"),
        # Bounded typed rows fetched through genie_get_query_result. The first table
        # remains the speech evidence above; all tables are available to structured
        # UI renderers without scraping Markdown.
        "query_results": result.get("query_results") or [],
    }

    if result.get("timeout"):
        return Evidence(
            source="genie_one",
            error=ErrorEvidence(
                code=ErrorCode.TIMEOUT,
                message="Genie One did not answer in time",
                retryable=True,
            ),
            meta=meta,
        )
    if status in {"failed", "cancelled"}:
        return Evidence(
            source="genie_one",
            error=ErrorEvidence(
                code=ErrorCode.UNSUPPORTED,
                message=f"Genie One response {status}",
                retryable=True,
            ),
            meta=meta,
        )

    prose_text = _prose_from_genie_one(result)
    columns, rows, table_sql = _genie_one_table(result)
    sql = table_sql or result.get("sql") or result.get("query")
    prose = (
        ProseEvidence(text=prose_text, citations=[f"genie_one:{cite_id}"])
        if prose_text and cite_id
        else None
    )

    if columns and rows:
        return Evidence(
            source="genie_one",
            table=TableEvidence(
                columns=columns,
                rows=rows,
                sql=str(sql) if sql else None,
                citations=[f"genie_one:{cite_id or 'unknown'}"],
            ),
            # Genie One's narrative is that service's OWN answer under this response
            # id, not narration we wrote over its rows, so it stays attributed even
            # when a table came back. The cells remain the cite for spoken numbers.
            prose=prose,
            display_prose=None if prose is not None else prose_text,
            meta=meta,
        )

    if prose is not None:
        return Evidence(source="genie_one", prose=prose, meta=meta)

    return Evidence(
        source="genie_one",
        # Unattributed: no id came back, so this text cannot be spoken as an answer.
        display_prose=prose_text,
        error=no_evidence_refuse(detail="Genie One returned nothing citable"),
        meta=meta,
    )


def evidence_from_agent_mode(
    tables: list[dict[str, Any]] | None,
    *,
    report_text: str | None = None,
) -> Evidence:
    """Agent Mode structured tables → Evidence. Report text is display-only."""
    prose = (report_text or "").strip() or None
    if not tables:
        return Evidence(
            source="agent_mode",
            display_prose=prose,
            error=no_evidence_refuse(detail="Agent Mode returned no tables"),
        )

    # Prefer the first table with both columns and rows.
    for idx, t in enumerate(tables):
        columns = [str(c) for c in (t.get("columns") or []) if c is not None]
        raw = t.get("preview_rows") if t.get("preview_rows") is not None else t.get("rows")
        rows = _normalize_rows(raw, columns)
        if columns and rows:
            table = TableEvidence(
                columns=columns,
                rows=rows,
                sql=str(t["sql"]) if t.get("sql") else None,
                citations=[f"agent_mode.table[{idx}]"],
            )
            return Evidence(
                source="agent_mode",
                table=table,
                display_prose=prose,
                meta={"table_index": idx, "table_count": len(tables)},
            )

    return Evidence(
        source="agent_mode",
        display_prose=prose,
        error=ErrorEvidence(
            code=ErrorCode.NO_EVIDENCE,
            message="Agent Mode tables lacked columns/rows",
            retryable=True,
        ),
    )
