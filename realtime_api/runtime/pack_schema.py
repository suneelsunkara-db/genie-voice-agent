"""Wrap legacy pack tool JSON into Evidence (cite-or-silence path)."""
from __future__ import annotations

from typing import Any

from .evidence import Evidence, TableEvidence
from .refuse import ErrorCode, ErrorEvidence, no_evidence_refuse, permission_refuse

_DISPLAY_ONLY_KEYS = {
    "answer",
    "description",
    "report",
    "refuse",
    "guidance",
}


def _rows_from_dict_list(items: list[Any]) -> tuple[list[str], list[list[Any]]] | None:
    records = [item for item in items if isinstance(item, dict)]
    if not records:
        return None
    columns: list[str] = []
    for record in records:
        for key, value in record.items():
            if key not in columns and key not in _DISPLAY_ONLY_KEYS and not isinstance(value, (dict, list)):
                columns.append(str(key))
    # Curated knowledge entries deliberately carry answer + citation as fields.
    if all("answer" in record and "citation" in record for record in records):
        for key in ("topic", "answer", "citation"):
            if key not in columns:
                columns.append(key)
    if not columns:
        return None
    return columns, [[record.get(column) for column in columns] for record in records]


def _flatten_scalars(value: Any, *, prefix: str = "") -> dict[str, Any]:
    """Flatten legacy tool JSON into citeable scalar field paths.

    This is the compatibility adapter promised by the L200 plan. It lets existing
    pack tools enter the one Evidence lane without making prose a source of truth.
    """
    out: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key) in _DISPLAY_ONLY_KEYS or str(key) in {"error_evidence", "obo_deny"}:
                continue
            path = f"{prefix}.{key}" if prefix else str(key)
            out.update(_flatten_scalars(child, prefix=path))
    elif isinstance(value, list):
        # Lists of objects are handled as real tables by the caller. Scalar lists
        # remain one citeable value rather than exploding into unstable columns.
        if value and all(not isinstance(item, (dict, list)) for item in value):
            out[prefix] = ", ".join(str(item) for item in value)
    elif prefix and value is not None:
        out[prefix] = value
    return out


def evidence_from_tool_result(
    name: str,
    result: Any,
    *,
    source: str | None = None,
) -> Evidence:
    """Map common tool JSON shapes to Evidence.

    Recognizes:
      - ``{columns, rows}`` / ``{columns, data}``
      - ``{answer, columns, rows}`` (Genie Space — prose is display-only)
      - ``{tables: [{columns, rows}, ...]}`` (Agent Mode)
      - empty / missing → ErrorEvidence.NO_EVIDENCE
    """
    src = source or f"tool:{name}"
    if not isinstance(result, dict):
        return Evidence(source=src, error=no_evidence_refuse(detail="non-object tool result"))

    # Failures are classified BEFORE any per-tool mapping: a denied OBO call must
    # surface as a permission refusal, not as "no evidence" from an empty result.
    if result.get("error") or result.get("denied"):
        detail = str(result.get("error") or result.get("message") or "denied")
        err = (
            permission_refuse(detail=detail)
            if result.get("denied")
            else ErrorEvidence(code=ErrorCode.UNSUPPORTED, message=detail, retryable=True)
        )
        return Evidence(
            source=src,
            display_prose=detail,
            error=err,
        )

    if name == "workspace_query":
        from .genie_adapters import evidence_from_genie_one

        return evidence_from_genie_one(result)

    tables = result.get("tables")
    if isinstance(tables, list) and tables:
        if name == "start_deep_dive":
            from .genie_adapters import evidence_from_agent_mode

            return evidence_from_agent_mode(
                tables,
                report_text=str(result.get("report") or ""),
            )
        first = tables[0] if isinstance(tables[0], dict) else {}
        cols = list(first.get("columns") or [])
        rows = list(first.get("rows") or first.get("data") or [])
        prose = result.get("report") or result.get("answer")
        if cols and rows:
            return Evidence(
                source=src,
                table=TableEvidence(columns=[str(c) for c in cols], rows=rows),
                display_prose=str(prose) if prose else None,
            )

    for value in result.values():
        if isinstance(value, list):
            tabular = _rows_from_dict_list(value)
            if tabular:
                list_columns, list_rows = tabular
                return Evidence(
                    source=src,
                    table=TableEvidence(
                        columns=list_columns,
                        rows=list_rows,
                        citations=[src],
                    ),
                    display_prose=str(result.get("answer") or result.get("report") or "") or None,
                )

    cols = result.get("columns")
    rows = result.get("rows") if "rows" in result else result.get("data")
    prose = result.get("answer")
    if isinstance(cols, list) and isinstance(rows, list) and cols and rows:
        return Evidence(
            source=src,
            table=TableEvidence(columns=[str(c) for c in cols], rows=list(rows)),
            display_prose=str(prose) if prose else None,
        )

    scalar_fields = _flatten_scalars(result)
    if scalar_fields:
        return Evidence(
            source=src,
            table=TableEvidence(
                columns=list(scalar_fields),
                rows=[list(scalar_fields.values())],
                citations=[src],
            ),
            display_prose=str(prose) if prose else None,
        )

    if prose:
        # Prose without tables — display only; composer will refuse speech.
        return Evidence(source=src, display_prose=str(prose))

    return Evidence(source=src, error=no_evidence_refuse(detail="empty tool result"))
