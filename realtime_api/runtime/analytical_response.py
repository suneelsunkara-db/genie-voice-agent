"""Use-case-neutral analytical result contract for realtime surfaces.

The contract hardcodes only safe presentation primitives—not business intents.
Forecasts, classifications, comparisons, and future analyses all travel as an
answer plus typed datasets. Optional views may be added by a governed upstream
source later; absent a validated view, clients fall back to a table.
"""
from __future__ import annotations

import re
from typing import Any

_NUMERIC_TYPE = re.compile(
    r"(tiny|small|big)?int|decimal|double|float|number|numeric|real",
    re.IGNORECASE,
)
_TEMPORAL_TYPE = re.compile(r"date|timestamp|time", re.IGNORECASE)


def _planned_views(
    question: str,
    datasets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Select conservative question-titled views from typed field semantics."""
    title = " ".join(question.strip().split())[:160] or None
    planned: list[dict[str, Any]] = []
    for dataset in datasets:
        if len(dataset["rows"]) < 2 or len(dataset["rows"]) > 40:
            continue
        fields = dataset["fields"]
        numeric = [
            field["name"]
            for field in fields
            if _NUMERIC_TYPE.search(field.get("type") or "")
        ]
        temporal = [
            field["name"]
            for field in fields
            if _TEMPORAL_TYPE.search(field.get("type") or "")
        ]
        categorical = [
            field["name"]
            for field in fields
            if field["name"] not in numeric and field["name"] not in temporal
        ]
        if len(temporal) == 1 and not categorical and numeric:
            planned.append(
                {
                    "dataset_id": dataset["id"],
                    "kind": "line",
                    "dimension": temporal[0],
                    "measures": numeric[:3],
                    "title": title,
                }
            )
        elif not temporal and len(categorical) == 1 and numeric:
            planned.append(
                {
                    "dataset_id": dataset["id"],
                    "kind": "bar",
                    "dimension": categorical[0],
                    "measures": numeric[:3],
                    "title": title,
                }
            )
    return planned


def analytical_response(
    *,
    source: str,
    answer: str | None,
    query_results: list[dict[str, Any]] | None,
    question: str = "",
    views: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    datasets: list[dict[str, Any]] = []
    for index, result in enumerate(query_results or []):
        if not isinstance(result, dict):
            continue
        raw_columns = result.get("columns")
        rows = result.get("rows")
        if not isinstance(raw_columns, list) or not isinstance(rows, list):
            continue
        fields: list[dict[str, str]] = []
        for column in raw_columns:
            if isinstance(column, dict):
                name = str(column.get("name") or "").strip()
                type_text = str(column.get("type_text") or "")
            else:
                name = str(column or "").strip()
                type_text = ""
            if name:
                fields.append({"name": name, "type": type_text})
        if not fields:
            continue
        datasets.append(
            {
                "id": str(result.get("item_id") or f"dataset_{index + 1}"),
                "fields": fields,
                "rows": [list(row) for row in rows if isinstance(row, (list, tuple))],
                "row_count": int(result.get("total_row_count") or len(rows)),
                "truncated": bool(result.get("truncated")),
                "sql": str(result.get("sql")) if result.get("sql") else None,
                # Upstream data is evidence until an explicit, validated view says
                # otherwise. Clients must not guess a business meaning from a name.
                "role": "evidence",
            }
        )
    fields_by_dataset = {
        dataset["id"]: {field["name"] for field in dataset["fields"]}
        for dataset in datasets
    }
    candidates = views if views is not None else _planned_views(question, datasets)
    validated_views: list[dict[str, Any]] = []
    for view in candidates:
        if not isinstance(view, dict):
            continue
        dataset_id = str(view.get("dataset_id") or "")
        kind = str(view.get("kind") or "")
        dimension = str(view.get("dimension") or "")
        measures = view.get("measures")
        fields = fields_by_dataset.get(dataset_id)
        if (
            fields is None
            or kind not in {"bar", "line"}
            or dimension not in fields
            or not isinstance(measures, list)
        ):
            continue
        clean_measures = [
            str(measure)
            for measure in measures
            if str(measure) in fields and str(measure) != dimension
        ][:3]
        if not clean_measures:
            continue
        validated_views.append(
            {
                "dataset_id": dataset_id,
                "kind": kind,
                "dimension": dimension,
                "measures": clean_measures,
                "title": str(view.get("title") or "").strip() or None,
            }
        )
    return {
        "version": "1",
        "source": source,
        "status": "complete" if (answer or datasets) else "unsupported",
        "answer": (answer or "").strip(),
        "datasets": datasets,
        "views": validated_views,
    }
