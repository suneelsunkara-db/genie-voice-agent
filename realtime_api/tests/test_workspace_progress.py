"""A minutes-long governed read is only tolerable if the caller can see it moving.

Genie reports internal steps while it works. Their field names are not a contract we
own and their text can contain reasoning, SQL, table names, and raw results. These
tests pin the business-safe normalization used by both TTS and the timeline.
"""
from __future__ import annotations

from realtime_api.runtime.genie_one import (
    current_progress_step,
    normalize_progress_steps,
)


def test_internal_trace_becomes_business_facing_stages():
    steps = normalize_progress_steps(
        [
            {
                "title": "Thinking: inspect system.ai_gateway.usage destination_model",
                "status": "COMPLETED",
            },
            {
                "title": "Running SQL: SELECT DISTINCT destination_model FROM system.ai_gateway.usage",
                "status": "completed",
            },
            {
                "title": "Query result: Query returned 6 rows with qwen3-next-80b",
                "status": "running",
            },
            {"title": "Formatting the answer", "status": "queued"},
        ]
    )
    assert [step["label"] for step in steps] == [
        "Analyzing usage",
        "Matching the requested item",
        "Checking the results",
        "Preparing your answer",
    ]
    assert [step["status"] for step in steps] == ["done", "done", "active", "pending"]
    assert current_progress_step(steps) == "Checking the results"


def test_steps_without_status_treat_the_latest_as_in_flight():
    steps = normalize_progress_steps(
        [{"name": "Thinking about the request"}, {"name": "Searching available sources"}]
    )
    assert [step["status"] for step in steps] == ["done", "active"]
    assert current_progress_step(steps) == "Finding the relevant information"


def test_plain_strings_and_a_bare_string_are_both_accepted():
    assert normalize_progress_steps(["Looking up billing usage"]) == [
        {"label": "Calculating the amount", "status": "active"}
    ]
    assert normalize_progress_steps("Scanning tables") == [
        {"label": "Finding the relevant information", "status": "active"}
    ]


def test_an_unrecognised_shape_degrades_to_no_steps():
    """The caller still gets the generic cadence; nothing crashes the turn."""
    assert normalize_progress_steps(None) == []
    assert normalize_progress_steps({"unexpected": "object"}) == []
    assert normalize_progress_steps([{"id": 7}, {}, ""]) == []
    assert current_progress_step([]) == ""


def test_repeated_internal_operations_collapse_into_one_business_stage():
    steps = normalize_progress_steps(
        [
            "Running SQL: SELECT * FROM private.table_a",
            "Running query against private.table_b",
            "Searching another schema",
        ]
    )
    assert steps == [
        {"label": "Finding the relevant information", "status": "active"}
    ]


def test_raw_sql_reasoning_results_and_identifiers_never_reach_the_wire():
    raw = [
        "Thinking: use `system`.`billing`.`usage` to find the SKU",
        "Running SQL: SELECT SUM(list_cost) FROM system.billing.usage",
        "Query result: returned 7 rows for ENTERPRISE_GEMINI_MODEL_SERVING",
    ]
    steps = normalize_progress_steps(raw)
    rendered = " ".join(step["label"] for step in steps).lower()

    for forbidden in (
        "thinking",
        "sql",
        "select",
        "system.",
        "billing.usage",
        "rows",
        "enterprise_gemini",
        "databricks",
    ):
        assert forbidden not in rendered
