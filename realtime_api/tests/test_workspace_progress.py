"""A minutes-long governed read is only tolerable if the caller can see it moving.

Genie reports internal steps while it works. Their field names are not a contract we
own and their text can contain reasoning, SQL, table names, and raw results. These
tests pin the business-safe pipeline used by both TTS and the on-screen timeline:
one ordered set of phases, always rendered in order, only ever moving forward.
"""
from __future__ import annotations

from realtime_api.runtime.genie_one import (
    current_progress_step,
    normalize_progress_steps,
)

PIPELINE = [
    "Understanding your question",
    "Finding the right data",
    "Running the analysis",
    "Checking the results",
    "Preparing your answer",
]

# The page renders these codes from its own message catalog, so they are a
# contract: renaming one silently drops that phase back to English on screen.
PIPELINE_CODES = [
    "understanding",
    "finding_data",
    "running_analysis",
    "checking_results",
    "preparing_answer",
]


def test_every_phase_travels_as_a_stable_code_the_ui_can_localize():
    steps = normalize_progress_steps(["Considering how to approach this"])
    assert [step["code"] for step in steps] == PIPELINE_CODES
    # The English label rides along for the spoken narration prompt and as the
    # fallback for a UI whose catalog does not know the code yet.
    assert [step["label"] for step in steps] == PIPELINE


def test_the_timeline_is_the_pipeline_in_order_with_one_phase_in_flight():
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
        ]
    )
    assert [step["label"] for step in steps] == PIPELINE
    assert [step["status"] for step in steps] == [
        "done",
        "done",
        "done",
        "active",
        "pending",
    ]
    assert current_progress_step(steps) == "Checking the results"


def test_a_step_naming_several_things_lands_on_the_furthest_phase_it_evidences():
    """"Running SQL: SELECT SUM(...)" is analysis, not still looking for the data."""
    steps = normalize_progress_steps(
        ["Running SQL: SELECT SUM(list_cost) FROM system.billing.usage"]
    )
    assert current_progress_step(steps) == "Running the analysis"


def test_finishing_the_last_reported_phase_moves_the_timeline_on():
    working = normalize_progress_steps([{"name": "Formatting the answer", "status": "running"}])
    assert current_progress_step(working) == "Preparing your answer"
    # Nothing follows the final phase, so a completed last step stays there rather
    # than claiming a phase that does not exist.
    finished = normalize_progress_steps(
        [{"name": "Formatting the answer", "status": "completed"}]
    )
    assert current_progress_step(finished) == "Preparing your answer"

    advanced = normalize_progress_steps(
        [{"name": "Searching available sources", "status": "completed"}]
    )
    assert current_progress_step(advanced) == "Running the analysis"


def test_the_timeline_never_walks_the_caller_backwards():
    """Upstream revisits earlier work; un-completing a done phase reads as a bug."""
    steps = normalize_progress_steps(
        [
            "Query result: returned 12 rows",
            "Running another query against a second table",
        ]
    )
    assert current_progress_step(steps) == "Checking the results"


def test_unrecognised_text_is_still_early_reasoning():
    steps = normalize_progress_steps(["Considering how to approach this"])
    assert current_progress_step(steps) == "Understanding your question"
    assert [step["status"] for step in steps] == [
        "active",
        "pending",
        "pending",
        "pending",
        "pending",
    ]


def test_plain_strings_and_a_bare_string_are_both_accepted():
    assert current_progress_step(normalize_progress_steps(["Scanning tables"])) == (
        "Finding the right data"
    )
    assert current_progress_step(normalize_progress_steps("Scanning tables")) == (
        "Finding the right data"
    )


def test_an_unrecognised_shape_degrades_to_no_steps():
    """The caller still gets the generic cadence; nothing crashes the turn."""
    assert normalize_progress_steps(None) == []
    assert normalize_progress_steps({"unexpected": "object"}) == []
    assert normalize_progress_steps([{"id": 7}, {}, ""]) == []
    assert current_progress_step([]) == ""


def test_raw_sql_reasoning_results_and_identifiers_never_reach_the_wire():
    raw = [
        "Thinking: use `system`.`billing`.`usage` to find the SKU",
        "Running SQL: SELECT SUM(list_cost) FROM system.billing.usage",
        "Query result: returned 7 rows for ENTERPRISE_GEMINI_MODEL_SERVING",
    ]
    rendered = " ".join(
        step["label"] for step in normalize_progress_steps(raw)
    ).lower()

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
