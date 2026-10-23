from realtime_api.runtime.analytical_response import analytical_response


def _result() -> list[dict]:
    return [
        {
            "item_id": "q1",
            "columns": [
                {"name": "month", "type_text": "DATE"},
                {"name": "usage", "type_text": "DOUBLE"},
            ],
            "rows": [["2026-01", 10], ["2026-02", 12]],
        }
    ]


def test_typed_temporal_dataset_gets_a_question_aligned_view():
    response = analytical_response(
        source="genie_one",
        answer="Usage increased.",
        query_results=_result(),
        question="How has governed usage changed?",
    )
    assert response["datasets"][0]["id"] == "q1"
    assert response["views"] == [
        {
            "dataset_id": "q1",
            "kind": "line",
            "dimension": "month",
            "measures": ["usage"],
            "title": "How has governed usage changed?",
        }
    ]


def test_only_a_field_validated_upstream_view_is_preserved():
    response = analytical_response(
        source="genie_one",
        answer="Usage increased.",
        query_results=_result(),
        views=[
            {
                "dataset_id": "q1",
                "kind": "line",
                "dimension": "month",
                "measures": ["usage", "missing"],
                "title": "Monthly usage",
            },
            {
                "dataset_id": "q1",
                "kind": "pie",
                "dimension": "month",
                "measures": ["usage"],
            },
        ],
    )
    assert response["views"] == [
        {
            "dataset_id": "q1",
            "kind": "line",
            "dimension": "month",
            "measures": ["usage"],
            "title": "Monthly usage",
        }
    ]


def test_ambiguous_multi_dimension_dataset_stays_a_table():
    response = analytical_response(
        source="genie_one",
        answer="Several scenarios are available.",
        question="Compare the governed scenarios.",
        query_results=[
            {
                "item_id": "q2",
                "columns": [
                    {"name": "entity", "type_text": "STRING"},
                    {"name": "scenario", "type_text": "STRING"},
                    {"name": "value", "type_text": "DOUBLE"},
                ],
                "rows": [["A", "base", 1], ["A", "high", 2]],
            }
        ],
    )
    assert response["views"] == []
