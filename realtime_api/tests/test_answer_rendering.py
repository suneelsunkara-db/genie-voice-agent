"""Shared FSI/Knowledge long-answer rendering contract."""
from __future__ import annotations

from realtime_api.runtime import answer_rendering


class _Serving:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def summarize(self, **kwargs):
        self.calls.append(kwargs)
        if "Translate" in kwargs["system"]:
            return "Informe traducido."
        return "Resumen breve para hablar."

    def summarize_stream(self, **kwargs):
        self.calls.append(kwargs)
        yield "Informe "
        yield "traducido."


def test_summary_uses_call_language_and_is_voice_bounded(monkeypatch):
    serving = _Serving()
    monkeypatch.setattr(
        "realtime_api.serving_factory.shared_serving",
        lambda: serving,
    )

    summary = answer_rendering.summarize_for_voice(
        "What can you answer?",
        "A long governed answer.",
        "es-ES",
    )

    assert summary == "Resumen breve para hablar."
    call = serving.calls[0]
    assert "es-ES" in call["system"]
    assert "no markdown" in call["system"].lower()
    assert call["max_tokens"] > 0


def test_full_translation_streams_in_order(monkeypatch):
    serving = _Serving()
    monkeypatch.setattr(
        "realtime_api.serving_factory.shared_serving",
        lambda: serving,
    )

    assert list(
        answer_rendering.localize_answer_stream("English report.", "es-ES")
    ) == ["Informe ", "traducido."]
    assert "Translate" in serving.calls[0]["system"]


def test_english_never_calls_the_translator(monkeypatch):
    """Knowledge (Genie One) and FSI deep-dive share this renderer.

    The on-screen report is already English. Hitting gpt-5-5 to "translate" it
    only adds latency and risks rewriting facts. The spoken 2-3 sentence summary
    is a different call (summarize_for_voice) and still runs for English so TTS
    does not read the full report.
    """
    serving = _Serving()
    monkeypatch.setattr(
        "realtime_api.serving_factory.shared_serving",
        lambda: serving,
    )

    for language in ("en-US", "en", "en-GB", None, ""):
        serving.calls.clear()
        assert list(answer_rendering.localize_answer_stream("English report.", language)) == []
        assert serving.calls == []


def test_english_spoken_summary_is_not_a_translation(monkeypatch):
    serving = _Serving()
    monkeypatch.setattr(
        "realtime_api.serving_factory.shared_serving",
        lambda: serving,
    )

    summary = answer_rendering.summarize_for_voice(
        "What is the cost?",
        "A long governed answer.",
        "en-US",
    )

    assert summary == "Resumen breve para hablar."
    assert len(serving.calls) == 1
    system = serving.calls[0]["system"]
    assert "Translate" not in system
    assert "en-US" in system


def test_a_narrative_answer_is_both_the_spoken_source_and_the_panel_report():
    from realtime_api.runtime.genie_adapters import evidence_from_genie_one

    evidence = evidence_from_genie_one(
        {
            "status": "completed",
            "response_id": "r1",
            "final_answer": "## Spending\n\nShopping leads at 416,660 SGD.",
            "query_results": [
                {
                    "item_id": "q1",
                    "columns": [{"name": "category", "type_text": "STRING"}],
                    "rows": [["Shopping"]],
                }
            ],
        }
    )

    spoken_source, panel_report = answer_rendering.governed_answer_render(evidence)
    assert "Shopping leads" in spoken_source
    assert panel_report == spoken_source


def test_a_table_only_answer_still_gets_rendered_instead_of_read_out():
    """Genie answered "top spending categories" with rows and no narrative.

    Before this, no narrative meant no summary, and the voice fell through to the
    row cites — "category: Shopping; total spend sgd: 416659.61; ..." — while the
    panel showed no detail at all, because both were gated on prose existing.
    """
    from realtime_api.runtime.genie_adapters import evidence_from_genie_one

    evidence = evidence_from_genie_one(
        {
            "status": "completed",
            "response_id": "r2",
            "query_results": [
                {
                    "item_id": "q1",
                    "sql": "select category, total_spend_sgd from spending",
                    "columns": [
                        {"name": "category", "type_text": "STRING"},
                        {"name": "total_spend_sgd", "type_text": "DECIMAL"},
                    ],
                    "rows": [["Shopping", "416659.61"], ["Groceries", "296301.32"]],
                }
            ],
        }
    )

    spoken_source, panel_report = answer_rendering.governed_answer_render(evidence)
    assert "| category | total_spend_sgd |" in spoken_source
    assert "416659.61" in spoken_source
    # The typed rows render as the table and chart, so the report would only be a
    # second copy of them.
    assert panel_report == ""


def test_an_unusable_result_renders_nothing_and_leaves_the_cites_to_speak():
    from realtime_api.runtime.evidence import Evidence

    assert answer_rendering.governed_answer_render(Evidence(source="genie_one")) == ("", "")


def test_a_table_only_answer_is_summarized_like_prose(monkeypatch):
    """Genie answers "top spending categories" with rows and no narrative.

    Those rows are still the answer, so they get the same short spoken rendering a
    narrative gets. Without this the voice reads the row cites out loud —
    "category: Shopping; total spend sgd: 416659.61; ..." — which is a cite list,
    not an answer.
    """
    serving = _Serving()
    monkeypatch.setattr(
        "realtime_api.serving_factory.shared_serving",
        lambda: serving,
    )

    report = answer_rendering.table_as_markdown(
        ["category", "total_spend_sgd", "total_transactions"],
        [["Shopping", "416659.61", 1645], ["Other", "365561.20", 1421]],
    )
    assert report.splitlines()[0] == "| category | total_spend_sgd | total_transactions |"
    assert "| Shopping | 416659.61 | 1645 |" in report

    summary = answer_rendering.summarize_for_voice("Top categories?", report, "en-US")

    assert summary == "Resumen breve para hablar."
    system = serving.calls[0]["system"]
    assert "row by row" in system
    assert report in serving.calls[0]["user"]


def test_table_markdown_is_bounded_and_survives_awkward_cells():
    rows = [[f"row-{index}", None, "a|b"] for index in range(40)]
    report = answer_rendering.table_as_markdown(["name", "value", "note"], rows)

    body = [line for line in report.splitlines() if line.startswith("| row-")]
    assert len(body) == 30
    assert "40 rows in total" in report
    # A raw pipe would break out of the cell it belongs to.
    assert "| row-0 |  | a\\|b |" in report


def test_table_markdown_needs_both_a_header_and_rows():
    assert answer_rendering.table_as_markdown([], [["x"]]) == ""
    assert answer_rendering.table_as_markdown(["name"], []) == ""


def test_language_helpers_use_bcp47_primary_tag():
    assert answer_rendering.is_english("en-US")
    assert answer_rendering.is_english("en")
    assert answer_rendering.is_english("en-GB")
    assert answer_rendering.is_english(None)
    assert not answer_rendering.is_english("fr-FR")
    assert not answer_rendering.is_english("hi-IN")
