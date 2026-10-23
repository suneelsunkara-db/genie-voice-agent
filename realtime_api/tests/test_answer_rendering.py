"""Shared FSI/Knowledge long-answer rendering contract."""
from __future__ import annotations

import pytest

from genie_voice.i18n import SUPPORTED_LANGUAGES
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

    assert summary == "Informe traducido."
    call = serving.calls[0]
    assert "es-ES" in call["system"]
    assert "no markdown" in call["system"].lower()
    assert "do not summarize" in call["system"].lower()
    assert call["max_tokens"] > 0


def test_already_localized_genie_answer_is_not_translated_again(monkeypatch):
    serving = _Serving()
    monkeypatch.setattr(
        "realtime_api.serving_factory.shared_serving",
        lambda: serving,
    )

    answer = "Los vuelos fueron el principal factor del aumento."
    summary = answer_rendering.summarize_for_voice(
        "¿Por qué aumentaron mis gastos?",
        answer,
        "es-ES",
        source_language="es",
    )

    assert summary == answer
    assert serving.calls == []


def test_empty_reasoning_summary_retries_with_larger_budget(monkeypatch):
    class _ReasoningServing(_Serving):
        def summarize(self, **kwargs):
            self.calls.append(kwargs)
            if kwargs["max_tokens"] < 512:
                return ""
            return "Die Ausgaben stiegen hauptsächlich wegen der Reise."

    serving = _ReasoningServing()
    monkeypatch.setattr(
        "realtime_api.serving_factory.shared_serving",
        lambda: serving,
    )
    monkeypatch.setattr(
        answer_rendering,
        "_render_knobs",
        lambda: (220, 1800, "reasoning-model"),
    )

    summary = answer_rendering.summarize_for_voice(
        "Warum sind meine Ausgaben gestiegen?",
        "Travel increased expenses.",
        "de-DE",
    )

    assert summary.startswith("Die Ausgaben")
    assert [call["max_tokens"] for call in serving.calls] == [220, 512]
    assert all(call["endpoint"] == "reasoning-model" for call in serving.calls)


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


def test_localized_genie_report_bypasses_full_translation(monkeypatch):
    serving = _Serving()
    monkeypatch.setattr(
        "realtime_api.serving_factory.shared_serving",
        lambda: serving,
    )

    assert list(
        answer_rendering.localize_answer_stream(
            "Informe de Genie.",
            "es-ES",
            source_language="es",
        )
    ) == []
    assert serving.calls == []


def test_agent_mode_question_is_canonicalized_to_english(monkeypatch):
    serving = _Serving()
    monkeypatch.setattr(
        "realtime_api.serving_factory.shared_serving",
        lambda: serving,
    )

    translated = answer_rendering.canonicalize_question_for_agent_mode(
        "Warum sind meine Ausgaben gestiegen?",
        "de-DE",
    )

    assert translated == "Informe traducido."
    call = serving.calls[0]
    assert "into English" in call["system"]
    assert "Do not answer" in call["system"]
    assert call["user"] == "Warum sind meine Ausgaben gestiegen?"


def test_english_agent_mode_question_bypasses_translation(monkeypatch):
    serving = _Serving()
    monkeypatch.setattr(
        "realtime_api.serving_factory.shared_serving",
        lambda: serving,
    )

    question = "Why did my expenses increase?"
    assert (
        answer_rendering.canonicalize_question_for_agent_mode(question, "en-US")
        == question
    )
    assert serving.calls == []


def test_english_never_calls_the_translator(monkeypatch):
    """Knowledge (Genie One) and FSI deep-dive share this renderer.

    The on-screen report is already English. Hitting gpt-5-5 to rewrite it adds
    latency and risks changing facts.
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


def test_english_spoken_excerpt_is_extractive_and_model_free(monkeypatch):
    serving = _Serving()
    monkeypatch.setattr(
        "realtime_api.serving_factory.shared_serving",
        lambda: serving,
    )

    summary = answer_rendering.summarize_for_voice(
        "What is the cost?",
        "First governed fact. Second governed fact. Third governed fact. Fourth fact.",
        "en-US",
    )

    assert summary == "First governed fact. Second governed fact. Third governed fact."
    assert serving.calls == []


def test_spoken_excerpt_preserves_genie_one_clarification_question(monkeypatch):
    serving = _Serving()
    monkeypatch.setattr(
        "realtime_api.serving_factory.shared_serving",
        lambda: serving,
    )
    answer = """
## I couldn't find a governed source for customer CPU/GPU spend

None of the governed sources I can reach break down CPU or GPU spend by customer,
so I can't answer this as asked yet.

**What I checked:**

- Customer records do not hold CPU or GPU costs.
- The closest match is account-level billing data.

**One question:** Do you have a customer-level cost source I should use?
Or would you like me to use the Databricks billing tables instead?
With that, I can show the requested monthly breakdown.
"""

    spoken = answer_rendering.summarize_for_voice(
        "How much are we spending on CPUs and GPUs for our customers?",
        answer,
        "en-US",
    )

    assert "can't answer this as asked yet" in spoken
    assert "Do you have a customer-level cost source I should use?" in spoken
    assert "Or would you like me to use the Databricks billing tables instead?" in spoken
    assert "What I checked" not in spoken
    assert serving.calls == []


def test_spoken_clarification_keeps_question_with_its_numbered_choices(monkeypatch):
    serving = _Serving()
    monkeypatch.setattr(
        "realtime_api.serving_factory.shared_serving",
        lambda: serving,
    )
    answer = """
I can't give you customer-level CPU and GPU spend because no governed source
breaks compute cost down by customer.

**What I checked:**
- Customer records do not hold compute costs.
- Only a small portion has customer tags.

**How would you like me to proceed?** I can:
1. use a customer cost-allocation source you point me to,
2. show account-wide CPU versus GPU spend by month, or
3. break down only the customer-tagged portion by customer.
"""

    spoken = answer_rendering.summarize_for_voice(
        "How much are we spending for CPUs and GPUs for our customers?",
        answer,
        "en-US",
    )

    assert "How would you like me to proceed?" in spoken
    assert "use a customer cost-allocation source you point me to" in spoken
    assert "show account-wide CPU versus GPU spend by month" in spoken
    assert "break down only the customer-tagged portion by customer" in spoken
    assert "Customer records do not hold compute costs" not in spoken
    assert serving.calls == []


@pytest.mark.parametrize(
    ("language", "outcome", "question"),
    [
        ("ar-SA", "لم أجد مصدرًا محكومًا.", "هل تريد استخدام بيانات الفوترة؟"),
        ("ja-JP", "管理対象の情報源が見つかりませんでした。", "請求データを使用しますか？"),
    ],
)
def test_localized_spoken_excerpt_preserves_follow_up_question(
    language, outcome, question
):
    answer = f"{outcome}\n\n追加の詳細です。\n\n{question}"

    spoken = answer_rendering.summarize_for_voice(
        "question",
        answer,
        language,
        source_language=language,
    )

    assert outcome in spoken
    assert question in spoken


def test_markdown_spend_answer_preserves_decimal_facts_for_speech(monkeypatch):
    serving = _Serving()
    monkeypatch.setattr(
        "realtime_api.serving_factory.shared_serving",
        lambda: serving,
    )
    answer = """
## Qwen ASR Model Spending: **$10,525.77 USD** Total

I interpreted "Quan ASR" as **Qwen ASR**. Across **7 endpoints**, your workspace
consumed **150,368.57 DBUs** totaling **$10,525.77 USD**
(April 2026 – October 2026). [Query Result](https://example.com/query)

| endpoint_name | total_cost_usd |
| --- | --- |
| qwen2-audio-asr | 4986.81 |

[View detailed query results](https://example.com/results)
"""

    spoken = answer_rendering.summarize_for_voice(
        "How much did I spend on Quan ASR model?",
        answer,
        "en-US",
    )

    assert "$10,525.77 USD" in spoken
    assert "150,368.57 DBUs" in spoken
    assert "**" not in spoken
    assert "|" not in spoken
    assert "https://" not in spoken
    assert "Query Result" not in spoken
    assert "from April 2026 through October 2026" in spoken
    assert serving.calls == []


def test_gpt_translation_restores_immutable_facts(monkeypatch):
    class _Translator(_Serving):
        def summarize(self, **kwargs):
            self.calls.append(kwargs)
            assert "[[FACT_0]]" in kwargs["user"]
            assert "[[FACT_1]]" in kwargs["user"]
            return "Gesamtausgaben: [[FACT_0]] für [[FACT_1]]."

    serving = _Translator()
    monkeypatch.setattr(
        "realtime_api.serving_factory.shared_serving",
        lambda: serving,
    )

    spoken = answer_rendering.summarize_for_voice(
        "Wie hoch waren meine Ausgaben?",
        "Total spend was $10,525.77 USD for qwen3-asr-1.7b.",
        "de-DE",
    )

    assert spoken == "Gesamtausgaben: $10,525.77 USD für qwen3-asr-1.7b."
    assert "[[FACT_" not in spoken


@pytest.mark.parametrize(
    "translated",
    [
        "Gesamtausgaben: [[FACT_0]].",  # dropped model identifier
        "Gesamtausgaben: [[FACT_0]] und [[FACT_0]] für [[FACT_1]].",  # duplicate
        "Gesamtausgaben: [[FACT_0]] für [[FACT_1]] plus 99.",  # invented number
        "**Gesamtausgaben:** [[FACT_0]] für [[FACT_1]].",  # residual Markdown
    ],
)
def test_gpt_translation_fails_closed_on_fact_or_markup_changes(
    monkeypatch, translated
):
    class _BadTranslator(_Serving):
        def summarize(self, **kwargs):
            self.calls.append(kwargs)
            return translated

    serving = _BadTranslator()
    monkeypatch.setattr(
        "realtime_api.serving_factory.shared_serving",
        lambda: serving,
    )

    with pytest.raises(answer_rendering.SpeechRenderError):
        answer_rendering.summarize_for_voice(
            "Wie hoch waren meine Ausgaben?",
            "Total spend was $10,525.77 USD for qwen3-asr-1.7b.",
            "de-DE",
        )
    assert len(serving.calls) == 2


@pytest.mark.parametrize(
    ("language", "answer", "fact"),
    [
        ("ar-SA", "بلغت التكلفة ١٠٬٥٢٥٫٧٧ دولارًا؟ هذه هي القيمة النهائية.", "١٠٬٥٢٥٫٧٧"),
        ("hi-IN", "कुल खर्च ₹10,525.77 था। यह अंतिम राशि है।", "₹10,525.77"),
        ("ja-JP", "合計費用は¥10,525.77です。これは確定額です。", "¥10,525.77"),
        ("th-TH", "ค่าใช้จ่ายทั้งหมดคือ $10,525.77 และเป็นยอดรวมสุดท้าย", "$10,525.77"),
        ("de-DE", "Die Gesamtkosten betrugen 10.525,77 EUR.", "10.525,77 EUR"),
    ],
)
def test_already_localized_speech_preserves_multilingual_numbers(
    language, answer, fact
):
    spoken = answer_rendering.summarize_for_voice(
        "question",
        answer,
        language,
        source_language=language,
    )
    assert fact in spoken


def test_fallback_tts_segmentation_never_splits_decimal_or_version():
    text = (
        "The total was $10,525.77 USD for qwen3-asr-1.7b. "
        + "Additional governed context remains available. " * 8
    )
    segments = answer_rendering.split_speech_segments(text, max_chars=180)

    assert "".join(segments).count("$10,525.77 USD") == 1
    assert "".join(segments).count("qwen3-asr-1.7b") == 1
    assert all(not segment.endswith("$10,525.") for segment in segments)


def test_fallback_tts_segmentation_never_splits_compact_decimal_units():
    text = (
        "CPU usage is likely to total about 0.5B–0.85B DBUs over the next "
        "12 months, while GPU usage remains lower."
    )
    segments = answer_rendering.split_speech_segments(text, max_chars=80)
    joined = " ".join(segments)

    assert "0.5B–0.85B DBUs" in joined
    assert "0. 5B" not in joined


def test_dense_accounting_sentence_splits_at_clauses_by_fact_count():
    text = (
        "This cycle (2025-12), CH-0001 posted $3,250.00 in purchases, "
        "plus $122.00 in fees and $28.00 in interest, bringing the statement "
        "to a new balance of $4,211.04 (up from $1,061.04 prior balance), "
        "after $250.00 in payments."
    )

    segments = answer_rendering.split_speech_segments(text)

    assert len(segments) >= 3
    assert all(
        len(answer_rendering._FACT_ATOM_RE.findall(segment)) <= 3
        for segment in segments
    )
    for fact in (
        "2025-12",
        "CH-0001",
        "$3,250.00",
        "$122.00",
        "$28.00",
        "$4,211.04",
        "$1,061.04",
        "$250.00",
    ):
        assert sum(segment.count(fact) for segment in segments) == 1


def test_escaped_agent_mode_citation_never_reaches_spoken_text():
    spoken = answer_rendering.summarize_for_voice(
        "How are my expenses?",
        "Purchases were $3,250.00 after $250.00 in payments \\[[1]\\].",
        "en-US",
        source_language="en",
    )

    assert spoken == "Purchases were $3,250.00 after $250.00 in payments."
    assert "\\[" not in spoken and "[1]" not in spoken


def test_dense_english_report_uses_verified_subset_for_voice(monkeypatch):
    class _DenseSummarizer(_Serving):
        def summarize(self, **kwargs):
            self.calls.append(kwargs)
            return (
                "Purchases were [[FACT_2]]. Fees were [[FACT_3]], "
                "and interest was [[FACT_4]]. The new balance is [[FACT_5]]."
            )

    serving = _DenseSummarizer()
    monkeypatch.setattr(
        "realtime_api.serving_factory.shared_serving",
        lambda: serving,
    )
    answer = (
        "This cycle (2025-12), CH-0001 posted $3,250.00 in purchases, "
        "plus $122.00 in fees and $28.00 in interest, bringing the statement "
        "to a new balance of $4,211.04 (up from a $1,061.04 prior balance), "
        "after $250.00 in payments."
    )

    spoken = answer_rendering.summarize_for_voice(
        "How are my expenses?",
        answer,
        "en-US",
        source_language="en",
    )

    assert spoken == (
        "Purchases were $3,250.00. Fees were $122.00, and interest was $28.00. "
        "The new balance is $4,211.04."
    )
    assert "CH-0001" not in spoken and "2025-12" not in spoken
    assert "at most four placeholders" in serving.calls[0]["system"]


@pytest.mark.parametrize("language", SUPPORTED_LANGUAGES)
def test_every_supported_language_uses_the_fact_placeholder_contract(
    monkeypatch, language
):
    if language.startswith("en"):
        spoken = answer_rendering.summarize_for_voice(
            "What was the cost?",
            "The governed total is $10,525.77 USD for qwen3-asr-1.7b.",
            language,
        )
    else:
        class _LanguageTranslator(_Serving):
            def summarize(self, **kwargs):
                self.calls.append(kwargs)
                return "Translated result: [[FACT_0]] for [[FACT_1]]."

        serving = _LanguageTranslator()
        monkeypatch.setattr(
            "realtime_api.serving_factory.shared_serving",
            lambda: serving,
        )
        spoken = answer_rendering.summarize_for_voice(
            "What was the cost?",
            "The governed total is $10,525.77 USD for qwen3-asr-1.7b.",
            language,
        )
        assert language in serving.calls[0]["system"]

    assert "$10,525.77 USD" in spoken
    assert "qwen3-asr-1.7b" in spoken
    assert "[[FACT_" not in spoken


def test_an_upstream_narrative_is_both_spoken_source_and_panel_report():
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

    spoken_source, panel_report = answer_rendering.upstream_answer_render(evidence)
    assert "Shopping leads" in spoken_source
    assert panel_report == spoken_source


def test_table_only_evidence_is_not_promoted_into_customer_prose():
    """Typed rows belong in tables; the conversational answer remains natural."""
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

    assert answer_rendering.upstream_answer_render(evidence) == ("", "")


def test_agent_mode_display_report_is_an_authoritative_upstream_answer():
    from realtime_api.runtime.genie_adapters import evidence_from_agent_mode

    evidence = evidence_from_agent_mode(
        [{"columns": ["driver", "amount_usd"], "preview_rows": [["flight", 1400]]}],
        report_text="Flights were the main expense driver at $1,400.",
    )

    spoken_source, panel_report = answer_rendering.upstream_answer_render(evidence)
    assert spoken_source == "Flights were the main expense driver at $1,400."
    assert panel_report == spoken_source


def test_an_unusable_result_renders_nothing():
    from realtime_api.runtime.evidence import Evidence

    assert answer_rendering.upstream_answer_render(Evidence(source="genie_one")) == ("", "")


def test_table_markdown_is_available_for_display_rendering():
    report = answer_rendering.table_as_markdown(
        ["category", "total_spend_sgd", "total_transactions"],
        [["Shopping", "416659.61", 1645], ["Other", "365561.20", 1421]],
    )
    assert report.splitlines()[0] == "| category | total_spend_sgd | total_transactions |"
    assert "| Shopping | 416659.61 | 1645 |" in report


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
