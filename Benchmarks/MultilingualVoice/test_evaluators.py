from evaluators import evaluate_asr, normalize_text


def test_normalize_text_preserves_thai_combining_marks() -> None:
    text = "กิ่!"

    assert normalize_text(text, keep_spaces=False) == "กิ่"


def test_evaluate_asr_reports_corpus_and_macro_error_separately() -> None:
    rows = [
        {"reference": "one", "transcript": ""},
        {"reference": "one two three four", "transcript": "one two three four"},
    ]

    scores = evaluate_asr(rows)

    assert scores["wer"] == 0.2
    assert scores["wer_macro"] == 0.5
    assert scores["aggregation"] == "corpus"
    assert scores["scoring_version"] == "unicode-v2-corpus"
    assert scores["wer_ci95"] is not None
