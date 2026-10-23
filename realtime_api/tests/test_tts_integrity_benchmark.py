from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _benchmark_module():
    root = Path(__file__).resolve().parents[2]
    scripts = root / "scripts" / "ml_asr"
    sys.path.insert(0, str(scripts))
    try:
        path = scripts / "benchmark_tts_streaming.py"
        spec = importlib.util.spec_from_file_location(
            "benchmark_tts_streaming_under_test", path
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(scripts))


def test_character_edits_measure_inserted_rambling_separately():
    benchmark = _benchmark_module()

    score = benchmark._character_edits(
        "I'm doing well. How can I help?",
        "random words I'm doing well. How can I help?",
    )

    assert score["insertions"] > 0
    assert score["insertion_rate"] > 0
    assert score["cer"] >= score["insertion_rate"]
    assert score["exact_normalized"] is False


def test_character_edits_are_unicode_and_punctuation_tolerant():
    benchmark = _benchmark_module()

    score = benchmark._character_edits("你好，这是测试。", "你好 这是测试")

    assert score["cer"] == 0
    assert score["exact_normalized"] is True


def test_integrity_gold_text_comes_from_all_app_locales():
    benchmark = _benchmark_module()

    texts = benchmark._locale_texts()

    assert len(texts) == 24
    assert {"en-US", "th-TH", "hi-IN", "fil-PH", "zh-CN"} <= set(texts)
    assert all(text.strip() for text in texts.values())


def test_summary_fails_samples_that_exceed_insertion_threshold():
    benchmark = _benchmark_module()
    rows = [
        {
            "mode": "stream",
            "ttfb_ms": 500,
            "total_ms": 2500,
            "quality": {"cer": 0.02, "insertion_rate": 0.0},
        },
        {
            "mode": "stream",
            "ttfb_ms": 600,
            "total_ms": 3000,
            "quality": {"cer": 0.3, "insertion_rate": 0.2},
        },
    ]

    summary = benchmark._summary(rows, max_cer=0.25, max_insertion_rate=0.08)

    assert summary["samples"] == 2
    assert summary["failed_samples"] == 1
    assert summary["pass_rate"] == 0.5
    assert summary["insertion_rate_max"] == 0.2
