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


def test_language_helpers_use_bcp47_primary_tag():
    assert answer_rendering.is_english("en-US")
    assert answer_rendering.is_english("en")
    assert answer_rendering.is_english("en-GB")
    assert answer_rendering.is_english(None)
    assert not answer_rendering.is_english("fr-FR")
    assert not answer_rendering.is_english("hi-IN")
