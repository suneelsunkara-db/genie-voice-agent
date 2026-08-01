"""Rendering a finished Agent-Mode report for the caller's language.

Agent Mode answers in English on purpose (asking it to write in the caller's
language killed the run for Hindi — see GenieAgentModeClient.ask), so the two
artifacts the caller gets are produced here: the report they READ, translated, and
the short "why" they HEAR. What must hold:

  * English calls are untouched — no translation call, no extra latency.
  * A failed translation shows the English report, never a blank panel.
  * The two renderings are independent: one failing does not blank the other.
  * The prompt names the language ("Hindi"), not the tag ("hi-IN").
"""
from __future__ import annotations

import time

import pytest

from realtime_api import deep_dive as dd


class _Serving:
    """Records summarize() calls; optionally fails or sleeps."""

    def __init__(self, reply: str = "translated", fail: bool = False, delay: float = 0.0) -> None:
        self.reply = reply
        self.fail = fail
        self.delay = delay
        self.calls: list[dict] = []

    def summarize(
        self, *, system: str, user: str, temperature: float, max_tokens: int, endpoint=None
    ) -> str:
        self.calls.append(
            {
                "system": system,
                "user": user,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "endpoint": endpoint,
            }
        )
        if self.delay:
            time.sleep(self.delay)
        if self.fail:
            raise RuntimeError("endpoint unavailable")
        return self.reply


@pytest.fixture()
def serving(monkeypatch):
    def _install(**kwargs) -> _Serving:
        s = _Serving(**kwargs)
        monkeypatch.setattr("realtime_api.serving_factory.shared_serving", lambda: s)
        return s

    return _install


@pytest.mark.parametrize("language", [None, "", "en", "en-US", "en-GB", "EN-us"])
def test_english_needs_no_translation(serving, language):
    s = serving()
    assert dd.is_english(language) is True
    assert dd.localize_report("Your expenses rose $2,450.", language) == ""
    assert s.calls == [], "an English call must not pay for a translation"


@pytest.mark.parametrize("language", ["hi-IN", "th-TH", "ja-JP", "es-ES"])
def test_non_english_is_translated(serving, language):
    s = serving(reply="अनुवादित रिपोर्ट")
    assert dd.is_english(language) is False
    assert dd.localize_report("Your expenses rose $2,450.", language) == "अनुवादित रिपोर्ट"
    assert len(s.calls) == 1
    # The report text is what gets translated, not the question.
    assert "$2,450" in s.calls[0]["user"]


def test_empty_report_is_not_sent_to_the_llm(serving):
    s = serving()
    assert dd.localize_report("   ", "hi-IN") == ""
    assert s.calls == []


def test_translation_failure_keeps_the_english_report(serving):
    serving(fail=True)
    # "" means "keep what the agent sent" — the panel shows English, not a blank.
    assert dd.localize_report("Your expenses rose $2,450.", "hi-IN") == ""


def test_prompt_names_the_language_and_protects_the_numbers(serving):
    s = serving()
    dd.localize_report("## Up $2,450 \\[[1]\\]", "hi-IN")
    system = s.calls[0]["system"]
    assert "Hindi" in system and "hi-IN" not in system
    for promise in ("markdown", "citation", "Do not summarize"):
        assert promise in system


def test_language_name_resolves_from_the_shared_catalog():
    assert dd.language_name("hi-IN") == "Hindi"
    assert dd.language_name("zh-CN") == "Chinese"
    assert dd.language_name("th-TH") == "Thai"
    # An unsupported tag still yields something usable rather than raising: the tag
    # in the prompt is a worse instruction, but far better than no report at all.
    assert dd.language_name("xx-YY") == "xx-YY"


def test_catalog_is_matched_on_the_tag_not_a_subtag_split():
    """nb-NO is the trap: the catalog is keyed by ISO base "no", so matching on a
    naive "nb" split would miss it and leak the raw tag into the prompt. Asserted
    against the catalog directly so the test doesn't depend on which languages a
    given deploy enables."""
    from genie_voice.i18n import LANGUAGE_CATALOG

    name = next((eng for (tag, eng) in LANGUAGE_CATALOG.values() if tag == "nb-NO"), None)
    assert name == "Norwegian"


_REPORT = {"kind": "report", "status": "completed", "report": "Expenses rose $2,450."}


def _stream(monkeypatch, language: str | None) -> list[dict]:
    out: list[dict] = []
    dd.stream_report_renderings("why?", dict(_REPORT), language, out.append)
    return out


def test_the_speakable_report_is_emitted_before_the_translation(serving, monkeypatch):
    """The voice must not wait on the translation: the customer is already ~40s in."""
    serving(reply="localized report", delay=0.25)
    monkeypatch.setattr(dd, "summarize_deepdive", lambda q, r, lang: f"spoken in {lang}")
    events = _stream(monkeypatch, "hi-IN")

    assert [e["kind"] for e in events] == ["report", "report_localized"]
    first, patch = events
    # Beat 1 carries the spoken line and the English text we already have, and warns
    # the client that the on-screen text is about to be replaced.
    assert first["spoken_summary"] == "spoken in hi-IN"
    assert first["report"] == "Expenses rose $2,450."
    assert first["report_language"] == "en"
    assert first["localization_pending"] is True
    # Beat 2 is the swap.
    assert patch["report"] == "localized report"
    assert patch["report_language"] == "hi-IN"


def test_english_emits_one_event_and_promises_no_swap(serving, monkeypatch):
    s = serving()
    monkeypatch.setattr(dd, "summarize_deepdive", lambda q, r, lang: "spoken")
    events = _stream(monkeypatch, "en-US")

    assert [e["kind"] for e in events] == ["report"]
    assert events[0]["localization_pending"] is False
    assert s.calls == []


def test_the_report_still_ships_when_the_summary_fails(serving, monkeypatch):
    serving(reply="localized report")

    def _boom(*_a, **_k):
        raise RuntimeError("summarizer down")

    monkeypatch.setattr(dd, "summarize_deepdive", _boom)
    events = _stream(monkeypatch, "hi-IN")
    assert [e["kind"] for e in events] == ["report", "report_localized"]
    assert "spoken_summary" not in events[0]


def test_a_failing_translation_leaves_the_english_report_standing(serving, monkeypatch):
    serving(fail=True)
    monkeypatch.setattr(dd, "summarize_deepdive", lambda q, r, lang: "spoken anyway")
    events = _stream(monkeypatch, "hi-IN")
    # No swap event: the caller keeps the English report and still hears the answer.
    assert [e["kind"] for e in events] == ["report"]
    assert events[0]["spoken_summary"] == "spoken anyway"


def test_the_two_renderings_overlap(serving, monkeypatch):
    """Run serially and the customer's total wait is both calls, not the longer one."""
    serving(reply="localized", delay=0.30)
    monkeypatch.setattr(dd, "summarize_deepdive", lambda q, r, lang: time.sleep(0.30) or "spoken")
    t0 = time.perf_counter()
    events = _stream(monkeypatch, "hi-IN")
    elapsed = time.perf_counter() - t0
    assert [e["kind"] for e in events] == ["report", "report_localized"]
    assert elapsed < 0.55, f"expected ~0.30s (overlapped), took {elapsed:.2f}s (serial?)"


def test_an_empty_report_is_never_promised_a_translation(serving, monkeypatch):
    serving()
    monkeypatch.setattr(dd, "summarize_deepdive", lambda q, r, lang: "")
    out: list[dict] = []
    dd.stream_report_renderings("why?", {"kind": "report", "report": ""}, "hi-IN", out.append)
    assert [e["kind"] for e in out] == ["report"]
    assert out[0]["localization_pending"] is False


def test_localize_max_tokens_is_config_sourced_and_bigger_than_the_summary_cap():
    _, summary_tokens, localize_tokens = dd._summary_knobs()
    # A whole report will not fit in the spoken summary's budget; if these ever
    # converge, translated reports come back truncated mid-sentence.
    assert localize_tokens > summary_tokens * 4


def test_conversions_route_to_the_configured_conversion_endpoint(serving, monkeypatch):
    """Both text->text conversions must hit the conversion model, not the voice LLM.

    The whole point of the split is that the deep-dive summary and report
    translation run on the small fast multilingual model while voice turns keep the
    80B. If either call forgot to pass the endpoint, it would silently regress onto
    the slow path — so assert the endpoint reaches the serving layer.
    """
    s = serving(reply="x")
    monkeypatch.setattr(dd, "_conversion_endpoint", lambda: "databricks-gemma-3-12b")
    dd.localize_report("Your expenses rose $2,450.", "hi-IN")
    dd.summarize_deepdive("why?", "Your expenses rose $2,450.", "hi-IN")
    assert [c["endpoint"] for c in s.calls] == ["databricks-gemma-3-12b"] * 2


def test_conversion_endpoint_none_falls_back_to_the_voice_llm(serving, monkeypatch):
    # Empty config -> None -> the serving layer uses its own llm_endpoint.
    s = serving(reply="x")
    monkeypatch.setattr(dd, "_conversion_endpoint", lambda: None)
    dd.localize_report("Your expenses rose $2,450.", "hi-IN")
    assert s.calls[0]["endpoint"] is None
