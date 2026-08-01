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
    """Records summarize()/summarize_stream() calls; optionally fails or sleeps.

    ``stream_chunks`` drives the streaming path: when set, ``summarize_stream``
    yields those pieces; when None it yields nothing so the caller falls back to the
    one-shot ``summarize`` (which is how the deep-dive streamer degrades on an
    endpoint that can't stream)."""

    def __init__(
        self,
        reply: str = "translated",
        fail: bool = False,
        delay: float = 0.0,
        stream_chunks: list[str] | None = None,
    ) -> None:
        self.reply = reply
        self.fail = fail
        self.delay = delay
        self.stream_chunks = stream_chunks
        self.calls: list[dict] = []

    def summarize(
        self, *, system: str, user: str, temperature: float | None = None, max_tokens: int, endpoint=None
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

    def summarize_stream(
        self, *, system: str, user: str, temperature: float | None = None, max_tokens: int, endpoint=None
    ):
        self.calls.append(
            {
                "system": system,
                "user": user,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "endpoint": endpoint,
                "stream": True,
            }
        )
        if self.fail:
            raise RuntimeError("endpoint unavailable")
        for piece in self.stream_chunks or []:
            if self.delay:
                time.sleep(self.delay)
            yield piece


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


def test_the_spoken_line_ships_first_and_the_panel_opens_empty(serving, monkeypatch):
    """The voice must not wait on the translation, and the panel must never flash
    English: beat 1 carries the spoken "why" with an EMPTY body + pending, then the
    translation streams in."""
    serving(reply="localized report", delay=0.05)
    monkeypatch.setattr(dd, "summarize_deepdive", lambda q, r, lang: f"spoken in {lang}")
    events = _stream(monkeypatch, "hi-IN")

    kinds = [e["kind"] for e in events]
    assert kinds[0] == "report"
    assert kinds[-1] == "report_localized"
    assert "report_localized_delta" in kinds
    first = events[0]
    assert first["spoken_summary"] == "spoken in hi-IN"
    # No English on screen — the body is empty and the panel is told to localize.
    assert first["report"] == ""
    assert first["report_language"] == "hi-IN"
    assert first["localization_pending"] is True
    # The terminal event carries the full localized text.
    assert events[-1]["report"] == "localized report"
    assert events[-1]["report_language"] == "hi-IN"


def test_the_translation_streams_in_as_deltas(serving, monkeypatch):
    """A streaming conversion endpoint paints the report progressively: each chunk
    is its own ``report_localized_delta`` in order, and the terminal event carries
    the joined whole."""
    serving(stream_chunks=["अनु", "वादित ", "रिपोर्ट"])
    monkeypatch.setattr(dd, "summarize_deepdive", lambda q, r, lang: "spoken")
    events = _stream(monkeypatch, "hi-IN")

    assert events[0]["kind"] == "report" and events[0]["report"] == ""
    deltas = [e["delta"] for e in events if e["kind"] == "report_localized_delta"]
    assert deltas == ["अनु", "वादित ", "रिपोर्ट"]
    assert events[-1]["kind"] == "report_localized"
    assert events[-1]["report"] == "अनुवादित रिपोर्ट"
    assert events[-1]["report_language"] == "hi-IN"


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
    kinds = [e["kind"] for e in events]
    assert kinds[0] == "report" and kinds[-1] == "report_localized"
    assert "spoken_summary" not in events[0]
    assert events[-1]["report"] == "localized report"


def test_a_failing_translation_resolves_pending_with_the_english_report(serving, monkeypatch):
    """A failed translation must NOT hang the UI on the "translating…" hint.

    Beat 1 announced ``localization_pending`` so the client is waiting for a swap;
    if the translation fails (endpoint down / SP lacks CAN_QUERY) we must still
    close that loop. Beat 2 arrives carrying the English report we already have and
    ``report_language: "en"``, which clears the pending state and leaves a readable
    English panel — and the caller still hears the spoken answer.
    """
    serving(fail=True)
    monkeypatch.setattr(dd, "summarize_deepdive", lambda q, r, lang: "spoken anyway")
    events = _stream(monkeypatch, "hi-IN")
    assert [e["kind"] for e in events] == ["report", "report_localized"]
    # The panel opened empty (no English flash) and the spoken line still played.
    assert events[0]["report"] == ""
    assert events[0]["spoken_summary"] == "spoken anyway"
    # With no deltas, the terminal event falls back to the English report + "en".
    fallback = events[1]
    assert fallback["report"] == "Expenses rose $2,450."
    assert fallback["report_language"] == "en"


def test_the_spoken_line_is_emitted_before_any_translation_delta(serving, monkeypatch):
    """The customer HEARS the answer before the on-screen translation starts: the
    report shell (with the spoken line) must precede the first streamed delta."""
    serving(stream_chunks=["local", "ized"])
    monkeypatch.setattr(dd, "summarize_deepdive", lambda q, r, lang: "spoken")
    events = _stream(monkeypatch, "hi-IN")
    first_delta = next(i for i, e in enumerate(events) if e["kind"] == "report_localized_delta")
    assert events[0]["kind"] == "report"
    assert events[0]["spoken_summary"] == "spoken"
    assert first_delta > 0, "the spoken shell must be emitted before any translation delta"


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
