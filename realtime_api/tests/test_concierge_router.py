"""Tests for the concierge's deterministic intent pre-router.

These guard the confidence contract behind always-confirm-then-route WITHOUT any
model or Databricks calls: short, unambiguous selection replies resolve to an
industry (so navigation never depends on the LLM emitting the tool call), while
cross-domain ties, questions, and long utterances defer to the LLM (return []).
"""
from __future__ import annotations

from realtime_api import concierge_tools
from realtime_api.profiles import ResolvedIntent, get_profile


def test_short_single_industry_replies_resolve():
    assert concierge_tools.resolve_industry("telco") == "telco"
    assert concierge_tools.resolve_industry("billing please") == "telco"
    assert concierge_tools.resolve_industry("the credit card one") == "fsi"
    assert concierge_tools.resolve_industry("let's do financial services") == "fsi"
    assert concierge_tools.resolve_industry("healthcare") == "healthcare"


def test_cross_domain_tie_is_ambiguous():
    # 'card'/'statement' -> fsi AND 'phone bill' -> telco: refuse to guess.
    assert concierge_tools.resolve_industry("my card statement and phone bill") is None


def test_questions_and_long_utterances_defer_to_llm():
    assert (
        concierge_tools.resolve_industry(
            "what is healthcare and how does it actually work for me today"
        )
        is None
    )
    assert concierge_tools.resolve_industry("") is None
    assert concierge_tools.resolve_industry("hello there how are you") is None


def test_resolver_emits_select_industry_with_confirmation():
    intents = concierge_tools._resolve_concierge_intents("credit card", "en-US")
    assert len(intents) == 1
    intent = intents[0]
    assert isinstance(intent, ResolvedIntent)
    assert intent.name == "select_industry"
    assert intent.arguments == {"industry": "fsi"}
    assert intent.confirm_intent  # a spoken confirmation instruction is present


def test_profile_exposes_resolver_hook():
    profile = get_profile("concierge")
    assert profile is not None
    assert profile.resolve_intents is not None
    assert profile.resolve_intents("telco", "en-US")[0].arguments["industry"] == "telco"
