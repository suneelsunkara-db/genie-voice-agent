"""Tests for the card profile's tool wiring.

These guard the *routing* contract without any Databricks calls: that the card
profile registers the two-lane tool set, and that the pure tool executors behave.
The navigator, not the system prompt, chooses which capability is in reach.
"""
from __future__ import annotations

import json

from realtime_api import card_tools
from realtime_api.contracts import SessionStart
from realtime_api.session import VoiceSession
from realtime_api.tool_registry import (
    ToolContext,
    attach_session_identity,
    genie_space_name,
)


def _ctx(customer_id: str | None = "CH-0001", **profile_state) -> ToolContext:
    return ToolContext(customer_id=customer_id, call_id="call-1", profile_state=dict(profile_state))


def test_card_profile_registers_two_lane_toolset():
    names = {t["function"]["name"] for t in card_tools._card_tools_spec()}
    assert names == {"select_use_case", "card_account_facts", "ask_card_genie", "start_deep_dive"}


def test_card_prompt_does_not_require_same_turn_dual_tools():
    p = card_tools.CARD_SYSTEM_PROMPT.lower()
    assert "card_account_facts()" not in p
    assert "start_deep_dive" not in p
    assert "why" not in p


def test_start_deep_dive_fails_closed_without_obo():
    out = json.loads(
        card_tools._run_start_deep_dive({"question": "Why did my expenses jump?"}, _ctx(use_case="statement_insights"))
    )
    assert out["obo_deny"] is True
    assert out["error_evidence"]["code"] == "permission"


def test_start_deep_dive_requires_question():
    out = json.loads(card_tools._run_start_deep_dive({"question": "   "}, _ctx()))
    assert "error" in out


def test_select_use_case_records_and_validates():
    ctx = _ctx()
    ok = json.loads(card_tools._run_select_use_case({"use_case": "rewards_optimizer"}, ctx))
    assert ok["ok"] is True and ok["use_case"] == "rewards_optimizer"
    assert ctx.profile_state["selected_use_case"] == "rewards_optimizer"

    bad = json.loads(card_tools._run_select_use_case({"use_case": "nonsense"}, ctx))
    assert "error" in bad


def test_session_space_name_overrides_the_demo_space():
    """A consumer points Space / Agent Mode at a space they can run, not ours."""
    assert genie_space_name(ToolContext(), "Genie Voice - Card Issuer") == (
        "Genie Voice - Card Issuer"
    )
    session = VoiceSession(
        SessionStart.from_event(
            {
                "language": "en-US",
                "profile": "card",
                "space_name": "My Team Space",
            }
        )
    )
    ctx = attach_session_identity(ToolContext(), session)
    assert genie_space_name(ctx, "Genie Voice - Card Issuer") == "My Team Space"
