"""Tests for the card profile's tool wiring and deep-dive trigger config.

These guard the *routing* contract without any Databricks calls: that the card
profile registers exactly the two-lane tool set, that the "why" / rewards
questions have explicit start_deep_dive guidance in the system prompt (the thing
that keeps the deep lane reachable), and that the pure tool executors behave.
"""
from __future__ import annotations

import json

from realtime_api import card_tools
from realtime_api.tool_registry import ToolContext


def _ctx(customer_id: str | None = "CH-0001", **profile_state) -> ToolContext:
    return ToolContext(customer_id=customer_id, call_id="call-1", profile_state=dict(profile_state))


def test_card_profile_registers_two_lane_toolset():
    names = {t["function"]["name"] for t in card_tools._card_tools_spec()}
    assert names == {"select_use_case", "card_account_facts", "ask_card_genie", "start_deep_dive"}


def test_prompt_has_expense_and_rewards_deep_dive_triggers():
    p = card_tools.CARD_SYSTEM_PROMPT.lower()
    # Expense "why" must route to the deep lane...
    assert "why" in p and "start_deep_dive" in p
    # ...and the rewards ask ("am I missing rewards / points on the table") too.
    assert "rewards" in p
    assert "points on the table" in p or "missing any" in p or "getting all my points" in p


def test_start_deep_dive_scopes_customer_and_returns_signal():
    out = json.loads(
        card_tools._run_start_deep_dive({"question": "Why did my expenses jump?"}, _ctx(use_case="statement_insights"))
    )
    assert out["started"] is True
    assert "CH-0001" in out["question"]  # scoped to the caller
    assert out["use_case"] == "statement_insights"


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
