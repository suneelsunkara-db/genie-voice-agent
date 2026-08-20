"""Follow-up turns belong to the conversation Genie One is already holding.

The failure these guard against: ask "sales per region", Genie replies asking which
domain to use, the caller says "Unity Catalog" — and that reply is treated as a brand
new question. It opens a fresh conversation with no memory of the clarifying question,
and its words alone read as a documentation topic, so the caller hears product docs
instead of their sales.

The fix is deliberately NOT a local interpretation of the reply. Genie One resolves
references against its own conversation, so this runtime carries the handle to that
conversation and offers the turn the choice of joining it. When upstream memory makes
the handle unnecessary, ``upstream_memory`` turns it off and nothing else changes.
"""
from __future__ import annotations

import json

from realtime_api.pipelines.speech_llm_toolassist_speech import _tools_and_prompt
from realtime_api.runtime import genie_one
from realtime_api.runtime.identity import SessionPrincipal
from realtime_api.runtime.workspace_conversation import WorkspaceConversation


def _tool_names(specs: list[dict]) -> set[str]:
    return {(spec.get("function") or {}).get("name") for spec in specs}


class _Profile:
    """Minimal stand-in for a pack: its own tool and its own prompt."""

    name = "knowledge"
    system_prompt = "Pack prompt."

    def tools_spec(self) -> list[dict]:
        return [{"type": "function", "function": {"name": "search_knowledge"}}]


# --- the conversation handle ------------------------------------------------- #

def test_a_new_call_has_no_conversation_to_join():
    conversation = WorkspaceConversation()
    assert not conversation.is_open
    assert conversation.handle is None
    assert conversation.briefing() == ""


def test_binding_adopts_the_id_genie_reported():
    conversation = WorkspaceConversation()
    conversation.bind("conv-1")
    conversation.record(question="sales per region", answer="APAC leads.")

    assert conversation.is_open
    assert conversation.handle == "conv-1"
    assert conversation.turns == 1
    briefing = conversation.briefing()
    assert "sales per region" in briefing and "APAC leads." in briefing


def test_nothing_is_invented_when_genie_reports_no_id():
    conversation = WorkspaceConversation()
    conversation.bind(None)
    assert conversation.handle is None
    assert conversation.turns == 0


def test_upstream_memory_retires_the_local_handle():
    """The replacement point: Genie One's own memory takes over continuity.

    The id stays recorded for observability, but nothing is carried into the next
    question and no turn is offered a conversation to join.
    """
    conversation = WorkspaceConversation(upstream_memory=True)
    conversation.bind("conv-1")
    conversation.record(question="sales per region", answer="APAC leads.")

    assert conversation.conversation_id == "conv-1"
    assert conversation.handle is None
    assert not conversation.is_open
    assert conversation.briefing() == ""


def test_closing_forgets_the_handle():
    conversation = WorkspaceConversation()
    conversation.bind("conv-1")
    conversation.close()
    assert not conversation.is_open
    assert conversation.handle is None


# --- what the turn is offered ----------------------------------------------- #

def test_a_first_pack_turn_sees_only_pack_tools():
    tools, prompt = _tools_and_prompt(
        _Profile(),
        route_adapter="pack_facts",
        workspace_selected=False,
        conversation_joinable=False,
    )
    assert _tool_names(tools) == {"search_knowledge"}
    assert prompt == "Pack prompt."


def test_a_continuing_turn_cannot_reach_the_pack_at_all():
    """The heart of the fix: the pack's tools are WITHDRAWN, not merely deprioritised.

    The knowledge pack's own prompt says to search its corpus for any question about
    Databricks, and it is right to. Leaving that tool in reach while a conversation is
    open is what answered "Personal finance and banking" — a reply to Genie's own
    question — out of a documentation corpus.
    """
    conversation = WorkspaceConversation()
    conversation.bind("conv-1")
    conversation.record(question="Sales per region", answer="Which domain should I use?")

    tools, prompt = _tools_and_prompt(
        _Profile(),
        route_adapter="genie_one_mcp",
        workspace_selected=True,
        conversation_joinable=True,
        briefing=conversation.briefing(),
    )

    assert _tool_names(tools) == {"workspace_query"}
    assert "UNCHANGED" in prompt
    assert "Sales per region" in prompt


def test_a_first_workspace_turn_asks_the_complete_question():
    tools, prompt = _tools_and_prompt(
        _Profile(),
        route_adapter="genie_one_mcp",
        workspace_selected=True,
        conversation_joinable=False,
    )
    assert _tool_names(tools) == {"workspace_query"}
    assert "complete question" in prompt
    assert "UNCHANGED" not in prompt


# --- conversation threading upstream ---------------------------------------- #

def _principal() -> SessionPrincipal:
    return SessionPrincipal(username="u", access_token="tok")


def test_the_handle_is_forwarded_to_genie_and_echoed_back(monkeypatch):
    seen: dict = {}

    async def _fake_query(
        question,
        principal,
        host,
        *,
        timeout_s,
        conversation_id=None,
        on_progress=None,
    ):
        seen["question"] = question
        seen["conversation_id"] = conversation_id
        return {"status": "completed", "conversation_id": "conv-1", "final_answer": "ok"}

    monkeypatch.setattr(genie_one, "_query", _fake_query)

    payload = json.loads(
        genie_one.run_workspace_query(
            "Unity Catalog",
            principal=_principal(),
            host="https://example.databricks.com",
            session_id="s",
            turn_id=2,
            conversation_id="conv-1",
        )
    )

    assert seen == {"question": "Unity Catalog", "conversation_id": "conv-1"}
    assert payload["conversation_id"] == "conv-1"


def test_a_first_question_opens_a_conversation_and_reports_its_id(monkeypatch):
    async def _fake_query(
        question,
        principal,
        host,
        *,
        timeout_s,
        conversation_id=None,
        on_progress=None,
    ):
        assert conversation_id is None
        return {"status": "completed", "conversation_id": "conv-new", "final_answer": "ok"}

    monkeypatch.setattr(genie_one, "_query", _fake_query)

    payload = json.loads(
        genie_one.run_workspace_query(
            "sales per region",
            principal=_principal(),
            host="https://example.databricks.com",
            session_id="s",
            turn_id=1,
        )
    )
    assert payload["conversation_id"] == "conv-new"


def test_a_failed_call_still_reports_the_conversation_so_the_thread_survives(monkeypatch):
    async def _boom(
        question,
        principal,
        host,
        *,
        timeout_s,
        conversation_id=None,
        on_progress=None,
    ):
        raise RuntimeError("transport died")

    monkeypatch.setattr(genie_one, "_query", _boom)

    payload = json.loads(
        genie_one.run_workspace_query(
            "Unity Catalog",
            principal=_principal(),
            host="https://example.databricks.com",
            session_id="s",
            turn_id=3,
            conversation_id="conv-7",
        )
    )
    assert payload["conversation_id"] == "conv-7"
    assert "transport died" in payload["error"]
