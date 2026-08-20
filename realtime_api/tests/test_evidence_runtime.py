"""Phase 3–5 Progressive Turn Runtime: Evidence, AgentRuntime, OBO, GoalFrame."""
from __future__ import annotations

import json
from typing import Any

import pytest

from realtime_api.runtime.agent_runtime import (
    AgentContext,
    AgentGoal,
    LiveToolRespondAdapter,
    RespondWithToolsAdapter,
)
from realtime_api.runtime.cancellation import CancellationToken
from realtime_api.runtime.evidence import Evidence, EvidenceComposer, SpokenClaim, TableEvidence
from realtime_api.runtime.events import AgentEventKind
from realtime_api.runtime.genie_adapters import (
    evidence_from_agent_mode,
    evidence_from_genie_one,
    evidence_from_genie_space,
)
from realtime_api.runtime.goal_frame import (
    GoalFrame,
    adapter_for,
    cascade_route,
    classify_barge_intent,
    classify_depth,
    enforce_effect,
)
from realtime_api.runtime.identity import SessionPrincipal, require_obo_token
from realtime_api.runtime.pack_schema import evidence_from_tool_result
from realtime_api.runtime.refuse import ErrorCode
from realtime_api.tool_registry import ToolContext, genie_obo_or_refuse
from realtime_api.tools import _run_ask_genie


# ---------------------------------------------------------------------------
# Phase 3 — Evidence / AgentRuntime
# ---------------------------------------------------------------------------


def test_spoken_claim_requires_cites():
    with pytest.raises(ValueError, match="field_path"):
        SpokenClaim(text="forty two dollars", field_paths=())


def test_composer_rejects_uncited_and_ignores_genie_prose():
    composer = EvidenceComposer()
    # Prose-only Genie payload → refuse, never speak prose numbers.
    ev = evidence_from_genie_space(
        {"answer": "The balance is $1,234.56", "rows": None, "columns": []}
    )
    assert ev.display_prose and "1,234" in ev.display_prose
    assert ev.error is not None and ev.error.code == ErrorCode.NO_EVIDENCE
    claims, refuse = composer.compose_or_refuse(ev)
    assert claims == []
    assert refuse
    assert "cite" in refuse.lower() or "data" in refuse.lower() or "find" in refuse.lower()


def test_empty_genie_refuses():
    composer = EvidenceComposer()
    ev = evidence_from_genie_space(None)
    claims, refuse = composer.compose_or_refuse(ev)
    assert claims == []
    assert refuse
    assert ev.error is not None and ev.error.code == ErrorCode.NO_EVIDENCE


def test_tabular_genie_produces_cited_claims():
    composer = EvidenceComposer()
    ev = evidence_from_genie_space(
        {
            "answer": "ignore this prose amount $999",
            "columns": ["customer", "balance"],
            "rows": [["CH-1", 42]],
            "message_id": "m1",
        }
    )
    assert ev.has_tabular
    claims, refuse = composer.compose_or_refuse(ev)
    assert refuse is None
    assert len(claims) == 1
    assert claims[0].field_paths
    assert "42" in claims[0].text
    assert "999" not in claims[0].text  # prose not spoken


def test_genie_one_prose_answer_is_attributed_and_speakable():
    # Many legitimate questions ("what can you answer for me?") have no tabular
    # answer at all. Genie One IS the governed answering service, and its answer
    # arrives under a response id, so it is cited evidence — refusing it would make
    # every descriptive question unanswerable.
    ev = evidence_from_genie_one(
        {
            "status": "completed",
            "conversation_id": "c1",
            "response_id": "r1",
            "answer": "I can answer revenue, churn and pipeline questions for you.",
        }
    )
    assert ev.has_attributed_prose
    assert ev.prose is not None and ev.prose.citations == ["genie_one:r1"]
    claims, refuse = EvidenceComposer().compose_or_refuse(ev)
    assert refuse is None
    assert len(claims) == 1
    assert "revenue" in claims[0].text
    assert claims[0].field_paths == (EvidenceComposer.PROSE_PATH,)


def test_genie_one_live_payload_shape_is_mapped():
    # Pinned to what the managed MCP server ACTUALLY returns (captured from a live
    # round-trip): the answer arrives as `final_answer`, not `answer`. Reading only
    # `answer` silently turned every real answer into a "nothing citable" refusal,
    # so this shape is a regression test, not an example.
    ev = evidence_from_genie_one(
        {
            "response_id": "f9ba376b9da448bab481989311dee05d",
            "conversation_id": "5ed74ad9a7414c24a7e1bbc2338f64f0",
            "status": "completed",
            "deep_link": "https://example.cloud.databricks.com/one/chat/threads/5ed7",
            "final_answer": "## Available domains\n\nBanking, contact center, card issuer.",
            "progress_steps": None,
            "narration_instruction": None,
            "query_items": None,
        }
    )
    assert ev.has_attributed_prose
    assert ev.prose is not None
    assert "Banking" in ev.prose.text
    assert ev.prose.citations == ["genie_one:f9ba376b9da448bab481989311dee05d"]
    assert ev.meta["deep_link"].endswith("threads/5ed7")
    claims, refuse = EvidenceComposer().compose_or_refuse(ev)
    assert refuse is None and claims


def test_genie_one_typed_query_results_become_table_evidence():
    ev = evidence_from_genie_one(
        {
            "status": "completed",
            "conversation_id": "c1",
            "response_id": "r1",
            "final_answer": "Shopping leads spending.",
            "query_results": [
                {
                    "item_id": "q1",
                    "sql": "select category, spend from monthly_spending",
                    "columns": [
                        {"name": "category", "type_text": "STRING"},
                        {"name": "spend", "type_text": "DECIMAL"},
                    ],
                    "rows": [["Shopping", "73998.64"], ["Groceries", "48115.80"]],
                    "total_row_count": 2,
                    "truncated": False,
                }
            ],
        }
    )

    assert ev.table is not None
    assert ev.table.columns == ["category", "spend"]
    assert ev.table.rows[0] == ["Shopping", "73998.64"]
    assert ev.table.sql == "select category, spend from monthly_spending"
    assert len(ev.meta["query_results"]) == 1


def test_genie_one_multi_step_answer_is_read_off_the_measure_not_the_first_step():
    # Captured from a live round-trip for "how much have we spent in tokens for Qwen
    # Three Next". Genie answers in steps: it first resolves how the model is spelled
    # in the data, THEN aggregates. Reading query_results[0] made the spoken answer a
    # list of model names — a plausible-looking answer to a question nobody asked —
    # and dropped the measured total entirely. The measure decides, not the order.
    ev = evidence_from_genie_one(
        {
            "status": "completed",
            "conversation_id": "4ab787fcdd7c4469a48f0ffe3f224358",
            "response_id": "r9",
            "final_answer": "Your account consumed ~57.85 million tokens across Qwen3 Next.",
            "query_results": [
                {
                    "item_id": "step1",
                    "sql": "SELECT DISTINCT `destination_model` FROM `system`.`ai_gateway`.`usage`",
                    "columns": [{"name": "destination_model", "type_text": "STRING"}],
                    "rows": [["Qwen3 Next Instruct"], ["qwen3-next-80b-a3b-instruct"]],
                    "total_row_count": 6,
                },
                {
                    "item_id": "step2",
                    "sql": "SELECT `destination_model`, SUM(`total_tokens`) AS total_tokens ...",
                    "columns": [
                        {"name": "destination_model", "type_text": "STRING"},
                        {"name": "total_tokens", "type_text": "BIGINT"},
                    ],
                    "rows": [["Qwen3 Next Instruct", "56999730"]],
                    "total_row_count": 1,
                },
            ],
        }
    )

    assert ev.table is not None
    assert ev.table.columns == ["destination_model", "total_tokens"]
    assert ev.table.sql is not None and "SUM" in ev.table.sql
    # Every step still reaches the screen; only the speech evidence is chosen.
    assert len(ev.meta["query_results"]) == 2

    claims, refuse = EvidenceComposer().compose_or_refuse(ev)
    assert refuse is None
    assert "56999730" in claims[0].text


def test_genie_one_answer_survives_a_table_so_it_can_be_rendered():
    # The runtime speaks a translated summary of Genie One's answer and paints the
    # full answer on screen; both read ``evidence.prose``. Demoting the answer to
    # display-only whenever rows existed silently disabled that whole path.
    ev = evidence_from_genie_one(
        {
            "status": "completed",
            "response_id": "r10",
            "final_answer": "## Token usage\n\nAbout 57.85 million tokens since February.",
            "query_results": [
                {
                    "item_id": "step1",
                    "columns": [{"name": "total_tokens", "type_text": "BIGINT"}],
                    "rows": [["56999730"]],
                }
            ],
        }
    )
    assert ev.has_tabular and ev.has_attributed_prose
    assert ev.prose is not None and "57.85 million" in ev.prose.text
    assert ev.display_prose is None


def test_genie_one_step_without_a_measure_is_still_usable_evidence():
    # A question whose honest answer IS a list of names has no measure anywhere. It
    # must not fall through to a refusal just because nothing numeric came back.
    ev = evidence_from_genie_one(
        {
            "status": "completed",
            "response_id": "r11",
            "query_results": [
                {
                    "item_id": "only",
                    "sql": "SELECT DISTINCT domain FROM catalog",
                    "columns": [{"name": "domain", "type_text": "STRING"}],
                    "rows": [["banking"], ["telco"]],
                }
            ],
        }
    )
    assert ev.has_tabular
    assert ev.table is not None and ev.table.columns == ["domain"]
    assert ev.table.sql == "SELECT DISTINCT domain FROM catalog"


def test_genie_one_prose_without_an_upstream_id_is_not_speakable():
    # No id means nothing to attribute the sentence to, so it stays display-only.
    ev = evidence_from_genie_one({"status": "completed", "answer": "Revenue is up 40%."})
    assert not ev.has_attributed_prose
    assert ev.display_prose == "Revenue is up 40%."
    claims, refuse = EvidenceComposer().compose_or_refuse(ev)
    assert claims == [] and refuse


def test_genie_one_cells_are_quoted_ahead_of_its_own_rounding():
    # A number spoken as a fact comes from a cell, so "1042" is quoted rather than
    # Genie's "roughly a thousand". The answer itself stays attributed: it is the
    # governed service's own reply under this response id, and the runtime renders it
    # for the caller's language and the screen.
    ev = evidence_from_genie_one(
        {
            "status": "completed",
            "response_id": "r2",
            "answer": "Roughly a thousand orders.",
            "columns": ["orders"],
            "rows": [[1042]],
            "sql": "select count(*) from orders",
        }
    )
    assert ev.has_tabular and ev.has_attributed_prose
    claims, refuse = EvidenceComposer().compose_or_refuse(ev)
    assert refuse is None
    assert "1042" in claims[0].text
    assert "thousand" not in claims[0].text


def test_genie_one_statement_shaped_result_is_read_as_a_table():
    ev = evidence_from_genie_one(
        {
            "status": "completed",
            "response_id": "r3",
            "query_result": {
                "manifest": {"schema": {"columns": [{"name": "domain"}, {"name": "tables"}]}},
                "data_array": [["finance", 12]],
            },
        }
    )
    assert ev.has_tabular
    assert ev.table is not None and ev.table.columns == ["domain", "tables"]


def test_genie_one_timeout_and_failure_are_typed_errors():
    timed_out = evidence_from_genie_one({"timeout": True, "error": "Genie One timed out"})
    assert timed_out.error is not None and timed_out.error.code == ErrorCode.TIMEOUT
    failed = evidence_from_genie_one({"status": "failed", "response_id": "r4"})
    assert failed.error is not None and failed.error.code == ErrorCode.UNSUPPORTED


def test_workspace_query_results_route_through_the_genie_one_adapter():
    ev = evidence_from_tool_result(
        "workspace_query",
        {"status": "completed", "response_id": "r5", "answer": "Sales and support domains."},
    )
    assert ev.source == "genie_one"
    assert ev.has_attributed_prose


def test_long_prose_is_trimmed_for_speech_but_kept_whole_on_the_wire():
    body = "First sentence about the data. " + ("Filler detail follows. " * 60)
    ev = evidence_from_genie_one(
        {"status": "completed", "response_id": "r6", "answer": f"## Heading\n\n- {body}"}
    )
    claims, refuse = EvidenceComposer().compose_or_refuse(ev)
    assert refuse is None
    spoken = claims[0].text
    assert len(spoken) <= 601
    assert "#" not in spoken and "- " not in spoken
    # The full answer still reaches the client for display.
    assert len(str(ev.as_dict()["prose"]["text"])) > len(spoken)


def test_agent_mode_tables_to_evidence():
    ev = evidence_from_agent_mode(
        [{"columns": ["cat", "amt"], "preview_rows": [["fees", 10]], "sql": "select 1"}],
        report_text="Fees rose because of X",
    )
    assert ev.has_tabular
    assert ev.display_prose and "Fees rose" in ev.display_prose
    composer = EvidenceComposer()
    claims, refuse = composer.compose_or_refuse(ev)
    assert refuse is None and claims


def test_tool_events_emit_before_final_text():
    import asyncio

    def fake_respond(transcript: str, **kwargs: Any) -> tuple[str, list[dict[str, Any]]]:
        return "Your balance is 42.", [
            {"name": "lookup_account", "arguments": {"customer_id": "C1"}, "result": {"ok": True}}
        ]

    async def _collect() -> list[AgentEventKind]:
        adapter = RespondWithToolsAdapter(fake_respond)
        kinds: list[AgentEventKind] = []
        async for ev in adapter.run(
            AgentGoal(utterance="what is my balance?", language="en"),
            AgentContext(turn_id=7),
            CancellationToken(),
        ):
            kinds.append(ev.kind)
        return kinds

    kinds = asyncio.run(_collect())
    assert AgentEventKind.ACTION_STARTED in kinds
    assert AgentEventKind.ACTION_COMPLETED in kinds
    assert AgentEventKind.ANSWER_FINAL in kinds
    assert kinds.index(AgentEventKind.ACTION_STARTED) < kinds.index(AgentEventKind.ANSWER_FINAL)
    assert kinds.index(AgentEventKind.ACTION_COMPLETED) < kinds.index(AgentEventKind.ANSWER_FINAL)


def test_live_adapter_emits_evidence_before_display_only_answer():
    import asyncio

    def fake_respond(transcript: str, **kwargs: Any):
        on_tool = kwargs["on_tool"]
        on_tool("started", "lookup_account", arguments={})
        on_tool("completed", "lookup_account", arguments={}, result={"balance": 42})
        return "The model can say anything here.", [{"name": "lookup_account"}]

    async def _collect():
        events = []
        async for event in LiveToolRespondAdapter(fake_respond).run(
            AgentGoal(utterance="balance", language="en"),
            AgentContext(turn_id=8),
            CancellationToken(),
        ):
            events.append(event)
        return events

    events = asyncio.run(_collect())
    kinds = [event.kind for event in events]
    assert kinds.index(AgentEventKind.EVIDENCE_AVAILABLE) < kinds.index(
        AgentEventKind.ANSWER_FINAL
    )
    final = next(event for event in events if event.kind == AgentEventKind.ANSWER_FINAL)
    assert final.payload["display_only"] is True


# ---------------------------------------------------------------------------
# Phase 4 — OBO fail-closed
# ---------------------------------------------------------------------------


def test_no_token_genie_not_invoked(monkeypatch):
    """Missing OBO token → ask_genie returns refuse JSON and never calls Genie."""
    called = {"ask": False}

    def boom_genie():
        called["ask"] = True
        raise AssertionError("Genie must not be invoked without OBO token")

    monkeypatch.setattr("api.app.deps.genie", boom_genie, raising=False)

    ctx = ToolContext(
        customer_id="C1",
        principal=SessionPrincipal(),  # no token
        session_id="sess-1",
        turn_id=3,
    )
    out = json.loads(_run_ask_genie({"question": "how many overdue?"}, ctx))
    assert out.get("obo_deny") is True
    assert out.get("error_evidence", {}).get("code") == "permission"
    assert called["ask"] is False


def test_require_obo_token_denies_empty_principal():
    err = require_obo_token(None, session_id="s1", turn_id=1)
    assert err is not None and err.code == ErrorCode.PERMISSION
    ok = require_obo_token(SessionPrincipal(access_token="tok"), session_id="s1")
    assert ok is None


def test_denied_workspace_query_maps_to_permission_refusal():
    """A fail-closed Genie One call must not be mistaken for an empty result.

    The caller has to hear "you're not authorized", not "I found nothing to cite",
    so failures are classified before any per-tool result mapping.
    """
    evidence = evidence_from_tool_result(
        "workspace_query",
        {"denied": True, "error": "missing_user_token"},
    )
    assert evidence.error is not None
    assert evidence.error.code == ErrorCode.PERMISSION
    assert evidence.table is None


def test_genie_obo_or_refuse_with_token_passes():
    ctx = ToolContext(
        principal=SessionPrincipal(access_token="user-token"),
        session_id="s",
    )
    assert genie_obo_or_refuse(ctx) is None


# ---------------------------------------------------------------------------
# Phase 5 — GoalFrame / misroutes
# ---------------------------------------------------------------------------


def test_misroute_lakebase_not_genie_one():
    """Lakebase-answerable pack facts must not escalate to Genie One."""
    d = cascade_route(
        "what is my overdue balance?",
        pack_id="billing",
        pack_intent_matched=True,
        has_fast_facts=True,
        has_obo=True,
    )
    assert d.adapter != "genie_one_mcp"
    assert d.adapter in ("pack_facts", "space_conversation")


def test_misroute_pack_fact_not_workspace():
    d = cascade_route(
        "how much is my statement total this cycle?",
        pack_id="card",
        pack_intent_matched=True,
        has_fast_facts=False,
        has_obo=True,
    )
    assert d.frame is not None and d.frame.scope == "pack"
    assert d.adapter != "genie_one_mcp"


def test_why_in_pack_routes_agent_mode():
    d = cascade_route(
        "why did my rewards points drop this cycle?",
        pack_id="card",
        pack_intent_matched=True,
        has_fast_facts=True,
        has_obo=True,
    )
    assert d.frame is not None
    assert d.frame.depth == "investigate"
    assert d.adapter == "agent_mode"
    assert adapter_for(d.frame) == "agent_mode"


def test_pack_fact_matrix_space_conversation():
    frame = GoalFrame(scope="pack", depth="fact", pack_id="card")
    assert adapter_for(frame) == "space_conversation"


def test_workspace_requires_obo_else_refuse():
    d = cascade_route(
        "what is my workspace cluster cost?",
        pack_id=None,
        has_obo=False,
    )
    assert d.adapter == "refuse"


def test_depth_classifier_and_ambiguous_clarify():
    assert classify_depth("why did fees jump?") == "investigate"
    assert classify_depth("how much is my balance?") == "fact"
    d = cascade_route(
        "tell me about that",
        pack_id="card",
        pack_intent_matched=True,
        has_obo=True,
    )
    assert d.adapter == "clarify" or (d.frame and d.frame.depth == "fact")


def test_barge_classifier_amend_new_stop():
    active = GoalFrame(scope="pack", depth="investigate", pack_id="card")
    assert classify_barge_intent("stop", active_goal=active) == "stop"
    assert classify_barge_intent("actually only last month", active_goal=active) == "amend"
    assert classify_barge_intent("what's the weather", active_goal=active) == "new"


def test_effect_gateway_confirm_mutate():
    assert enforce_effect("read") is None
    blocked = enforce_effect("confirm_mutate", user_confirmed=False)
    assert blocked is not None and blocked.code == ErrorCode.AMBIGUOUS
    assert enforce_effect("confirm_mutate", user_confirmed=True) is None
    assert enforce_effect("mutate", mutate_succeeded=False) is not None
    assert enforce_effect("mutate", mutate_succeeded=True) is None


def test_evidence_table_direct():
    ev = Evidence(
        source="lakebase",
        table=TableEvidence(columns=["a"], rows=[[1]], citations=["lakebase"]),
    )
    composer = EvidenceComposer()
    claims, refuse = composer.compose_or_refuse(ev)
    assert refuse is None and claims[0].field_paths[0].endswith(".a")
