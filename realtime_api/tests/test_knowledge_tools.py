"""Tests for the Databricks Knowledge Agent profile (knowledge_tools).

These guard the CITE-OR-SILENCE contract without any model or Databricks calls:
an on-corpus question returns entries that each carry a citation, and an
off-corpus question returns NO matches so the agent has nothing to answer from.
"""
from __future__ import annotations

import json

from realtime_api import knowledge_tools
from realtime_api.profiles import get_profile


def _search(query: str) -> dict:
    return json.loads(knowledge_tools._run_knowledge_search({"query": query}, None))


def test_on_corpus_questions_rank_their_own_topic_first():
    assert _search("what is unity catalog")["matches"][0]["topic"] == "Unity Catalog"
    assert _search("why delta lake instead of parquet")["matches"][0]["topic"] == "Delta Lake"
    assert _search("how do I serve a model behind an api")["matches"][0]["topic"] == "Model Serving"


def test_every_corpus_entry_has_a_known_category():
    for entry in knowledge_tools.KNOWLEDGE_CORPUS:
        assert entry["category"] in knowledge_tools.KNOWLEDGE_CATEGORIES


def test_workspace_questions_are_not_claimed_by_documentation_retrieval():
    # Routing is now a multilingual semantic capability decision. The docs tool
    # still has to return empty evidence for organization-specific questions so a
    # future regression cannot turn an accidental tool call into a factual answer.
    for question in (
        "which jobs failed in the last 24 hours",
        "what is the DBU usage in this workspace this month",
        "which tables are tagged as containing sensitive data",
        "what were the slowest sql queries today",
    ):
        assert _search(question)["matches"] == []


def test_documentation_questions_are_citable_by_the_docs_tool():
    for question in ("what is unity catalog", "what is photon", "what does vector search do"):
        assert _search(question)["matches"]


def test_profile_has_no_keyword_lane_resolver():
    profile = get_profile("knowledge")
    assert profile is not None
    assert profile.resolve_lane is None


def test_published_topics_are_all_live_and_carry_no_canned_answer():
    topics = knowledge_tools.knowledge_topics()
    ids = [t["id"] for t in topics]
    assert ids, "the page must publish questions"
    assert len(ids) == len(set(ids))
    for topic in topics:
        # Only the live lane is published, and it never ships a canned answer: the
        # answer must come from the workspace, so the page cannot show a number
        # Genie One did not produce.
        assert topic["lane"] == "workspace"
        assert "answer" not in topic
        assert topic["preview"].strip()
        assert topic["question"].strip()
        assert topic["source"].strip()
        assert topic["category"] in knowledge_tools.KNOWLEDGE_CATEGORIES


def test_every_match_carries_a_citation():
    # The agent can only attribute what the tool hands it, so a match without a
    # citation would silently become an unattributed spoken claim.
    for match in _search("what does vector search do")["matches"]:
        assert match["citation"].strip()
        assert match["answer"].strip()


def test_off_corpus_question_returns_no_matches():
    # Silence, not a guess: the empty result plus guidance is what the system
    # prompt leans on to make the agent say it does not know.
    result = _search("what is the weather in paris")
    assert result["matches"] == []
    assert result["guidance"]
    assert "Unity Catalog" in result["topics"]


def test_empty_query_matches_nothing():
    assert _search("")["matches"] == []


def test_stopwords_alone_do_not_match_everything():
    # "what is it about" is all stopwords; scoring them would rank the whole corpus.
    assert _search("what is it about")["matches"] == []


def test_search_is_capped_so_the_agent_gets_a_short_evidence_set():
    assert len(knowledge_tools.search_knowledge("databricks catalog delta genie vector", limit=3)) <= 3


def test_profile_registers_with_the_search_tool():
    profile = get_profile("knowledge")
    assert profile is not None
    names = {spec["function"]["name"] for spec in profile.tools_spec()}
    assert names == {"knowledge_search"}


def test_corpus_entries_are_uniquely_identified():
    ids = [entry["id"] for entry in knowledge_tools.KNOWLEDGE_CORPUS]
    assert len(ids) == len(set(ids))
