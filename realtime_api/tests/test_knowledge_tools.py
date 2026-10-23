"""Tests for Knowledge starter metadata and the Genie One-only profile."""
from __future__ import annotations

import hashlib
import json

from genie_voice.i18n import LANGUAGE_SPECS, content_language
from realtime_api import knowledge_tools
from realtime_api.profiles import get_profile


def test_every_corpus_entry_has_a_known_category():
    for entry in knowledge_tools.KNOWLEDGE_CORPUS:
        assert entry["category"] in knowledge_tools.KNOWLEDGE_CATEGORIES


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
        assert topic["canonical_question"] == topic["question"]


def test_localized_topics_keep_the_canonical_genie_query_separate():
    english = {topic["id"]: topic for topic in knowledge_tools.knowledge_topics("en-US")}
    hindi = {topic["id"]: topic for topic in knowledge_tools.knowledge_topics("hi-IN")}

    assert set(hindi) == set(english)
    for topic_id, localized in hindi.items():
        assert localized["canonical_question"] == english[topic_id]["question"]
        assert localized["question"] != english[topic_id]["question"]
        assert localized["preview"] != english[topic_id]["preview"]
        assert localized["source"] != english[topic_id]["source"]


def test_localized_topic_asset_covers_every_product_language():
    catalogs = knowledge_tools._topic_locales()
    expected = {content_language(language) for language in LANGUAGE_SPECS}
    assert expected <= set(catalogs)

    source = {
        "categories": list(knowledge_tools.KNOWLEDGE_CATEGORIES),
        "topics": {
            prompt["id"]: {
                field: prompt[field]
                for field in ("topic", "question", "preview", "source")
            }
            for prompt in knowledge_tools.WORKSPACE_PROMPTS
        },
    }
    source_hash = hashlib.sha256(
        json.dumps(source, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]

    expected_ids = {prompt["id"] for prompt in knowledge_tools.WORKSPACE_PROMPTS}
    for language in expected:
        catalog = catalogs[language]
        assert catalog["_sourceHash"] == source_hash
        assert set(catalog["categories"]) == set(knowledge_tools.KNOWLEDGE_CATEGORIES)
        assert set(catalog["topics"]) == expected_ids
        for topic in catalog["topics"].values():
            assert set(topic) == {"topic", "question", "preview", "source"}
            assert all(str(value).strip() for value in topic.values())


def test_profile_exposes_no_local_answering_tools():
    profile = get_profile("knowledge")
    assert profile is not None
    assert profile.tools_spec() == []
    assert "Genie One" in profile.system_prompt


def test_corpus_entries_are_uniquely_identified():
    ids = [entry["id"] for entry in knowledge_tools.KNOWLEDGE_CORPUS]
    assert len(ids) == len(set(ids))
