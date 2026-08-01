"""The concierge routing table is config, and it is validated on load.

This table decides which assistant a caller is handed to. A missing or malformed
block must therefore be a loud startup failure — a default table quietly filling
in would route real callers on rules nobody reviewed.
"""
from __future__ import annotations

import textwrap

import pytest

from realtime_api.concierge_tools import resolve_industry
from realtime_api.config import concierge_router_config
from realtime_api.guardrails import GuardLedger

_VALID = """
realtime_voice:
  llm_endpoint: llm
  stt_candidates: {a: {endpoint: stt}}
  tts_candidates: {a: {endpoint: tts}}
  concierge_router:
    languages: [en]
    max_selection_words: 8
    industries:
      telco:
        label: Telco
        confirm_label: Telco billing support
        cues: [telco, phone bill]
      fsi:
        label: Financial Services
        confirm_label: the credit-card assistant
        cues: [card, bank]
"""


def _config(tmp_path, yaml_text: str):
    (tmp_path / "config.yaml").write_text(textwrap.dedent(yaml_text), encoding="utf-8")
    return tmp_path


def test_loads_the_configured_table(tmp_path):
    cfg = concierge_router_config(_config(tmp_path, _VALID))
    assert cfg.languages == ("en",)
    assert cfg.max_selection_words == 8
    assert cfg.keys == ("telco", "fsi")
    assert cfg.industry("telco").confirm_label == "Telco billing support"
    assert cfg.industry("nope") is None


def test_repo_config_is_valid():
    # The shipped config must load: this is what the app boots on.
    cfg = concierge_router_config()
    assert "telco" in cfg.keys and cfg.max_selection_words >= 1


_HEAD = (
    "realtime_voice:\n"
    "  llm_endpoint: llm\n"
    "  stt_candidates: {a: {endpoint: stt}}\n"
    "  tts_candidates: {a: {endpoint: tts}}\n"
)


@pytest.mark.parametrize(
    "block, expected",
    [
        ("  concierge_router: {}\n", "not configured"),
        ("  concierge_router:\n    max_selection_words: 8\n", "languages"),
        ("  concierge_router:\n    languages: []\n    max_selection_words: 8\n", "languages"),
        ("  concierge_router:\n    languages: [en]\n", "max_selection_words"),
        ("  concierge_router:\n    languages: [en]\n    max_selection_words: 0\n", ">= 1"),
        (
            "  concierge_router:\n    languages: [en]\n    max_selection_words: 8\n"
            "    industries: {}\n",
            "at least one industry",
        ),
        (
            "  concierge_router:\n    languages: [en]\n    max_selection_words: 8\n"
            "    industries:\n      telco:\n        label: T\n        cues: [telco]\n",
            "confirm_label",
        ),
    ],
)
def test_malformed_config_fails_fast_with_the_offending_key(tmp_path, block, expected):
    with pytest.raises(RuntimeError, match=expected):
        concierge_router_config(_config(tmp_path, _HEAD + block))


def test_missing_block_is_an_error_not_an_empty_router(tmp_path):
    with pytest.raises(RuntimeError, match="concierge_router is not configured"):
        concierge_router_config(_config(tmp_path, _HEAD))


def test_a_cue_claimed_by_two_industries_is_rejected(tmp_path):
    # Such a cue can never route: the ambiguity gate declines every time it
    # matches, so it would be a dead entry rather than a visible mistake.
    yaml_text = _VALID.replace("cues: [card, bank]", "cues: [card, telco]")
    with pytest.raises(RuntimeError, match="could never route"):
        concierge_router_config(_config(tmp_path, yaml_text))


def test_router_declines_a_language_its_cues_were_not_authored_for():
    # The home page pins STT to English today. If that pin is ever lifted, English
    # cues must not be matched against, say, a Hindi transcript — the router steps
    # aside and the model handles the turn.
    ledger = GuardLedger()
    assert resolve_industry("telco", ledger, "hi-IN") is None
    entry = next(e for e in ledger.entries if e.guard_id == "selection_language_scope")
    assert entry.outcome == "not_evaluated"
    assert "hi" in (entry.reason or "")
    # ...and it still routes the language it IS authored for.
    assert resolve_industry("telco", GuardLedger(), "en-US") == "telco"
