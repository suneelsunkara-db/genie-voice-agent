"""The concierge destination allowlist is config and validated on load.

This table decides which assistant a caller is handed to. A missing or malformed
block must therefore be a loud startup failure — a default table quietly filling
in would route real callers on rules nobody reviewed.
"""
from __future__ import annotations

import textwrap

import pytest

from realtime_api.config import concierge_router_config

_VALID = """
realtime_voice:
  llm_endpoint: llm
  stt_candidates: {a: {endpoint: stt}}
  tts_candidates: {a: {endpoint: tts}}
  concierge_router:
    industries:
      telco:
        label: Telco
        confirm_label: Telco billing support
      fsi:
        label: Financial Services
        confirm_label: the credit-card assistant
"""


def _config(tmp_path, yaml_text: str):
    (tmp_path / "config.yaml").write_text(textwrap.dedent(yaml_text), encoding="utf-8")
    return tmp_path


def test_loads_the_configured_table(tmp_path):
    cfg = concierge_router_config(_config(tmp_path, _VALID))
    assert cfg.keys == ("telco", "fsi")
    assert cfg.industry("telco").confirm_label == "Telco billing support"
    assert cfg.industry("nope") is None


def test_repo_config_is_valid():
    # The shipped config must load: this is what the app boots on.
    cfg = concierge_router_config()
    assert "telco" in cfg.keys


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
        (
            "  concierge_router:\n    industries: {}\n",
            "at least one industry",
        ),
        (
            "  concierge_router:\n"
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


