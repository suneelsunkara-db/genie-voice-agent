"""Shared config access for the realtime voice model scripts.

Single source of truth is ``config/config.yaml`` deep-merged with the gitignored
``config/config.local.yaml`` (local values win) — the same precedence the app
loader uses. The standalone ``realtime_api/model_serving.yaml`` is retired; all
registry/serving settings now live under the ``realtime_voice:`` block.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BASE = _REPO_ROOT / "config" / "config.yaml"
_LOCAL = _REPO_ROOT / "config" / "config.local.yaml"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config() -> dict[str, Any]:
    config: dict[str, Any] = {}
    if _BASE.exists():
        config = yaml.safe_load(_BASE.read_text(encoding="utf-8")) or {}
    if _LOCAL.exists():
        config = _deep_merge(config, yaml.safe_load(_LOCAL.read_text(encoding="utf-8")) or {})
    if not config:
        raise FileNotFoundError("No config found under config/config.yaml or config/config.local.yaml")
    return config


def realtime_voice(config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = config or load_config()
    block = config.get("realtime_voice")
    if not block:
        raise KeyError("Missing 'realtime_voice' block in config")
    return block


def databricks(config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = config or load_config()
    return config.get("databricks") or {}


def find_candidate(candidate_id: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    rv = realtime_voice(config)
    candidate = (rv.get("stt_candidates") or {}).get(candidate_id) or (
        rv.get("tts_candidates") or {}
    ).get(candidate_id)
    if not candidate:
        raise KeyError(f"Unknown realtime voice candidate: {candidate_id}")
    return candidate


def all_candidates(config: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    rv = realtime_voice(config)
    return {**(rv.get("stt_candidates") or {}), **(rv.get("tts_candidates") or {})}


def registered_model_name(candidate: dict[str, Any], config: dict[str, Any] | None = None) -> str:
    db = databricks(config)
    return f"{db['catalog']}.{db['schema']}.{candidate['registered_model']}"
