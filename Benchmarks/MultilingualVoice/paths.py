"""Resolve benchmark paths + config + auth from a single explicit contract.

The entry-point dir is an explicit ``--benchmark-dir`` parameter (set by the
job submitter, or the local cwd). No ``__file__`` / ``sys.argv[0]`` / cwd
ladder — one input, one contract.

Config is loaded once and cached; resolvers read the cache.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any


def benchmark_dir() -> Path:
    """Directory holding these benchmark modules (explicit contract)."""
    raw = os.getenv("MLV_BENCHMARK_DIR")
    if not raw:
        raise RuntimeError(
            "MLV_BENCHMARK_DIR must be set (the job submitter passes --benchmark-dir; "
            "local runs set it in run_benchmark.py)."
        )
    return Path(raw)


def repo_root() -> Path:
    raw = os.getenv("GENIE_REPO_ROOT")
    if raw:
        return Path(raw)
    return benchmark_dir().parents[1]


def config_dir() -> Path:
    raw = os.getenv("GENIE_CONFIG")
    if raw:
        path = Path(raw)
        return path.parent if path.suffix else path
    return repo_root() / "config"


@lru_cache(maxsize=1)
def _merged_config() -> dict[str, Any]:
    import yaml

    merged: dict[str, Any] = {}
    for name in ("config.yaml", "config.local.yaml"):
        path = config_dir() / name
        if path.exists():
            block = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            merged = _deep_merge(merged, block)
    return merged


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def resolve_volume_path(template: str) -> str:
    cfg = _merged_config()
    db = cfg.get("databricks") or {}
    vol = cfg.get("volume") or {}
    return template.format(
        catalog=db.get("catalog", ""),
        schema=db.get("schema", ""),
        batch_volume=vol.get("batch_name", "raw_batch_data"),
        streaming_volume=vol.get("streaming_name", "raw_streaming_data"),
    )


def benchmark_results_dir() -> Path:
    """UC Volume path for benchmark results/logs (the canonical store)."""
    override = os.getenv("MLV_RESULTS_DIR")
    if override:
        return Path(override)
    template = str((_merged_config().get("volume") or {}).get("multilingual_voice_benchmark_path") or "").strip()
    if not template:
        raise RuntimeError("volume.multilingual_voice_benchmark_path is not configured")
    return Path(resolve_volume_path(template))


def _benchmark_block() -> dict[str, Any]:
    return (_merged_config().get("realtime_voice") or {}).get("benchmark") or {}


def benchmark_api_host() -> str:
    override = os.getenv("MLV_API_HOST")
    if override:
        return override.rstrip("/")
    host = str(_benchmark_block().get("api_host") or "").strip()
    if host:
        return host.rstrip("/")
    raise RuntimeError("Set MLV_API_HOST or realtime_voice.benchmark.api_host")


def benchmark_api_prefix() -> str:
    if os.getenv("MLV_API_PREFIX") is not None:
        return os.getenv("MLV_API_PREFIX", "").strip("/")
    return str(_benchmark_block().get("api_prefix") or "realtime").strip("/")


def databricks_host() -> str:
    for var in ("DATABRICKS_HOST", "MLV_HOST"):
        value = os.getenv(var)
        if value:
            return value.rstrip("/")
    return str((_merged_config().get("databricks") or {}).get("host") or "").rstrip("/")


def databricks_profile() -> str | None:
    profile = os.getenv("DATABRICKS_CONFIG_PROFILE") or os.getenv("DATABRICKS_PROFILE")
    if profile:
        return profile.strip() or None
    value = str((_merged_config().get("databricks") or {}).get("profile") or "").strip()
    return value or None


def benchmark_sp_credentials() -> tuple[str | None, str | None]:
    """Service-principal (client_id, client_secret) for app M2M auth."""
    client_id = os.getenv("MLV_SP_CLIENT_ID")
    client_secret = os.getenv("MLV_SP_CLIENT_SECRET")
    if client_id and client_secret:
        return client_id.strip(), client_secret.strip()
    auth = _benchmark_block().get("auth") or {}
    return (
        str(auth.get("client_id") or "").strip() or None,
        str(auth.get("client_secret") or "").strip() or None,
    )


def delta_catalog() -> str:
    return str((_merged_config().get("databricks") or {}).get("catalog") or "")


def delta_schema() -> str:
    return str((_merged_config().get("databricks") or {}).get("schema") or "")
