from __future__ import annotations

import os
from pathlib import Path


def is_volume_mode() -> bool:
    return os.environ.get("ML_ASR_RUN_MODE", "orchestrator").lower() in {"serverless", "volume"}


def databricks_profile() -> str | None:
    """Databricks CLI/SDK profile for local ml_asr tooling, from config.

    Config (``databricks.profile`` in config.local.yaml) is the single source of
    truth. Returns ``None`` on serverless/volume runs so the ambient job identity
    is used instead of a profile.
    """
    if is_volume_mode():
        return None
    from genie_voice.config import get_settings

    return get_settings().databricks.profile or None


def volume_path(path: str) -> str:
    if path.startswith("/Volumes/"):
        return path
    if path.startswith("dbfs:/Volumes/"):
        return path.removeprefix("dbfs:")
    return path


def training_root_from_config_path(path: str | Path) -> str | None:
    config_path = Path(path)
    if not str(config_path).startswith("/Volumes/"):
        return None
    parts = config_path.parts
    if "evaluations" not in parts:
        return None
    idx = parts.index("evaluations")
    if idx <= 1:
        return None
    return str(Path(*parts[:idx]))
