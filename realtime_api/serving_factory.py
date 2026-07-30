"""Single, config-driven builder for the realtime ``DatabricksServing``.

One place constructs the STT/LLM/TTS serving adapter from ``RealtimeSettings`` so
every consumer shares the SAME instance and the SAME (config-sourced) knobs —
the WebSocket voice loop (mounted in ``api/app/main.py``) and any off-loop LLM
use such as the deep-dive spoken-summary. This removes the previous second,
hardcoded ``DatabricksServing`` that lived in the card router.
"""
from __future__ import annotations

import os
from functools import lru_cache

from .config import RealtimeSettings, databricks_profile
from .services import DatabricksServing


def serving_profile() -> str | None:
    """CLI profile to auth the SDK with — or None when app SP creds are present.

    Databricks Apps inject service-principal OAuth via env (no CLI profile);
    locally we use the configured CLI profile because ``~/.databrickscfg`` may
    hold several profiles for one host (``profile=None`` would make the SDK raise).
    """
    if os.getenv("DATABRICKS_CLIENT_ID") or os.getenv("DATABRICKS_APP_NAME"):
        return None
    profile = databricks_profile()
    if not profile or profile.startswith("<"):  # unfilled placeholder
        return None
    return profile


def build_serving(settings: RealtimeSettings) -> DatabricksServing:
    """Construct a ``DatabricksServing`` from resolved config (not cached)."""
    return DatabricksServing.from_sdk(
        stt_endpoint=settings.stt_endpoint,
        llm_endpoint=settings.llm_endpoint,
        tts_endpoint=settings.tts_endpoint,
        profile=serving_profile(),
        llm_temperature=settings.llm_temperature,
        llm_max_tokens=settings.llm_max_tokens,
        llm_tools_enabled=settings.llm_tools_enabled,
        llm_max_tool_iterations=settings.llm_max_tool_iterations,
        tts_inference_timesteps=settings.tts_inference_timesteps,
        tts_cfg_value=settings.tts_cfg_value,
        predict_timeout_s=settings.predict_timeout_s,
        tts_stream_timeout_s=settings.tts_stream_timeout_s,
    )


@lru_cache(maxsize=1)
def shared_serving() -> DatabricksServing:
    """Process-wide singleton serving, built once from resolved config."""
    return build_serving(RealtimeSettings.resolve())
