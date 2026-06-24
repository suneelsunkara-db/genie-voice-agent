from __future__ import annotations

from fastapi import APIRouter

from genie_voice.config import get_settings

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict:
    s = get_settings()
    return {
        "status": "ok",
        "deployment": s.deployment,
        "mode": s.mode,
        "stt_provider": s.providers.stt.active,
        "tts_provider": s.providers.tts.active,
        "databricks_host": s.databricks_host,
        "catalog": s.databricks.catalog,
        "schema": s.databricks.schema_name,
    }
