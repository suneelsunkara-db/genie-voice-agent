"""Run the realtime voice API (backend only; the UI is a separate app).

    python -m realtime_api.server           # ws://localhost:8001/v1/realtime/voice
    PORT=9000 python -m realtime_api.server

The browser client lives in ``realtime_test_ui/`` and connects over the WebSocket;
serve it independently (see README.md at the repo root -> "Realtime Voice API + UI").

Uses the Databricks SDK serving client (no local mlflow needed), authenticating
with the profile from config (``databricks.profile``) or DATABRICKS_CONFIG_PROFILE.
"""
from __future__ import annotations

import logging
import os

from .app import create_app
from .config import RealtimeSettings, databricks_profile
from .services import DatabricksServing
from .session import VoicePipeline


def _pipeline_factory(profile: str | None):
    def factory(settings: RealtimeSettings) -> VoicePipeline:
        serving = DatabricksServing.from_sdk(
            stt_endpoint=settings.stt_endpoint,
            llm_endpoint=settings.llm_endpoint,
            tts_endpoint=settings.tts_endpoint,
            profile=profile,
            llm_temperature=settings.llm_temperature,
            llm_max_tokens=settings.llm_max_tokens,
            llm_tools_enabled=settings.llm_tools_enabled,
            llm_max_tool_iterations=settings.llm_max_tool_iterations,
            tts_inference_timesteps=settings.tts_inference_timesteps,
            tts_cfg_value=settings.tts_cfg_value,
        )
        return VoicePipeline(stt=serving, llm=serving, tts=serving, verify_mode=settings.verify_mode)

    return factory


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logging.getLogger("realtime_voice").setLevel(logging.INFO)

settings = RealtimeSettings.resolve()
app = create_app(settings=settings, pipeline_factory=_pipeline_factory(databricks_profile()))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8001")))
