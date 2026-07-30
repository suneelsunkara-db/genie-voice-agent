"""Run the realtime voice API (backend only; the UI is a separate app).

    python -m realtime_api.server           # ws://localhost:8001/v1/speech-llm-toolassist-speech
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
from .config import RealtimeSettings
from .pipelines import ServingBundle
from .serving_factory import shared_serving


def _factory(settings: RealtimeSettings) -> ServingBundle:
    # Process-wide singleton serving (shared with the mounted app): one auth client
    # for every connection, and warm-up primes the same replicas that serve turns.
    serving = shared_serving()
    return ServingBundle(stt=serving, llm=serving, tts=serving)


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logging.getLogger("realtime_voice").setLevel(logging.INFO)

settings = RealtimeSettings.resolve()
app = create_app(settings=settings, bundle_factory=_factory)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8001")))
