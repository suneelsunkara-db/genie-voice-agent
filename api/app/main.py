"""FastAPI application entrypoint.

All settings (host, port, CORS) come from config. Run:
    uvicorn app.main:app --reload --port 8000   (cwd = api/)
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# Configure the realtime_voice logger so pipeline events (turn lifecycle,
# errors, latency) are visible in the uvicorn console output.
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logging.getLogger("realtime_voice").setLevel(logging.INFO)

# The standalone ``realtime_api`` package lives at the repo root (not pip-installed).
# uvicorn is launched with cwd=api/ (start_app.sh), so add the repo root to the
# import path here; otherwise ``import realtime_api`` fails and the /realtime mount
# is silently skipped, falling through to the SPA catch-all.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from genie_voice.config import get_settings

from .deps import serving
from .routers import accounts, agent_assist, asr_benchmark, genie, health, mic_stream, pipeline_status

# Built React UI (populated by deploy_app.sh: `frontend build` -> here). When
# present (e.g. on Databricks Apps, which runs a single web process) the API also
# serves the SPA so UI and API share one origin. Absent in local dev, where Vite
# serves the UI on :5173 separately.
_FRONTEND_DIST = Path(__file__).resolve().parent / "static"


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="Genie Voice Agent API", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(agent_assist.router)
    app.include_router(mic_stream.router)
    app.include_router(accounts.router)
    app.include_router(genie.router)
    app.include_router(asr_benchmark.router)
    app.include_router(pipeline_status.router)

    @app.on_event("startup")
    def _ensure_lakebase_serving_schema() -> None:
        """Create/upgrade shared Lakebase serving tables on API startup.

        Keeps table creation centralized in LakebaseServing.ensure_schema()
        so new serving tables (like resolution_events) are provisioned once.

        Runs OFF-THREAD: schema provisioning hits Lakebase Postgres + the SQL
        warehouse, both of which can be cold (tens of seconds) or slow. Doing it
        synchronously here blocks the event loop and prevents the server from ever
        binding (observed as a startup hang at the Databricks auth step). It is
        idempotent and only needed before the first billing/resolution write, so
        we provision it in the background and let the server come up immediately.
        """
        import threading

        def _work() -> None:
            try:
                serving().ensure_schema()
            except Exception as exc:  # noqa: BLE001
                print(f"[api-startup] Lakebase schema ensure skipped: {exc}")
            if settings.lakebase.enabled and not settings.databricks.sql_warehouse_id:
                print(
                    "[api-startup] WARNING: lakebase.enabled requires databricks.sql_warehouse_id "
                    "for governed UC billing writes; close/billing will fail until configured."
                )
            try:
                from genie_voice.databricks.warehouse_sql import (
                    ensure_billing_adjustments_table,
                    warehouse_configured,
                )

                if warehouse_configured(settings):
                    ensure_billing_adjustments_table(settings)
            except Exception as exc:  # noqa: BLE001
                print(f"[api-startup] UC billing_adjustments ensure skipped: {exc}")

        threading.Thread(target=_work, daemon=True, name="api-schema-ensure").start()

    @app.on_event("startup")
    def _warm_databricks_paths() -> None:
        """Warm slow Databricks dependencies in the background so the first page
        load doesn't pay cold-start costs: the workspace-client OAuth handshake and
        the SQL Warehouse + Jobs API lookups behind `/status`. The Lakebase pool is
        already warmed by `ensure_schema()` above. Runs off-thread so startup (and
        readiness) is not delayed.
        """
        import threading

        def _work() -> None:
            try:
                from genie_voice.databricks.client import get_workspace_client

                get_workspace_client(settings)
            except Exception as exc:  # noqa: BLE001
                print(f"[api-startup] workspace client warm skipped: {exc}")
            try:
                from .routers.pipeline_status import warm_meta

                warm_meta(settings)
            except Exception as exc:  # noqa: BLE001
                print(f"[api-startup] status meta warm skipped: {exc}")

        threading.Thread(target=_work, daemon=True, name="api-warm").start()

    _mount_realtime(app)
    _mount_realtime_test_ui(app)
    _mount_frontend(app)
    return app


def _mount_realtime(app: FastAPI) -> None:
    """Additively mount the standalone Realtime Voice API under ``/realtime``.

    This is fully isolated from the contact-center routes: it exposes
    ``/realtime/v1/speech-to-text``, ``/realtime/v1/speech-llm-toolassist-speech``,
    ``/realtime/v1/text-to-speech``, and ``/realtime/v1/languages``.
    Auth differs by environment: on Databricks Apps the injected service
    principal creds (``DATABRICKS_CLIENT_ID``/``_SECRET`` + host) are picked up
    with ``profile=None``; locally there is no injected SP and ``~/.databrickscfg``
    may hold several profiles for the same host, so ``profile=None`` makes the SDK
    raise "Use --profile" and the WS handshake fails. We therefore use the
    configured CLI profile locally and ``profile=None`` only when app SP creds are
    present. No-op (logged) if the realtime package/config is unavailable.
    """
    try:
        from realtime_api.app import create_app as create_realtime_app
        from realtime_api.app import warm_serving
        from realtime_api.config import RealtimeSettings, databricks_profile
        from realtime_api.pipelines import ServingBundle
        from realtime_api.services import DatabricksServing

        rt_settings = RealtimeSettings.resolve()

        def _serving_profile() -> str | None:
            # Databricks Apps inject SP OAuth creds via env -> no CLI profile.
            if os.getenv("DATABRICKS_CLIENT_ID") or os.getenv("DATABRICKS_APP_NAME"):
                return None
            profile = databricks_profile()
            # Ignore an unfilled placeholder like "<your-databricks-profile>".
            if not profile or profile.startswith("<"):
                return None
            return profile

        def _factory(s: RealtimeSettings) -> ServingBundle:
            serving = DatabricksServing.from_sdk(
                stt_endpoint=s.stt_endpoint,
                llm_endpoint=s.llm_endpoint,
                tts_endpoint=s.tts_endpoint,
                profile=_serving_profile(),  # CLI profile locally; SP OAuth in-app
                llm_temperature=s.llm_temperature,
                llm_max_tokens=s.llm_max_tokens,
                llm_tools_enabled=s.llm_tools_enabled,
                llm_max_tool_iterations=s.llm_max_tool_iterations,
                tts_inference_timesteps=s.tts_inference_timesteps,
                tts_cfg_value=s.tts_cfg_value,
            )
            return ServingBundle(stt=serving, llm=serving, tts=serving)

        app.mount("/realtime", create_realtime_app(settings=rt_settings, bundle_factory=_factory))

        @app.on_event("startup")
        def _warm_realtime_serving() -> None:
            # Mounted sub-apps don't get their own startup events from Starlette,
            # so prime the STT/LLM/TTS replicas from the parent's startup. This is
            # the path that runs both locally (start_app.sh) and on Databricks
            # Apps, so the first browser voice turn is never the one paying the
            # replica warm-up. Off-thread — never delays readiness.
            warm_serving(rt_settings, _factory)
    except Exception as exc:  # noqa: BLE001
        print(f"[api-startup] realtime API mount skipped: {exc}")


def _mount_realtime_test_ui(app: FastAPI) -> None:
    """Serve the standalone realtime test client at ``/realtime-test`` (optional).

    The page auto-targets the ``/realtime`` mount above when served from here.
    No-op when the folder is absent.
    """
    ui_dir = Path(__file__).resolve().parents[2] / "realtime_test_ui"
    if ui_dir.is_dir():
        # A bare "/realtime-test" (no trailing slash) is NOT matched by the Mount
        # below, so without this it falls through to the SPA catch-all in
        # _mount_frontend and serves the contact-center app instead of the test
        # client. Registered before the catch-all (this runs before
        # _mount_frontend), so typing the URL lands on the test UI.
        @app.get("/realtime-test", include_in_schema=False)
        def _realtime_test_index() -> RedirectResponse:
            return RedirectResponse(url="/realtime-test/")

        app.mount("/realtime-test", StaticFiles(directory=ui_dir, html=True), name="realtime-test")


def _mount_frontend(app: FastAPI) -> None:
    """Serve the built React SPA from the same process/origin as the API.

    Registered AFTER all API routers so explicit API routes (and the mic-stream
    WebSocket) always take precedence; the catch-all only handles UI navigation
    and static assets. No-op when the build output is absent (local dev).
    """
    if not _FRONTEND_DIST.is_dir():
        return

    assets_dir = _FRONTEND_DIST / "assets"
    if assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    index_file = _FRONTEND_DIST / "index.html"

    @app.get("/{full_path:path}", include_in_schema=False)
    def _spa(full_path: str) -> FileResponse:
        candidate = (_FRONTEND_DIST / full_path).resolve()
        # Serve real files (favicon, etc.); otherwise fall back to index.html so
        # client-side routing works on refresh/deep-link.
        if full_path and candidate.is_file() and _FRONTEND_DIST in candidate.parents:
            return FileResponse(candidate)
        return FileResponse(index_file)


app = create_app()
