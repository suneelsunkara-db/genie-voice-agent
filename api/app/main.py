"""FastAPI application entrypoint.

All settings (host, port, CORS) come from config. Run:
    uvicorn app.main:app --reload --port 8000   (cwd = api/)
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
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
        """
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

    This is fully isolated from the contact-center routes: it only exposes
    ``/realtime/v1/realtime/voice`` (WebSocket) and ``/realtime/v1/languages``.
    In the Databricks app the pipeline must authenticate as the injected service
    principal (OAuth), so we build the SDK serving client with ``profile=None``
    rather than the CLI profile from config. No-op (logged) if the realtime
    package or its config is unavailable, so the main app still boots.
    """
    try:
        from realtime_api.app import create_app as create_realtime_app
        from realtime_api.config import RealtimeSettings
        from realtime_api.services import DatabricksServing
        from realtime_api.session import VoicePipeline

        rt_settings = RealtimeSettings.resolve()

        def _factory(s: RealtimeSettings) -> VoicePipeline:
            serving = DatabricksServing.from_sdk(
                stt_endpoint=s.stt_endpoint,
                llm_endpoint=s.llm_endpoint,
                tts_endpoint=s.tts_endpoint,
                profile=None,  # in-app: use the injected SP OAuth creds, not a CLI profile
                llm_temperature=s.llm_temperature,
                llm_max_tokens=s.llm_max_tokens,
                llm_tools_enabled=s.llm_tools_enabled,
                llm_max_tool_iterations=s.llm_max_tool_iterations,
                tts_inference_timesteps=s.tts_inference_timesteps,
                tts_cfg_value=s.tts_cfg_value,
            )
            return VoicePipeline(stt=serving, llm=serving, tts=serving, verify_mode=s.verify_mode)

        app.mount("/realtime", create_realtime_app(settings=rt_settings, pipeline_factory=_factory))
    except Exception as exc:  # noqa: BLE001
        print(f"[api-startup] realtime API mount skipped: {exc}")


def _mount_realtime_test_ui(app: FastAPI) -> None:
    """Serve the standalone realtime test client at ``/realtime-test`` (optional).

    The page auto-targets the ``/realtime`` mount above when served from here.
    No-op when the folder is absent.
    """
    ui_dir = Path(__file__).resolve().parents[2] / "realtime_test_ui"
    if ui_dir.is_dir():
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
