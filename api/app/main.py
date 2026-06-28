"""FastAPI application entrypoint.

All settings (host, port, CORS) come from config. Run:
    uvicorn app.main:app --reload --port 8000   (cwd = api/)
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from genie_voice.config import get_settings

from .deps import serving
from .routers import accounts, agent_assist, asr_benchmark, genie, health, mic_stream, pipeline_status


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

    return app


app = create_app()
