"""FastAPI application entrypoint.

All settings (host, port, CORS) come from config. Run:
    uvicorn app.main:app --reload --port 8000   (cwd = api/)
"""
from __future__ import annotations

import logging
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
from .routers import (
    accounts,
    agent_assist,
    asr_benchmark,
    card,
    concierge,
    genie,
    health,
    knowledge,
    languages,
    me,
    mic_stream,
    pipeline_status,
    traces,
)

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
    app.include_router(me.router)
    app.include_router(languages.router)
    app.include_router(concierge.router)
    app.include_router(knowledge.router)
    app.include_router(agent_assist.router)
    app.include_router(mic_stream.router)
    app.include_router(accounts.router)
    app.include_router(genie.router)
    app.include_router(card.router)
    app.include_router(asr_benchmark.router)
    app.include_router(pipeline_status.router)
    app.include_router(traces.router)

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
            try:
                # voice_traces is owned by whichever principal created it, so only
                # that principal's process can migrate it. Doing it at startup means
                # the owner (the deployed app) applies pending columns once, rather
                # than every writer racing to discover them on its first trace.
                added = serving().migrate_voice_traces()
                if added:
                    print(f"[api-startup] voice_traces columns added: {', '.join(added)}")
            except Exception as exc:  # noqa: BLE001
                print(f"[api-startup] voice_traces migration skipped: {exc}")
            if settings.lakebase.enabled and not settings.databricks.sql_warehouse_id:
                print(
                    "[api-startup] WARNING: lakebase.enabled requires databricks.sql_warehouse_id "
                    "for governed UC billing writes; close/billing will fail until configured."
                )
            try:
                from genie_voice.databricks.warehouse_sql import (
                    ensure_billing_adjustments_table,
                    ensure_voice_traces_table,
                    warehouse_configured,
                )

                if warehouse_configured(settings):
                    ensure_billing_adjustments_table(settings)
                    ensure_voice_traces_table(settings)
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

    @app.get("/__mcp_status", include_in_schema=False)
    def _mcp_status_route() -> dict:
        # Lightweight ops probe: is the in-process MCP endpoint mounted, and if
        # not, why (import / build / start error). Registered before the SPA
        # catch-all so it isn't shadowed.
        return dict(getattr(app.state, "mcp_status", {"mounted": False, "error": "not attempted"}))

    _mount_mcp(app)
    _mount_realtime(app)
    _mount_realtime_test_ui(app)
    _mount_frontend(app)
    return app


def _mount_mcp(app: FastAPI) -> None:
    """Host the MCP server over HTTP at EXACTLY ``/realtime/mcp`` (remote MCP).

    Exposes the realtime voice API as Model Context Protocol tools so remote MCP
    clients (Cursor, Claude Desktop) can reach it via URL + a Databricks token
    (the app's normal ingress auth). It runs IN-PROCESS: its tools call the
    ``/realtime`` routes over loopback (see ``mcp_server.server._resolve_target``),
    so there's no second auth hop.

    We deliberately avoid both ``FastMCP.streamable_http_app()`` (its Starlette app
    mounts the transport at ``streamable_http_path``, default ``/mcp``, → the
    endpoint would nest at ``/realtime/mcp/mcp``) and ``app.mount("/realtime/mcp")``
    (a Starlette Mount only FULL-matches ``/realtime/mcp/<sub>`` and 307-redirects
    the bare path, so ``/realtime/mcp`` with no trailing slash falls through to the
    ``/realtime`` mount). Instead we drive the transport directly with a
    ``StreamableHTTPSessionManager`` over the FastMCP low-level server, dispatched
    from a thin pure-ASGI middleware that matches the exact path (see below), so the
    endpoint is precisely ``/realtime/mcp`` with no redirect and no nested sub-path.

    The session manager needs a task group that must be started inside a running
    lifespan, so we start it from the parent's startup event and keep it open for
    the app's life via an AsyncExitStack. Fully guarded: any failure degrades to
    "no MCP endpoint" and never blocks the app.
    """
    status: dict = {"mounted": False, "path": "/realtime/mcp", "via": None, "error": None}
    app.state.mcp_status = status

    try:
        from mcp_server.server import mcp
    except Exception as exc:  # noqa: BLE001
        status["error"] = f"import mcp_server.server: {type(exc).__name__}: {exc}"
        print(f"[api-startup] MCP mount skipped: {status['error']}")
        return

    try:
        from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

        low_level = getattr(mcp, "_mcp_server", None) or getattr(mcp, "server", None)
        if low_level is None:
            raise RuntimeError("cannot access FastMCP low-level server")
        manager = StreamableHTTPSessionManager(app=low_level)
    except Exception as exc:  # noqa: BLE001
        status["error"] = f"build session manager: {type(exc).__name__}: {exc}"
        print(f"[api-startup] MCP mount skipped: {status['error']}")
        return

    # Serve the transport at EXACTLY "/realtime/mcp". We do NOT use app.mount():
    # a Starlette Mount only FULL-matches "/realtime/mcp/<sub>" and 307-redirects
    # the bare path, so "/realtime/mcp" (no trailing slash) would fall through to
    # the "/realtime" mount. Instead a thin PURE-ASGI middleware (kept pure so the
    # SSE stream isn't buffered) intercepts the exact path + any subpath, rewrites
    # the scope to the transport root, and hands off to the path-agnostic manager.
    mount = "/realtime/mcp"

    class _MCPPathMiddleware:
        # Starlette instantiates middleware as ``cls(app=<next-asgi-app>, **opts)``,
        # so the first parameter MUST be named ``app``.
        def __init__(self, app) -> None:
            self._inner = app

        async def __call__(self, scope, receive, send) -> None:
            if scope.get("type") == "http":
                path = scope.get("path", "")
                if path == mount or path.startswith(mount + "/"):
                    scope = dict(scope)
                    scope["path"] = "/"
                    scope["raw_path"] = b"/"
                    scope["root_path"] = mount
                    await manager.handle_request(scope, receive, send)
                    return
            await self._inner(scope, receive, send)

    app.add_middleware(_MCPPathMiddleware)
    status["via"] = "asgi-middleware"

    @app.on_event("startup")
    async def _mcp_start() -> None:
        from contextlib import AsyncExitStack

        try:
            stack = AsyncExitStack()
            await stack.enter_async_context(manager.run())
            app.state._mcp_stack = stack
            status["mounted"] = True
            print(f"[api-startup] MCP endpoint live at {mount} (via {status['via']})")
        except Exception as exc:  # noqa: BLE001
            status["error"] = f"start: {type(exc).__name__}: {exc}"
            print(f"[api-startup] MCP session start skipped: {status['error']}")

    @app.on_event("shutdown")
    async def _mcp_stop() -> None:
        stack = getattr(app.state, "_mcp_stack", None)
        if stack is not None:
            try:
                await stack.aclose()
            except Exception as exc:  # noqa: BLE001
                print(f"[api-startup] MCP shutdown error: {exc}")


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
        from realtime_api.config import RealtimeSettings
        from realtime_api.pipelines import ServingBundle
        from realtime_api.serving_factory import shared_serving

        rt_settings = RealtimeSettings.resolve()

        def _factory(s: RealtimeSettings) -> ServingBundle:
            # Shared, config-driven singleton: the WS loop and the card deep-dive
            # summarizer use the SAME serving instance and the SAME knobs.
            serving = shared_serving()
            return ServingBundle(stt=serving, llm=serving, tts=serving)

        app.mount("/realtime", create_realtime_app(settings=rt_settings, bundle_factory=_factory))

        # A bare "/realtime" (no trailing slash) is NOT matched by the Mount above,
        # so without this it falls through to the SPA catch-all in _mount_frontend
        # and serves the web app instead of the API. Redirect to "/realtime/" so the
        # base path lands on the sub-app's JSON API descriptor. Registered before the
        # catch-all (this runs before _mount_frontend); same pattern as /realtime-test.
        @app.get("/realtime", include_in_schema=False)
        def _realtime_index() -> RedirectResponse:
            return RedirectResponse(url="/realtime/")

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
