"""Deployment readiness for the in-app Setup page.

`GET /readiness` runs the shared checks (config, warehouse, voice model
endpoints, Genie spaces, Lakebase CDF, Lakebase serving, AI Gateway guardrails,
Apps User Authorization) and reports whether the install is usable yet.

`POST /readiness/fix` runs cheap, idempotent remediations that the app can
perform as its own identity (currently: snapshot the Lakebase reference cache).
The voice-serving endpoints expose a separate, explicit operator action which
uses the App's narrowly scoped voice-endpoint management grant to disable
scale-to-zero and can be polled while Databricks applies the billable update.

The endpoint is available to any authenticated app user (read-mostly ops view);
viewer-scoped checks use the forwarded user token when present.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from genie_voice import readiness as _readiness
from genie_voice.config import get_settings

router = APIRouter(prefix="/readiness", tags=["readiness"])

_FORWARDED_TOKEN = "x-forwarded-access-token"


def _viewer(request: Request, obo: str | None) -> dict:
    return {
        "email": (request.headers.get("x-forwarded-email") or "").strip() or None,
        "username": (
            request.headers.get("x-forwarded-preferred-username") or ""
        ).strip() or None,
        "user_authorization_token": bool(obo),
    }


def _add_viewer_objects(payload: dict, viewer: dict) -> None:
    identity = viewer.get("email") or viewer.get("username")
    if not identity:
        return
    for check in payload.get("checks", []):
        if check.get("id") in {"obo", "viewer_genie", "agent_mode"}:
            check.setdefault("objects", []).append(f"Signed-in viewer: {identity}")
        if check.get("id") == "viewer_genie":
            check.setdefault("objects", []).append(
                f"Verified current-viewer App permission: {identity} → CAN_USE"
            )


@router.get("")
def get_readiness(request: Request, full: bool = True) -> dict:
    obo = (request.headers.get(_FORWARDED_TOKEN) or "").strip() or None
    payload = _readiness.run_checks(get_settings(), obo_token=obo, full=full).to_dict()
    payload["validation_mode"] = "full" if full else "quick"
    payload["viewer"] = _viewer(request, obo)
    _add_viewer_objects(payload, payload["viewer"])
    return payload


@router.get("/voice-serving")
def get_voice_serving() -> dict:
    return _readiness.voice_serving_status(get_settings())


@router.post("/voice-serving/disable-scale-to-zero")
def post_disable_scale_to_zero() -> dict:
    try:
        return _readiness.disable_voice_scale_to_zero(get_settings())
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        detail = str(exc)
        status = 403 if "PERMISSION_DENIED" in detail or "403" in detail else 502
        raise HTTPException(
            status_code=status,
            detail=f"could not disable scale-to-zero: {detail}",
        ) from exc


@router.post("/voice-serving/enable-scale-to-zero")
def post_enable_scale_to_zero() -> dict:
    try:
        return _readiness.enable_voice_scale_to_zero(get_settings())
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        detail = str(exc)
        status = 403 if "PERMISSION_DENIED" in detail or "403" in detail else 502
        raise HTTPException(
            status_code=status,
            detail=f"could not enable scale-to-zero: {detail}",
        ) from exc


class FixRequest(BaseModel):
    action: str


@router.post("/fix")
def post_fix(req: FixRequest, request: Request) -> dict:
    try:
        result = _readiness.apply_fix(req.action, get_settings())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"fix '{req.action}' failed: {exc}") from exc
    obo = (request.headers.get(_FORWARDED_TOKEN) or "").strip() or None
    readiness = _readiness.run_checks(get_settings(), obo_token=obo, full=True).to_dict()
    readiness["validation_mode"] = "full"
    readiness["viewer"] = _viewer(request, obo)
    _add_viewer_objects(readiness, readiness["viewer"])
    return {
        "fix": result,
        "readiness": readiness,
    }
