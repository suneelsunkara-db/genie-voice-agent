"""Deployment readiness for the in-app Setup page.

`GET /readiness` runs the shared checks (config, warehouse, voice model
endpoints, Genie spaces, Lakebase CDF, Lakebase serving, AI Gateway guardrails,
Apps User Authorization) and reports whether the install is usable yet.

`POST /readiness/fix` runs only cheap, idempotent remediations that the app can
perform as its own identity (currently: snapshot the Lakebase reference cache).
Owner-only grants and the long/billable GPU model deploy stay in `deploy_app.sh`
and are surfaced here as guidance, never executed from a page.

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
