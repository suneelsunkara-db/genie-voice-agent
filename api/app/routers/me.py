"""Logged-in user identity for the SPA.

Databricks Apps sit behind an identity-aware proxy that injects the viewing
workspace user's identity on every request as ``X-Forwarded-*`` headers. This
endpoint surfaces that to the browser so the voice concierge can greet the user by
name. Locally those headers are absent, so we fall back to ``APP_DEV_USER`` (if
set) and otherwise report an unauthenticated, nameless user — the client then
greets generically instead of by name.
"""
from __future__ import annotations

import os

from fastapi import APIRouter, Request

router = APIRouter(tags=["me"])


def _first_name(email: str, username: str) -> str:
    """Best-effort human first name from an email / preferred-username.

    'suneel.sunkara@x.com' / 'suneel_sunkara' -> 'Suneel'. Returns "" when there's
    nothing usable, so the caller can greet generically.
    """
    local = (username or email or "").split("@")[0].strip()
    if not local:
        return ""
    token = local.replace("_", ".").replace("-", ".").split(".")[0]
    return token[:1].upper() + token[1:] if token else ""


@router.get("/me")
def me(request: Request) -> dict:
    """The signed-in user, from Databricks Apps' forwarded identity headers."""
    email = (request.headers.get("x-forwarded-email") or "").strip()
    username = (request.headers.get("x-forwarded-preferred-username") or "").strip()
    if not email and not username:
        dev = os.environ.get("APP_DEV_USER", "").strip()
        if dev:
            username = dev
    return {
        "email": email,
        "username": username,
        "name": _first_name(email, username),
        "authenticated": bool(email or username),
    }
