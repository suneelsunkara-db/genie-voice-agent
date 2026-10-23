"""Session principal / OBO helpers for Genie workspace paths."""
from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field, replace
from typing import Any, Mapping

from ..observability import log_event
from .refuse import ErrorEvidence, permission_refuse, refuse_speech

logger = logging.getLogger("realtime_voice")

# Header Databricks Apps injects when user authorization is enabled.
FORWARDED_ACCESS_TOKEN = "x-forwarded-access-token"
FORWARDED_EMAIL = "x-forwarded-email"
FORWARDED_PREFERRED_USERNAME = "x-forwarded-preferred-username"


@dataclass(frozen=True)
class SessionPrincipal:
    """User identity bound at session.start for OBO Genie / SQL calls."""

    access_token: str | None = field(default=None, repr=False)
    email: str | None = None
    username: str | None = None
    name: str | None = None
    # True when token came from Apps forwarded header (not local U2M stand-in).
    from_forwarded_header: bool = False

    @property
    def has_token(self) -> bool:
        return bool(self.access_token and self.access_token.strip())

    @property
    def display_name(self) -> str | None:
        return self.name or self.email or self.username


# Cache only non-secret identity attributes. Access tokens are deliberately never
# used as keys or values, so refreshing a browser session cannot retain a bearer
# credential in process memory beyond the VoiceSession that needs it.
_IDENTITY_CACHE: dict[str, tuple[str | None, str | None, str | None]] = {}
_IDENTITY_CACHE_LOCK = threading.Lock()


def enrich_principal_identity(principal: SessionPrincipal) -> SessionPrincipal:
    """Resolve the authoritative workspace display name once for a signed-in user.

    Apps forwards email/username with the WebSocket request. ``current_user.me``
    supplies the full display name; failures are non-fatal and retain the forwarded
    identity. The caller runs this function off the event loop.
    """
    if not principal.has_token:
        return principal
    key = (principal.email or principal.username or "").strip().casefold()
    if key:
        with _IDENTITY_CACHE_LOCK:
            cached = _IDENTITY_CACHE.get(key)
        if cached is not None:
            name, email, username = cached
            return replace(
                principal,
                name=name or principal.name,
                email=email or principal.email,
                username=username or principal.username,
            )
    try:
        me = workspace_client_for_principal(principal).current_user.me()
        name = str(getattr(me, "display_name", "") or "").strip() or principal.name
        user_name = str(getattr(me, "user_name", "") or "").strip() or principal.username
        email = principal.email or (user_name if "@" in (user_name or "") else None)
        enriched = replace(principal, name=name, email=email, username=user_name)
        if key:
            with _IDENTITY_CACHE_LOCK:
                _IDENTITY_CACHE[key] = (enriched.name, enriched.email, enriched.username)
        return enriched
    except Exception:  # noqa: BLE001 - identity enrichment must not block a call
        logger.warning("current_user.me identity enrichment failed", exc_info=True)
        return principal


def principal_from_headers(headers: Mapping[str, str] | None) -> SessionPrincipal:
    """Bind principal from WS/HTTP headers (Apps user authorization)."""
    if not headers:
        return SessionPrincipal()
    # Starlette headers are case-insensitive; normalize keys for plain dicts.
    lower = {str(k).lower(): str(v) for k, v in headers.items()}
    token = (lower.get(FORWARDED_ACCESS_TOKEN) or "").strip() or None
    email = (lower.get(FORWARDED_EMAIL) or "").strip() or None
    username = (lower.get(FORWARDED_PREFERRED_USERNAME) or "").strip() or None
    return SessionPrincipal(
        access_token=token,
        email=email,
        username=username,
        from_forwarded_header=bool(token),
    )


def local_u2m_stand_in() -> SessionPrincipal | None:
    """Local-dev stand-in: use CLI/U2M token when Apps headers are absent.

    Set ``GENIE_OBO_LOCAL_TOKEN`` to an explicit PAT/U2M access token, or leave
    unset to signal "no OBO principal" (fail-closed for Genie paths in tests /
    production-shaped local runs). Never invents a token from the SP env.
    """
    token = (os.environ.get("GENIE_OBO_LOCAL_TOKEN") or "").strip()
    if not token:
        return None
    return SessionPrincipal(
        access_token=token,
        email=os.environ.get("APP_DEV_USER") or None,
        username=os.environ.get("APP_DEV_USER") or None,
        from_forwarded_header=False,
    )


def resolve_session_principal(
    headers: Mapping[str, str] | None,
    *,
    allow_local_stand_in: bool = True,
) -> SessionPrincipal:
    """Prefer Apps forwarded token; optionally fall back to local U2M stand-in."""
    principal = principal_from_headers(headers)
    if principal.has_token:
        return principal
    if allow_local_stand_in:
        stand_in = local_u2m_stand_in()
        if stand_in is not None:
            return stand_in
    return principal


def obo_deny(
    *,
    session_id: str,
    turn_id: int | None = None,
    reason: str = "missing_user_token",
) -> ErrorEvidence:
    """Log + metric for fail-closed Genie paths; return permission ErrorEvidence."""
    log_event(
        "obo_deny",
        session_id=session_id,
        turn_id=turn_id,
        reason=reason,
        metric="obo_deny",
    )
    logger.warning("obo_deny session=%s turn=%s reason=%s", session_id, turn_id, reason)
    return permission_refuse(detail=reason)


def require_obo_token(
    principal: SessionPrincipal | None,
    *,
    session_id: str = "",
    turn_id: int | None = None,
) -> ErrorEvidence | None:
    """Return ErrorEvidence when Genie must not be invoked; else None."""
    if principal is not None and principal.has_token:
        return None
    return obo_deny(session_id=session_id or "unknown", turn_id=turn_id)


def workspace_client_for_principal(principal: SessionPrincipal, settings: Any = None):
    """Build a user-scoped WorkspaceClient isolated from the app SP.

    Databricks Apps injects ``DATABRICKS_CLIENT_ID``/``CLIENT_SECRET`` for the
    app service principal. Without an explicit auth strategy, Unified Auth mixes
    those ambient OAuth credentials with the forwarded user bearer token and
    rejects the client as having both ``oauth`` and ``pat`` configured.

    ``pat`` is the SDK's static-bearer strategy; despite its historical name, it
    simply sends the supplied token as ``Authorization: Bearer``. Selecting it
    explicitly ensures this OBO client uses only the forwarded user token.
    """
    if not principal.has_token:
        raise PermissionError("OBO token required for workspace client")
    from databricks.sdk import WorkspaceClient

    if settings is None:
        from genie_voice.config import get_settings

        settings = get_settings()
    host = getattr(settings, "databricks_host", None) or ""
    if not host and hasattr(settings, "databricks"):
        host = getattr(settings.databricks, "host", "") or ""
    if not host:
        raise RuntimeError("Databricks host is not configured for OBO client")
    return WorkspaceClient(
        host=host,
        token=principal.access_token,
        auth_type="pat",
    )


def refuse_text_for_obo(error: ErrorEvidence | None = None, *, language: str = "en") -> str:
    err = error or permission_refuse()
    return refuse_speech(err.code, language=language)
