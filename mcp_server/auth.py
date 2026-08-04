"""Bearer-token providers for calling the realtime voice API behind a Databricks App.

A Databricks App gates every request (HTTP and WebSocket) on a workspace OAuth /
OIDC token, so the MCP server attaches ``Authorization: Bearer <token>`` to each
call. Three ways to obtain one, strongest first:

  1. static token  — ``GENIE_VOICE_TOKEN`` / ``DATABRICKS_TOKEN`` (a pre-minted
     OAuth token or PAT); never refreshed.
  2. service principal (M2M) — ``DATABRICKS_HOST`` + ``DATABRICKS_CLIENT_ID`` +
     ``DATABRICKS_CLIENT_SECRET``; the SDK mints and refreshes OAuth tokens.
  3. CLI profile (U2M) — ``DATABRICKS_CONFIG_PROFILE`` (local dev); shells out to
     ``databricks auth token``.

``HOST`` is the *workspace* host that issues the token (e.g.
``https://my-workspace.cloud.databricks.com``), NOT the ``*.databricksapps.com``
app URL. If nothing is configured the provider is ``None`` and requests go out
unauthenticated (only useful against a local ``python -m realtime_api.server``).
"""
from __future__ import annotations

import json
import os
import subprocess
from typing import Callable

TokenProvider = Callable[[], str]


class _StaticToken:
    def __init__(self, token: str) -> None:
        self._token = token

    def __call__(self) -> str:
        return self._token


class _M2MToken:
    """Service-principal client-credentials tokens (SDK-managed refresh)."""

    def __init__(self, host: str, client_id: str, client_secret: str) -> None:
        from databricks.sdk.core import Config

        self._config = Config(
            host=host,
            client_id=client_id,
            client_secret=client_secret,
            auth_type="oauth-m2m",
        )

    def __call__(self) -> str:
        headers = self._config.authenticate() or {}
        auth = headers.get("Authorization") or headers.get("authorization") or ""
        token = auth.removeprefix("Bearer ").strip()
        if not token:
            raise RuntimeError("M2M OAuth returned an empty token")
        return token


class _CliProfileToken:
    """U2M tokens via the Databricks CLI (local dev only)."""

    def __init__(self, profile: str) -> None:
        self._profile = profile

    def __call__(self) -> str:
        raw = subprocess.check_output(
            ["databricks", "auth", "token", "--profile", self._profile],
            stderr=subprocess.STDOUT,
            text=True,
        )
        token = json.loads(raw).get("access_token")
        if not token:
            raise RuntimeError(f"databricks auth token empty for profile {self._profile!r}")
        return str(token)


def token_provider_from_env() -> tuple[TokenProvider | None, str]:
    """Build the strongest available token provider from environment variables.

    Returns ``(provider, mode)``; ``provider`` is ``None`` (mode ``"none"``) when
    no auth is configured, so callers can decide whether that is acceptable.
    """
    static = os.environ.get("GENIE_VOICE_TOKEN") or os.environ.get("DATABRICKS_TOKEN")
    if static:
        return _StaticToken(static.strip()), "static-token"

    host = os.environ.get("DATABRICKS_HOST")
    client_id = os.environ.get("DATABRICKS_CLIENT_ID")
    client_secret = os.environ.get("DATABRICKS_CLIENT_SECRET")
    if host and client_id and client_secret:
        return _M2MToken(host, client_id, client_secret), "service-principal (m2m)"

    profile = os.environ.get("DATABRICKS_CONFIG_PROFILE")
    if profile:
        return _CliProfileToken(profile), f"cli-profile ({profile})"

    return None, "none"
