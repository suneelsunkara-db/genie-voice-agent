"""Databricks OAuth token providers + workspace client + secret reads.

One provider per sweep, one WorkspaceClient per process. The SDK manages
OAuth token caching/refresh internally, so providers are plain callables that
return a fresh-enough bearer on each call — no manual ``refresh()`` dance.
"""
from __future__ import annotations

import base64
import json
import subprocess
from functools import lru_cache
from typing import Callable


@lru_cache(maxsize=1)
def workspace_client():
    """Process-wide singleton WorkspaceClient (one auth handshake)."""
    from databricks.sdk import WorkspaceClient

    return WorkspaceClient()


def read_workspace_secret(scope: str, key: str) -> str:
    """Read a Databricks secret via the job's ambient identity."""
    secret = workspace_client().secrets.get_secret(scope, key)
    return base64.b64decode(secret.value).decode()


class StaticTokenProvider:
    """Fixed token (e.g. injected via ``MLV_AUTH_TOKEN``); never refreshes."""

    def __init__(self, token: str) -> None:
        self._token = token

    def __call__(self) -> str:
        return self._token


class M2MTokenProvider:
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


class CliProfileTokenProvider:
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


TokenProvider = Callable[[], str]


def build_token_provider(
    *,
    host: str | None = None,
    client_id: str | None = None,
    client_secret: str | None = None,
    profile: str | None = None,
    static_token: str | None = None,
) -> TokenProvider:
    """Pick the strongest available auth mode.

    Priority: explicit static token > SP M2M > CLI profile (U2M).
    """
    if static_token:
        return StaticTokenProvider(static_token)
    if host and client_id and client_secret:
        return M2MTokenProvider(host, client_id, client_secret)
    if profile:
        return CliProfileTokenProvider(profile)
    raise RuntimeError(
        "No benchmark auth configured. Set a service-principal client_id/secret "
        "(realtime_voice.benchmark.auth or MLV_SP_CLIENT_ID/MLV_SP_CLIENT_SECRET) "
        "for job runs, or a Databricks CLI profile for --local."
    )
