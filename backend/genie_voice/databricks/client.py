"""Databricks SDK client factory.

Auth is config-driven (`databricks.auth_type`):
  - default -> SDK unified-auth credential chain. With OAuth U2M this means the
               token cached by `databricks auth login --host <host>` (run AS the
               user via OAuth U2M). No secrets in .env.
               An optional `databricks.profile` selects a ~/.databrickscfg profile.
  - pat     -> DATABRICKS_TOKEN (personal access token).
  - oauth   -> DATABRICKS_CLIENT_ID / DATABRICKS_CLIENT_SECRET (service principal M2M).
"""
from __future__ import annotations

import os

from genie_voice.config import Settings, get_settings

# One client per process. We can't @lru_cache on `settings` (a pydantic model is
# unhashable), and there's a single workspace per run anyway, so cache the built
# client in a module global instead.
_CLIENT = None


def get_workspace_client(settings: Settings | None = None):
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = _build_workspace_client(settings or get_settings())
    return _CLIENT


def _build_workspace_client(settings: Settings):
    from databricks.sdk import WorkspaceClient

    host = settings.databricks_host
    auth = settings.databricks.auth_type

    if auth == "pat":
        if not host:
            raise RuntimeError("Databricks host is not configured (DATABRICKS_HOST / config).")
        return WorkspaceClient(host=host, token=os.environ.get("DATABRICKS_TOKEN", ""))

    if auth == "oauth":
        if not host:
            raise RuntimeError("Databricks host is not configured (DATABRICKS_HOST / config).")
        return WorkspaceClient(
            host=host,
            client_id=os.environ.get("DATABRICKS_CLIENT_ID", ""),
            client_secret=os.environ.get("DATABRICKS_CLIENT_SECRET", ""),
        )

    # auth == "default": let the SDK resolve credentials.
    profile = settings.databricks.profile
    if profile:
        return WorkspaceClient(profile=profile)
    if host:
        # Host-scoped resolution picks up the OAuth U2M token cache for this host.
        return WorkspaceClient(host=host)
    return WorkspaceClient()


def current_user(client) -> str:
    """Email/username of the authenticated identity (for GRANTs / Lakebase role)."""
    try:
        return client.current_user.me().user_name
    except Exception:  # noqa: BLE001
        return ""
