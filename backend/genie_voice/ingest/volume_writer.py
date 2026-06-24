"""Write raw payloads to a Unity Catalog Volume.

Supports two backends, chosen automatically:
  - Databricks SDK `files.upload` to a /Volumes/... path (default).
  - A local filesystem directory (when GENIE_LOCAL_VOLUME_DIR is set) - useful
    for fully offline dev / tests without a workspace.

Files are newline-delimited JSON (one event per line) so Auto Loader can ingest
them with the JSON reader.
"""
from __future__ import annotations

import io
import json
import os
from typing import Any, Iterable

from genie_voice.config import Settings, get_settings


def _local_dir() -> str | None:
    return os.environ.get("GENIE_LOCAL_VOLUME_DIR")


def write_events(
    call_id: str,
    events: Iterable[dict[str, Any]],
    settings: Settings | None = None,
    meta: dict[str, Any] | None = None,
) -> str:
    """Write all events for one call as a single JSONL object. Returns the path.

    `meta` fields (e.g. customer_id) are merged into every event so bronze keeps
    provenance the vendor payload itself doesn't carry.
    """
    settings = settings or get_settings()
    base_meta = {"_call_id": call_id, **(meta or {})}
    payload = "\n".join(json.dumps({**e, **base_meta}) for e in events)
    filename = f"{call_id}.json"

    local_dir = _local_dir()
    if local_dir:
        os.makedirs(local_dir, exist_ok=True)
        path = os.path.join(local_dir, filename)
        with open(path, "w") as fh:
            fh.write(payload)
        return path

    # Databricks Volume upload.
    from genie_voice.databricks.client import get_workspace_client

    client = get_workspace_client(settings)
    path = f"{settings.raw_stt_path}/{filename}"
    client.files.upload(path, io.BytesIO(payload.encode()), overwrite=True)
    return path


def write_json_record(path: str, record: dict[str, Any], settings: Settings | None = None) -> str:
    """Write one JSON record to a configured Volume path."""
    settings = settings or get_settings()
    payload = json.dumps(record, default=str)
    local_dir = _local_dir()
    if local_dir:
        os.makedirs(local_dir, exist_ok=True)
        local_path = os.path.join(local_dir, os.path.basename(path))
        with open(local_path, "w") as fh:
            fh.write(payload)
        return local_path

    from genie_voice.databricks.client import get_workspace_client

    client = get_workspace_client(settings)
    client.files.upload(path, io.BytesIO(payload.encode()), overwrite=True)
    return path
