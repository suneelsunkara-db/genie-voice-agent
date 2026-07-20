"""UC Volume file I/O via the Databricks SDK (in-job) or pathlib (local).

Uses a process-wide WorkspaceClient singleton. Volume writes ``mkdir`` the
parent directory first (the Files API does not auto-create directories).
"""
from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path

_SCRATCH_DIR: Path | None = None


def local_scratch_dir() -> Path:
    """Per-process, uniquely-owned local scratch dir for L1 log/cache writes.

    UC Volumes don't support append/seek, so logs are written to a local file
    first and copied to the Volume at checkpoints. We use ``mkdtemp`` (honours
    TMPDIR, unique owner) rather than a shared ``/tmp/mlv_logs`` parent: on
    serverless a node recycled from a run under a different identity leaves that
    parent owned by someone else, so ``mkdir`` of any child fails with EACCES.
    Cached module-wide so all logs from one process land in the same dir.
    """
    global _SCRATCH_DIR
    if _SCRATCH_DIR is None:
        _SCRATCH_DIR = Path(tempfile.mkdtemp(prefix="mlv_logs_"))
    return _SCRATCH_DIR


def is_volume_path(path: Path | str) -> bool:
    return str(path).startswith("/Volumes/")


def _use_sdk(path: Path | str) -> bool:
    # DATABRICKS_RUNTIME_VERSION is set inside Databricks serverless/pro runs.
    return is_volume_path(path) and bool(os.getenv("DATABRICKS_RUNTIME_VERSION"))


def _client():
    from databricks_auth import workspace_client

    return workspace_client()


def _mkdirs_volume(path: Path) -> None:
    """Create parent directories on a UC Volume (Files API has no auto-mkdir)."""
    client = _client()
    # Walk up from the parent, creating each directory segment.
    parts = str(path.parent).strip("/").split("/")
    cur = ""
    for part in parts:
        cur = f"{cur}/{part}"
        try:
            client.files.create_directory(cur)
        except Exception:
            # Already exists or permission error — the upload will surface real failures.
            pass


def volume_exists(path: Path | str) -> bool:
    """True if the path exists (UC Volume via SDK metadata, else local pathlib)."""
    path = Path(path)
    if _use_sdk(path):
        try:
            _client().files.get_metadata(str(path))
            return True
        except Exception:
            return False
    return path.exists()


def read_text(path: Path | str, *, encoding: str = "utf-8") -> str:
    path = Path(path)
    if _use_sdk(path):
        resp = _client().files.download(str(path))
        data = resp.contents.read() if hasattr(resp, "contents") else resp.read()
        return data.decode(encoding)
    return path.read_text(encoding=encoding)


def write_text(path: Path | str, text: str, *, encoding: str = "utf-8") -> None:
    path = Path(path)
    if _use_sdk(path):
        _mkdirs_volume(path)
        _client().files.upload(str(path), io.BytesIO(text.encode(encoding)), overwrite=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding=encoding)


def copy_to_volume(local_path: Path, volume_path: Path) -> None:
    """Upload a local file to a UC Volume path (whole-file overwrite)."""
    if _use_sdk(volume_path):
        _mkdirs_volume(volume_path)
        _client().files.upload(str(volume_path), local_path.open("rb"), overwrite=True)
        return
    volume_path.parent.mkdir(parents=True, exist_ok=True)
    volume_path.write_bytes(local_path.read_bytes())
