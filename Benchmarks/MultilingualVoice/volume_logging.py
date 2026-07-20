"""Run-log handler: local file + checkpoint-copy to the UC Volume.

UC Volumes don't support append/seek, so we write to a local file continuously
(via stdlib ``logging.FileHandler``) and copy the whole file to the Volume at
each checkpoint. No in-memory buffer — logs land on disk immediately and stream
to stdout for the live Databricks job log.
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from volume_io import copy_to_volume, local_scratch_dir


def run_log_path(out_dir: Path, *, run_label: str | None = None) -> Path:
    stamp = run_label or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return out_dir / "logs" / f"run_{stamp}.log"


def issues_log_path(out_dir: Path) -> Path:
    return out_dir / "logs" / "issues.jsonl"


def _local_log_path(run_label: str | None) -> Path:
    """Local scratch path for the continuous FileHandler (L1).

    Uses a per-process, uniquely-owned scratch dir (see ``local_scratch_dir``)
    to avoid PermissionError when a serverless node was recycled from a run
    under a different identity that left a shared ``/tmp`` parent behind.
    """
    stamp = run_label or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return local_scratch_dir() / f"run_{stamp}.log"


class _CheckpointCopyHandler(logging.Handler):
    """No-op handler; the local FileHandler writes, and we copy at checkpoints."""

    def __init__(self) -> None:
        super().__init__()

    def emit(self, record: logging.LogRecord) -> None:
        return  # handled by the FileHandler


class RunLogManager:
    """Owns the local FileHandler and copies to the Volume at checkpoints."""

    def __init__(self, local_path: Path, volume_path: Path) -> None:
        self.local_path = local_path
        self.volume_path = volume_path
        self._file_handler = logging.FileHandler(local_path, encoding="utf-8")
        self._file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))

    @property
    def file_handler(self) -> logging.Handler:
        return self._file_handler

    def checkpoint_to_volume(self) -> None:
        """Copy the local log file to the UC Volume (whole-file overwrite)."""
        if self.local_path.exists():
            copy_to_volume(self.local_path, self.volume_path)


def setup_run_logging(
    out_dir: Path,
    *,
    run_label: str | None = None,
    logger_name: str = "mlv",
) -> RunLogManager:
    """Attach stdout + local-file handlers; return the manager for checkpoints."""
    local = _local_log_path(run_label)
    volume = run_log_path(out_dir, run_label=run_label)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    manager = RunLogManager(local, volume)
    logger.addHandler(manager.file_handler)

    logger.info("benchmark run started out_dir=%s log=%s", out_dir, volume)
    return manager
