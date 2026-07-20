"""Structured issue log: local JSONL append + checkpoint-copy to the UC Volume.

Issues append to a local file (O(1) per record) and the whole file is copied
to the Volume at each checkpoint. No in-memory accumulation.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from volume_io import copy_to_volume, local_scratch_dir


class IssueTracker:
    """Append issues to a local JSONL file; copy to the Volume on ``flush()``."""

    def __init__(self, volume_path: Path) -> None:
        self.volume_path = volume_path
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.local_path = local_scratch_dir() / f"issues_{stamp}.jsonl"
        self._count = 0
        self._by_kind: dict[str, int] = {}

    def record(
        self,
        *,
        dataset: str,
        language: str,
        kind: str,
        message: str,
        sample_index: int | None = None,
        phase: str = "inference",
        context: dict[str, Any] | None = None,
    ) -> None:
        entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dataset": dataset,
            "language": language,
            "phase": phase,
            "kind": kind,
            "message": message,
        }
        if sample_index is not None:
            entry["sample_index"] = sample_index
        if context:
            entry["context"] = context
        with self.local_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._count += 1
        self._by_kind[kind] = self._by_kind.get(kind, 0) + 1

    def run_summary(self) -> dict[str, Any]:
        return {"count": self._count, "by_kind": dict(self._by_kind)}

    def reset_run(self) -> None:
        self._count = 0
        self._by_kind.clear()

    def flush(self) -> None:
        """Copy the local JSONL to the UC Volume (whole-file overwrite)."""
        if self.local_path.exists():
            copy_to_volume(self.local_path, self.volume_path)
