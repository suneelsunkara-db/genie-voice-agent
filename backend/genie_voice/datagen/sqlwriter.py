"""Offline export helper for the modeled tables.

Online ingestion is done by the serverless jobs (which read the files the
producers land in the Volume), so there is no SQL-insert path here anymore. This
only supports the OFFLINE emulator: dump each table to JSON + emit its DDL for
inspection alongside the local run.
"""
from __future__ import annotations

import json
import os

from genie_voice.config import Settings

from .schema import MODEL


def export_local(settings: Settings, out_dir: str, tables: dict[str, list[dict]]) -> None:
    """Write each table to JSON + emit its DDL, for offline inspection."""
    os.makedirs(out_dir, exist_ok=True)
    for name, rows in tables.items():
        with open(os.path.join(out_dir, f"{name}.json"), "w") as fh:
            json.dump(rows, fh, indent=2)
        with open(os.path.join(out_dir, f"{name}.sql"), "w") as fh:
            fh.write(MODEL[name].render_ddl(settings.fqtn) + ";\n")
