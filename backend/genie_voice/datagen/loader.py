"""Produce demo reference data and land call artifacts in UC Volumes.

Reference/customer/billing rows land in raw_batch_data for UC Delta ingestion.
Call transcript/audio artifacts land in raw_streaming_data and are referenced by
Lakebase-owned call_facts.

This module is the producer for reference source files. The
`batch_reference_ingest` job task reads `reference/<table>/*.json` into governed UC
Delta tables. It also writes the per-call transcript/audio artifacts so
call_facts.transcript_path/audio_path are real file links.

Offline (GENIE_LOCAL_VOLUME_DIR set): exports tables to JSON + writes artifacts
to the local dir so the in-process emulator (enrich.derive) can run with no
Databricks.
"""
from __future__ import annotations

import io
import json
import os

from genie_voice.config import Settings, get_settings

from .build import Dataset, build_dataset
from .schema import REFERENCE_TABLES
from .sqlwriter import export_local


def land_to_volume(dataset: Dataset, settings: Settings) -> None:
    """Land each reference table as newline-delimited JSON in the Volume, one
    sub-dir per table (reference/<table>/<table>.json), plus the call artifacts.
    These files are audit/demo exports; Lakebase CDF is the UC analytics source."""
    from genie_voice.databricks.client import get_workspace_client

    client = get_workspace_client(settings)
    for name in REFERENCE_TABLES:
        rows = dataset.table(name)
        payload = "\n".join(json.dumps(r, default=str) for r in rows)
        path = f"{settings.reference_table_path(name)}/{name}.json"
        client.files.upload(path, io.BytesIO(payload.encode()), overwrite=True)
        print(f"  landed {len(rows):>4} rows -> {path}")
    _write_artifacts(dataset, settings, local=False)


def export_local_tables(dataset: Dataset, settings: Settings, out_dir: str) -> None:
    export_local(settings, out_dir, {name: dataset.table(name) for name in REFERENCE_TABLES})
    _write_artifacts(dataset, settings, local=True)


def _write_artifacts(dataset: Dataset, settings: Settings, local: bool) -> None:
    """Write transcript (.txt) and a placeholder audio (.wav) per call."""
    local_dir = os.environ.get("GENIE_LOCAL_VOLUME_DIR")
    if local and local_dir:
        base = os.path.dirname(local_dir.rstrip("/")) or local_dir
        for c in dataset.calls:
            _write_local(os.path.join(base, "transcripts"), f"{c['call_id']}.txt", c["transcript_text"])
            _write_local(os.path.join(base, "audio"), f"{c['call_id']}.wav", f"MOCK_AUDIO::{c['call_id']}")
        return
    if local:
        return

    from genie_voice.databricks.client import get_workspace_client

    client = get_workspace_client(settings)
    for c in dataset.calls:
        client.files.upload(c["transcript_path"], io.BytesIO(c["transcript_text"].encode()), overwrite=True)
        client.files.upload(c["audio_path"], io.BytesIO(f"MOCK_AUDIO::{c['call_id']}".encode()), overwrite=True)


def _write_local(folder: str, name: str, content: str) -> None:
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, name), "w") as fh:
        fh.write(content)


def main() -> None:
    settings = get_settings()
    dataset = build_dataset(settings)
    local_dir = os.environ.get("GENIE_LOCAL_VOLUME_DIR")
    if local_dir:
        out = os.path.normpath(os.path.join(local_dir, "..", "tables"))
        export_local_tables(dataset, settings, out)
        print(f"Exported reference tables ({len(dataset.calls)} calls) locally to {out}.")
    else:
        land_to_volume(dataset, settings)
        print(f"Landed reference tables ({len(dataset.calls)} calls) -> {settings.reference_path}.")


if __name__ == "__main__":
    main()
