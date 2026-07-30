"""Land the credit-card reference tables into the card-issuer UC Volume.

Mirrors ``datagen/loader.py`` but for the card-issuer domain: each reference
table is written as newline-delimited JSON under
``/Volumes/<catalog>/<card_schema>/<batch_volume>/reference/<table>/<table>.json``,
where the ``batch_reference_ingest`` step reads it into governed UC Delta.

Offline (GENIE_LOCAL_VOLUME_DIR set): export tables to JSON + emit each table's
DDL for inspection, with no Databricks calls.
"""
from __future__ import annotations

import io
import json
import os

from genie_voice.config import Settings, get_settings

from .build_card import CardDataset, build_card_dataset
from .schema_card import CARD_MODEL, CARD_REFERENCE_TABLES


def land_to_volume_card(dataset: CardDataset, settings: Settings) -> None:
    from genie_voice.databricks.client import get_workspace_client

    client = get_workspace_client(settings)
    for name in CARD_REFERENCE_TABLES:
        rows = dataset.table(name)
        payload = "\n".join(json.dumps(r, default=str) for r in rows)
        path = f"{settings.card_reference_table_path(name)}/{name}.json"
        client.files.upload(path, io.BytesIO(payload.encode()), overwrite=True)
        print(f"  landed {len(rows):>5} rows -> {path}")


def export_local_tables_card(dataset: CardDataset, settings: Settings, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for name in CARD_REFERENCE_TABLES:
        rows = dataset.table(name)
        with open(os.path.join(out_dir, f"{name}.json"), "w") as fh:
            json.dump(rows, fh, indent=2, default=str)
        with open(os.path.join(out_dir, f"{name}.sql"), "w") as fh:
            fh.write(CARD_MODEL[name].render_ddl(settings.card_fqtn) + ";\n")


def main() -> None:
    settings = get_settings()
    if not settings.card_issuer.enabled:
        print("card_issuer.enabled is false; skipping card-issuer data load.")
        return
    dataset = build_card_dataset(settings)
    local_dir = os.environ.get("GENIE_LOCAL_VOLUME_DIR")
    if local_dir:
        out = os.path.normpath(os.path.join(local_dir, "..", "card_tables"))
        export_local_tables_card(dataset, settings, out)
        print(f"Exported card-issuer reference tables locally to {out}.")
    else:
        land_to_volume_card(dataset, settings)
        print(f"Landed card-issuer reference tables -> {settings.card_reference_path}.")


if __name__ == "__main__":
    main()
