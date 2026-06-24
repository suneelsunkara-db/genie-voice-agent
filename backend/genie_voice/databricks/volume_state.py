"""Small helpers for checking whether demo Volume inputs already exist."""
from __future__ import annotations

import os

from genie_voice.config import Settings, get_settings


def _local_has_files(path: str) -> bool:
    if not os.path.exists(path):
        return False
    if os.path.isfile(path):
        return True
    for _, _, files in os.walk(path):
        if files:
            return True
    return False


def _volume_has_files(path: str, settings: Settings) -> bool:
    if os.environ.get("GENIE_LOCAL_VOLUME_DIR"):
        return _local_has_files(path)

    from genie_voice.databricks.client import get_workspace_client

    client = get_workspace_client(settings)
    try:
        entries = list(client.files.list_directory_contents(path.rstrip("/")))
    except Exception:
        return False
    for entry in entries:
        entry_path = getattr(entry, "path", None) or str(entry)
        if not entry_path.endswith("/"):
            return True
        if _volume_has_files(entry_path.rstrip("/"), settings):
            return True
    return False


def reference_inputs_present(settings: Settings | None = None) -> bool:
    settings = settings or get_settings()
    return all(
        _volume_has_files(settings.reference_table_path(table), settings)
        for table in ("customers", "agents", "invoices", "payments")
    )


def streaming_inputs_present(settings: Settings | None = None) -> bool:
    settings = settings or get_settings()
    return _volume_has_files(settings.raw_stt_path, settings) and _volume_has_files(
        settings.call_facts_path, settings
    )


def main() -> None:
    settings = get_settings()
    print(f"reference={'1' if reference_inputs_present(settings) else '0'}")
    print(f"streaming={'1' if streaming_inputs_present(settings) else '0'}")


if __name__ == "__main__":
    main()
