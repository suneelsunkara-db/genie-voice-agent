"""Compatibility wrapper for older callers; delegates to Volume-backed call ingest."""
from __future__ import annotations

from genie_voice.config import Settings, get_settings
from genie_voice.lakebase.call_ingest import ingest_call_stream


def seed_serving_tables(settings: Settings | None = None) -> None:
    ingest_call_stream(settings or get_settings())


if __name__ == "__main__":
    seed_serving_tables()
