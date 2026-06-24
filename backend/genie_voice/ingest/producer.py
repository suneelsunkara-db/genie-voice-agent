"""Mock producer: render call scripts via the ACTIVE STT provider and land the
vendor-shaped events into the Volume.

This is vendor-agnostic: it never names Deepgram/ElevenLabs. It asks the
configured provider for `mock_events()` and writes whatever shape it returns.
Swapping providers in config changes the payloads with zero code change here.
"""
from __future__ import annotations

import time
from typing import Any

from genie_voice.config import Settings, get_settings
from genie_voice.datagen import build_dataset
from genie_voice.providers import get_stt_provider

from .volume_writer import write_events, write_json_record


def _render_opts(settings: Settings) -> dict[str, Any]:
    return {
        "channels": settings.mock.channels,
        "interim_words_step": settings.mock.interim_words_step,
        "inject_low_confidence": settings.mock.inject_low_confidence,
    }


def produce_all(settings: Settings | None = None, pace: bool | None = None) -> list[str]:
    settings = settings or get_settings()
    provider = get_stt_provider(settings)
    render = _render_opts(settings)
    pace = settings.mock.realtime_pacing if pace is None else pace

    written: list[str] = []
    dataset = build_dataset(settings)
    facts = {row["call_id"]: row for row in dataset.call_facts_rows()}
    for call in dataset.calls:
        events = list(provider.mock_events(call["turns"], render=render))
        path = write_events(
            call["call_id"],
            events,
            settings,
            meta={"_customer_id": call.get("customer_id"), "_agent_id": call.get("agent_id")},
        )
        written.append(path)
        fact_path = f"{settings.call_facts_path}/{call['call_id']}.json"
        write_json_record(fact_path, {"event_type": "call_ended", **facts[call["call_id"]]}, settings)
        if pace:
            time.sleep(1.0)  # stagger calls so the UI shows a live trickle
    return written


if __name__ == "__main__":
    for p in produce_all():
        print("wrote", p)
