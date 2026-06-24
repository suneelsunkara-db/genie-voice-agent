"""Offline live-state emulator for the local UI.

Online deployments write call state through the Lakebase ingest task. This module
is only used when Databricks is unavailable and Lakebase falls back to memory.
"""
from __future__ import annotations

from typing import Any

from genie_voice.config import Settings, get_settings
from genie_voice.enrich.engine import enrich_utterance, summarize_call
from genie_voice.mock.call_scripts import get_scripts
from genie_voice.providers import get_stt_provider
from genie_voice.serve import LakebaseServing


def run(settings: Settings | None = None) -> list[dict[str, Any]]:
    settings = settings or get_settings()
    provider = get_stt_provider(settings)
    serving = LakebaseServing(settings)
    serving.ensure_schema()

    render = {
        "channels": settings.mock.channels,
        "interim_words_step": settings.mock.interim_words_step,
        "inject_low_confidence": settings.mock.inject_low_confidence,
    }

    gold_rows: list[dict[str, Any]] = []
    for script in get_scripts():
        call_id = script["call_id"]
        customer_id = script.get("customer_id")

        # Normalize -> utterances (keep complete turns).
        utterances: list[dict[str, Any]] = []
        for raw in provider.mock_events(script["turns"], render=render):
            ev = provider.normalize(raw, call_id=call_id)
            if ev.is_utterance_end:
                utterances.append({"text": ev.text, "speaker": ev.speaker})

        # Call-level gold record from the Foundation Model contract.
        gold = summarize_call(utterances, settings)
        gold_row = {"call_id": call_id, "customer_id": customer_id, **gold}
        gold_rows.append(gold_row)

        # Live state for the agent UI (same engine as gold).
        live = enrich_utterance(utterances[-1]["text"], settings) if utterances else {}
        serving.upsert_call_state(
            call_id,
            customer_id,
            {"gold": gold, "live": live, "utterances": utterances},
        )
    return gold_rows


if __name__ == "__main__":
    for row in run():
        print(row["call_id"], "->", row["primary_intent"], row["disposition"])
