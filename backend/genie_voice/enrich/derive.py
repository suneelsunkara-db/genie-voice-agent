"""OFFLINE emulator of the Databricks pipeline (no workspace required).

When there is no Databricks auth, `local-deploy.sh` sets GENIE_LOCAL_VOLUME_DIR
and this module stands in for BOTH serverless jobs so the UI/Genie demo still
works end to end on a laptop:

    /<run>/volume/raw_stt/*.json
        --provider.normalize-->    silver_call_utterances   (diarized turns)
        --enrich.summarize_call--> gold_call_insights        (NLP insights)

It writes the derived tables as local JSON exports (alongside the reference
tables exported by `genie_voice.datagen.loader`).

Silver (diarization) is fully offline. gold insights use the Foundation Model
engine (enrich.engine), so they require Databricks auth + a reachable serving
endpoint; with no auth the insight fields come back as "unavailable" (null) -
there is no heuristic fallback.

ONLINE, this is NOT used: orchestration tasks produce final UC call/gold tables
from Lakebase CDF history. Calling
`derive()` without GENIE_LOCAL_VOLUME_DIR raises, so there is never a second
writer of the derived tables.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

from genie_voice.config import Settings, get_settings
from genie_voice.datagen.schema import T_GOLD
from genie_voice.datagen.sqlwriter import export_local
from genie_voice.enrich.engine import summarize_call
from genie_voice.providers import get_stt_provider


def _read_raw_events(local_dir: str) -> dict[str, list[dict]]:
    """Read the landed STT JSONL files (one call per file) from the local dir."""
    out: dict[str, list[dict]] = defaultdict(list)
    for fn in sorted(os.listdir(local_dir)):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(local_dir, fn)) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    ev = json.loads(line)
                    out[ev.get("_call_id") or fn[:-5]].append(ev)
    return out


def _silver_and_gold(settings: Settings, local_dir: str) -> tuple[list[dict], list[dict]]:
    """Normalize raw events -> silver utterances; roll up -> gold insights."""
    provider = get_stt_provider(settings)
    raw_by_call = _read_raw_events(local_dir)

    silver: list[dict] = []
    gold: list[dict] = []
    for call_id, events in raw_by_call.items():
        utterances: list[dict] = []
        customer_id = None
        turn = 0
        for raw in events:
            customer_id = customer_id or raw.get("_customer_id")
            ev = provider.normalize(raw, call_id=call_id)
            if not ev.is_utterance_end:  # keep complete turns only
                continue
            role = "agent" if ev.channel == 0 else "customer"
            silver.append({
                "utterance_id": f"{call_id}-{turn}",
                "call_id": call_id,
                "turn_index": turn,
                "channel": ev.channel,
                "speaker_role": role,
                "start_sec": round(ev.start, 4),
                "end_sec": round(ev.end, 4),
                "text": ev.text,
                "confidence": round(ev.confidence, 4),
            })
            utterances.append({"text": ev.text, "speaker_role": role})
            turn += 1
        insights = summarize_call(utterances)
        insights.pop("available", None)  # not a gold column; engine bookkeeping only
        gold.append({"call_id": call_id, "customer_id": customer_id, **insights})
    return silver, gold


def derive(settings: Settings | None = None) -> dict[str, int]:
    settings = settings or get_settings()
    local_dir = os.environ.get("GENIE_LOCAL_VOLUME_DIR")
    if not local_dir:
        raise RuntimeError(
            "enrich.derive is the OFFLINE emulator (set GENIE_LOCAL_VOLUME_DIR). "
            "Online, silver is produced by the streaming voice job and gold by the "
            "batch job - deploy them via infra/jobs/deploy_pipeline.py."
        )

    silver, gold = _silver_and_gold(settings, local_dir)
    out = os.path.normpath(os.path.join(local_dir, "..", "tables"))
    export_local(settings, out, {"silver_call_utterances": silver, T_GOLD: gold})
    return {"silver_call_utterances": len(silver), "gold_call_insights": len(gold)}


def main() -> None:
    print("derived (offline):", derive())


if __name__ == "__main__":
    main()
