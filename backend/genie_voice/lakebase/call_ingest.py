"""Ingest landed live-call operational data from raw_streaming_data into Lakebase.

The producer writes vendor-shaped STT events and call_facts records into the
streaming Volume. This task is the source-to-serving bridge: it reads those
landed files, normalizes complete utterances, and upserts the Lakebase primary
tables that CDF publishes back to UC.
"""
from __future__ import annotations

import io
import json
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any, Iterable

from genie_voice.config import Settings, get_settings
from genie_voice.databricks.client import get_workspace_client
from genie_voice.providers import get_stt_provider
from genie_voice.serve import LakebaseServing


def _marker_path(settings: Settings) -> str:
    return f"{settings.checkpoint_path}/cdf_markers/latest_call_ingest.json"


def _list_files(client, path: str) -> list[str]:
    out: list[str] = []
    try:
        entries = list(client.files.list_directory_contents(path.rstrip("/")))
    except Exception:
        return out
    for entry in entries:
        entry_path = getattr(entry, "path", None) or str(entry)
        if entry_path.endswith("/"):
            out.extend(_list_files(client, entry_path.rstrip("/")))
        else:
            out.append(entry_path)
    return sorted(out)


def _read_text(client, path: str) -> str:
    resp = client.files.download(path)
    contents = getattr(resp, "contents", resp)
    data = contents.read() if hasattr(contents, "read") else contents
    if isinstance(data, bytes):
        return data.decode()
    return str(data)


def _iter_json_lines(client, path: str) -> Iterable[dict[str, Any]]:
    for line in _read_text(client, path).splitlines():
        line = line.strip()
        if line:
            yield json.loads(line)


def _load_call_facts(client, settings: Settings) -> dict[str, dict[str, Any]]:
    facts: dict[str, dict[str, Any]] = {}
    for path in _list_files(client, settings.call_facts_path):
        for record in _iter_json_lines(client, path):
            call_id = record.get("call_id")
            if call_id:
                facts[str(call_id)] = record
    return facts


def _load_utterances(client, settings: Settings) -> dict[str, list[dict[str, Any]]]:
    provider = get_stt_provider(settings)
    by_call: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in _list_files(client, settings.raw_stt_path):
        for raw in _iter_json_lines(client, path):
            call_id = raw.get("_call_id") or raw.get("call_id")
            if not call_id:
                continue
            event = {k: v for k, v in raw.items() if not str(k).startswith("_")}
            ev = provider.normalize(event, call_id=str(call_id))
            if not ev.is_utterance_end:
                continue
            turn_index = len(by_call[ev.call_id])
            role = "agent" if ev.channel == settings.mock.channels.get("agent", 0) else "customer"
            by_call[ev.call_id].append(
                {
                    "utterance_id": f"{ev.call_id}-U{turn_index:02d}",
                    "call_id": ev.call_id,
                    "turn_index": turn_index,
                    "channel": ev.channel,
                    "speaker_role": role,
                    "start_sec": ev.start,
                    "end_sec": ev.end,
                    "text": ev.text,
                    "confidence": ev.confidence,
                }
            )
    return by_call


def _write_ingest_marker(client, settings: Settings, started_at: str, calls: int, utterances: int) -> None:
    payload = json.dumps(
        {
            "started_at_utc": started_at,
            "completed_at_utc": datetime.now(UTC).replace(microsecond=0).strftime("%Y-%m-%d %H:%M:%S"),
            "calls": calls,
            "utterances": utterances,
        }
    )
    path = _marker_path(settings)
    try:
        client.files.create_directory(path.rsplit("/", 1)[0])
    except Exception:
        pass
    client.files.upload(path, io.BytesIO(payload.encode()), overwrite=True)


def ingest_call_stream(settings: Settings | None = None) -> None:
    s = settings or get_settings()
    if not s.lakebase.enabled:
        print("lakebase.enabled=false -> skipping call Lakebase ingest.")
        return

    client = get_workspace_client(s)
    started_at = datetime.now(UTC).replace(microsecond=0).strftime("%Y-%m-%d %H:%M:%S")
    facts = _load_call_facts(client, s)
    utterances_by_call = _load_utterances(client, s)
    if not facts:
        raise RuntimeError(f"No call_facts records found in {s.call_facts_path}")
    if not utterances_by_call:
        raise RuntimeError(f"No final STT utterances found in {s.raw_stt_path}")

    lb = LakebaseServing(s)
    lb.ensure_schema()
    print(f"Ingesting call stream into Lakebase ({s.lakebase.database}.{s.lakebase.schema_name}) ...")

    utterance_count = 0
    for call_id, fact in sorted(facts.items()):
        utterances = utterances_by_call.get(call_id, [])
        lb.upsert_call_fact(fact)
        lb.replace_live_utterances(call_id, utterances)
        lb.upsert_call_state(
            call_id,
            fact.get("customer_id"),
            {
                "call_id": call_id,
                "customer_id": fact.get("customer_id"),
                "agent_id": fact.get("agent_id"),
                "utterances": utterances,
            },
        )
        utterance_count += len(utterances)
        print(f"  ingested call {call_id}: {len(utterances)} utterances")
    _write_ingest_marker(client, s, started_at, len(facts), utterance_count)


if __name__ == "__main__":
    ingest_call_stream()
