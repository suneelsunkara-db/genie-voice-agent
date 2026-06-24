"""build_dataset(): the single source of truth.

Produces a fully self-consistent enterprise dataset where every entity links
correctly:

  customers ──< invoices ──< payments
      │             ^
      │             │ (mentioned_invoice_id, derived)
  call_facts >── agents          gold_call_insights ──(call_id)──> call_facts
      │  (datagen-owned: agent, ts, duration, csat, file paths)
      └──(customer_id)
                       audio_path / transcript_path -> files in the UC Volume

datagen produces the source-system tables (customers/agents/invoices/payments)
and the telephony grain `call_facts`. The dialogue (turns/utterances) drives the
producer + the enrichment pipeline, which derives silver/gold. Speech, text, and
files are all keyed by `call_id`, and every call references a REAL
customer/agent (and a real invoice where one is discussed).
"""
from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from .generators import gen_agents, gen_customers, gen_invoices_and_payments
from .scenarios import build_call

if TYPE_CHECKING:  # avoid importing config (yaml/pydantic) just to build data
    from genie_voice.config import Settings

_WORD_SEC = 0.32
_GAP_SEC = 0.4


@dataclass
class Dataset:
    agents: list[dict] = field(default_factory=list)
    customers: list[dict] = field(default_factory=list)
    invoices: list[dict] = field(default_factory=list)
    payments: list[dict] = field(default_factory=list)
    # Full per-call records: telephony facts + turns + utterances + artifacts.
    calls: list[dict] = field(default_factory=list)

    def call_facts_rows(self) -> list[dict]:
        """Telephony/CTI metadata per call (the datagen-owned `call_facts` table).

        Deliberately excludes any transcript-derived field - those belong to the
        enrichment pipeline's gold_call_insights, not to seeded reference data.
        """
        keep = ["call_id", "customer_id", "agent_id", "call_ts", "duration_sec",
                "csat", "audio_path", "transcript_path"]
        return [{k: c.get(k) for k in keep} for c in self.calls]

    def table(self, logical_name: str) -> list[dict]:
        """Rows for a REFERENCE table (datagen-owned)."""
        return {
            "agents": self.agents,
            "customers": self.customers,
            "invoices": self.invoices,
            "payments": self.payments,
            "call_facts": self.call_facts_rows(),
        }[logical_name]

    def call_scripts(self) -> list[dict]:
        """Shape consumed by the producer / enrichment / local_runner."""
        return [
            {"call_id": c["call_id"], "customer_id": c["customer_id"], "turns": c["turns"]}
            for c in self.calls
        ]


def _annotate_timing_and_artifacts(call: dict, settings: "Settings", call_ts: datetime) -> None:
    audio_base = settings.resolve_volume_path(settings.volume.audio_path)
    transcript_base = settings.resolve_volume_path(settings.volume.transcript_path)
    cid = call["call_id"]

    utterances: list[dict] = []
    t = 0.0
    transcript_lines: list[str] = []
    for idx, turn in enumerate(call["turns"]):
        words = max(1, len(turn["text"].split()))
        start = round(t, 3)
        end = round(t + words * _WORD_SEC, 3)
        role = turn["speaker"]
        utterances.append({
            "utterance_id": f"{cid}-U{idx:02d}",
            "call_id": cid,
            "turn_index": idx,
            "channel": 0 if role == "agent" else 1,
            "speaker_role": role,
            "start_sec": start,
            "end_sec": end,
            "text": turn["text"],
            "confidence": round(0.93 + (idx % 5) * 0.012, 4),
        })
        transcript_lines.append(f"[{role}] {turn['text']}")
        t = end + _GAP_SEC

    call["utterances"] = utterances
    call["transcript_text"] = "\n".join(transcript_lines)
    call["duration_sec"] = int(round(t))
    call["call_ts"] = call_ts.replace(microsecond=0).isoformat()
    call["audio_path"] = f"{audio_base}/{cid}.wav"
    call["transcript_path"] = f"{transcript_base}/{cid}.txt"


def build_dataset(settings: "Settings | None" = None) -> Dataset:
    if settings is None:
        from genie_voice.config import get_settings

        settings = get_settings()
    dg = settings.datagen
    rng = random.Random(dg.seed)

    agents = gen_agents(rng, dg.num_agents)
    customers = gen_customers(rng, dg.num_customers)
    invoices, payments = gen_invoices_and_payments(rng, customers, dg.months_history)

    inv_by_cust: dict[str, list[dict]] = defaultdict(list)
    for inv in invoices:
        inv_by_cust[inv["customer_id"]].append(inv)
    pay_by_cust: dict[str, list[dict]] = defaultdict(list)
    for p in payments:
        pay_by_cust[p["customer_id"]].append(p)

    calls: list[dict] = []
    now = datetime.now()
    for k in range(dg.num_calls):
        cust = customers[k % len(customers)]
        agent = agents[k % len(agents)]
        call_id = f"CALL-{2000 + k}"
        call = build_call(
            call_id=call_id,
            customer=cust,
            invoices=inv_by_cust[cust["customer_id"]],
            payments=pay_by_cust[cust["customer_id"]],
            agent=agent,
            rng=rng,
        )
        call_ts = now - timedelta(days=rng.randint(0, 27), minutes=rng.randint(0, 1440))
        _annotate_timing_and_artifacts(call, settings, call_ts)
        calls.append(call)

    return Dataset(
        agents=agents,
        customers=customers,
        invoices=invoices,
        payments=payments,
        calls=calls,
    )
