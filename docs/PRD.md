# PRD — Databricks Genie Voice Agent (Contact-Center Intelligence)

**Status:** Draft v0.1
**Owner:** TBD
**Last updated:** 2026-06-14
**Target Databricks workspace:** configure `databricks.host` in `config/config.yaml`
(e.g. `https://<your-workspace>.cloud.databricks.com`).
**Companions:** architecture → `docs/ARCHITECTURE.md` · data model & relationships → `docs/DATA_MODEL.md` · revenue use cases → `docs/REVENUE_USE_CASES.md`

---

## 1. Overview

A contact-center intelligence system that captures live agent↔customer phone
conversations, enriches them in **real time** for in-call agent assistance, and
makes the resulting structured data available for **post-call analytics through
Databricks AI/BI Genie**, joined to the enterprise's existing structured data
(customer, billing, invoices).

The product is delivered in two phases, in this order:

1. **Phase 1 — Millisecond live enrichment.** During a call, stream the
   transcript, derive live signals (intent, sentiment, compliance/keyword
   triggers, next-best-action), and surface the customer's account facts to the
   agent with sub-second latency.
2. **Phase 2 — Post-call Genie analytics.** Persist conversations into a
   medallion architecture, extract a structured per-call record with an LLM,
   join it to existing billing/customer/invoice tables, and expose it through a
   Genie space for natural-language analytics.

### Build strategy: mock-first against real contracts
The entire pipeline is built and validated against **mocked data that matches
the real Deepgram (listening) and ElevenLabs (speaking) API response
contracts**. Swapping the mock for the live vendors is a configuration change,
not a rewrite. Mocking validates the *pipeline and analytics*; live ASR
latency/quality can be validated separately with vendor tooling when the demo
moves to real capture.

---

## 2. Problem statement

Enterprises run large contact centers where agents speak to customers about
billing, invoices, and account issues. The conversation itself is rich,
unstructured signal that today is lost or, at best, manually summarized. Three
gaps:

- **In-call:** agents lack instant, contextual access to the caller's account
  facts and no real-time guidance, increasing handle time and errors.
- **Post-call:** there is no structured, queryable record of *what was discussed*
  that can be correlated with billing/invoice data.
- **Analytics:** business users cannot self-serve insights ("how many
  billing_dispute calls correlated with overdue invoices last week?") without a
  data team.

---

## 3. Goals & non-goals

### Goals
- G1. Capture diarized (per-speaker) call transcripts via a swappable STT layer.
- G2. Deliver live in-call enrichment (intent, sentiment, triggers, NBA) at
  millisecond-class processing latency.
- G3. Surface the caller's existing account facts (customer/billing/invoices) to
  the agent with low-latency lookups.
- G4. Persist a governed, structured per-call record joined to existing
  enterprise tables.
- G5. Enable natural-language analytics over the joined data via a Genie space.
- G6. Be buildable end-to-end on mock data without live telephony/ASR access.

### Non-goals (initial)
- N1. Replacing the telephony/CCaaS platform.
- N2. Fully automated voice bots replacing human agents (future option via
  ElevenLabs Agents).
- N3. Advanced ML (forecasting/clustering/anomaly detection) — Genie does
  NL→SQL, not modeling. ML is a separate future workstream.
- N4. Real-time analytics via Genie (Genie is async; it is post-call/analyst-only).

---

## 4. Users & personas

| Persona | Needs | Phase |
|---|---|---|
| **Contact-center agent** | Live account facts + guidance during the call | 1 |
| **Team lead / QA** | Compliance flags, call review, dispositions | 1 + 2 |
| **Analyst / manager** | Self-serve NL analytics over calls + billing | 2 |
| **Data/platform engineer** | Governed, maintainable pipeline | 1 + 2 |
| **Compliance officer** | PII/PCI redaction, retention, audit | 1 + 2 |

---

## 5. Key use cases

- **U1 (live):** Caller mentions an invoice; agent UI instantly shows that
  invoice, balance, and last 3 payments.
- **U2 (live):** Customer sentiment drops / risky keyword detected → real-time
  alert and next-best-action to agent.
- **U3 (post-call):** Auto-generate a structured call record (intent,
  disposition, resolution, sentiment, summary, entities).
- **U4 (analytics):** "How many billing_dispute calls last week, average
  resolution time by agent?" answered in Genie.
- **U5 (analytics):** "Of customers who called about invoice errors, how many
  have an overdue balance?" — answered by joining call data to billing/invoices.

---

## 6. Functional requirements

### Phase 1 — enrichment (BUILD DECISION: Lakebase-first orchestration)
> The host is only a **producer** (`deployment: local|live` picks synthetic vs.
> real capture). Reference rows land in `raw_batch_data` and become UC Delta
> tables. Live call rows land in `raw_streaming_data`, are ingested into Lakebase
> call tables, and Lakebase CDF publishes governed call history into UC. One
> serverless orchestration job gates on CDF readiness, then task-based UC refresh
> builds analytics/gold for Genie.

- F1.1 Ingest transcript events in the **Deepgram streaming** contract
  (`is_final`/`speech_final`, per-channel, per-word confidence/speaker), produced
  by a swappable provider (synthetic for `deployment=local`, live for `live`).
- F1.2 Land reference data into `raw_batch_data` and live call/transcript data
  into `raw_streaming_data`.
- F1.3 Orchestration: ingest streaming Volume files into Lakebase → verify
  Lakebase CDF has published fresh `lb_<table>_history` to UC →
  `gold_insights_refresh` derives `gold_call_insights` directly from call and
  utterance history.
- F1.4 Upsert live call state to **Lakebase (Autoscaling)** for the agent UI.
- F1.5 Serve the caller's existing account facts from governed UC reference
  tables (`customers`, `invoices`, `payments`, etc.).
- F1.6 (later) Upgrade hot path to **Real-Time Mode + Kafka** for millisecond
  latency, reusing the same enrichment logic.

### Phase 2 — Post-call Genie analytics
- F2.1 Lakebase CDF history→Gold in Unity Catalog
  (`lb_live_call_utterances_history` + `call_facts` → `gold_call_insights`).
- F2.2 Lakebase `live_call_utterances` contains normalized, diarized utterances
  and CDF publishes the history that gold consumes.
- F2.3 Gold (`gold_call_insights`): one row per call via **LLM extraction**
  (Foundation Model API): `call_id, customer_id, primary_intent, all_intents,
  disposition, resolution_status, sentiment_score, sentiment_label,
  next_best_action, mentioned_invoice_id, mentioned_amount, summary`.
- F2.4 Join `gold_call_insights` to `call_facts`/`customers`/`invoices` on
  `call_id` / `customer_id`.
- F2.5 Configure a **Genie space** over gold + existing tables (column comments,
  sample questions, instructions) and expose via the **Genie Conversation API**.
- F2.6 Optional voice front-end for analyst questions (ElevenLabs TTS for spoken
  answers).

### Cross-cutting
- F3.1 STT/TTS are **swappable layers** behind a stable internal contract.
- F3.2 **PII/PCI redaction** applied before storage; verbatim (post-redaction)
  retained.
- F3.3 All data governed by **Unity Catalog** (masking, RLS, lineage, audit).

---

## 7. Data contracts (mocked, real-shaped)

- **Listening (live):** Deepgram **streaming** response — `type`,
  `channel_index`, `start`, `duration`, `is_final`, `speech_final`,
  `channel.alternatives[].{transcript,confidence,words[]}`, optional `entities`.
- **Listening (batch, optional):** Deepgram pre-recorded —
  `results.channels[].alternatives[].{transcript,confidence,words[]}` +
  `results.utterances[]`.
- **Speaking:** ElevenLabs TTS — `audio_base64` + `alignment`
  (`characters`, `character_start_times_seconds`, `character_end_times_seconds`).

See `docs/ARCHITECTURE.md` for how each contract maps to tables.

---

## 8. Tech stack

| Layer | Choice | Notes |
|---|---|---|
| Telephony (prod) | Twilio (Media Streams / Flex) | Cloud-neutral; dual-channel. Mocked in dev |
| STT — listening | **Deepgram** | Telephony-grade, diarization, redaction, self-hostable |
| TTS — speaking | **ElevenLabs** | Spoken answers / future voice agent |
| Ingestion (Phase 1) | **UC Volumes + one serverless orchestration job** | Batch reference files to UC; streaming call files to Lakebase, then CDF to UC |
| Live enrichment (later) | **Databricks Real-Time Mode + Kafka** | Millisecond upgrade; shared logic |
| Low-latency serving | **Lakebase Autoscaling / Online Feature Store** | Serverless Postgres, sub-ms lookups |
| Storage | **UC Volumes + Delta medallion** | Unstructured + structured |
| Extraction | **Databricks Foundation Model API** | Transcript → gold record |
| Analytics | **Genie space + Conversation API** | NL→SQL over joined data |
| Governance | **Unity Catalog** | Masking, RLS, lineage |
| Packaging | **Databricks App / FastAPI** | OAuth M2M service principal |

---

## 9. Milestones

- **M0 — Foundation:** mock Deepgram streaming generator + sample call scripts;
  Redpanda up; produce events to Kafka.
- **M1 — Live hot path:** RTM enrichment job (Kafka→enrich→Lakebase/Kafka sink).
- **M2 — Live serving:** Lakebase/Online Feature Store account lookups + agent UI.
- **M3 — Persistence:** micro-batch Kafka→bronze Delta.
- **M4 — Medallion + extraction:** silver/gold + LLM extraction.
- **M5 — Genie:** join + Genie space + Conversation API.
- **M6 — Live vendor swap:** replace mock with Deepgram/Twilio; ElevenLabs TTS.

---

## 10. Success metrics

- **Live:** enrichment processing p99 latency (target sub-100ms in RTM); account
  lookup p99 (target < 50ms); trigger precision/recall.
- **Extraction:** gold-record completeness, valid controlled vocabularies, and
  business consistency checks before Genie publication.
- **Genie:** answer correctness rate on a fixed analyst question set.
- **Adoption:** agent usage, analyst questions/week.

---

## 11. Compliance & security

- PII/PCI redaction before storage (PCI-DSS for card data on invoices/billing).
- Recording-consent / call-monitoring law compliance; retention policy.
- Unity Catalog access controls, column masking, row-level security, lineage.
- Encryption at rest/in transit; OAuth M2M for service access (PAT for local dev only).
- Keep **verbatim** (post-redaction) transcript for audit/QA.

---

## 12. Risks & mitigations

| Risk | Mitigation |
|---|---|
| RTM can't use Delta as source/sink | Kafka backbone; `forEach`/Lakebase sink; separate micro-batch to Delta |
| RTM only on classic compute (cost) | Reserve RTM for hot path only; batch elsewhere |
| Genie quality depends on schema curation | Rich column comments, sample questions, instructions |
| Mock too clean → over-optimistic | Inject realistic confidence variance, redaction, disfluencies, cross-talk |
| Genie ≠ ML | Set expectations; ML is separate workstream |
| Vendor lock-in | STT/TTS behind swappable contract |

---

## 13. Open questions

- Which CCaaS in production (Twilio assumed)?
- Unity Catalog naming (catalog/schema) and where existing billing/customer/invoice tables live?
- Multilingual requirements (would tilt STT toward ElevenLabs Scribe)?
- Do we need spoken answers (TTS) in Phase 2, or text-only Genie?
- Data residency / self-hosting requirements for STT?
