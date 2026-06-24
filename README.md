# Databricks Genie Voice Agent

Contact-center voice intelligence on Databricks. Captures agent↔customer calls,
serves live agent assist from **Lakebase**, and publishes governed analytics to
Unity Catalog for **AI/BI Genie**.

See [`docs/PRD.md`](docs/PRD.md), [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), and
[`docs/DATA_MODEL.md`](docs/DATA_MODEL.md) (entities, relationships, Genie sample questions).

## Problem statement: Genie for voice use cases

Voice contact centers need **account-aware, governed intelligence during the call** —
not after it. Agents still spend hold time searching CRM, billing, and ticket systems
while the customer waits. Generic LLM chatbots are not the answer: they are expensive
at call volume, hard to audit, and disconnected from the warehouse of record.

This project shows how **Databricks Genie**, **streaming capture**, and **Lakebase**
work together so customer context and insights arrive **while the agent is on the
line**, grounded in Unity Catalog — without token-maxing every syllable.

## Today's tools and gaps

| What exists today | Gap on live voice calls |
|---|---|
| CRM / billing portals | Agent leaves the call flow to look up account facts |
| Post-call analytics & QA | Insights arrive too late to change the outcome |
| Generic agentic chat (full-context LLM per turn) | High token cost, weak governance, latency on every utterance |
| Batch Genie / BI dashboards | Great for portfolio questions, not millisecond agent assist |

**Core gap:** missing **customer context and insights in live calls** — overdue
balances, dispute signals, waiver eligibility, and resolution state are not
streamed to the agent at the moment they matter.

## Proposed approach: streaming, Genie, and Lakebase

```
Voice (Deepgram)  →  utterance-bound inference  →  Lakebase live serving
Governed UC data  →  Genie fact validation      →  Agent Assist UI
Lakebase CDF      →  UC analytics + gold         →  Genie space (portfolio intelligence)
```

1. **Streaming capture** — STT turns continuous audio into **final utterances**
   (not per-chunk LLM calls). Local mode uses synthetic producer; live mode uses Deepgram.
2. **Lakebase (hot path)** — sub-second reads/writes for `call_state`, account facts
   overlay, `resolution_events`, and `billing_adjustments`. No warehouse round-trip
   on every UI poll.
3. **Foundation Model (per turn)** — one structured FM call per customer utterance
   (intent, sentiment, `customer_signal`, waiver/plan flags) plus short agent prose.
4. **Genie (governed facts)** — NL→SQL over curated UC tables to **validate** account
   metrics and power the Genie console; not every spoken word goes through Genie chat.
5. **Unity Catalog (cold path)** — batch reference ingest, CDF history, gold insights,
   data quality gate, then Genie space reconcile.

> `deployment` (`local` | `live`) selects who generates capture data. Lakebase is the
> low-latency serving system; UC is the asynchronous analytics path.

## Impact

- **Streaming customer insights with Genie while the agent engages the customer** —
  overdue exposure, at-risk status, recommended next action, and resolution journey
  update on each customer turn; Genie validates facts against governed tables.
- **Less hold time and faster issue resolution** — account context is pre-loaded from
  Lakebase; FM enrichment and billing close run in one assist transaction per utterance.
- **Avoids token-maxing for agentic solutions** — LLM spend scales with **meaningful
  customer turns**, not audio frames or full transcript re-summarization; Lakebase
  serves state without tokens; Genie runs for analytics validation, not live prose.

## Architecture (Lakebase first)

```
HOST / SERVERLESS JOBS                    DATABRICKS
producer (local|live) ------------->       UC Volumes (raw_batch_data/raw_streaming_data)
batch_reference_ingest ------------>       UC reference Delta tables
call_lakebase_ingest -------------->       Lakebase call_state/call_facts/live_call_utterances
gold_insights_refresh ------------->       gold Delta table from CDF history
Genie space ----------------------->       curated UC analytics tables
backend API + observability UI <----       Lakebase for live serving
```

**Serving first, analytics second.** Lakebase owns live agent-assist serving.
Unity Catalog owns governed analytics and Genie. No duplicate
`*_serving` tables are used.

Everything is **config-driven** (`config/config.yaml` + `.env`) and external
vendors (STT/TTS) are **swappable** behind a provider interface — no vendor name
appears in core code.

## Repository layout

```
config/         config.yaml (non-secret) + .env.example (secrets)
backend/        genie_voice package (core library)
  genie_voice/
    config/       settings loader (all tunables)
    models/       canonical vendor-neutral contracts
    providers/    swappable STT/TTS adapters + dynamic registry
    mock/         call scripts (sourced from the data generator)
    datagen/      enterprise dataset generator (schema, relationships, file producer)
    ingest/       voice producer + Volume writer
    databricks/   SDK client + UC bootstrap (schema/volume/DDL/grants)
    pipeline/     wheel task CLI
    lakebase/     Lakebase-first seed/load helpers
    enrich/       Foundation Model enrichment (utterance + call summary; no heuristic fallback)
    assist/       Live resolution, billing, Genie validation, alignment checks
    serve/        Lakebase autoscaling serving (call state, resolution events, billing adjustments)
    genie/        Genie Conversation API client
api/            FastAPI service (health, agent-assist, accounts, genie, status)
frontend/       Vite/React agent-assist cockpit (customers with issues, resolution journey, Genie panel)
infra/lakebase/ Lakebase Autoscaling provisioning
local-deploy.sh end-to-end local deploy
```

## Authentication & permissions (U2M, no PAT)

The app runs **as your Databricks user** via OAuth U2M — no tokens or secrets in
`.env`. `local-deploy.sh` runs `databricks auth login --host <host>` for you
(opens a browser) and everything thereafter runs under that identity.

Set workspace values in **`config/config.local.yaml`** (gitignored full config,
deep-merged over `config/config.yaml`). Copy `config/config.yaml` as a starting
point and replace placeholders with your workspace values.

```yaml
databricks:
  host: "https://<your-workspace>.cloud.databricks.com"
  profile: "<your-databricks-profile>"
  run_as: "user@example.com"
  catalog: "<your-catalog>"
  sql_warehouse_id: "<your-sql-warehouse-id>"
lakebase:
  instance: "<your-lakebase-instance>"
```

`bootstrap` then creates the schema + Volume in the existing catalog and applies
the GRANTs the app needs (`USE CATALOG` + `ALL PRIVILEGES ON SCHEMA`). Lakebase
uses **runtime-minted Postgres tokens** (no stored password); set
`lakebase.enabled: true` to use the real instance.

> Prefer PAT or a service principal? Set `auth_type: pat` (then `DATABRICKS_TOKEN`)
> or `auth_type: oauth` (then `DATABRICKS_CLIENT_ID/SECRET`).

## Quick start

```bash
cp config/.env.example .env              # optional for U2M; add vendor keys for live mode
cp config/config.yaml config/config.local.yaml   # then edit with your workspace values
# Edit config/config.local.yaml — host, catalog, sql_warehouse_id, lakebase instance, secrets
./local-deploy.sh                # logs you in, sets up perms, runs flow, starts API+UI
# UI:  http://localhost:5173
# API: http://localhost:8000/health
./local-undeploy.sh              # stop API + UI
```

`config/config.yaml` in git is a placeholder template only. **`config/config.local.yaml`**
(gitignored) is your full local profile and is deep-merged on top at runtime.

One-command startup with optional Deepgram validation:

```bash
./start_app.sh                   # auth-only Deepgram check (if key exists) + start app
./start_app.sh --live            # force live mode + require DEEPGRAM_API_KEY
./start_app.sh --live --listen-once   # exactly one prerecorded STT test then start
```

If you skip the Databricks login the script runs in **offline mode** (local
volume dir + in-process serving) so you can see the full flow immediately.

## API endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness |
| GET | `/status` | Medallion stages + table row counts + live call states |
| GET | `/calls` | List live call states (Lakebase) |
| GET | `/calls/{call_id}/assist` | Read persisted call enrichment + resolution state |
| GET | `/accounts/with-issues` | Customers with billing/account risk (sidebar queue) |
| POST | `/calls/{call_id}/assist` | Enrich one utterance (FM), advance resolution, optional billing close, FM agent reply |
| POST | `/calls/{call_id}/mic-transcribe` | Deepgram mic blob → same flow as `POST /assist` |
| WS | `/calls/{call_id}/mic-stream` | Streaming mic → Deepgram → `POST /assist` |
| GET | `/calls/{call_id}/account` | Account facts for the call's customer (Lakebase overlay + billing adjustments) |
| GET | `/calls/{call_id}/resolution-events` | Issue status timeline for the call |
| GET | `/calls/{call_id}/alignment` | Lakebase resolution + billing vs account facts consistency check |
| POST | `/calls/{call_id}/reset-demo-session` | Revert billing, clear resolution/timeline/utterances for replay |
| GET | `/accounts/{customer_id}` | Customer + invoices + recent payments |
| POST | `/genie/ask` | Ask the Genie space a question (NL → SQL) |

Account facts are served from governed UC reference tables merged with persisted
`billing_adjustments` when Lakebase is enabled; offline mode uses the local
datagen export.

### Live assist flow (`POST /assist`, customer turn)

1. **FM enrich** — one Foundation Model call returns utterance signals plus
   `customer_signal`, `payment_plan_requested`, and `waiver_requested` (no keyword
   heuristics; unavailable FM returns `available: false`).
2. **Resolution** — FM-driven transitions: `open` → `in_progress` → `closed`.
   Close requires customer `confirm_proceed` and validated account facts.
3. **FM agent reply** — Foundation Model phrases a customer-facing reply using a
   small validated fact block from Lakebase. **Genie cross-checks metrics** when
   needed; Genie Conversation API is for analytics Q&A (`POST /genie/ask`), not
   live spoken prose. If validation fails, `agent_reply` is `null` (no template fallback).
4. **Billing commit** — waiver/payment-plan writes to Lakebase
   `billing_adjustments` and UC `invoices` run **after** the agent reply on
   customer turns. Issue status moves to `closed` only when billing succeeds.
5. **Timeline** — one `resolution_events` row per status transition; duplicates
   are suppressed. The UI **issue resolution journey** reflects business steps
   (describe → understand → review → offer → apply → close). **Reset scenario**
   clears timeline, billing, and call state.

Spotlight demo customer: **CUST-4028 / CALL-2028 (Omar Patel)** — overdue invoice
with late-fee waiver + payment plan path.

## Swapping a provider (no code changes)

Edit `config/config.yaml`:

```yaml
providers:
  stt:
    adapters:
      deepgram: "genie_voice.providers.stt.deepgram:DeepgramSTT"
      assemblyai: "genie_voice.providers.stt.assemblyai:AssemblyAISTT"   # add file + line
    active: assemblyai
```

Drop in `backend/genie_voice/providers/stt/assemblyai.py` implementing
`STTProvider` (a `normalize()` + optional `mock_events()`), and you're done.

## Deployment: local → live

One flag, `deployment` (top of `config/config.yaml`), selects the producer:

- `deployment: local` (default): the synthetic `datagen` generator produces
  vendor-shaped Deepgram/ElevenLabs payloads + reference records. No vendor calls.
- `deployment: live`: set `DEEPGRAM_API_KEY` / `ELEVENLABS_API_KEY` and wire the
  live `stream()`/API paths in the adapters.

The serving and analytics flow is identical for both; only the capture source
changes.

## Serverless Orchestration

`infra/jobs/deploy_pipeline.py` builds the `genie_voice` wheel, copies it +
`config.yaml` into the workspace, uploads the runtime wheel to a stable UC Volume
path, and creates/updates:

- `pipeline.orchestration_job_name`: reference UC ingest + Lakebase call ingest
  in parallel → CDF freshness check → gold refresh → UC constraints → data
  quality → Genie reconcile.

```bash
python infra/jobs/deploy_pipeline.py                 # deploy + run orchestration
python infra/jobs/deploy_pipeline.py --full-refresh  # accepted for compatibility
python infra/jobs/deploy_pipeline.py --no-run        # deploy only
```

`local-deploy.sh` runs this automatically online. See
[`infra/jobs/README.md`](infra/jobs/README.md). Provision serving with
`infra/lakebase/setup_lakebase.py`.

## Genie space (created dynamically by name)

No hardcoded space id. The space is recreated by `databricks.genie_space_name`
after the data quality gate passes, with entity matching on categorical columns,
example SQL, instructions, and benchmark questions. Joins are inferred from the
post-refresh UC PK/FK metadata task.

```bash
python -m genie_voice.genie.space     # runs DQ, recreates by name, prints the URL
```

The orchestration job runs this automatically online after constraints and DQ.
At query time `GenieClient` just resolves the space by name.
