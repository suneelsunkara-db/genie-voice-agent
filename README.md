# Databricks Genie Voice Agent

Contact-center voice intelligence on Databricks. Captures agent↔customer calls,
serves live agent assist from **Lakebase**, publishes governed analytics to
Unity Catalog for **AI/BI Genie**, and includes a standalone **Realtime Voice API**
(OSS STT → LLM → TTS on Databricks Model Serving) with a browser test UI.

See [`docs/PRD.md`](docs/PRD.md), [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), and
[`docs/DATA_MODEL.md`](docs/DATA_MODEL.md) (entities, relationships, Genie sample questions).
Future/optional work is tracked in [`ROADMAP.md`](ROADMAP.md).

This is the **single source of truth** README for the repo. Component-level docs
that used to live in subfolders have been merged here (see the table of contents).

## Contents

- [Problem statement](#problem-statement-genie-for-voice-use-cases)
- [Today's tools and gaps](#todays-tools-and-gaps)
- [Proposed approach](#proposed-approach-streaming-genie-and-lakebase)
- [Impact](#impact)
- [Architecture (Lakebase first)](#architecture-lakebase-first)
- [Repository layout](#repository-layout)
- [Authentication & permissions](#authentication--permissions)
- [Running the app: two setups](#running-the-app-two-setups)
- [Setup A — Local (dev)](#setup-a--local-dev)
- [Setup B — Databricks App (hosted)](#setup-b--databricks-app-hosted)
- [Agent-assist API endpoints](#agent-assist-api-endpoints)
- [Swapping a provider](#swapping-a-provider-no-code-changes)
- [Capture mode: local vs live](#capture-mode-local-vs-live-data-producer)
- [Realtime Voice API + UI](#realtime-voice-api--ui)
- [ML ASR pipeline & model serving](#ml-asr-pipeline--model-serving)
- [ASR model-training harness](#asr-model-training-harness)
- [Lakebase (Autoscaling) serving layer](#lakebase-autoscaling-serving-layer)
- [Serverless orchestration job](#serverless-orchestration-job)
- [Genie space](#genie-space-created-dynamically-by-name)

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
config/            config.yaml (non-secret) + config.local.yaml (gitignored, holds secrets)
backend/           genie_voice package (core library)
  genie_voice/
    config/          settings loader (all tunables)
    models/          canonical vendor-neutral contracts
    providers/       swappable STT/TTS adapters + dynamic registry
    mock/            call scripts (sourced from the data generator)
    datagen/         enterprise dataset generator (schema, relationships, file producer)
    ingest/          voice producer + Volume writer
    databricks/      SDK client + UC bootstrap (schema/volume/DDL/grants)
    pipeline/        wheel task CLI
    lakebase/        Lakebase-first seed/load helpers
    enrich/          Foundation Model enrichment (utterance + call summary)
    assist/          Live resolution, billing, Genie validation, alignment checks
    serve/           Lakebase autoscaling serving
    genie/           Genie Conversation API client
    ml_asr/          multilingual ASR bake-off library (serving, eval, jobs)
    asr_eval/        ASR model-training/benchmark harness
api/               FastAPI service (health, agent-assist, accounts, genie, status)
  app/static/      built React SPA served by FastAPI (populated by deploy_app.sh)
frontend/          Vite/React agent-assist cockpit
realtime_api/      standalone Realtime Voice API (WebSocket STT→LLM→TTS) — see below
realtime_test_ui/  standalone browser test client for the Realtime Voice API — see below
scripts/ml_asr/    OSS model register/deploy + realtime-voice agent packaging
scripts/asr/       legacy EN-centric training + registration jobs
infra/lakebase/    Lakebase Autoscaling provisioning
infra/jobs/        serverless orchestration job deploy
infra/apps/        grant_app_sp.py — grants UC + Lakebase + Genie to the app service principal
local-deploy.sh    end-to-end local deploy (U2M, runs as your user)
start_app.sh       one-command local start (API + UI)
deploy_app.sh      one-command deploy to Databricks Apps
app.yaml           Databricks Apps runtime config
```

## Authentication & permissions

- **Local (Setup A):** runs **as your Databricks user** via OAuth U2M — no tokens
  or secrets in `.env`. `local-deploy.sh` runs `databricks auth login` for you.
- **Databricks App (Setup B):** runs as the app's **service principal** via M2M
  OAuth (platform-injected `DATABRICKS_CLIENT_ID/SECRET/HOST`); `deploy_app.sh`
  grants that SP its UC/Lakebase/Genie access and the `workspace-access` entitlement.

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

## Running the app: two setups

There are two ways to run the cockpit. Both use the **same** FastAPI backend and
React frontend and the same `config/config.local.yaml`; they differ only in where
the process runs and how it authenticates.

| | Local (dev) | Databricks App (hosted) |
|---|---|---|
| Processes | Vite dev server + uvicorn (two processes) | one uvicorn process serving API **and** built SPA |
| Auth to Databricks | U2M OAuth **as your user** (`databricks auth login`) | M2M OAuth as the app's **service principal** (injected creds) |
| Vendor keys (Deepgram) | from `config/config.local.yaml` / `.env` | Databricks **secret scope** → injected as env via app resources |
| Frontend URL | `http://localhost:5173` | `https://<app>.<region>.databricksapps.com` (same origin as API) |
| Command | `./local-deploy.sh` / `./start_app.sh` | `./deploy_app.sh` |

`config/config.yaml` in git is a placeholder template only. **`config/config.local.yaml`**
(gitignored) is your full profile and is deep-merged on top at runtime. Both setups
read defaults from it.

---

## Setup A — Local (dev)

Runs **as your Databricks user** via OAuth U2M — no tokens or secrets in git.
`local-deploy.sh` runs `databricks auth login --host <host>` for you (opens a
browser); everything thereafter runs under that identity.

```bash
cp config/config.yaml config/config.local.yaml   # then edit with your workspace values
# Edit config/config.local.yaml — host, catalog, sql_warehouse_id, lakebase instance, secrets
./local-deploy.sh                # logs you in, sets up perms, runs flow, starts API+UI
# UI:  http://localhost:5173
# API: http://localhost:8000/health
./local-undeploy.sh              # stop API + UI
```

One-command startup with optional Deepgram validation:

```bash
./start_app.sh                   # auth-only Deepgram check (if key exists) + start app
./start_app.sh --live            # force live mode + require DEEPGRAM_API_KEY
./start_app.sh --live --listen-once   # exactly one prerecorded STT test then start
```

If you skip the Databricks login the script runs in **offline mode** (local
volume dir + in-process serving) so you can see the full flow immediately.

---

## Setup B — Databricks App (hosted)

One command turns the local two-process app into a single Databricks App web
process: the React SPA is **built and served by FastAPI** from `api/app/static`,
and the app authenticates as its own **service principal** (no personal tokens
at runtime).

```bash
./deploy_app.sh                  # zero-arg: reads defaults from config.local.yaml
```

### Prerequisites

- Databricks CLI (`brew install databricks`) authenticated to the target profile:
  `databricks auth login --profile <profile>`.
- `npm` (builds the frontend) and the repo virtualenv at `.venv/` (the script
  uses `.venv/bin/python` so backend deps are available when reading config).
- `deepgram_api_key` present in `config/config.local.yaml` (or `DEEPGRAM_API_KEY`
  in the env) — required for mic STT.
- The `partner_demo_catalog`, Lakebase instance, SQL warehouse, and the Claude /
  Whisper serving endpoints already exist and are owned by (or grantable by) you.

### What `deploy_app.sh` does (idempotent)

1. **Builds the frontend** for same-origin (`VITE_API_BASE_URL=""`) and copies
   `frontend/dist` → `api/app/static`.
2. **Pushes vendor keys** from `config.local.yaml` into the `genie-voice` secret
   scope (`deepgram_api_key`; `elevenlabs_api_key` only if set).
3. **Creates/updates the app** with declared **resources** (SQL warehouse,
   Deepgram secret, Claude + Whisper serving endpoints) so the app's service
   principal is auto-granted access. ElevenLabs is included only when its key exists.
4. **Grants the service principal** its runtime access:
   - `workspace-access` **entitlement** via SCIM (needed to mint Lakebase Postgres
     OAuth tokens at runtime).
   - **UC + Lakebase + Genie** grants via `infra/apps/grant_app_sp.py`
     (catalog/schema/volume `SELECT`/`MODIFY`/`READ VOLUME`, Lakebase role +
     table/sequence grants, Genie `CAN_RUN`).
5. **Syncs source** to `/Workspace/Users/<you>/genie-voice-agent` (respects
   `.gitignore`) and **deploys** in `SNAPSHOT` mode, printing the app URL.

### Configuration

Defaults live at the top of `deploy_app.sh` (sourced from `config.local.yaml`);
override any of them via env vars before running:

```bash
APP_NAME=genie-voice-agent \
DATABRICKS_PROFILE=<profile> \
SECRET_SCOPE=genie-voice \
SQL_WAREHOUSE_ID=<warehouse-id> \
CLAUDE_ENDPOINT=databricks-claude-opus-4-8 \
WHISPER_ENDPOINT=voice_asr_en_finetuned_whisper_lora \
./deploy_app.sh
```

Runtime behaviour is defined in **`app.yaml`**: it runs `uvicorn app.main:app` on
`0.0.0.0:$DATABRICKS_APP_PORT` and injects vendor secrets via `valueFrom` against
the declared app resources. All configuration (`deployment: live`, `auth_type`,
catalog, warehouse, providers, …) comes from **`config/config.yaml`**, which is
synced with the app — there are no `GENIE_*` env overrides. `run_as` is intentionally left empty so the app uses
its **service principal** identity (`config/config.yaml` ships `run_as: ""`; local
dev sets a real email in `config.local.yaml`).

### Notes / gotchas

- **File-size limit:** Databricks Apps reject any single synced file > 10 MB.
  Large non-runtime assets are `.gitignore`d (e.g. `deck-framework/`, `.run/`) so
  `databricks sync` skips them.
- **`websockets` version:** the mic→Deepgram proxy detects whether the installed
  `websockets` uses `additional_headers` (v13+) or `extra_headers` (v12), so the
  same code works locally and on the Apps runtime.
- **Logs:** Compute → Apps → `genie-voice-agent` → Logs.
- **Egress:** the app calls `api.deepgram.com` directly over the serverless egress
  plane (validated); no proxy/warehouse round-trip for STT.

## Agent-assist API endpoints

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

## Capture mode: local vs live (data producer)

Independent of *where* the app runs (Setup A/B above), one flag — `deployment`
(top of `config/config.yaml`) — selects the capture producer:

- `deployment: local` (default): the synthetic `datagen` generator produces
  vendor-shaped Deepgram/ElevenLabs payloads + reference records. No vendor calls.
- `deployment: live`: set `DEEPGRAM_API_KEY` / `ELEVENLABS_API_KEY` and wire the
  live `stream()`/API paths in the adapters.

The serving and analytics flow is identical for both; only the capture source
changes.

---

# Realtime Voice API + UI

A **standalone** realtime voice loop, independent of the contact-center app above.
It is a thin orchestration layer over OSS models on Databricks Model Serving:

```text
browser PCM frames → STT endpoint → LLM endpoint → TTS endpoint → browser audio
     (16 kHz)        Qwen3-ASR      Qwen3-Next       VoxCPM2       (streamed)
```

It does **not** import or expose the existing app's call, billing, provider, or UI
routes. All API code lives in `realtime_api/`; the browser test client lives in
`realtime_test_ui/`. Model packaging/registration/deployment lives in
`scripts/ml_asr/`. Model-serving settings live in the shared `realtime_voice:`
block of `config/config.yaml` (+ `config/config.local.yaml`).

### Two ways to run

1. **Standalone** (local dev) — run `realtime_api.server` and serve
   `realtime_test_ui/` from any static server (below).
2. **Inside the Databricks app** — `api/app/main.py` additively **mounts** the
   realtime API at **`/realtime`** and serves the test client at **`/realtime-test`**,
   so `deploy_app.sh` ships them alongside the contact-center cockpit without
   touching it. In the app the pipeline authenticates as the injected service
   principal (OAuth). Endpoints become:
   - `GET /realtime` (and `/realtime/`) — a JSON **API descriptor** that names the
     service and lists every endpoint below (links are mount-prefix aware). The bare
     `/realtime` 307-redirects to `/realtime/` so the base path lands on the API, not
     the web SPA.
   - `WS /realtime/v1/speech-to-text`, `WS /realtime/v1/speech-llm-toolassist-speech`, `WS /realtime/v1/text-to-speech`, `GET /realtime/v1/capabilities`, `GET /realtime/v1/languages`, `GET /realtime/v1/benchmarks`
   - `POST /realtime/mcp` — the MCP endpoint (voice API exposed as MCP tools); see [MCP server](#mcp-server-remote-over-http)
   - Test UI: `https://<app-host>/realtime-test/` (auto-targets the `/realtime` mount)

   All of the above are behind Databricks Apps auth (SSO in a browser; a Bearer
   token from an identity with access to the app for programmatic callers).

   `deploy_app.sh` attaches the realtime STT/LLM/TTS serving endpoints (from the
   `realtime_voice:` config block) as app resources so the service principal gets
   `CAN_QUERY`. Missing endpoints are skipped with a warning.

### External / cross-workspace access

A Databricks App is protected by its **host workspace's** OAuth/OIDC — there is no
anonymous access and no API key. Two rules follow:

1. **Auth is workspace-scoped.** The caller must present an OAuth token issued by
   the **app's** workspace host. A token minted against a different workspace is
   rejected (an unauthenticated request 302-redirects to the app workspace's OIDC).
2. **The principal needs `CAN_USE` on the app** (App → Permissions). A valid token
   without app permission still gets `403`.

**Option A — Service principal (M2M), recommended for programmatic / cross-workspace
callers.** This is how the multilingual benchmark reaches the app
(`M2MTokenProvider` in `Benchmarks/MultilingualVoice/databricks_auth.py`). The
realtime API calls its STT/LLM/TTS endpoints as the **app's own** service principal,
so the *caller's* SP only needs `CAN_USE` on the app — no serving grants.

1. In the app's account, create a service principal and generate an **OAuth secret**
   (`client_id` + `client_secret`). Secret creation is an **account-admin** action
   (Account Console → *Service principals* → *Secrets*, or
   `databricks account service-principal-secrets create --service-principal-id <id>`).
2. Grant that SP `CAN_USE` on the app:
   ```bash
   databricks apps update-permissions genie-voice-agent -p <profile> --json '{
     "access_control_list": [
       {"service_principal_name": "<client-id>", "permission_level": "CAN_USE"}
     ]}'
   ```
3. The caller mints a token against the **app's workspace host** and sends it as a
   Bearer on every request (HTTP and the WebSocket upgrade):
   ```python
   from databricks.sdk.core import Config
   cfg = Config(host="https://<app-workspace-host>", client_id="<client-id>",
                client_secret="<secret>", auth_type="oauth-m2m")
   token = cfg.authenticate()["Authorization"].removeprefix("Bearer ").strip()
   # ws:  wss://<app-host>/realtime/v1/speech-to-text   header: Authorization: Bearer <token>
   ```
   Raw equivalent: `POST https://<app-workspace-host>/oidc/v1/token` with
   `grant_type=client_credentials&scope=all-apis` and HTTP Basic `client_id:secret`.

**Option B — Existing account user (U2M).** If the caller is already a user in the
app's account (this app grants `CAN_USE` to *account users*), they authenticate to
the app's workspace host directly — `databricks auth login --host https://<app-workspace-host>`
then `databricks auth token` — or simply SSO in the browser. No shared secret.

### MCP server (remote, over HTTP)

The realtime voice API is also exposed as a **Model Context Protocol** endpoint so
MCP clients (Cursor, Claude Desktop, any MCP host) can use it as tools. It is hosted
**in-process** by the app at the exact path:

```
https://<app-host>/realtime/mcp
```

Transport is **streamable HTTP** (the MCP `StreamableHTTPSessionManager`). The server
code lives in `mcp_server/`; `api/app/main.py:_mount_mcp` wires it at `/realtime/mcp`,
and `deploy_app.sh` ships/updates it with every deploy (no extra step). Its tools call
the co-hosted `/realtime` routes over **loopback**, so there is no second auth hop.

**Tools exposed:** `describe_api`, `get_capabilities`, `list_languages`,
`get_benchmarks`, `health`, `synthesize_speech`, `transcribe_audio`, `ask_voice_agent`.

**Auth is the app's normal ingress auth** — attach `Authorization: Bearer <token>`
where `<token>` is a Databricks OAuth token from the app's workspace with `CAN_USE`
on the app (same tokens as [External / cross-workspace access](#external--cross-workspace-access)).
Opening `/realtime/mcp` in a **browser** returns
`{"error":{"code":-32600,"message":"Not Acceptable: Client must accept text/event-stream"}}`
— that is **expected**: a browser `GET` isn't an MCP client. Only clients that send
`Accept: application/json, text/event-stream` can speak the protocol.

**Connect a remote MCP client** (e.g. Cursor `~/.cursor/mcp.json`):

```jsonc
{
  "mcpServers": {
    "genie-voice": {
      "url": "https://<app-host>/realtime/mcp",
      "headers": { "Authorization": "Bearer <databricks-oauth-token>" }
    }
  }
}
```

**Run locally over stdio** (the same package, pointed at the deployed app or a local
`python -m realtime_api.server`):

```bash
pip install -r mcp_server/requirements.txt      # mcp>=1.9,<2 (bundles FastMCP)
GENIE_VOICE_API_URL=https://<app-host>/realtime \
DATABRICKS_HOST=https://<app-workspace-host> \
DATABRICKS_CLIENT_ID=<sp-client-id> DATABRICKS_CLIENT_SECRET=<sp-secret> \
python -m mcp_server.server                      # or DATABRICKS_CONFIG_PROFILE=<profile>
```

**Verify the deployed endpoint** (JSON-RPC over streamable HTTP):

```bash
TOKEN=$(databricks auth token -p <profile> | jq -r .access_token)
MCP=https://<app-host>/realtime/mcp
# initialize -> 200 + an `mcp-session-id` response header
curl -s -D- -o/dev/null -X POST "$MCP" \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  --data '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"probe","version":"0"}}}'
# then tools/list (reuse the returned session id via header `mcp-session-id: <id>`)
```

A liveness/mount probe is available at `GET /__mcp_status` →
`{"mounted":true,"path":"/realtime/mcp","via":"asgi-middleware"}`.

> Note: `mcp` is pinned `>=1.9,<2` — the `mcp` 2.0 release removed the bundled
> `FastMCP` (`mcp.server.fastmcp`) this server builds on.

## Capabilities

- **Auto language detection** — Qwen3-ASR detects the spoken language per turn; the
  LLM and TTS follow it. No language picker required.
- **End-to-end language coverage: 24 languages** — the round-trippable set is the
  **intersection of STT (Qwen3-ASR, 30) and TTS (VoxCPM2, 30)**. A language must be
  supported by both to work end to end. The API computes this dynamically and exposes
  it at `GET /v1/languages`; the UI shows it on page load.
  Current 24: Arabic, Chinese, Danish, Dutch, English, Finnish, French, German,
  Greek, Hindi, Indonesian, Italian, Japanese, Korean, Malay, Polish, Portuguese,
  Russian, Spanish, Swedish, Filipino, Thai, Turkish, Vietnamese.
- **Streaming TTS** — VoxCPM2 chunks are streamed as generated (`predict_stream`),
  cutting time-to-first-audio ~3.5–4.5× vs full-sentence synthesis.
- **LLM with tools + temperature** — the middle stage is a Databricks foundation-model
  chat endpoint (default `qwen3-next-80b`) that supports `temperature` and tool
  calling (a generic `get_current_time` tool is wired as an example).
- **Three WebSocket capabilities** — `speech-to-text` (STT only), `speech-llm-toolassist-speech`
  (full dialog with LLM + tools + TTS), and `text-to-speech` (synthesis only). See
  `GET /v1/capabilities` for paths and per-route language lists.
- **Server-side VAD/endpointing + barge-in** — the API owns turn-taking; the UI never
  reimplements it. Barge-in is opt-in (`realtime_voice.allow_barge_in: true` in config,
  needs headphones/AEC).
- **Per-endpoint latency** — the API reports `stt_ms`, `llm_ms`, and `tts_first_ms`
  per turn; the UI shows both client-observed and server endpoint timings.

## Run the API

```bash
pip install -r realtime_api/requirements.txt
# Endpoints (and all knobs) come from the realtime_voice: config block. Edit
# config/config.yaml or config/config.local.yaml — there are no env overrides.
python -m realtime_api.server            # ws://localhost:8001/v1/speech-llm-toolassist-speech
# PORT=9000 python -m realtime_api.server   # PORT is only the listen port
```

Auth uses the Databricks SDK serving client (no local mlflow needed), authenticating
with the profile from config (`databricks.profile`) or `DATABRICKS_CONFIG_PROFILE`.

HTTP endpoints: `GET /` (JSON API descriptor listing all routes), `GET /healthz`,
`GET /v1/capabilities`, `GET /v1/languages` (end-to-end supported languages), and
`GET /v1/benchmarks` (latest multilingual scores from Delta).

## Run the UI (standalone client)

The UI is a single-file browser client that only speaks the WebSocket protocol, so
it can be hosted anywhere and pointed at any API host.

```bash
python -m http.server 8000 -d realtime_test_ui
# open:
#   http://localhost:8000/index.html?api=localhost:8001
```

API host + base-path resolution (priority order, see `connect()` in `index.html`):

1. `?api=host:port` and `?apiPrefix=/prefix` query params
2. `window.REALTIME_API_HOST` / `window.REALTIME_API_PREFIX`
3. same origin — and when served from the app under `/realtime-test`, the base
   path defaults to `/realtime` (the app mount), so no query params are needed

**UI responsibilities (and nothing more):** microphone capture + downsample to
16 kHz PCM16, streaming PCM frames over the WebSocket, rendering server events,
playing back streamed audio chunks, and presentation (status, language badge,
latency panel). All turn-taking, VAD/endpointing, barge-in, language handling, and
STT→LLM→TTS orchestration live in the **API**.

> Mic tips: the UI disables browser `echoCancellation`/`noiseSuppression`/`autoGainControl`
> (they degrade ASR) and downsamples with anti-aliased box-averaging. Use headphones
> if you enable barge-in, so the assistant doesn't hear its own voice.

## WebSocket protocol

Connect to `WS /v1/speech-llm-toolassist-speech` (or `WS /v1/speech-to-text` / `WS /v1/text-to-speech` for single-capability routes), send `session.start`, then binary mono
`pcm_s16le` frames (8/16/24/48 kHz accepted; 16 kHz recommended). The service
finalizes a turn automatically after configured speech silence or maximum duration;
`audio.end` remains available for push-to-talk clients.

**Turn ownership (`endpointing`).** By default the server owns the end-of-turn
boundary (Silero VAD + smart-turn). A `session.start` may set `endpointing: false`
to take **client-managed** control: the server does no automatic finalization and
ends the turn only on `audio.end` (with `max_turn_seconds` as a safety ceiling).
This is for offline/batch callers that already hold a whole utterance — e.g. the
multilingual benchmark — so a natural mid-utterance pause can't split the clip into
truncated turns. Omitting the flag preserves the live-call behavior exactly.

Server → client events:

- `session.ready` (includes `supported_languages`)
- `speech.started`
- `turn.started`
- `transcript.final` (includes `stt_ms`)
- `response.text` (includes `llm_ms`)
- `response.audio` (streamed chunks; first carries `tts_first_ms`; last is `final`)
- `playback.stop`
- `error`

`barge_in` immediately cancels the current turn and increments the turn ID,
preventing late inference responses from reaching the client.

## Tuning (config only)

All knobs live in the `realtime_voice:` block of `config/config.yaml`
(`config.local.yaml` overrides locally) — there are no env overrides.

| Config key (`realtime_voice.*`) | Default | Meaning |
|---|---|---|
| `allow_barge_in` | `false` | Allow sustained talk-over to interrupt the reply |
| `warmup` | `true` | Prime the STT/LLM/TTS replicas at startup |
| `stt_warmup_passes` | `3` | STT warm-up passes fired at startup |
| `debug_audio` / `debug_audio_dir` | `false` / `/tmp/realtime_audio` | Save each finalized turn's PCM to WAV |
| `stt_candidates` / `tts_candidates` / `llm_endpoint` | — | Serving endpoints |

Other VAD/LLM/TTS defaults (silence window, min speech, temperature, diffusion
steps) are `RealtimeSettings` fields in `realtime_api/config.py`, populated from
the same `realtime_voice:` block.

## Realtime voice model serving (candidates)

Every realtime candidate is packaged as an MLflow **`ResponsesAgent`** (Databricks
Agent Framework, `task = agent/v1/responses`) and deployed as an agent Model Serving
endpoint — the raw `dataframe_records` pyfunc path is intentionally **not** used.
Audio travels through the Responses `custom_inputs`/`custom_outputs` channel.

Registration/deployment code (`scripts/ml_asr/`):

- `realtime_stt_agent.py` — `RealtimeSTTAgent`, the OSS ASR (Qwen3-ASR) wrapper.
- `realtime_tts_agent.py` — `RealtimeTTSAgent`, the OSS TTS (VoxCPM2) wrapper.
- `register_realtime_voice_agent.py` — registers a UC candidate alias per modality
  (`--modality stt|tts`).
- `deploy_realtime_voice_models.py` — promotes a `candidate` alias to its agent
  Model Serving endpoint.
- `submit_realtime_voice_jobs.py` — stages the agents to a UC Volume and submits
  **serverless** register→deploy jobs (run locally against the configured profile).
- `smoke_realtime_voice_agents.py` — exercises deployed endpoints with the Responses
  contract + latency.

The initial models are **candidates**, not active routes. Promote them only after
isolated latency and multilingual-quality testing (see [`ROADMAP.md`](ROADMAP.md)).

---

# ML ASR pipeline & model serving

Config-driven multilingual ASR bake-off. All artifacts live on a UC Volume; dataset
and eval steps run on **serverless** by default. Entry point: **`scripts/ml_asr.sh`**.

| Step | Command | What it does |
|------|---------|--------------|
| 1 | `./scripts/ml_asr.sh datasets` | Build FLEURS holdouts (business + acoustic) on a UC Volume |
| 2 | `./scripts/ml_asr.sh quality` | Semantic dataset quality gates |
| 3 | `./scripts/ml_asr.sh register all` | UC registration (**Databricks models only**) |
| 4 | `./scripts/ml_asr.sh serve deploy-all` | Deploy Model Serving endpoints (SDK from laptop) |
| 5 | `./scripts/ml_asr.sh eval` | Score all eval routes (serverless) |
|   | `./scripts/ml_asr.sh status` | Eval pipeline state |

```bash
./scripts/ml_asr.sh datasets
./scripts/ml_asr.sh quality
./scripts/ml_asr.sh register all
./scripts/ml_asr.sh serve deploy-all
./scripts/ml_asr/04_serve.sh smoke databricks_en_finetuned_whisper_lora
./scripts/ml_asr.sh eval
./scripts/ml_asr.sh status
```

**Config:** `config/ml_asr_eval.yaml` · **Python:** `backend/genie_voice/ml_asr/`

### Model routes

| Route prefix | Kind | Register / serve? |
|--------------|------|-------------------|
| `deepgram_nova3` | Commercial API (Deepgram Nova-3) | No — API key only, eval in step 5 |
| `databricks_*` | UC-registered models on Model Serving | Yes — steps 3–4, then eval in step 5 |

`eval_matrix` in config lists every route scored at eval time. `model_serving` lists
only `databricks_*` models for register/deploy, mapping each to:

- `registered_model_leaf` — UC model name
- `register.type` — `oss` (OSS baseline) or `finetuned_whisper`
- `serve.workload_type` / `workload_size` — endpoint compute

Endpoint names come from `models.*.endpoint` (same names `05_eval.sh` uses).
`03_register.sh` uses `genie_voice.ml_asr.serving` (OSS models register on serverless;
EN finetuned Whisper bridges to `scripts/asr/05_register*` until migrated).

### Benchmark UI

After eval completes, sync the Volume index for the cockpit ASR benchmark page:

```bash
./scripts/ml_asr/05_eval.sh sync-index    # or: ./scripts/ml_asr/sync_benchmark_index.sh
```

Open **http://localhost:5173/#/asr-benchmark** — prefers `ml_asr` FLEURS results when
`.run/ml_asr_eval/index.json` exists; falls back to legacy `voice_model_deep_eval`
holdout. Use the **Eval tier** dropdown for business (entity readiness) vs acoustic
(WER/CER). Optional API: `?source=ml_asr|legacy|auto&tier=business|acoustic`.

### Dev-only local runs

```bash
./scripts/ml_asr/01_datasets.sh local
./scripts/ml_asr/02_quality.sh local
```

### Legacy ASR scripts (`scripts/asr/`)

Lower-level training, registration, and bake-off scripts from the original EN-centric
workflow. The ML ASR pipeline **reuses** some for UC registration; you normally do not
run the full legacy sequence for eval.

| Script | Purpose |
|--------|---------|
| `01_asr_model_training.sh` | Data prep, manifests, GPU cluster lifecycle, baseline runs |
| `02_asr_baseline_runs.sh` | Candidate evaluation (Deepgram, Whisper baselines) |
| `03_asr_model_finetuning.sh` | Whisper LoRA fine-tuning |
| `04_asr_real_audio_holdout.sh` | Realistic held-out evaluation gate |
| `05_register_asr_model_candidate.sh` | Register EN finetuned Whisper LoRA in UC |
| `06_deploy_asr_model_serving_endpoint.sh` | Deploy a single ASR serving endpoint |
| `07_deep_voice_model_eval.sh` | Deepgram vs serving endpoint eval |
| `10_register_multilingual_asr_candidates.sh` | Register multilingual OSS baselines in UC |
| `11_multilingual_asr_finetuning.sh` | Multilingual LoRA fine-tuning |
| `12_multilingual_business_holdout_loop.sh` | Business holdout build + base vs LoRA loop |
| `13_multilingual_asr_promotion_gate.sh` | Read-only promotion decision report |

**When to use legacy vs ML ASR:**

- **Multilingual FLEURS bake-off** → `scripts/ml_asr.sh` (steps 1–5).
- **EN LoRA training from a custom manifest** → `scripts/asr/01` → `03` → `05` → `06`.
- **Promotion gate after multilingual fine-tune** → `scripts/asr/13`.

---

# ASR model-training harness

`backend/genie_voice/asr_eval/` prepares and benchmarks the utterance-level ASR
dataset used for Whisper fine-tuning and model selection. The first benchmark is
Deepgram Nova-3 on a locked training/evaluation manifest; Whisper and Databricks
model-serving baselines use the same manifest and scoring functions for fair
comparison.

### Manifest

JSONL, one utterance-level clip per line. Required: `clip_id`, `audio_path`,
`reference_transcript`. Recommended: `call_id`, `speaker`, `audio_format`,
`sample_rate_hz`, `duration_seconds`, `scenario`, `split`, `dataset_version`,
`expected_entities`. See `docs/asr_model_training_manifest.example.jsonl`.

### Workflow

```bash
scripts/asr/01_asr_model_training.sh        # data prep, manifest, Volume layout, GPU lifecycle
scripts/asr/02_asr_baseline_runs.sh whisper-full   # full-manifest Whisper baseline (no training)
scripts/asr/02_asr_baseline_runs.sh fair-compare   # rescore + Deepgram Nova-3 on same manifest
scripts/asr/03_asr_model_finetuning.sh preflight
scripts/asr/03_asr_model_finetuning.sh dry-run
scripts/asr/03_asr_model_finetuning.sh train-lora  # only after dry-run succeeds
```

`01_asr_model_training.sh` with no arguments runs all safe repeatable steps and stops
when it needs real audio or corrected transcripts. Useful subcommands: `next`,
`volume`, `prepare`, `validate`, `augment`, `deepgram`, `whisper`, `whisper-db`,
`gpu-status`, `gpu-start`, `gpu-stop`, `summarize`, `all`.

External acoustic data: Common Voice (if a Mozilla archive is placed under
`/Volumes/<catalog>/<schema>/<streaming_volume>/asr_model_training/external_raw/common_voice`),
otherwise LibriSpeech `dev-clean` from OpenSLR is used automatically.

### Scoring

`score_transcript()` computes WER, CER, invoice-ID accuracy, amount accuracy, date
accuracy, billing-action phrase accuracy, and confirmation/refusal phrase accuracy.
**Business entity accuracy is the main promotion signal**; generic WER is supporting
evidence only.

---

# Lakebase (Autoscaling) serving layer

Lakebase Autoscaling is a **serverless Postgres** project (scale-to-zero, instant
restore, Unity Catalog governed), used for the low-latency reads the Agent Assist UI
needs:

- **`call_state`** — live enrichment per call, upserted by `genie_voice.serve.LakebaseServing`.
- **operational call tables** — `call_facts` and `live_call_utterances`.
- **Lakebase CDF** — started in the Lakebase UI to publish `lb_<table>_history` into
  Unity Catalog for task-based analytics refresh.

### Provisioning

```bash
python infra/lakebase/setup_lakebase.py
```

Resolves the Autoscaling project and ensures the configured Postgres schema exists.
If the project does not exist, create it in the UI, then re-run setup.

### Connecting (U2M — default)

No password needed. With OAuth U2M the serving layer **mints a short-lived Postgres
token at runtime** via `/api/2.0/postgres/credentials` and connects as
`databricks.run_as`. Set `lakebase.enabled: true` in `config/config.yaml`.

To pin a static connection instead, set these in `.env` (overrides the minted token):

```
LAKEBASE_HOST=...
LAKEBASE_PORT=5432
LAKEBASE_DATABASE=databricks_postgres
LAKEBASE_USER=...
LAKEBASE_PASSWORD=...
```

With `lakebase.enabled: false` the serving layer falls back to an in-process store so
the local end-to-end flow still works offline.

> New Lakebase instances are **Autoscaling** projects by default (2026+). UC analytics
> reads Lakebase CDF history; this repo does not create duplicate UC-to-Lakebase
> managed synced tables.

---

# Serverless orchestration job

`infra/jobs/deploy_pipeline.py` deploys one serverless orchestration job running as
your U2M identity:

- **Reference UC ingest** — reads `raw_batch_data` files into UC reference Delta tables.
- **Lakebase ingest** — reads `raw_streaming_data` files and upserts primary Lakebase
  call tables (runs in parallel with reference UC ingest).
- **Lakebase CDF sync check** — resolves project/branch, verifies `REPLICA IDENTITY FULL`,
  requires `wal2delta.tables` status `STREAMING`/`SNAPSHOTTING`, then waits for
  `lb_<table>_history` tables in UC.
- **Gold insights refresh** — creates UC Delta `gold_call_insights` from Lakebase call
  and utterance history.
- **UC constraints** — adds informational PK/FK metadata so Genie sees relationships.
- **Data quality** — validates PK/FK metadata, integrity, vocabularies, call consistency.
- **Genie reconcile** — recreates the Genie space only after DQ passes.

```bash
python infra/jobs/deploy_pipeline.py                 # deploy + run orchestration
python infra/jobs/deploy_pipeline.py --full-refresh  # accepted for compatibility
python infra/jobs/deploy_pipeline.py --no-run        # deploy only
python infra/jobs/deploy_pipeline.py --paused        # create paused
```

Wheel tasks run on **serverless** compute: a job environment whose only dependency is
the `genie_voice` wheel installed from a stable UC Volume path (`pyspark`/`pandas` are
preinstalled). The task reads config from the `config.yaml` copied into the same
workspace folder (`--config /Workspace/.../config.yaml`; workspace files are
FUSE-mounted). `GENIE_<SECTION>__<KEY>` overrides still apply.

**Prerequisites:** the UC schema + Volume + typed tables exist
(`genie_voice.databricks.bootstrap`, which `local-deploy.sh` runs); `pip` can build the
wheel; your identity can create serverless jobs and write to its workspace home.

`local-deploy.sh` runs this automatically online.

---

# Genie space (created dynamically by name)

No hardcoded space id. The space is recreated by `databricks.genie_space_name` after
the data quality gate passes, with entity matching on categorical columns, example SQL,
instructions, and benchmark questions. Joins are inferred from the post-refresh UC
PK/FK metadata task.

```bash
python -m genie_voice.genie.space     # runs DQ, recreates by name, prints the URL
```

The orchestration job runs this automatically online after constraints and DQ. At
query time `GenieClient` just resolves the space by name.
