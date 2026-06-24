# Genie Voice Contact Center Architecture

The architecture separates enterprise reference data from live call data.

- Reference/customer/billing data is batch-ingested from `raw_batch_data` into
  governed Unity Catalog Delta tables.
- Live call data is streamed into Lakebase first for low-latency agent assist.
- Live agent-assist **resolution and billing** are written to Lakebase
  (`resolution_events`, `billing_adjustments`) and mirrored to UC for Genie.
- Lakebase CDF publishes call history into Unity Catalog.
- Job tasks build final UC `call_facts` and `gold_call_insights`.
- Genie reads curated Unity Catalog business tables plus `billing_adjustments`.

## Architecture

```mermaid
flowchart LR
    subgraph RAW["UC Volumes"]
        BATCH["raw_batch_data<br/>customers, agents, invoices, payments"]
        STREAM["raw_streaming_data<br/>Deepgram events + call_facts records"]
    end

    subgraph UCREF["Unity Catalog Reference Delta"]
        CUSTOMERS["customers"]
        AGENTS["agents"]
        INVOICES["invoices"]
        PAYMENTS["payments"]
        BILLING["billing_adjustments<br/>audit mirror"]
    end

    subgraph LB["Lakebase Operational Call Store"]
        STATE["call_state<br/>live nudge + resolution"]
        LBFacts["call_facts"]
        LBTurns["live_call_utterances"]
        REV["resolution_events"]
        ADJ["billing_adjustments"]
    end

    subgraph HIST["Unity Catalog Lakebase CDF History"]
        HFacts["lb_call_facts_history"]
        HTurns["lb_live_call_utterances_history"]
    end

    subgraph CURATED["Unity Catalog Curated Analytics"]
        FACTS["call_facts<br/>latest current state"]
        GOLD["gold_call_insights<br/>FM-derived"]
    end

    subgraph SERVE["Consumption"]
        UI["Agent Assist UI<br/>Lakebase + API"]
        API["FastAPI<br/>POST /assist"]
        GENIE["Genie Space<br/>curated UC tables"]
    end

    BATCH --> CUSTOMERS
    BATCH --> AGENTS
    BATCH --> INVOICES
    BATCH --> PAYMENTS

    STREAM --> LBFacts
    STREAM --> LBTurns
    STREAM --> STATE

    API --> STATE
    API --> REV
    API --> ADJ
    API --> BILLING
    API --> INVOICES
    STATE --> UI
    LBFacts --> UI
    LBTurns --> UI
    REV --> UI
    CUSTOMERS --> UI
    INVOICES --> UI
    ADJ --> UI

    LBFacts --> HFacts
    LBTurns --> HTurns
    HFacts --> FACTS
    HTurns --> GOLD

    CUSTOMERS --> GOLD
    AGENTS --> GOLD
    INVOICES --> GOLD
    PAYMENTS --> GOLD
    FACTS --> GOLD

    CUSTOMERS --> GENIE
    AGENTS --> GENIE
    INVOICES --> GENIE
    PAYMENTS --> GENIE
    BILLING --> GENIE
    FACTS --> GENIE
    GOLD --> GENIE
```

## Live agent assist flow

Each customer utterance on `POST /calls/{call_id}/assist` runs this pipeline.
There are no keyword fallbacks or canned agent templates.

```mermaid
sequenceDiagram
    participant UI as Agent Assist UI
    participant API as FastAPI
    participant FM as Foundation Model
    participant LB as Lakebase
    participant WH as SQL Warehouse
    participant G as Genie

    UI->>API: POST /assist (customer utterance)
    API->>FM: enrich customer utterance + resolution signals
    FM-->>API: intent, sentiment, customer_signal, waiver/plan flags
    API->>API: evaluate resolution (open / in_progress / pending_close)
    API->>G: compose agent reply (Lakebase metrics authoritative)
    G-->>API: prose reply (validated against account facts)
    alt pending close and reply available
        API->>LB: persist billing_adjustments
        API->>WH: MERGE billing_adjustments + UPDATE invoices
        API->>API: finalize closed status
    end
    API->>LB: upsert call_state + resolution_events (status transitions only)
    API-->>UI: live nudge, resolution, agent_reply, billing, validation
```

**Ordering guarantees**

- Billing writes and `closed` status commit **after** Genie agent reply on
  customer turns, so KPIs and invoice overlays do not change while the UI still
  shows "Genie is preparing the agent response…".
- Close is blocked if billing UC/Lakebase writes fail or if Genie cannot produce
  a validated reply (`agent_reply: null`, `close_block_reason` set).
- `GET /calls/{call_id}/alignment` cross-checks resolution, active billing
  adjustments (call-scoped), and account summary.

## Job Flow

```mermaid
flowchart LR
    REF["batch_reference_ingest<br/>raw_batch_data -> UC Delta"]
    CALL["call_lakebase_ingest<br/>raw_streaming_data -> Lakebase"]
    CDF["call_cdf_sync_check<br/>verify fresh call history tables"]
    GOLD["gold_insights_refresh<br/>call + utterance history -> gold"]
    CONS["uc_constraints<br/>add PK/FK metadata"]
    DQ["data_quality_check<br/>PK/FK + call consistency"]
    GENIE["recreate_genie_space"]

    CALL --> CDF
    CDF --> GOLD
    REF --> GOLD
    GOLD --> CONS
    CONS --> DQ
    DQ --> GENIE
```

## Genie Tables

Genie reads:

- `customers`
- `agents`
- `invoices`
- `payments`
- `billing_adjustments` (live assist waiver / payment-plan writes)
- `gold_call_insights`

Genie does not read raw `lb_*_history`, `call_state`, `resolution_events`, or
raw transcript events.

## Data Quality Gate

Before Genie is recreated, `data_quality_check` validates:

- primary keys are non-null and unique
- foreign keys are not orphaned
- every call has call facts and utterances
- every call has a gold insight row
- required gold insight fields are populated
- mentioned invoices belong to the same customer as the call

## Demo reset

`POST /calls/{call_id}/reset-demo-session` reverts active billing adjustments
(UC + Lakebase), deletes `resolution_events` and live utterances for the call,
and clears resolution state in `call_state` so the spotlight scenario can be
replayed from `open`.
