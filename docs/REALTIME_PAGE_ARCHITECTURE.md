# Realtime API and Page Architecture

## Status

This document is the architecture contract for the application's conversational
and analytical surfaces. It supersedes page-local routing conventions. Changes
to profiles, tools, handoffs, conversation state, evidence, or visualization
must preserve these boundaries and their contract tests.

## Architectural principles

1. The application has one shared realtime transport, not one universal agent.
2. Every conversational page declares a profile and surface at session ingress.
3. A profile exposes a language-neutral capability allowlist.
4. Policy validates identity, effect, confirmation, and backend ownership before
   exposing exactly one tool lane.
5. One upstream backend owns a conversational thread. Conversation identifiers
   never cross backend types.
6. Tools return typed evidence. Speech and visual presentation are derived from
   the same evidence.
7. Business intent is not encoded as application keywords or artifact classes.
8. Operational and benchmark pages do not participate in realtime conversation
   state.

## Product planes

### Conversation plane

The conversation plane contains Home, Knowledge, Telco, and Card. They share:

- WebSocket session lifecycle
- Qwen ASR and endpointing
- selected-language handling
- capability navigation and policy
- OBO identity
- AgentRuntime event ordering
- half-duplex TTS and barge-in
- evidence envelopes
- actor/session/turn tracing

They do not share domain tools, governed data sources, mutation rules, or
upstream conversation identifiers.

### Operations plane

Setup, Traces, and Guardrails are HTTP control and observability surfaces. They
do not open voice sessions or inherit conversational state.

Permissions remain as currently configured. This architecture does not add an
authorization redesign.

### Evidence plane

Voice Benchmarks and ASR Benchmark are read-only evaluation surfaces. Their
datasets and jobs remain separate, while navigation and presentation may group
them under one Benchmarks area.

## Page authority

### Home (`profile=concierge`, `surface=home`)

Home owns destination routing only:

- application help and navigation
- Telco destination selection
- Card destination selection
- exact Databricks question handoff to Knowledge

Home does not answer Databricks, billing, or card questions. It does not decide
whether analytical details are sufficient. A substantive Databricks request is
transferred to Knowledge even when it is incomplete.

The handoff is a typed session continuation, not a browser storage side channel:

```json
{
  "type": "session.start",
  "profile": "knowledge",
  "surface": "knowledge",
  "continuation": {
    "kind": "knowledge_question",
    "text": "the exact caller request",
    "language": "en-US"
  }
}
```

The Knowledge session processes that continuation as its first text turn after
`session.ready`. It must not play a greeting first or ask the caller to repeat.

### Knowledge (`profile=knowledge`, `surface=knowledge`)

Knowledge uses the managed Genie One MCP through Unity Gateway:

```text
/ai-gateway/mcp-services/system.ai.genie_one_mcp
```

It is the default authority for:

- general Databricks product questions
- governed workspace questions
- usage and spend
- forecasting and comparison
- classification
- multi-step analysis
- clarification of incomplete analytical requests

Knowledge has one execution lane:

```text
request -> genie_ask -> poll -> answer/evidence
        -> same conversation_id for the next caller turn
```

The local application does not own analytical clarification and does not expose
`knowledge_search` or generic Agent Mode tools on this profile.

A scoped Genie Agent is not selected from analytical wording. It requires an
explicitly configured workflow. The existing Card deep-investigation workflow is
such a workflow; Knowledge is not.

### Telco (`profile=billing`, `surface=telco`)

Telco uses:

- Lakebase for low-latency customer, invoice, payment, and account facts
- the configured Telco Genie Space for governed billing analysis
- a two-phase prepare/confirm/apply contract for account mutations

Both voice and Telco analytical UI use the Telco Genie Space. Telco never routes
general analysis to Genie One.

An account-changing action is valid only when:

1. a read-only preparation resolves the exact account item and outcome,
2. the proposal is spoken,
3. the caller explicitly confirms the same proposal, and
4. effect policy authorizes the mutation.

### Card (`profile=card`, `surface=card`)

Card uses:

- Card Lakebase for low-latency account facts
- the configured Card Genie Space for governed card questions
- the explicitly configured Card Genie Agent for deep investigation

Card Agent Mode is intentional and remains isolated from Knowledge.

## Backend ownership

Session state records one backend owner:

```text
none
genie_one
genie_space
genie_agent
```

Each owner has its own conversation handle. A handle is never interpreted as a
different backend's handle.

- Knowledge follow-ups remain in the Genie One conversation.
- Telco questions remain in the Telco Genie Space.
- Card questions remain in the Card Genie Space.
- Card investigations remain in the selected Card Genie Agent conversation.

## Turn execution contract

```text
session ingress
  -> profile capability catalog
  -> semantic capability proposal
  -> policy validation
  -> one execution adapter
  -> typed evidence
  -> spoken summary
  -> visual presentation
  -> durable trace
```

Execution adapters are explicit:

- `pack_facts`: Lakebase and deterministic pack tools
- `space_conversation`: configured Telco or Card Genie Space
- `agent_mode`: explicitly configured deep investigation
- `genie_one_mcp`: Knowledge only
- `confirm_mutate`: confirmed account-changing action

Latency, progress support, cancellation, and timeout are adapter properties, not
page keyword rules.

## Source and entity integrity

Governed analytical execution follows these upstream rules:

1. Use evidence for the entity and scope named by the caller.
2. Do not silently substitute another customer, account, workspace, or dataset.
3. When the requested source or entity cannot be resolved, ask for the missing
   source or state that it is unavailable.
4. Proxy analysis requires explicit caller approval.
5. The application does not detect violations using response keywords or
   punctuation.

These rules belong in governed Genie instructions/ontology or an explicit
pre-execution policy contract. They are not post-response text filters.

## Analytical response and adaptive presentation

All analytical use cases use one generic contract:

```text
answer
datasets[]
views[]
provenance
status/error
```

There are no business-specific artifacts such as `ForecastArtifact`.

A validated view contains:

```json
{
  "dataset_id": "result identifier",
  "kind": "kpi | table | bar | line | area | scatter",
  "dimension": "optional field",
  "measures": ["field"],
  "series": ["optional field"],
  "units": {"field": "unit"},
  "title": "question-aligned title"
}
```

View selection uses:

- the original caller question
- typed field metadata
- result shape
- units and temporal semantics
- Genie-provided visualization metadata when available

The presentation planner may choose a KPI, table, or chart. It must validate
that every referenced field exists and that charted measures are numeric. The
frontend renders the validated view and never independently guesses business
meaning from column names.

Examples:

- forecast -> temporal series with scenarios when the result supports them
- classification -> category plus score/probability
- comparison -> grouped measures when units are compatible
- single value -> KPI
- unsupported or ambiguous shape -> table

## Multilingual boundary

- ASR preserves the selected language and named entities.
- Routing uses language-neutral capability identifiers.
- Governed tools receive the caller request without business-intent rewriting.
- English-only upstream reports are localized after evidence is returned.
- Numbers, identifiers, product names, citations, and SQL provenance remain
  unchanged during localization.
- The spoken summary and written answer derive from the same evidence.

## Frontend state

The realtime event reducer is the source of truth for:

- active turn
- transcript
- progress
- evidence
- committed speech
- final answer
- errors and refusals

Pages render selectors over this shared state. They do not maintain parallel
interpretations of the same events.

Page-specific UI remains page-specific:

- Home destination cards
- Telco resolution journey
- Card analytical canvas
- Knowledge structured answer

## Operations and evidence pages

- Setup owns readiness and existing scale-to-zero controls.
- Traces owns turn and actor observability.
- Guardrails owns policy coverage and incident rollups.
- Voice Benchmarks owns end-to-end realtime evaluation.
- ASR Benchmark owns focused transcription/provider evaluation.

These pages use HTTP APIs and do not reuse conversational routing or state.

## Trace contract

Every conversation turn records:

- `profile`
- `surface`
- `traffic_class`
- `session_id`, `call_id`, and `turn_id`
- actor name, username, and email when available
- selected capability and policy reason
- backend owner and conversation identifier
- tool names and outcomes
- evidence source
- STT, routing, tool, answer, and TTS timing

Progress text, internal reasoning, raw chain-of-thought, and private tool payloads
are not exposed to the browser.

## Migration rules

Implementation proceeds in this order:

1. Align Knowledge profile tools and prompts to Genie One only.
2. Replace browser storage handoff with typed session continuation.
3. Keep Genie One conversation ownership for every Knowledge follow-up.
4. Remove Knowledge-only clarification and generic Agent Mode remnants.
5. Implement question-adaptive validated presentation views.
6. Replace domain-specific navigation fixtures with neutral contract fixtures.
7. Run profile, transport, evidence, multilingual, and end-to-end tests.
8. Deploy once after the complete contract passes.

Telco, Card, permissions, Setup controls, and benchmark data sources do not
change as part of the Knowledge migration.
