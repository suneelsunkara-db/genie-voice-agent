# Guardrails architecture

`config/guardrails.yaml` is the policy control plane. It defines policy identity,
version, bundles, and assignments independently of model names.

## Invariants

1. Policies attach to stable trust boundaries, not transient model versions.
2. Every STT result crosses the shared post-STT boundary before model or tool use.
3. Every synthesized response crosses the shared pre-TTS boundary.
4. Tool authorization, confirmation, and state validation remain in the action
   boundary; an LLM or content policy cannot authorize a mutation.
5. Every decision uses the normalized ledger contract: `policy_id`,
   `policy_version`, `enforcer`, `phase`, `resource`, `outcome`, and redacted
   `reason`.
6. The Guardrails UI reads the manifest and deployment state from the API. It
   does not define a second policy catalog.

## Control, enforcement, and evidence planes

- **Control:** `config/guardrails.yaml` and the typed loader in
  `genie_voice.guardrails`.
- **Enforcement:** Unity AI Gateway service policies for foundation-model text;
  shared application boundaries for speech and tools.
- **Evidence:** `GuardLedger` → `TurnTrace.guard_roster` → Lakebase/Delta →
  `/traces/guardrails`.

## Databricks service-policy beta

Model-service routing, rate limits, inference tables, and grants are reconciled
by `deploy_app.sh`. Service-policy attachment writes remain UI-only, but the
public model-service read API exposes the deployed handlers, phases, and ranks.
Deployment fails closed on missing or drifted policy attachments, and the
Guardrails page reports that observed state directly.

Do not use private UI endpoints or browser automation in deployment. When
Databricks publishes policy attachment APIs, implement that capability behind
the reconciler without changing the policy manifest, enforcement boundaries, or
UI contract.
