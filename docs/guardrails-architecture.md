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
7. Greetings, wait prompts, progress narration, language prompts, and navigation
   confirmations are committed localized application copy, never runtime inference.
8. Factual speech is cite-or-silence: only tool cells or an attributed governed
   answer can cross the response boundary. Unsupported model prose becomes a
   localized refusal.

## Control, enforcement, and evidence planes

- **Control:** `config/guardrails.yaml` and the typed loader in
  `genie_voice.guardrails`.
- **Enforcement:** Unity AI Gateway service policies for foundation-model text;
  shared application boundaries for speech and tools.
- **Evidence:** in-turn decisions use `GuardLedger` → `TurnTrace.guard_roster`;
  decisions without a turn use the append-only `voice_guard_events` sink. Both
  persist to Lakebase/Delta and merge in `/traces/guardrails`.

## Purpose-specific policy contracts

- Interactive Qwen traffic and GPT text transformations enforce unsafe-content
  and jailbreak policies.
- Contact identifiers (email and phone) are redacted, not denied. Credential and
  government identifiers are denied. The post-STT boundary applies the same risk
  split before browser, history, model, tool, or persistence admission.
- Hallucination is a log-mode observer. It is not the factuality gate because an
  output-only judge cannot establish the application's tool evidence. The
  application evidence boundary is authoritative for factual speech.
- Fixed product speech is loaded from `realtime_api/phrases/runtime.json`. The
  offline translation script validates complete language/key coverage and exact
  placeholder preservation.

## Databricks service-policy beta

Model-service routing, rate limits, inference tables, and grants are reconciled
by `deploy_app.sh`. Service-policy attachment writes remain UI-only, but the
public model-service read API exposes the deployed handlers and options.
Deployment compares the complete attachment set, including phase, rank, mode,
action, categories, evaluator, and turn window. Missing, drifted, or unexpected
attachments fail closed. A behavioral conformance matrix then verifies normal
allow, jailbreak/unsafe/credential deny, contact redaction, and log-mode
hallucination pass-through before the app is deployed.

Do not use private UI endpoints or browser automation in deployment. When
Databricks publishes policy attachment APIs, implement that capability behind
the reconciler without changing the policy manifest, enforcement boundaries, or
UI contract.
