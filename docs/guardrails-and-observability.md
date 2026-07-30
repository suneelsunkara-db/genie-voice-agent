# Voice Guardrails & Observability — Design

> Status: **Design / not yet implemented.** Captured for later use. This document
> covers (1) the pluggable, async guardrail layer for the realtime voice API, (2) how
> it complements Qwen3-ASR's built-in behaviors, (3) how guardrails surface in the UI,
> and (4) an observability/traces UI plan (including concrete improvements to the
> current Trace Explorer).

Related code map references are inline as `path:line` so this survives refactors only
loosely — re-check line numbers before implementing.

---

## 0. Non-negotiable principles

**No fallbacks. No hardcodings. No hotfixes.** These apply to every part of this design.

- **No hardcodings** — model endpoints, table/schema names, regex/blocklist patterns,
  language maps, thresholds, cadences, and severities live in **config**, never as
  literals in code. Guards are loaded by import path from config (§9); their patterns and
  parameters are passed in, not baked in.
- **No silent fallbacks / no degrade-and-continue** — if required config or a dependency
  (e.g. the observer LLM endpoint) is missing or misconfigured, **fail fast at startup**
  with a clear error. Do not swap in a default, do not run in a reduced mode, do not
  swallow the error to keep going. Config is **validated on load**; missing required keys
  are a startup failure, not a runtime surprise.
- **No hotfixes / no special-casing** — no `if language == "vi"` style one-offs, no
  per-customer patches, no "temporary" branches. Fix root causes; express variation
  through config and well-typed guard plugins.
- **Errors are explicit and observable, never hidden** — a guard error is surfaced as a
  `guardrail.error` span + event (§6), never quietly ignored. Any per-guard error policy
  is an **explicit, configured, logged** decision (see the note below), not an implicit
  catch-all.
- **No implicit defaults for behavior-changing flags** — e.g. `trust_stt_lid` /
  `trust_stt_no_speech` must be set explicitly in config and validated; the engine never
  *assumes* what Qwen covered — it reads the STT output and acts on declared config.

**Tension to resolve explicitly (Open Question #6):** "async while the voice speaks"
wants the observer to never stall a turn, which usually implies tolerating a failed
observer call. Under the no-silent-fallback rule this is allowed **only** as an
*explicitly configured, per-guard, logged and flagged* error policy — the failure is
emitted as `guardrail.error` and shown in the UI. It is **not** a hidden `try/except:
pass`. If the product prefers strict correctness over availability, guards are
`fail_closed` and a failure blocks the turn loudly. Pick one deliberately; don't default.

### Existing code that violates these principles (remediate, don't extend)

These predate this design and should be fixed as part of the work, not copied:

- **Trace JSONL fallback** — `backend/genie_voice/serve/lakebase.py:567` silently writes
  to `/tmp/genie_voice_traces/...` when Lakebase is disabled. That is a silent fallback;
  make trace storage explicit and fail-fast if the configured sink is unavailable.
- **Silent trace drop** — `TraceSink` drops traces when its 512-item queue is full
  (`realtime_api/tracing.py:204`) with only a warning. Must be an explicit, surfaced
  metric (§8.7), not a silent loss.
- **Hardcoded language-name map** — `_LANGUAGE_NAMES` in
  `scripts/ml_asr/realtime_stt_agent.py:44` is a literal map; language handling should be
  config-driven and shared, not duplicated literals.
- **Optional MLflow mirror via env flag** (`GENIE_TRACE_MLFLOW`) — env-gated optional
  behavior is a soft fallback; promote to validated config with an explicit on/off.

---

## 1. Goals

- **Pluggable / configurable** guardrails — declared in config, discovered by import
  path (same pattern as STT/LLM/TTS providers), individually toggled and ordered.
- **Async while the voice speaks** — expensive (LLM) policy checks run off the turn's
  critical path and steer the *next* turn; only cheap deterministic checks run inline.
- **Complement, don't duplicate, Qwen3-ASR** — skip what the STT model already owns
  (language ID, no-speech suppression, ITN); add what it deliberately refuses to do
  (PII, toxicity, downstream prompt-injection defense, policy/safety, output checks).
- **Observable** — every guardrail decision is a first-class `GUARD` span + a
  `guardrail.flagged` wire event, visible in the Trace Explorer and live UI.

---

## 2. What Qwen3-ASR already covers (complementarity)

Qwen3-ASR is intentionally an **ASR-only** model (see the
[Qwen3-ASR technical report](https://arxiv.org/html/2601.21337v1)). It is trained to
*not* follow natural-language instructions embedded in audio, "to mitigate instruction
injection and instruction-following failures." That protects **Qwen's own decoding** —
it does **not** protect our downstream LLM, because the transcript is still returned
verbatim and flows into `respond_with_tools`.

| Behavior | Qwen owns it? | Runtime signal | Our stance |
|---|---|---|---|
| Language ID (52 langs) | ✅ | `detected_language` in `custom_outputs` | Trust + skip our own LID |
| No-speech / noise hallucination suppression | ✅ | **empty transcript** | Trust + skip |
| Inverse text normalization (clean digits) | ✅ | normalized transcript | Leverage for PII regex |
| STT-layer instruction-injection resistance | ✅ (self only) | none (no flag) | **Do NOT rely on** for the LLM |
| Context biasing / hotwords (`prompt`) | ✅ (knob, unused) | n/a | Optional lever |
| PII / PCI redaction | ❌ | — | **We add** |
| Profanity / toxicity | ❌ | — | **We add** |
| Downstream prompt-injection defense | ❌ | — | **We add** |
| Content / safety / fraud semantics | ❌ (by design) | — | **We add (observer)** |
| Output-side checks (reply / TTS text) | ❌ | — | **We add** |

**Runtime detection rule:** the only guardrail state Qwen surfaces is `detected_language`
(LID ran) and an empty transcript (no-speech guard fired). There is no "injection
resisted" flag — it is a model property, not an event. The engine therefore *trusts and
skips* LID + no-speech, and *never assumes* injection safety for the downstream LLM.

STT invocation today: `DatabricksServing.transcribe` (`realtime_api/services.py:264`)
returns `(transcript, detected_language)`.

---

## 3. Guardrail catalog

Grouped by pipeline stage. **Type** = deterministic (regex/blocklist, ~ms) vs LLM.

| Stage | Guardrail | Catches | Type | Enforcement |
|---|---|---|---|---|
| Input (inline) | PII / PCI redaction | card no., CVV, SSN / national ID | deterministic | redact, fail-closed |
| Input (inline) | Profanity / abuse | slurs, harassment | deterministic | flag / mask |
| Input (inline) | Prompt-injection / jailbreak | "ignore your instructions", role override | deterministic (+opt LLM) | flag / block |
| Input (inline) | Out-of-scope | requests outside billing domain | lightweight | flag / steer |
| Observer (async) | Fraud / claim manipulation | dispute gaming, social engineering | LLM | inject directive → next turn |
| Observer (async) | Safety / self-harm / threats | caller in danger, threats | LLM | escalate directive |
| Observer (async) | Policy nuance ("no financial advice") | off-policy commitments | LLM | inject directive |
| Output (pre-TTS) | PII in reply | agent about to speak sensitive data | deterministic | block / redact, fail-closed |
| Output (pre-TTS) | Unsupported promises | "I've refunded you" w/o tool call | deterministic phrase-check | block / replace |
| Output (pre-TTS) | Wrong-language reply | reply not in caller's language | reuse `i18n` check | block / regenerate |

Already-handled by Qwen (skipped, zero cost): language ID, no-speech suppression, ITN.

---

## 4. Architecture

Two execution tiers so latency-sensitive work stays cheap and semantic work stays off
the hot path.

```
Client WS audio
  → process_turn (assist)
    → transcript.final ──▶ [INPUT_TRANSCRIPT guards]  (inline, deterministic)
                             │  redact / block / flag
                             ├─▶ publish → [OBSERVER_ASYNC guards] (detached, LLM)
                             │        └─ verdict → session.guardrail_directives (next turn)
    → session.history.append
    → respond_with_tools  (merges guardrail_directives as a 2nd system message)
    → response.text ──▶ [PRE_TTS guards] (inline, deterministic; optional barge-in cutoff)
    → stream_tts
  ── guardrail.flagged events → UI (live) + GUARD spans → TurnTrace
```

### Stages & hook points

| Stage | Hook (file:line today) | Execution | Can mutate? |
|---|---|---|---|
| `INPUT_TRANSCRIPT` | after `transcript.final` yield — `speech_llm_toolassist_speech.py:182`, before `session.history.append` (`:184`) | inline, fast | transcript + history |
| `OBSERVER_ASYNC` | detached task off the transcript | background (while TTS plays) | next turn only (directives) |
| `PRE_TTS` | between `response.text` (`:285`) and `stream_tts` (`:294`) | inline, fast (+opt cutoff) | reply text / abort |

Directive injection point: system-message assembly in
`DatabricksServing.respond_with_tools` (`realtime_api/services.py:315`). Because context
is rebuilt every turn, a directive naturally applies from the next turn and can be
expired — no LiveKit-style read-only-copy dance required.

### Verdict model (proposed)

```python
class Stage(Enum):
    INPUT_TRANSCRIPT = "input_transcript"
    OBSERVER_ASYNC = "observer_async"
    PRE_TTS = "pre_tts"

@dataclass
class Verdict:
    action: Literal["allow", "redact", "flag", "inject", "block", "escalate"]
    redacted_text: str | None = None   # redact
    directive: str | None = None       # inject (merged into next system msg)
    reason: str = ""
    dedup_key: str | None = None       # inject at most once per session
    severity: Literal["info", "warn", "critical"] = "info"

class Guardrail(Protocol):
    id: str
    stage: Stage
    blocking: bool        # inline-blocking vs advisory
    fail_closed: bool     # error → block (PII) vs allow (LLM nuance)
    async def evaluate(self, ctx: "GuardContext") -> Verdict | None: ...
```

### Engine

Per-session `GuardrailEngine` built from config:
- Runs `INPUT_TRANSCRIPT` + `PRE_TTS` guards inline, honoring `blocking` / `fail_closed`.
- Schedules `OBSERVER_ASYNC` as a detached task with a single-flight guard
  (`_evaluating` / `_pending_eval`, per the observer pattern) so LLM calls never stack.
- Writes deduped directives to `session.guardrail_directives` (new field on
  `VoiceSession`, `realtime_api/session.py`).
- Emits `guardrail.flagged` to the client and records a `GUARD` span per decision.
- Skips LID / no-speech guards when STT `custom_outputs` show Qwen handled them.

### Pluggability (config-declared, provider-style)

Guards are discovered by import path, exactly like the existing providers
(`genie_voice.providers.stt.deepgram:DeepgramSTT`). See §9 for the config schema.

---

## 5. Latency budget

| Stage | On critical path? | Added latency |
|---|---|---|
| Input deterministic guards | Yes (before LLM) | ~1–5 ms (negligible vs STT + LLM) |
| Async observer (LLM) | **No** — runs while current turn's TTS plays | **0 ms perceived**; steers next turn |
| Pre-TTS deterministic guard | Yes (before first audio) | ~1–5 ms added to TTFB |
| Optional LLM output check | Only when a cheap pre-filter flags | rare; stream optimistically + barge-in cutoff |

Safeguards: an **explicitly configured, per-guard error policy** (see §0 — a failed LLM
guard emits `guardrail.error` and, if configured `fail_closed: false`, is skipped with a
visible flag; it is never a silent `try/except: pass`), config toggles to tune
coverage/latency, and deterministic-first design. The one honest trade-off: strict
pre-TTS blocking adds a few ms to first-audio; never put a heavy LLM check inline there —
use the async path or barge-in cutoff.

---

## 6. Wire protocol additions

New outbound event (`realtime_api/contracts.py`, string-typed like the rest):

```jsonc
{
  "type": "guardrail.flagged",
  "turn_id": 12,
  "stage": "input_transcript",
  "guard_id": "pci_pii",
  "action": "redact",
  "severity": "critical",
  "reason": "card number detected",
  "redacted": true        // never include the raw sensitive value
}
```

GUARD span additions (`realtime_api/tracing.py` — kind `"GUARD"` already exists and is
used by `language.gate` and `history`): one span per guard decision with
`input = {stage, guard_id}`, `output = {action, severity, reason, dedup_key}`,
`attributes = {blocking, fail_closed, latency_ms}`, and `status` = `ok` / `blocked` /
`error`. This is the persisted record the Trace Explorer renders.

---

## 7. How guardrails reflect in the UI

Three surfaces, cheapest first:

### 7a. Trace Explorer (`frontend/src/components/TracesPage.tsx`)
- GUARD spans already have styling (`traces.css` `.tv-kind.GUARD`) and a `KindTag`. New
  guard spans appear in the span **waterfall** with their action/severity in `SpanDetail`.
- Add a **guardrail summary strip** on the trace detail: chips per fired guard
  (id + action + severity), colored by severity.
- Add a **filter**: `all / ok / issues / guardrail` and a per-guard facet. Today filters
  are billing-centric (`apply_billing_action_called`); generalize to guardrail signals.
- Promote `guard_flags` to a list column on `TraceSummary` so the session/turn list can
  badge flagged turns without fetching span bodies.

### 7b. Live call UI (`CockpitPage` / Sentient)
- Consume the `guardrail.flagged` WS event during a live call and show a transient badge
  / toast on the active turn (severity-colored). This is the "operator sees it happen"
  view — analogous to the existing `language.mismatch` banner.

### 7c. New Observability / Guardrails page (see §8)
- Aggregate view: guardrail hit-rates by type, severity distribution, per-language, and
  recent flagged turns with deep-links into the Trace Explorer.

---

## 8. Observability UI plan (and Trace Explorer improvements)

### Current state (grounded)
- Backend: `TurnTrace` → `submit_trace` → `TraceSink` daemon → Lakebase Postgres
  `voice_traces` (`backend/genie_voice/serve/lakebase.py:608`), optional UC Delta mirror
  + optional MLflow (`GENIE_TRACE_MLFLOW`).
- API: `api/app/routers/traces.py` → `GET /traces`, `GET /traces/{id}`,
  `GET /traces/sessions` (the last is **unused** by the UI today).
- Frontend: `TracesPage` (hash `#/traces`) — session list + span waterfall +
  `SpanDetail`. `FlowTracker.tsx` exists but is **orphaned** (and expects `jobs.lakeflow`
  while the backend returns `jobs.orchestration`).
- Operational events: `log_event` (`realtime_api/observability.py:39`) → JSON log lines
  only; **not** visualized anywhere.

### Concrete improvements (prioritized)
1. **Live polling** on `#/traces` (Cockpit already has `POLL_INTERVAL_MS`) so new turns
   appear without a manual Refresh. Add auto-refresh toggle + "new turns" indicator.
2. **Nested span waterfall** — the span model is a **flat list** with no `parent_id`.
   Add an optional `parent_id` to `Span` (`tracing.py:46`) so LLM iterations, tool calls,
   filler-TTS-overlapping-LLM, and guard spans nest correctly. Backfill parent by
   producer (e.g. `tool.*` under `llm.iteration.n`).
3. **Use the `/traces/sessions` rollup** instead of client-side re-aggregation; add
   guardrail counts to the rollup (`api/app/routers/traces.py:24`).
4. **Guardrail facets & badges** (see §7a) — generalize the billing-centric filters.
5. **Deep-link** Cockpit turn → Trace Explorer (`call_id` / `session_id` filters already
   exist in `GET /traces`; just wire the link).
6. **Operational log stream panel** — surface `log_event` output (ws.open, turn.error,
   turn.discarded, barge_in, …) as a live tail beside spans, so runtime issues that never
   produce a persisted trace are visible.
7. **Queue-drop visibility** — `TraceSink` silently drops when its 512 queue is full
   (`tracing.py:204`); expose a dropped-count metric + a UI warning.
8. **Retire or fix `FlowTracker`** — either delete the dead component or repair the
   `jobs.lakeflow` vs `jobs.orchestration` mismatch and wire it into `#/status`.
9. **Sampling / retention controls** — add `realtime_voice.tracing` config (sampling
   rate, enable flag, retention) and reflect current settings in the UI. Today tracing is
   always-on with no yaml knob.

### New page slot
Follow the `VoiceBenchmarksPage` pattern (full-screen dark surface, own CSS, hero +
highlight cards + sections). Wire in `frontend/src/App.tsx` as a new hash
(`#/observability` or `#/guardrails`) with a nav pill next to **Traces**/**Benchmarks**,
or as tabs within `#/traces` (Spans | Guardrails | Logs).

---

## 9. Config schema (proposed)

Top-level under `realtime_voice:` in `config/config.yaml`:

```yaml
realtime_voice:
  guardrails:
    enabled: true
    trust_stt_lid: true          # skip our LID; Qwen owns it
    trust_stt_no_speech: true    # skip no-speech guard; Qwen owns it
    observer:
      endpoint: <llm-endpoint>   # reuse main LLM or a cheaper/stronger one
      cadence_turns: 1           # evaluate every N user turns
    guards:
      - id: pci_pii
        impl: genie_voice.guardrails.pii:PciPiiRedactor
        stage: input_transcript
        blocking: true
        fail_closed: true
      - id: jailbreak
        impl: genie_voice.guardrails.injection:JailbreakDetector
        stage: input_transcript
        blocking: false
      - id: policy_observer
        impl: genie_voice.guardrails.policy:LlmPolicyObserver
        stage: observer_async
        fail_closed: false
      - id: output_pii
        impl: genie_voice.guardrails.pii:OutputPiiGuard
        stage: pre_tts
        blocking: true
        fail_closed: true

  tracing:               # NEW — make observability configurable (§8.9)
    enabled: true
    sampling: 1.0
    retention_days: 30
```

---

## 10. Phased plan

- **Phase 0 — scaffolding + principle cleanup (no behavior change):** config block +
  flag (off by default), `session.guardrail_directives`, `GuardrailEngine` + registry
  (import-path plugins), `guardrail.flagged` + `guardrail.error` contract events, GUARD
  span helper. No-op guards. **In scope (per §0 — remediate existing violations):**
  1. Remove the trace JSONL `/tmp` fallback (`backend/genie_voice/serve/lakebase.py:567`)
     — make the trace sink explicit and fail-fast if unavailable.
  2. Replace the silent `TraceSink` queue drop (`realtime_api/tracing.py:204`) with a
     surfaced dropped-count metric (§8.7).
  3. Move the hardcoded `_LANGUAGE_NAMES` map (`scripts/ml_asr/realtime_stt_agent.py:44`)
     to shared, config-driven language handling.
  4. Promote the env-gated MLflow mirror (`GENIE_TRACE_MLFLOW`) to validated
     `realtime_voice.tracing` config with an explicit on/off.
- **Phase 1 — deterministic input guards:** PII/PCI, profanity, jailbreak. Redact + flag.
  Closes the realtime-STT PII gap (Qwen does not redact).
- **Phase 2 — async LLM observer (assist only):** rolling-transcript policy eval →
  directive injection into next turn; single-flight + dedup.
- **Phase 3 — pre-TTS output guards:** PII-in-output, unsupported promises, reuse i18n
  language check; barge-in cutoff for heavier checks.
- **Phase 4 — observability UI:** GUARD span rendering + guardrail facets, live polling,
  nested waterfall, log-stream panel, new Guardrails/Observability page, deep-links.

---

## 11. Open questions

1. Observer model — reuse the main LLM endpoint, or budget a separate cheaper/stronger one?
2. Default `PRE_TTS` enforcement — inline-blocking (slightly higher TTFB) or
   optimistic-with-barge-in-cutoff?
3. First guards to ship — PII/PCI redaction (compliance) or the LLM policy observer (safety)?
4. Scope — assist route only, or also redact on the STT-only route?
5. Observability page — standalone hash, or tabs inside `#/traces`?
6. Guard/observer **error policy** (see §0) — strict `fail_closed` (block loudly) vs
   explicitly-configured skip-with-flag. Must be chosen deliberately, not defaulted.

---

## 12. Reference index

**Backend**
- `realtime_api/pipelines/speech_llm_toolassist_speech.py` — assist turn loop; GUARD spans
  (`language.gate`, `history`); hook points at `:182`, `:184`, `:285`, `:294`.
- `realtime_api/services.py` — `transcribe` (`:264`), `respond_with_tools` +
  system-prompt assembly (`:303`, `:315`), LLM/TOOL spans.
- `realtime_api/tracing.py` — `Span` (`:46`), `TurnTrace` (`:104`), `TraceSink` (`:180`),
  `submit_trace` (`:293`).
- `realtime_api/observability.py` — `log_event` (`:39`).
- `realtime_api/contracts.py` — wire events, `SessionStart`.
- `realtime_api/session.py` — `VoiceSession` (add `guardrail_directives`).
- `backend/genie_voice/serve/lakebase.py` — `insert_voice_trace` (`:608`), DDL (`:522`).
- `api/app/routers/traces.py` — `GET /traces`, `/traces/{id}`, `/traces/sessions`.
- `api/app/routers/pipeline_status.py` — `GET /status` (medallion stages).
- `backend/genie_voice/i18n.py` — `generated_text_language_check` (reuse for output guard).

**Frontend**
- `frontend/src/components/TracesPage.tsx` + `styles/traces.css` — Trace Explorer.
- `frontend/src/components/VoiceBenchmarksPage.tsx` — good page pattern to mirror.
- `frontend/src/components/CockpitPage.tsx` / `sentient/*` — live call UI.
- `frontend/src/components/FlowTracker.tsx` — orphaned; fix or delete.
- `frontend/src/api/client.ts` — `voiceTraces`, `voiceTrace`, `voiceTraceSessions`.
- `frontend/src/App.tsx` — hash routing + nav pills.

**External**
- Qwen3-ASR technical report — https://arxiv.org/html/2601.21337v1
- LiveKit observer pattern — https://livekit.com/blog/observer-pattern-voice-agent-guardrails
