# Voice Guardrails & Observability — Design

> Status: **Design / not yet implemented.** The *architecture decisions are now
> settled* (see §4 — "two seams, one ledger") and **Phase 0 is approved to build**.
> Everything below Phase 0 remains design. This document covers (1) the two
> guardrail seams for the realtime voice API and the shared ledger that unifies
> their observability, (2) how they complement Qwen3-ASR's actual behaviors,
> (3) how guardrails surface in the UI, and (4) an observability/traces UI plan
> (including concrete improvements to the current Trace Explorer).

Related code map references are inline as `path:line`. Every reference in this
revision was re-verified against the working tree; they still drift, so re-check
before implementing.

---

## 0. Non-negotiable principles

**No fallbacks. No hardcodings. No hotfixes.** These apply to every part of this design.

- **No hardcodings** — model endpoints, table/schema names, regex/blocklist patterns,
  language maps, cue tables, thresholds, cadences, and severities live in **config**,
  never as literals in code. Guards are loaded by import path from config (§9); their
  patterns and parameters are passed in, not baked in.
- **No silent fallbacks / no degrade-and-continue** — if required config or a dependency
  (e.g. the observer LLM endpoint) is missing or misconfigured, **fail fast at startup**
  with a clear error. Do not swap in a default, do not run in a reduced mode, do not
  swallow the error to keep going. Config is **validated on load**; missing required keys
  are a startup failure, not a runtime surprise.
- **No hotfixes / no special-casing** — no `if language == "vi"` style one-offs, no
  per-customer patches, no "temporary" branches. Fix root causes; express variation
  through config and well-typed guard plugins.
- **Errors are explicit and observable, never hidden** — a guard error is recorded in the
  turn's roster with outcome `error` and surfaced as a `guardrail.error` event (§6), never
  quietly ignored. Any per-guard error policy is an **explicit, configured, logged**
  decision, not an implicit catch-all.
- **No implicit defaults for behavior-changing flags** — e.g. `trust_stt_lid` /
  `trust_stt_no_speech` must be set explicitly in config and validated; the engine never
  *assumes* what Qwen covered — it reads the STT output and acts on declared config.

**Tension to resolve explicitly (Open Question #4):** "async while the voice speaks"
wants the observer to never stall a turn, which usually implies tolerating a failed
observer call. Under the no-silent-fallback rule this is allowed **only** as an
*explicitly configured, per-guard, logged and flagged* error policy — the failure is
recorded as roster outcome `error` and shown in the UI. It is **not** a hidden `try/except:
pass`. If the product prefers strict correctness over availability, guards are
`fail_closed` and a failure blocks the turn loudly. Pick one deliberately; don't default.

### Existing code that violates these principles (remediate, don't extend)

These predate this design and should be fixed as part of the work, not copied:

- **Trace JSONL fallback** — `backend/genie_voice/serve/lakebase.py:687` builds a
  `tempfile.gettempdir()/genie_voice_traces` path, and `insert_voice_trace` (`:718`)
  silently appends there when Lakebase is disabled, swallowing `OSError` (`:726`). That is
  a silent fallback; make trace storage explicit and fail-fast if the configured sink is
  unavailable.
- **Silent trace drop** — `TraceSink` drops traces when its 512-item queue is full
  (`realtime_api/tracing.py:264`) with only a warning. Must be an explicit, surfaced
  metric (§8.7), not a silent loss.
- **Hardcoded language-name map** — `_LANGUAGE_NAMES` in
  `scripts/ml_asr/realtime_stt_agent.py:44` is a literal map; language handling should be
  config-driven and shared, not duplicated literals.
- **Optional MLflow mirror via env flag** (`GENIE_TRACE_MLFLOW`,
  `realtime_api/tracing.py:295`) — env-gated optional behavior is a soft fallback; promote
  to validated config with an explicit on/off.
- **Concierge router hardcodings** (new — this list predates the concierge profile) —
  `realtime_api/concierge_tools.py` holds `_MAX_SELECTION_WORDS = 8` (`:100`), the
  English-only `_INDUSTRY_CUES` allowlist (`:85`–`:89`), and `_CONFIRM_LABELS` (`:92`).
  These are exactly the thresholds and allowlists §0 says must live in config, and they
  gate a *decision* (which industry the caller gets routed to), so a silent change in
  behavior is a routing bug.
  **The English-only coupling is deliberate today** — `HomePage.tsx` pins STT to English
  (`:309`) and renders the language bar disabled (`:355`) after a Hindi mis-detection
  incident on short replies like "Telco". But that dependency currently lives only in a
  code comment (`concierge_tools.py:80`–`:84`). It must be *expressed and validated in
  config* (e.g. the concierge router declares `languages: [en]` and load-time validation
  fails if the profile's session language can fall outside it), so enabling the picker
  can never silently degrade routing.

---

## 1. Goals

- **Two seams, one ledger** — an execution seam (guards that permit/modify/block) and a
  decision seam (routing/gating that *chooses an action*) keep their own logic, but both
  publish every outcome into a single per-turn ledger (§4).
- **Pluggable / configurable** guardrails — declared in config, discovered by import
  path (same pattern as STT/LLM/TTS providers), scoped **per profile**, individually
  toggled and ordered.
- **Async while the voice speaks** — expensive (LLM) policy checks run off the turn's
  critical path and steer the *next* turn; only cheap deterministic checks run inline.
- **Complement, don't duplicate, Qwen3-ASR** — delegate only what the STT model
  demonstrably owns (language ID *when it actually runs*, no-speech suppression); add
  what it deliberately refuses to do (PII, toxicity, downstream prompt-injection defense,
  policy/safety, output checks).
- **Prove the negative** — the roster records checks that *passed* and *were delegated*,
  not just ones that fired, so the UI can state "N checks ran, M delegated to Qwen, none
  fired" instead of showing an empty list.

---

## 2. What Qwen3-ASR already covers (complementarity)

Qwen3-ASR is intentionally an **ASR-only** model. Its published claims are robustness
under "complex acoustic environments and challenging text patterns" and **52-language
LID**. It is trained to *not* follow natural-language instructions embedded in audio, to
mitigate instruction injection and instruction-following failures. That protects **Qwen's
own decoding** — it does **not** protect our downstream LLM, because the transcript is
returned verbatim and flows into `respond_with_tools` (`realtime_api/services.py:281`,
called from the pipeline at `:306`).

| Behavior | Qwen owns it? | Runtime signal | Our stance |
|---|---|---|---|
| Language ID (52 langs) | ✅ **but only when unpinned** | `detected_language` in `custom_outputs` | Delegate **conditionally** — see below |
| No-speech / noise hallucination suppression | ✅ | **empty transcript** | Delegate |
| Inverse text normalization (clean digits) | ⚠️ **observed, not contracted** | none | Do **not** assume normalized digits |
| Context biasing / hotwords | ❌ **not available to us** | n/a | Not a lever today |
| STT-layer instruction-injection resistance | ✅ (self only) | none (no flag) | **Do NOT rely on** for the LLM |
| PII / PCI redaction | ❌ | — | **We add** |
| Profanity / toxicity | ❌ | — | **We add** |
| Downstream prompt-injection defense | ❌ | — | **We add** |
| Content / safety / fraud semantics | ❌ (by design) | — | **We add (observer)** |
| Output-side checks (reply / TTS text) | ❌ | — | **We add** |

**Runtime detection rule:** the only guardrail state Qwen surfaces is `detected_language`
and an empty transcript. The STT wrapper returns `custom_outputs` of exactly
`transcript`, `language`, `detected_language`, `inference_device`
(`scripts/ml_asr/realtime_stt_agent.py:141`–`:146`). There is no "injection resisted"
flag — it is a model property, not an event.

### 2.1 Language ID is conditional, not a checkbox

LID delegation is **live only when the session leaves STT on auto**:

- `realtime_api/pipelines/_shared.py:79`–`:80` computes
  `stt_language = None if (not pref or pref == "auto") else pref` from
  `session.config.language`, and passes it into `transcribe` (`:82`–`:87`).
- `DatabricksServing.transcribe` (`realtime_api/services.py:242`) forwards it as
  `custom_inputs.language` (`:252`).
- The wrapper turns it into a **forced** decode: `_forced_language(requested_language)`
  (`realtime_stt_agent.py:131`, definition at `:150`–`:153`) is passed as
  `model.transcribe(..., language=forced)` (`:133`). When a language is pinned, Qwen's LID
  never runs, and `custom_outputs.language` is literally `requested_language or detected`
  (`:143`), i.e. an echo of our own pin.
- Separately, `language_mismatch` (`_shared.py:53`) returns `None` unless
  `session.config.expected_language` is set (`:64`–`:66`).

What that means per surface, as wired today:

| Surface | `session.start` sends | Qwen LID runs? | Mismatch gate armed? |
|---|---|---|---|
| Home concierge (`HomePage.tsx:309`) | `language: "en-US"` (pinned) + `expected_language` | **No** — forced to English | Armed, but can't fire (detection is forced) |
| Billing / telco (`CallList.tsx:1200`) | `language: "auto"` + `expected_language: <picker>` | **Yes** | **Yes** |
| Card (`CardIssuerPage.tsx:538`) | `language: "auto"` + `expected_language: <picker>` | **Yes** | **Yes** |
| Knowledge Agent (`KnowledgeAgentPage.tsx`) | `language: "auto"` + `expected_language: <picker>` | **Yes** | **Yes** |

`startRealtimeVoice` only pins STT when the caller passes `sttLanguage`
(`frontend/src/lib/realtimeVoice.ts:234`); the picker value is sent as
`expected_language` (`:241`), which drives the *reply* language via `resolve_language`
(`_shared.py:35`–`:50`) and arms the gate — it does **not** pin STT. Only the home
concierge passes `sttLanguage`.

Consequences for this design:

1. The roster must distinguish **`delegated`** (Qwen's LID ran and we consumed
   `detected_language`) from **`not_evaluated` — reason `session pinned the language`**
   (concierge) from **`we own it`** (any future case where we run our own LID).
2. The UI must **never** show a green "language ✓ Qwen" badge on a pinned turn. On the
   concierge, the honest row is "language ID — not evaluated (English pinned by the
   session)".
3. `trust_stt_lid` in config is not a global boolean truth; it is permission to delegate
   *when the session left STT on auto*. The engine reads the actual STT request/response,
   never the flag alone.

### 2.2 Two Qwen claims, softened

- **Inverse text normalization** — the `Qwen/Qwen3-ASR-1.7B` model card does not document
  ITN. We have *observed* clean digits, but that is behavior, not a contract. **PII
  matching must not assume normalized digits** (see §2b(a)).
- **Context biasing / hotwords** — not documented on the model card, and **not reachable
  from our wrapper**: `custom_inputs` are parsed at `realtime_stt_agent.py:112`–`:116`
  (`audio_b64`, `language`, `sample_rate_hz`) plus a `max_new_tokens` generation ceiling
  (`:125`). There is no `prompt` / hotword channel. Treat it as unavailable, not as an
  "optional lever".

---

## 2b. What makes voice guardrails different

Three properties drive the design and explain why a text-chat guardrail library does not
transfer directly.

**(a) Spoken PII is not written PII.** Callers say "four one one one, one one one one,
one one one one, one one one one" — or "triple four two". A regex like `\b4\d{15}\b`
matches nothing. Combined with §2.2 (ITN is not contracted), PII guards need
**spoken-numeral normalization** as a first-class, configured preprocessing step, and
their patterns must be tested against spoken forms, not just digit strings.

**(b) You can't unsay audio.** `stream_tts` (`_shared.py:99`) yields chunks as the
endpoint produces them (~80 ms of audio each). Blocking only *works* before the first
chunk yields; after that the best available action is a barge-in-style cutoff, which the
caller hears as the agent being interrupted mid-word. Therefore pre-TTS guards must be
**cheap and deterministic** — never a heavy LLM call inline.

**(c) A blocked turn must still speak.** Dead air reads as a broken call. The existing
language gate is the reference pattern: it blocks the turn, *speaks* a localized switch
prompt, and sets a cooldown (`speech_llm_toolassist_speech.py:186`–`:208`). Any "loud"
guard needs the same shape — hence the spoken-remediation extension to `Verdict` in §4.

---

## 3. Guardrail catalog

Split three ways by who owns the check today. **Type** = deterministic (regex/allowlist,
~ms) vs LLM. **Surface** = whether the Guardrails UI shows it (§7).

**Totals: 10 already running · 2 delegated to Qwen · 11 to add.**

### Group A — already shipped (Phase 0 only makes them report)

No new logic, no behavior change. These already run on every relevant turn; today most of
them leave no record when they *decline* to act.

| Guard id | Where | Behavior | Surface |
|---|---|---|---|
| `language_gate` | `_shared.py:53`, invoked `speech_llm_toolassist_speech.py:185`, span `:188` | Off-language turn → block + speak switch prompt (`:198`) + 1.5 s cooldown (`:207`) | guardrail |
| `selection_allowlist` | `concierge_tools.py:85`–`:89` (`_INDUSTRY_CUES`) | Only allowlisted cues can route | guardrail |
| `selection_length` | `concierge_tools.py:100` (`_MAX_SELECTION_WORDS = 8`), applied `:116` | Utterances over 8 words defer to the LLM | guardrail |
| `selection_ambiguity` | `concierge_tools.py:118`–`:119` | Cross-domain tie → no deterministic route | guardrail |
| `tool_markup_strip` | `_strip_tool_markup` `services.py:724`, used `:362` and `:404` | Strips `<tool_call>…` markup out of spoken text | guardrail |
| `tool_arg_enum` | `concierge_tools.py:68` | Rejects tool args outside the declared enum | guardrail |
| `empty_transcript` | `speech_llm_toolassist_speech.py:177` | Blank STT result → drop turn | **internal** |
| `noise_timeout` | `ws/handler.py:153` (`session.is_noise_timeout`, `session.py:160`), config `realtime_voice.endpointing.noise_discard_seconds` (`config/config.yaml:290`) | Ambient noise → discard buffer without transcribing | **internal** |
| `stale_turn` | `speech_llm_toolassist_speech.py:174` (and `:265`, `:344`, `:369`) | Superseded turn → abandon | **internal** |
| `retrigger_cooldown` | `session.set_cooldown` (`session.py:139`), honored in `should_finalize` (`:129`) | Suppresses echo-triggered follow-up turns | **internal** |

**Also in this bucket but not yet on the realtime path:** `_BAD_AGENT_REPLY_MARKERS`
(`api/app/routers/agent_assist.py:190`, used by `_looks_like_bad_agent_reply` at `:211`)
catches the agent reading raw SQL / schema identifiers aloud. It lives only on the legacy
agent-assist route, which the UI no longer calls. **Port it** to the realtime path as the
`output_schema_leak` pre-TTS guard (Group C) rather than writing a new one.

### Group B — delegated to Qwen3-ASR

| Guard id | Signal | Notes |
|---|---|---|
| `language_id` | `detected_language` in `custom_outputs` | **Conditional** — see §2.1. Roster records `delegated` only when STT ran unpinned. |
| `no_speech_suppression` | empty transcript + our VAD (`session.py`, `ws/handler.py:153`) | Joint: Qwen returns "" for noise, our endpointer discards ambient buffers before STT. |

**Not contracted, so not delegated:** inverse text normalization; instruction-injection
resistance (self-only — it does not cover our downstream LLM).

### Group C — to add

**Input stage** (deterministic, inline, ~1–5 ms, before the `transcript.final` yield):

| Guard id | Catches | Enforcement | Profiles |
|---|---|---|---|
| `pci_pii_input` | card number, CVV, SSN / national ID — including spoken-as-words digits | redact, **fail-closed** | billing, card |
| `prompt_injection` | "ignore your instructions", role override, tool-name coaxing | flag / block | all |
| `off_scope` | requests outside the profile's domain | flag / steer | all |
| `profanity_abuse` | slurs, harassment | flag / mask | installed, **default OFF in config** |

**Observer stage** (async LLM, 0 ms perceived, steers the *next* turn):

| Guard id | Catches | Enforcement | Profiles |
|---|---|---|---|
| `fraud_social_engineering` | dispute gaming, social engineering | inject directive | billing, card |
| `safety_escalation` | caller in danger, threats, self-harm | escalate directive | all |
| `policy_nuance` | off-policy commitments ("no financial advice") | inject directive | billing, card |

**Pre-TTS stage** (deterministic, before the first audio chunk):

| Guard id | Catches | Enforcement | Notes |
|---|---|---|---|
| `pii_in_reply` | agent about to speak sensitive data | block / redact, **fail-closed** | |
| `unsupported_promise` | "I've refunded you" with no tool call behind it | block / replace | cross-check against the turn's `tool_names` |
| `reply_language` | reply not in the caller's language | block / regenerate | reuse `generated_text_language_check` (`backend/genie_voice/i18n.py:366`) |
| `output_schema_leak` | agent reading SQL / schema identifiers aloud | block / replace | **port** `_BAD_AGENT_REPLY_MARKERS` (`agent_assist.py:190`) |

---

## 4. Architecture — two seams, one ledger

### 4.1 The fork, and how it is resolved

The tempting simplification is to make *everything* a guard plugin: fold the language gate
and the concierge intent router into a single `GuardrailEngine`. **We are not doing that.**

A guard **permits, modifies, or blocks** something already in flight. The intent router
**decides an action**: it picks a tool, runs it through `profile.tool_runner`, emits
`tool.called`, speaks a generated confirmation, and short-circuits the LLM
(`speech_llm_toolassist_speech.py:240`–`:283`). Folding routing into the engine forces
`Verdict` to grow an "…and run this tool, and speak this, and skip the LLM" arm, at which
point `Verdict` is no longer a verdict.

The evidence that the current `Verdict` model is *already* insufficient is in shipped
code: the **language gate** blocks the turn **and** speaks a localized prompt **and** sets
a cooldown (`:186`–`:208`). None of `allow / redact / flag / inject / block / escalate`
can express that. So even the "obvious" guard doesn't fit the guard shape today.

**Conclusion: unify *observability*, not *execution*.**

| Seam | What it is | Where it lives | Registered via |
|---|---|---|---|
| **Execution** | `GuardrailEngine` — checks that permit / modify / block (PII, injection, output checks, async policy observer) | new `genie_voice.guardrails.*` | `VoiceProfile` (§4.4) |
| **Decision** | `language_gate` and the concierge intent router — logic that *chooses an action* | stays where it is (`_shared.py:53`, `concierge_tools.py:109`) | unchanged |
| **Ledger** | one per-turn roster record both seams write to | `TurnTrace` (`realtime_api/tracing.py:111`) | — |

The UI reads the **ledger**. It never reads span kinds (§4.5).

### 4.2 The roster (the ledger)

Every turn records **every check**, not just the ones that fired. One entry per check:

```jsonc
{
  "guard_id": "language_id",
  "seam": "execution",              // execution | decision
  "stage": "stt",                   // stt | input_transcript | routing | observer_async | pre_tts
  "surface": "guardrail",           // guardrail | internal   ← the UI filter (§7)
  "owner": "qwen",                  // us | qwen
  "outcome": "delegated",           // see table below
  "latency_ms": 0.0,
  "reason": "detected_language=en-US (session language auto)"   // redacted, never raw PII
}
```

| Outcome | Meaning |
|---|---|
| `passed` | Ran, found nothing. **The most common and most valuable outcome.** |
| `fired` | Ran, took action (blocked / redacted / flagged / routed / injected). |
| `delegated` | We deliberately did not run it; an upstream model owns it and we consumed its signal. |
| `not_evaluated` | Preconditions absent (e.g. LID with a pinned session language). `reason` explains why. |
| `disabled` | Present in the catalog, switched off for this profile in config. |
| `error` | Raised. `reason` carries the error; the per-guard error policy (§0) decides the turn's fate. |

**Storage:** one compact JSONB column `guard_roster` on `voice_traces`
(`backend/genie_voice/serve/lakebase.py:570` DDL, written in `insert_voice_trace` at
`:718`, mirrored to UC Delta at `:767`), plus `guard_roster` in the trace JSON
(`tracing.py:196`–`:233`). **Only `fired` entries additionally get their own span** — a
`passed` roster entry is one small dict, not a span, so a full roster costs a fraction of
a KB and does not bloat the waterfall.

**Why a roster and not just incidents.** A UI built only on fired events looks empty and
proves nothing. The claim worth making to an operator is "23 checks ran on this turn, 2
were delegated to Qwen3-ASR, none fired." That is a compliance statement; "no events" is
not.

**The concrete blindness this fixes.** The pipeline emits the `intent.router` span **only
inside `if resolved:`** (`speech_llm_toolassist_speech.py:242`–`:244`). A *decline* — too
many words (`concierge_tools.py:116`), no cue match, or a cross-domain tie (`:118`) —
records **nothing at all**. That is exactly why diagnosing "Telco isn't working" required
manual trace archaeology: the trace showed an LLM turn and no explanation of why the
deterministic route didn't take. With the roster, a decline is a first-class
`selection_length: fired (11 words > 8)` row.

### 4.3 Execution flow

```
Client WS audio
  → process_turn (assist)
    → transcribe  ──▶ roster: language_id (delegated | not_evaluated)
    │                 roster: no_speech_suppression (passed | fired)
    → [INPUT_TRANSCRIPT guards]  ◀── BEFORE the transcript.final yield
    │     redact / block / flag                (see §4.6 — placement correction)
    │     └─▶ publish → [OBSERVER_ASYNC guards] (detached, LLM)
    │              └─ verdict → session.guardrail_directives (next turn)
    → transcript.final yield (redacted text)   :215
    → session.history.append (redacted text)   :223
    → language gate / intent router ──▶ roster (decision seam, incl. DECLINES)
    → respond_with_tools  (merges guardrail_directives as a 2nd system message)
    → response.text
    → stream_tts(primary=True) ──▶ [PRE_TTS guards] before the first chunk
  ── guard_roster → TurnTrace → Lakebase; guardrail.flagged events → live UI
```

### 4.4 Registration seam

Guards are registered **per profile** via `VoiceProfile` (`realtime_api/profiles.py:38`),
the same seam that already carries `tools_spec` (`:43`), `tool_runner` (`:44`),
`make_context` (`:46`), `after_turn` (`:49`) and `resolve_intents` (`:55`):

```python
@dataclass(frozen=True)
class VoiceProfile:
    ...
    # NEW: guards for this profile, built from config (§9). None = no engine.
    guardrails: "GuardrailEngine | None" = None
```

A new industry is then **a config block**, not a pipeline edit — the same property that
makes `tool_registry` keying on `(profile, name)` (`realtime_api/tool_registry.py:77`,
`:83`) work today.

### 4.5 The UI must not filter on span kind

`kind == "GUARD"` is already worn by three unrelated things:

| Span | Where | Actually a guard? |
|---|---|---|
| `language.gate` | `speech_llm_toolassist_speech.py:188` | Yes |
| `intent.router` | `speech_llm_toolassist_speech.py:243` | It's a *decision*, and only recorded on success |
| `history` | `speech_llm_toolassist_speech.py:299` | **No** — a context snapshot of the messages the LLM saw |

Filtering `kind == "GUARD"` in the UI would list every turn's history snapshot as a
guardrail event. `Span` has no guard identity field at all (`tracing.py:54`–`:63`; the
kind comment is at `:56`). **Requirement:** guardrail identity lives in the roster
(`guard_id` + `surface`), and the UI reads the roster. Span kinds stay a debugging detail.

### 4.6 Placement correction — input guards run *before* the `transcript.final` yield

The previous revision of this document said the input hook sits "after the
`transcript.final` yield, before `session.history.append`". **That is too late.** In
`realtime_api/pipelines/speech_llm_toolassist_speech.py` the order is:

| Line | What happens |
|---|---|
| `:161` | `transcribe(...)` returns the raw transcript |
| `:165` | `trace.input_transcript = transcript` — **raw text enters the trace** |
| `:215`–`:221` | `transcript.final` yielded **to the client** |
| `:223` | `session.history.append({"role": "user", ...})` |
| `:240`–`:241` | intent pre-router runs |

Redaction must run **between `:161` and `:165`**. Two independent reasons:

1. **The client already has it.** Yielding at `:215` puts the raw transcript on the wire
   and into the browser's live caption before any guard could touch it.
2. **The trace becomes a PII store in three places.** `trace.input_transcript` /
   `trace.output_text` (`tracing.py:131`–`:132`) are serialized into the trace doc
   (`:210`–`:211`) and persisted to Lakebase (`lakebase.py:718`), mirrored to UC Delta
   (`:767`), and optionally to MLflow (`tracing.py:295`–`:296`, mirroring at `:313`).
   Redacting late means the card number the caller read aloud is durably stored in three
   systems.

### 4.7 Pre-TTS placement — inside `stream_tts`, gated on `primary`

Put output guards inside `stream_tts` (`_shared.py:99`), gated on the existing `primary`
boolean parameter (`:109`), which was added for TTFT tracing. `primary=True` already means
"the actual reply rather than a filler":

| Call site | `primary` | Guarded? | Right answer? |
|---|---|---|---|
| Answer TTS (`speech_llm_toolassist_speech.py:407`) | `True` (default) | ✅ | yes |
| Router confirmation (`:273`) | `True` (default) | ✅ | yes — it's a real spoken reply |
| Filler (`:332`–`:334`) | `False` (explicit) | ❌ | yes — canned latency cover |
| Language-switch prompt (`:198`) | `True` (default) | ⚠️ | see note |

Note: the switch prompt currently passes `primary` implicitly as `True`. It is a
*remediation* utterance produced by a guard, so guarding it again is at best redundant and
at worst a loop. Phase 2 should pass `primary=False` there, or (cleaner) add an explicit
`guarded: bool` parameter derived from `primary` so the two concerns don't stay welded
together.

Placing the hook here means **every pipeline that speaks gets output guarding for free**,
with no per-pipeline edits.

### 4.8 Verdict model (proposed)

```python
class Stage(Enum):
    INPUT_TRANSCRIPT = "input_transcript"
    OBSERVER_ASYNC = "observer_async"
    PRE_TTS = "pre_tts"

@dataclass
class SpokenRemediation:
    """A 'loud' guard's spoken response — the language-gate pattern, generalized.

    ``intent`` is a natural-language instruction for the phrase model (never a
    literal line), so it renders in every supported language — mirroring
    ResolvedIntent.confirm_intent (profiles.py:29).
    """
    intent: str
    cooldown_s: float = 0.0

@dataclass
class Verdict:
    action: Literal["allow", "redact", "flag", "inject", "block", "escalate"]
    redacted_text: str | None = None       # redact
    directive: str | None = None           # inject (merged into next system msg)
    speak: SpokenRemediation | None = None # NEW — block without dead air (§2b(c))
    reason: str = ""                       # redacted; goes into the roster
    dedup_key: str | None = None           # inject at most once per session
    severity: Literal["info", "warn", "critical"] = "info"

class Guardrail(Protocol):
    id: str
    stage: Stage
    surface: Literal["guardrail", "internal"]
    blocking: bool        # inline-blocking vs advisory
    error_policy: Literal["fail_closed", "skip_with_flag"]   # explicit, never defaulted
    async def evaluate(self, ctx: "GuardContext") -> Verdict | None: ...
```

`speak` + `cooldown_s` is the minimum needed to express what `language_gate` already does.
Without it, "block" means dead air, which §2b(c) rules out.

### 4.9 Engine

Per-profile `GuardrailEngine` built from config:
- Runs `INPUT_TRANSCRIPT` + `PRE_TTS` guards inline, honoring `blocking` / `error_policy`.
- Schedules `OBSERVER_ASYNC` as a detached task with a single-flight guard
  (`_evaluating` / `_pending_eval`) so LLM calls never stack.
- Writes deduped directives to `session.guardrail_directives` (new field on
  `VoiceSession`, `realtime_api/session.py:33`).
- Appends a roster entry for **every** registered guard, including ones it skipped —
  `disabled`, `not_evaluated`, `delegated` are outcomes, not omissions.
- Emits `guardrail.flagged` to the client and a span **only** for `fired` entries.
- Reads the actual STT request/response to decide `delegated` vs `not_evaluated` for LID
  (§2.1); it never infers coverage from a config flag alone.

Directive injection point: system-message assembly in `respond_with_tools`
(`services.py:301`). Because context is rebuilt every turn, a directive naturally applies
from the next turn and can be expired — no LiveKit-style read-only-copy dance required.

---

## 5. Latency budget

| Stage | On critical path? | Added latency |
|---|---|---|
| Roster bookkeeping | Yes | dict appends; well under 1 ms per turn |
| Input deterministic guards | Yes (before the yield + LLM) | ~1–5 ms (negligible vs STT + LLM) |
| Async observer (LLM) | **No** — runs while the current turn's TTS plays | **0 ms perceived**; steers the next turn |
| Pre-TTS deterministic guard | Yes (before first audio) | ~1–5 ms added to TTFB |
| Optional LLM output check | Only when a cheap pre-filter flags | rare; stream optimistically + barge-in cutoff |

Safeguards: an **explicitly configured, per-guard error policy** (§0 — a failed LLM guard
records `error` in the roster and, if configured `skip_with_flag`, is skipped with a
visible flag; it is never a silent `try/except: pass`), config toggles to tune
coverage/latency, and deterministic-first design. The one honest trade-off: strict pre-TTS
blocking adds a few ms to first-audio; per §2b(b), never put a heavy LLM check inline
there — use the async path or a barge-in cutoff.

---

## 6. Wire protocol & persistence additions

New outbound event (`realtime_api/contracts.py`, string-typed like the rest), emitted only
for `fired` entries with `surface: "guardrail"`:

```jsonc
{
  "type": "guardrail.flagged",
  "turn_id": 12,
  "stage": "input_transcript",
  "guard_id": "pci_pii_input",
  "action": "redact",
  "severity": "critical",
  "reason": "card number detected",
  "redacted": true        // never include the raw sensitive value
}
```

**Trace additions** (`realtime_api/tracing.py`):

1. `TurnTrace.guard_roster: list[dict]` (alongside `input_transcript` at `:131`),
   serialized in `to_dict` (`:196`–`:233`).
2. A promoted `guard_roster` JSONB column on `voice_traces` (DDL `lakebase.py:570`; write
   mapping `:730`–`:749`) so listing/filtering doesn't need the full span body. Note the
   existing migration constraint: `_trace_columns` (`:607`) only adds columns when the
   role owns the table, and currently hardcodes `DOUBLE PRECISION` (`:637`) — adding a
   JSONB column means generalizing that helper, not copying it.
3. A span **per fired guard only**, with `input = {stage, guard_id}`,
   `output = {action, severity, reason, dedup_key}`,
   `attributes = {blocking, error_policy, latency_ms}`, and `status` = `ok` / `blocked` /
   `error`.

---

## 7. How guardrails reflect in the UI

### 7.0 Scope (hard constraint)

The Guardrails UI shows **only**:

- rows **delegated to Qwen3-ASR** (`language_id`, `no_speech_suppression`), and
- the **Input**, **Observer**, and **Pre-TTS** stage guardrails.

It must **not** surface turn-integrity mechanics: `empty_transcript`, `noise_timeout`,
`stale_turn`, `retrigger_cooldown`. Those remain genuinely useful when reading a span
waterfall or debugging a dropped turn, so they stay in the trace and in the roster — they
are simply not *guardrail rows*.

**Implement this declaratively, not as a hardcoded exclusion list.** Each roster entry
carries `surface: "guardrail" | "internal"` (§4.2), set from the guard catalog in config
(§9). The UI filter is then `entry.surface === "guardrail"`, and a new internal mechanic
never has to be remembered and added to a denylist in the frontend.

### 7a. Trace Explorer (`frontend/src/components/TracesPage.tsx`)
- Fired-guard spans appear in the span **waterfall** with their action/severity in
  `SpanDetail`; `GUARD` styling already exists (`styles/traces.css:100`, `KindTag` at
  `TracesPage.tsx:135`). This is the debug view, so it keeps showing *everything*,
  internal mechanics included.
- Add a **guardrail summary strip** on the trace detail, driven by `guard_roster`: "N ran ·
  M delegated · K fired", then chips per fired guard (id + action + severity), colored by
  severity. Filter to `surface == "guardrail"` for the strip.
- Add a **filter**: `all / ok / issues / guardrail` plus a per-guard facet. Today filters
  are billing-centric (`apply_billing_action_called`, `TracesPage.tsx:330` and `:512`;
  typed at `client.ts:393`, `:422`); generalize to guardrail signals.
- Promote a `guard_flags` list column onto `TraceSummary` so the session/turn list can
  badge flagged turns without fetching span bodies.

### 7b. Live call UI (`CockpitPage` / Sentient)
- Consume the `guardrail.flagged` WS event during a live call and show a transient badge /
  toast on the active turn (severity-colored) — analogous to the existing
  `language.mismatch` banner. Only `surface: "guardrail"` events are sent on the wire, so
  no client-side filtering is needed here.

### 7c. New Observability / Guardrails page (see §8)
- Aggregate view over `guard_roster`: coverage ("checks run per turn"), delegation rate,
  hit-rates by guard, severity distribution, per-language breakdown, and recent flagged
  turns with deep-links into the Trace Explorer.
- **Honest rendering rule (§2.1):** a `not_evaluated` LID row on a pinned session renders
  as "not evaluated — English pinned by the session", never as a green Qwen checkmark.

---

## 8. Observability UI plan (and Trace Explorer improvements)

### Current state (grounded)
- Backend: `TurnTrace` (`tracing.py:111`) → `submit_trace` (`:349`) → `TraceSink` daemon
  (`:236`) → Lakebase Postgres `voice_traces` (`lakebase.py:718`), UC Delta mirror
  (`:767`), optional MLflow (`tracing.py:295`).
- API: `api/app/routers/traces.py` → `GET /traces` (`:17`), `GET /traces/sessions` (`:24`),
  `GET /traces/{id}` (`:67`). The sessions rollup is **unused** by the UI today.
- Frontend: `TracesPage` (hash `#/traces`) — session list + span waterfall + `SpanDetail`.
  `FlowTracker.tsx` is **orphaned** (defined at `:64`, imported nowhere) and expects
  `jobs.lakeflow` (`:33`, `:37`) while the backend returns `jobs.orchestration`
  (`api/app/routers/pipeline_status.py:158`, `:252`). `client.ts` still exposes the same
  stale shape (`:118`).
- `voiceTraceSessions` (`frontend/src/api/client.ts:600`) has **no component caller** —
  confirmed; the sessions rollup is re-derived client-side in `TracesPage.tsx:512`.
- Operational events: `log_event` (`realtime_api/observability.py:39`) → JSON log lines
  only; **not** visualized anywhere.

### Concrete improvements (prioritized)
1. **Live polling** on `#/traces` (`POLL_INTERVAL_MS` already exists,
   `frontend/src/config.ts:21`, used in `App.tsx:57`) so new turns appear without a manual
   Refresh. Add an auto-refresh toggle + "new turns" indicator.
2. **Nested span waterfall** — the span model is a **flat list** with no `parent_id`
   (`tracing.py:54`–`:63`). Add an optional `parent_id` to `Span` so LLM iterations, tool
   calls, filler-TTS-overlapping-LLM, and fired-guard spans nest correctly. Backfill parent
   by producer (e.g. `tool.*` under `llm.iteration.n`). Note `_mirror_to_mlflow` already
   assumes a single flat level (`:323`–`:328`) and needs updating with it.
3. **Use the `/traces/sessions` rollup** (`api/app/routers/traces.py:24`) instead of
   client-side re-aggregation; add guardrail counts (checks run / delegated / fired) to
   the rollup.
4. **Guardrail facets & badges** (see §7a) — generalize the billing-centric filters.
5. **Deep-link** Cockpit turn → Trace Explorer (`call_id` / `session_id` filters already
   exist in `GET /traces`, `traces.py:18`; just wire the link).
6. **Operational log stream panel** — surface `log_event` output (`speech.started`,
   `barge_in` at `ws/handler.py:142`, `turn.discarded` at `:155`, …) as a live tail beside
   spans, so runtime issues that never produce a persisted trace are visible.
7. **Queue-drop visibility** — `TraceSink` silently drops when its 512 queue is full
   (`tracing.py:264`); expose a dropped-count metric + a UI warning.
8. **Retire or fix `FlowTracker`** — either delete the dead component or repair the
   `jobs.lakeflow` vs `jobs.orchestration` mismatch and wire it into `#/status`.
9. **Sampling / retention controls** — add `realtime_voice.tracing` config (sampling rate,
   enable flag, retention) and reflect current settings in the UI. Today tracing is
   always-on with no yaml knob.

### New page slot
Follow the `VoiceBenchmarksPage` pattern (full-screen dark surface, own CSS, hero +
highlight cards + sections). Wire in `frontend/src/App.tsx` as a new hash (`#/guardrails`)
with a nav pill next to **Traces**/**Benchmarks**, or as tabs within `#/traces`
(Spans | Guardrails | Logs). Either placement is fine — §7.0's *scope* is the binding
constraint, not the URL.

---

## 9. Config schema (proposed)

The previous flat `guards:` list could not express that the concierge needs **no PII
guards** while billing and card do, and that a future regulated profile needs a different
(e.g. PHI) set. Guards are therefore **profile-scoped**: one *catalog* of definitions, plus
per-profile enable/override lists — mirroring how `tool_registry` keys on
`(profile, name)` (`realtime_api/tool_registry.py:77`).

Top-level under `realtime_voice:` in `config/config.yaml` (which today ends at the
`deep_dive` block, `:269`):

```yaml
realtime_voice:
  guardrails:
    enabled: true
    # Permission to delegate WHEN the session left STT on auto. The engine still
    # reads the actual STT request/response; it never trusts the flag alone (§2.1).
    trust_stt_lid: true
    trust_stt_no_speech: true

    observer:
      endpoint: <llm-endpoint>       # required when any observer_async guard is enabled
      cadence_turns: 1

    # ---- Catalog: definitions only. Enabling happens per profile, below. ----
    catalog:
      pci_pii_input:
        impl: genie_voice.guardrails.pii:PciPiiRedactor
        stage: input_transcript
        surface: guardrail
        blocking: true
        error_policy: fail_closed     # required; no default (§0)
        severity: critical
        params:
          # ITN is observed, not contracted (§2.2) — match spoken digits too (§2b(a)).
          spoken_numerals: true
          patterns: [<card>, <cvv>, <national_id>]

      prompt_injection:
        impl: genie_voice.guardrails.injection:JailbreakDetector
        stage: input_transcript
        surface: guardrail
        blocking: false
        error_policy: skip_with_flag
        severity: warn
        params: { patterns: [...] }

      profanity_abuse:
        impl: genie_voice.guardrails.toxicity:ProfanityGuard
        stage: input_transcript
        surface: guardrail
        blocking: false
        error_policy: skip_with_flag
        severity: info
        params: { blocklist: [...] }

      policy_nuance:
        impl: genie_voice.guardrails.policy:LlmPolicyObserver
        stage: observer_async
        surface: guardrail
        error_policy: skip_with_flag   # explicit: availability over strictness, logged
        severity: warn
        params: { rubric: <policy-text-ref> }

      pii_in_reply:
        impl: genie_voice.guardrails.pii:OutputPiiGuard
        stage: pre_tts
        surface: guardrail
        blocking: true
        error_policy: fail_closed
        severity: critical

      output_schema_leak:
        impl: genie_voice.guardrails.output:SchemaLeakGuard   # ported from agent_assist.py:190
        stage: pre_tts
        surface: guardrail
        blocking: true
        error_policy: fail_closed
        severity: warn
        params: { markers: [...] }

      # Turn-integrity mechanics are catalogued so they can REPORT into the roster,
      # but surface=internal keeps them out of the Guardrails UI (§7.0).
      empty_transcript:   { stage: stt,              surface: internal, reports_only: true }
      noise_timeout:      { stage: stt,              surface: internal, reports_only: true }
      stale_turn:         { stage: stt,              surface: internal, reports_only: true }
      retrigger_cooldown: { stage: stt,              surface: internal, reports_only: true }

    # ---- Per-profile enablement. A new industry is a config block, not a code edit. ----
    profiles:
      concierge:
        enable: [prompt_injection, off_scope, reply_language, output_schema_leak]
        # No PII guards: the concierge never handles account data.
      billing:
        enable:
          - pci_pii_input
          - prompt_injection
          - off_scope
          - fraud_social_engineering
          - safety_escalation
          - policy_nuance
          - pii_in_reply
          - unsupported_promise
          - reply_language
          - output_schema_leak
      card:
        enable: [ ... same as billing ... ]
        override:
          policy_nuance:
            params: { rubric: <card-issuer-policy-ref> }
      knowledge:
        enable: [prompt_injection, off_scope, reply_language, output_schema_leak]
        # No PII guards: the knowledge agent never handles account data.

  # ---- Decision seam: still not guards, but no longer hardcoded (§0, Phase 0). ----
  profiles:
    concierge:
      intent_router:
        max_selection_words: 8          # was concierge_tools.py:100
        languages: [en]                 # was an English-only assumption in a comment
        cues:                           # was concierge_tools.py:85-89
          telco: [telco, telecom, telecoms, billing, "phone bill", phone, mobile, wireless, cellular]
          fsi:   [fsi, financial, finance, "credit card", credit-card, card, bank, banking, statement]
          knowledge: [knowledge, databricks, platform, docs, documentation, lakehouse, "unity catalog", "delta lake"]
        confirm_labels:                 # was concierge_tools.py:92
          telco: "Telco billing support"
          fsi: "the credit-card assistant"
          knowledge: "the Databricks Knowledge Agent"

  tracing:               # NEW — make observability configurable (§8.9)
    enabled: true
    sampling: 1.0
    retention_days: 30
    mlflow_mirror: false  # replaces the GENIE_TRACE_MLFLOW env flag (tracing.py:295)
```

**Load-time validation (fail-fast, §0):**
- every `enable` entry exists in `catalog`;
- every catalog entry declares `stage`, `surface`, and `error_policy` explicitly;
- `observer.endpoint` is present iff any enabled guard has `stage: observer_async`;
- the concierge router's `languages` list is consistent with the surfaces that can start
  that profile — so re-enabling the language picker fails loudly instead of silently
  breaking routing.

---

## 10. Phased plan

- **Phase 0 — the ledger (approved to build; no new guards, no behavior change).**
  1. `TurnTrace.guard_roster` + the `guard_roster` JSONB column (§6) and the promoted
     column migration.
  2. Make **every Group A guard report** into the roster — critically **including
     declines**: `selection_length`, `selection_allowlist`, and `selection_ambiguity` must
     emit a roster entry when they *decline* to route, which today records nothing
     (`speech_llm_toolassist_speech.py:242`). Turn-integrity mechanics report with
     `surface: internal`.
  3. Record Group B delegation honestly: `delegated` vs `not_evaluated (session pinned the
     language)`, read from the actual STT request/response (§2.1).
  4. Move the concierge thresholds and cue/label tables into config (`_MAX_SELECTION_WORDS`
     `:100`, `_INDUSTRY_CUES` `:85`, `_CONFIRM_LABELS` `:92`), with the English-only
     constraint declared and validated.
  5. Ship the Guardrails UI reading the roster, filtered on `surface == "guardrail"`
     (§7.0).
  6. **Existing-violation cleanup (§0):** remove the `/tmp` trace fallback
     (`lakebase.py:687`); replace the silent `TraceSink` drop (`tracing.py:264`) with a
     surfaced metric; move `_LANGUAGE_NAMES` (`realtime_stt_agent.py:44`) to shared
     config-driven language handling; promote `GENIE_TRACE_MLFLOW` (`tracing.py:295`) to
     validated `realtime_voice.tracing` config.

  **Payoff:** immediately fixes the "why didn't it route?" blindness, and gives the UI
  something true to show, before a single new guard exists.

- **Phase 1 — engine + input guards.** `GuardrailEngine`, the `Verdict` model with
  `SpokenRemediation` (§4.8), `session.guardrail_directives`, config-driven registration
  on `VoiceProfile` (§4.4). Guards placed **before the `transcript.final` yield** (§4.6).
  Billing and card get `pci_pii_input`; the concierge does not. `prompt_injection` and
  `off_scope` on all profiles; `profanity_abuse` installed but off.

- **Phase 2 — pre-TTS guards** inside `stream_tts` on `primary=True` (§4.7):
  `pii_in_reply`, `unsupported_promise`, `reply_language` (reuse `i18n.py:366`),
  `output_schema_leak` (ported from `agent_assist.py:190`).

- **Phase 3 — async LLM observer.** Rolling-transcript policy eval → directive injection
  into the next turn; single-flight + dedup. `safety_escalation`, `policy_nuance`,
  `fraud_social_engineering`.

- **Phase 4 — remaining observability items.** Nested spans via `parent_id` (§8.2), live
  polling (§8.1), operational log-stream panel (§8.6), Cockpit→Trace deep-links (§8.5),
  queue-drop metric surfaced end-to-end (§8.7), `FlowTracker` fixed or deleted (§8.8).

---

## 11. Open questions

*Resolved and removed:* the architecture fork (settled in §4 — two seams, one ledger); the
observability-page placement (settled by §7.0 — the binding constraint is scope, not URL);
default pre-TTS enforcement (settled by §2b(b) — deterministic and inline before the first
chunk); scope across routes (settled by §9 — profile-scoped config).

Genuinely open:

1. **Observer model** — reuse the main LLM endpoint
   (`realtime_voice.llm_endpoint`, `config/config.yaml:233`), or budget a separate cheaper
   (latency/cost) or stronger (judgment) one? Affects both the config shape and the
   fail-open/fail-closed calculus.
2. **Does `profanity_abuse` ship at all** for these demos? It is the one Group C guard with
   no compliance or safety story behind it, and a false positive on an accented transcript
   is a bad demo moment. Currently specced as installed-but-off; the alternative is not
   shipping it.
3. **Which observer guard ships first** — `policy_nuance` demos best for a card issuer
   (visible, specific, obviously hard to do with regex), while `safety_escalation` is the
   one you would never ship a real voice agent without.
4. **Per-guard error policy** — `fail_closed` for compliance guards (`pci_pii_input`,
   `pii_in_reply`) vs explicitly-configured `skip_with_flag` for LLM nuance guards. The
   rule is settled (it must be **declared per guard in config, never defaulted**, §0); what
   is open is the actual value chosen for each of the eleven Group C guards.

---

## 12. Reference index

**Backend**
- `realtime_api/pipelines/speech_llm_toolassist_speech.py` — assist turn loop.
  `stale_turn` (`:174`), `empty_transcript` (`:177`), language gate (`:185`, span `:188`,
  spoken prompt `:198`, cooldown `:207`), `transcript.final` yield (`:215`),
  `history.append` (`:223`), profile resolution (`:228`), intent pre-router (`:240`, span
  only on success `:243`), `history` GUARD span (`:299`), LLM call (`:306`), filler TTS
  (`:332`), answer TTS (`:407`).
- `realtime_api/pipelines/_shared.py` — `language_mismatch` (`:53`, gated on
  `expected_language` `:64`), `transcribe` (`:76`, STT pin `:80`), `stream_tts` (`:99`,
  `primary` `:109`), `resolve_language` (`:35`).
- `realtime_api/profiles.py` — `ResolvedIntent` (`:20`), `VoiceProfile` (`:38`) with
  `tools_spec` (`:43`), `tool_runner` (`:44`), `resolve_intents` (`:55`); add `guardrails`.
- `realtime_api/concierge_tools.py` — `tool_arg_enum` (`:68`), `_INDUSTRY_CUES` (`:85`),
  `_CONFIRM_LABELS` (`:92`), `_MAX_SELECTION_WORDS` (`:100`), `resolve_industry` (`:109`,
  length `:116`, ambiguity `:118`).
- `realtime_api/tool_registry.py` — `(profile, name)` registry (`:77`, `:83`).
- `realtime_api/services.py` — `transcribe` (`:242`), `respond_with_tools` (`:281`) +
  system-prompt assembly (`:301`), `_strip_tool_markup` (`:724`, used `:362`/`:404`),
  `_extract_inline_tool_calls` (`:696`).
- `realtime_api/tracing.py` — `Span` (`:54`, kind comment `:56`), `TurnTrace` (`:111`),
  `input_transcript`/`output_text` (`:131`–`:132`, serialized `:210`–`:211`), `span()`
  (`:182`), `TraceSink` (`:236`), silent queue drop (`:264`), `GENIE_TRACE_MLFLOW` (`:295`),
  MLflow flat-span mirror (`:323`), `submit_trace` (`:349`).
- `realtime_api/session.py` — `VoiceSession` (`:33`, add `guardrail_directives`),
  `should_finalize` cooldown check (`:129`), `set_cooldown` (`:139`), `is_noise_timeout`
  (`:160`).
- `realtime_api/ws/handler.py` — `barge_in` (`:142`), noise discard (`:153`).
- `realtime_api/observability.py` — `log_event` (`:39`).
- `realtime_api/contracts.py` — wire events, `SessionStart`, `expected_language` (`:90`).
- `backend/genie_voice/serve/lakebase.py` — `voice_traces` DDL (`:570`), latency-column
  migration (`:607`, `DOUBLE PRECISION` hardcode `:637`), `/tmp` fallback path (`:687`),
  `insert_voice_trace` (`:718`), UC Delta mirror (`:767`), `list_voice_traces` (`:769`).
- `api/app/routers/traces.py` — `GET /traces` (`:17`), `/traces/sessions` (`:24`),
  `/traces/{id}` (`:67`).
- `api/app/routers/pipeline_status.py` — `GET /status`, `jobs.orchestration` (`:158`,
  `:252`).
- `api/app/routers/agent_assist.py` — `_BAD_AGENT_REPLY_MARKERS` (`:190`),
  `_looks_like_bad_agent_reply` (`:211`) — port to `output_schema_leak`.
- `backend/genie_voice/i18n.py` — `generated_text_language_check` (`:366`).
- `scripts/ml_asr/realtime_stt_agent.py` — `_LANGUAGE_NAMES` (`:44`), `custom_inputs`
  parsing (`:112`–`:116`, `max_new_tokens` `:125`), forced decode (`:131`, `:133`),
  `custom_outputs` (`:141`–`:146`), `_forced_language` (`:150`).
- `config/config.yaml` — `realtime_voice:` (`:228`), `llm_endpoint` (`:233`),
  `noise_discard_seconds` (`:290`).

**Frontend**
- `frontend/src/components/TracesPage.tsx` (`KindTag` `:135`, billing-centric filters
  `:330`, `:512`) + `styles/traces.css` (`.tv-kind.GUARD` `:100`) — Trace Explorer.
- `frontend/src/components/VoiceBenchmarksPage.tsx` — good page pattern to mirror.
- `frontend/src/components/CockpitPage.tsx` / `sentient/*` — live call UI.
- `frontend/src/components/HomePage.tsx` — concierge; STT pinned to English (`:309`),
  language bar disabled (`:355`).
- `frontend/src/components/CardIssuerPage.tsx` (`:538`), `KnowledgeAgentPage.tsx`,
  `CallList.tsx` (`:1200`) — all pass the picker as `expected_language`, leaving STT on
  auto.
- `frontend/src/lib/realtimeVoice.ts` — `language` / `sttLanguage` (`:234`),
  `expected_language` (`:241`).
- `frontend/src/components/FlowTracker.tsx` — orphaned (`:64`); expects `jobs.lakeflow`
  (`:33`, `:37`).
- `frontend/src/api/client.ts` — `voiceTraces`, `voiceTrace`, `voiceTraceSessions` (`:600`,
  no component caller), stale `jobs.lakeflow` type (`:118`).
- `frontend/src/config.ts` — `POLL_INTERVAL_MS` (`:21`).
- `frontend/src/App.tsx` — hash routing + nav pills, polling (`:57`).

**External**
- Qwen3-ASR model card (`Qwen/Qwen3-ASR-1.7B`) — documents 52-language LID and robustness
  under complex acoustic environments; does **not** document ITN or context biasing.
- LiveKit observer pattern — https://livekit.com/blog/observer-pattern-voice-agent-guardrails
