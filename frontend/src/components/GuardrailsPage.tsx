/**
 * Guardrails view — the rail *catalog* first, live detections overlaid.
 *
 * The Trace Explorer answers "did a guardrail fire on THIS turn". This page
 * answers the prior question: "what can we detect at all, and how often has it
 * fired across every recorded turn". So the spine of the page is a static
 * catalog of rails — the ones delegated to Qwen3-ASR and the custom rails this
 * service owns — and each rail is then decorated with its aggregate outcomes
 * from `guard_roster` (via `GET /traces/guardrails`).
 *
 * Two families, per the design scope (docs/guardrails-and-observability.md §3, §7):
 *   - Delegated to Qwen3-ASR — the ASR model owns these; we consume its signal.
 *   - Custom rails — what this service adds, grouped by pipeline stage. Rails
 *     whose engine hasn't shipped yet are shown as `planned` with the phase that
 *     lands them, so the page reads as the guardrail *program*, not just today's
 *     hits.
 *
 * Honesty rules kept from the roster contract:
 *   - `not_evaluated` never renders as a pass (e.g. language-ID on a pinned
 *     session did NOT run).
 *   - `planned` rails carry no fake numbers; they show a phase, not a "0 passed".
 *   - only `surface === "guardrail"` rows exist in the rollup (the backend filters
 *     turn-integrity mechanics out), so nothing here inflates "checks ran".
 */
import { useCallback, useEffect, useMemo, useState } from "react";

import { api, type GuardRollup, type GuardRollupGuard } from "../api/client";
import "../styles/traces.css";

type RailStatus = "live" | "delegated" | "planned";
type RailKind = "deterministic" | "llm" | "model-signal";

interface RailDef {
  guard_id: string;
  title: string;
  owner: "qwen" | "us";
  /** Which top-level family the rail belongs to. */
  family: "qwen" | "custom";
  /** Sub-heading within the custom family (ignored for the qwen family). */
  group: string;
  /** Human-readable pipeline stage. */
  stage: string;
  kind: RailKind;
  status: RailStatus;
  /** For planned rails: the phase that ships it. */
  phase?: string;
  /** Where the rail applies, when it isn't every profile. */
  profiles?: string;
  blurb: string;
}

/**
 * The catalog. Mirrors docs/guardrails-and-observability.md §3 (Group A/B/C).
 * This is UI descriptor data (labels, blurbs, status) — no thresholds or
 * patterns — and will be sourced from the config-driven guard catalog (§9) once
 * the enforcement engine ships.
 */
const CATALOG: RailDef[] = [
  // ---- Group B — delegated to Qwen3-ASR ----
  {
    guard_id: "language_id",
    title: "Spoken-language identification",
    owner: "qwen",
    family: "qwen",
    group: "Delegated to Qwen3-ASR",
    stage: "STT",
    kind: "model-signal",
    status: "delegated",
    blurb:
      "Qwen3-ASR reports the language it heard (52-language LID). It only runs when the session leaves STT on auto — a pinned language means no detection took place, which the roster records as “not evaluated”, never as a pass.",
  },
  {
    guard_id: "no_speech_suppression",
    title: "Silence / no-speech suppression",
    owner: "qwen",
    family: "qwen",
    group: "Delegated to Qwen3-ASR",
    stage: "STT",
    kind: "model-signal",
    status: "delegated",
    blurb:
      "Qwen3-ASR returns an empty transcript for silence, echo, or breath; combined with our endpointer, the turn is dropped before any downstream model sees it.",
  },

  // ---- Group A — custom rails already live ----
  {
    guard_id: "language_gate",
    title: "Reply-language gate",
    owner: "us",
    family: "custom",
    group: "Live today",
    stage: "Routing",
    kind: "deterministic",
    status: "live",
    blurb:
      "Drops a turn whose spoken language doesn't match the selected one and speaks a localized switch prompt, instead of answering in the wrong language.",
  },
  {
    guard_id: "selection_allowlist",
    title: "Selection cue allowlist",
    owner: "us",
    family: "custom",
    group: "Live today",
    stage: "Routing",
    kind: "deterministic",
    status: "live",
    blurb: "Routes only on known industry cues, never on an unrecognized phrase.",
  },
  {
    guard_id: "selection_length",
    title: "Selection length limit",
    owner: "us",
    family: "custom",
    group: "Live today",
    stage: "Routing",
    kind: "deterministic",
    status: "live",
    blurb:
      "Only short, selection-style replies route deterministically. A longer utterance is a question, not a choice, so it defers to the model.",
  },
  {
    guard_id: "selection_ambiguity",
    title: "Selection ambiguity gate",
    owner: "us",
    family: "custom",
    group: "Live today",
    stage: "Routing",
    kind: "deterministic",
    status: "live",
    blurb: "Refuses to route when a reply matches more than one destination, rather than guessing between them.",
  },
  {
    guard_id: "tool_arg_enum",
    title: "Tool-argument enum check",
    owner: "us",
    family: "custom",
    group: "Live today",
    stage: "Routing",
    kind: "deterministic",
    status: "live",
    blurb: "Rejects tool arguments the model invents outside the declared enum.",
  },
  {
    guard_id: "tool_markup_strip",
    title: "Tool-markup strip",
    owner: "us",
    family: "custom",
    group: "Live today",
    stage: "Pre-TTS",
    kind: "deterministic",
    status: "live",
    blurb: "Strips leaked <tool_call> markup out of text before it is ever spoken.",
  },

  // ---- Group C — custom rails to add: Input stage ----
  {
    guard_id: "pci_pii_input",
    title: "PCI / PII redaction (input)",
    owner: "us",
    family: "custom",
    group: "Input stage",
    stage: "Input transcript",
    kind: "deterministic",
    status: "planned",
    phase: "Phase 1",
    profiles: "Billing · Card",
    blurb:
      "Card number, CVV, and national ID — including spoken-as-words digits (“four one one one…”). Redacts before the transcript is stored or shown; fail-closed.",
  },
  {
    guard_id: "prompt_injection",
    title: "Prompt-injection detector",
    owner: "us",
    family: "custom",
    group: "Input stage",
    stage: "Input transcript",
    kind: "deterministic",
    status: "planned",
    phase: "Phase 1",
    profiles: "All profiles",
    blurb: "“Ignore your instructions”, role overrides, and tool-name coaxing aimed at the downstream LLM.",
  },
  {
    guard_id: "off_scope",
    title: "Off-scope request",
    owner: "us",
    family: "custom",
    group: "Input stage",
    stage: "Input transcript",
    kind: "deterministic",
    status: "planned",
    phase: "Phase 1",
    profiles: "All profiles",
    blurb: "Flags or steers requests outside the profile's domain back on track.",
  },
  {
    guard_id: "profanity_abuse",
    title: "Profanity / abuse",
    owner: "us",
    family: "custom",
    group: "Input stage",
    stage: "Input transcript",
    kind: "deterministic",
    status: "planned",
    phase: "Phase 1",
    profiles: "Installed · off by default",
    blurb: "Slurs and harassment. Shipped switched off — a false positive on an accented transcript is a worse outcome than a miss.",
  },

  // ---- Group C — Observer stage (async LLM) ----
  {
    guard_id: "fraud_social_engineering",
    title: "Fraud / social engineering",
    owner: "us",
    family: "custom",
    group: "Observer stage",
    stage: "Async observer",
    kind: "llm",
    status: "planned",
    phase: "Phase 3",
    profiles: "Billing · Card",
    blurb: "Watches the rolling transcript for dispute-gaming and social engineering, and injects a directive into the next turn.",
  },
  {
    guard_id: "safety_escalation",
    title: "Safety escalation",
    owner: "us",
    family: "custom",
    group: "Observer stage",
    stage: "Async observer",
    kind: "llm",
    status: "planned",
    phase: "Phase 3",
    profiles: "All profiles",
    blurb: "Caller in danger, threats, or self-harm — escalates rather than continuing a scripted flow.",
  },
  {
    guard_id: "policy_nuance",
    title: "Policy nuance",
    owner: "us",
    family: "custom",
    group: "Observer stage",
    stage: "Async observer",
    kind: "llm",
    status: "planned",
    phase: "Phase 3",
    profiles: "Billing · Card",
    blurb: "Off-policy commitments (“no financial advice”) that a regex can't catch — steered on the next turn.",
  },

  // ---- Group C — Pre-TTS stage ----
  {
    guard_id: "pii_in_reply",
    title: "PII in the reply",
    owner: "us",
    family: "custom",
    group: "Pre-TTS stage",
    stage: "Pre-TTS",
    kind: "deterministic",
    status: "planned",
    phase: "Phase 2",
    blurb: "Blocks the agent from speaking sensitive data aloud; fail-closed, before the first audio chunk.",
  },
  {
    guard_id: "unsupported_promise",
    title: "Unsupported promise",
    owner: "us",
    family: "custom",
    group: "Pre-TTS stage",
    stage: "Pre-TTS",
    kind: "deterministic",
    status: "planned",
    phase: "Phase 2",
    blurb: "“I've refunded you” with no tool call behind it — cross-checked against the turn's actual tool calls.",
  },
  {
    guard_id: "reply_language",
    title: "Reply-language check",
    owner: "us",
    family: "custom",
    group: "Pre-TTS stage",
    stage: "Pre-TTS",
    kind: "deterministic",
    status: "planned",
    phase: "Phase 2",
    blurb: "Catches a reply that drifted out of the caller's language before it is spoken, and regenerates it.",
  },
  {
    guard_id: "output_schema_leak",
    title: "Schema / SQL leak",
    owner: "us",
    family: "custom",
    group: "Pre-TTS stage",
    stage: "Pre-TTS",
    kind: "deterministic",
    status: "planned",
    phase: "Phase 2",
    blurb: "Stops the agent reading raw SQL or schema identifiers aloud.",
  },
];

/** Order of custom sub-groups, most-shipped first. */
const CUSTOM_GROUPS = ["Live today", "Input stage", "Observer stage", "Pre-TTS stage"] as const;

/** Outcomes in the order an operator reads them: hits first, then coverage. */
const OUTCOME_ORDER = ["fired", "error", "passed", "delegated", "not_evaluated", "disabled"] as const;

const OUTCOME_LABEL: Record<string, string> = {
  passed: "passed",
  fired: "fired",
  delegated: "delegated",
  not_evaluated: "not evaluated",
  disabled: "disabled",
  error: "error",
};

const OUTCOME_HINT: Record<string, string> = {
  passed: "Ran and found nothing. The most common and most valuable outcome.",
  fired: "Ran and took action: blocked, redacted, flagged, or declined to route.",
  delegated: "We deliberately did not run it — Qwen3-ASR owns it and we consumed its signal.",
  not_evaluated: "Preconditions absent, so the check never ran. Not a pass.",
  disabled: "In the catalog, switched off for this profile.",
  error: "The check itself raised.",
};

const STATUS_LABEL: Record<RailStatus, string> = {
  live: "live",
  delegated: "delegated",
  planned: "planned",
};

const KIND_LABEL: Record<RailKind, string> = {
  deterministic: "deterministic",
  llm: "LLM",
  "model-signal": "model signal",
};

function pct(part: number, whole: number): string {
  if (!whole) return "0%";
  return `${Math.round((part / whole) * 100)}%`;
}

function OutcomeBadge({ outcome, count }: { outcome: string; count?: number }) {
  return (
    <span className={`gr-outcome is-${outcome}`} title={OUTCOME_HINT[outcome] ?? outcome}>
      {OUTCOME_LABEL[outcome] ?? outcome}
      {count !== undefined && <strong>{count}</strong>}
    </span>
  );
}

/** Proportional bar of a rail's outcomes, so a rare `fired` stays visible. */
function OutcomeBar({ outcomes, runs }: { outcomes: Record<string, number>; runs: number }) {
  const parts = OUTCOME_ORDER.filter((o) => outcomes[o]);
  return (
    <div className="gr-bar" role="img" aria-label={parts.map((o) => `${o} ${outcomes[o]}`).join(", ")}>
      {parts.map((o) => (
        <span
          key={o}
          className={`gr-bar-seg is-${o}`}
          // A single hit among hundreds of passes still has to be findable.
          style={{ width: `${Math.max(4, (outcomes[o] / runs) * 100)}%` }}
          title={`${OUTCOME_LABEL[o]}: ${outcomes[o]} of ${runs}`}
        />
      ))}
    </div>
  );
}

/** One rail: its catalog identity, decorated with aggregate detections if any. */
function RailRow({ def, live }: { def: RailDef; live?: GuardRollupGuard }) {
  const fired = live?.outcomes.fired ?? 0;
  const runs = live?.runs ?? 0;
  return (
    <div className={`gr-guard gr-rail is-${def.status}`}>
      <div className="gr-guard-head">
        <span className="gr-guard-name">{def.title}</span>
        <code className="gr-guard-id">{def.guard_id}</code>
        <span className={`gr-owner${def.owner === "qwen" ? " is-qwen" : ""}`}>
          {def.owner === "qwen" ? "Qwen3-ASR" : "this service"}
        </span>
        <span className="gr-stage">{def.stage}</span>
        <span className="gr-kind">{KIND_LABEL[def.kind]}</span>
        <span className={`gr-status is-${def.status}`}>{STATUS_LABEL[def.status]}</span>
        <span className="gr-guard-runs">
          {runs > 0 ? (
            <>
              {runs} {runs === 1 ? "check" : "checks"}
              {fired > 0 && <em> · {pct(fired, runs)} fired</em>}
            </>
          ) : def.status === "planned" ? (
            <span className="gr-phase">{def.phase ?? "planned"}</span>
          ) : (
            <span className="gr-idle">no turns recorded yet</span>
          )}
        </span>
      </div>
      <div className="gr-guard-blurb">
        {def.profiles && <span className="gr-profiles">{def.profiles}</span>}
        {def.blurb}
      </div>
      {runs > 0 && live && (
        <>
          <OutcomeBar outcomes={live.outcomes} runs={runs} />
          <div className="gr-guard-foot">
            {OUTCOME_ORDER.filter((o) => live.outcomes[o]).map((o) => (
              <OutcomeBadge key={o} outcome={o} count={live.outcomes[o]} />
            ))}
            {live.last_reason && <span className="gr-reason">latest: {live.last_reason}</span>}
          </div>
        </>
      )}
    </div>
  );
}

export function GuardrailsPage() {
  const [rollup, setRollup] = useState<GuardRollup | null>(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);

  const load = useCallback(() => {
    setLoading(true);
    api
      .guardRollup(300)
      .then((r) => {
        setRollup(r);
        setErr(null);
      })
      .catch((e) => setErr(e instanceof Error ? e.message : "failed"))
      .finally(() => setLoading(false));
  }, []);

  useEffect(load, [load]);

  // Index live detections by guard_id so each catalog rail can find its own.
  const liveById = useMemo(() => {
    const map = new Map<string, GuardRollupGuard>();
    for (const g of rollup?.guards ?? []) map.set(g.guard_id, g);
    return map;
  }, [rollup]);

  const qwenRails = CATALOG.filter((r) => r.family === "qwen");
  const customRails = CATALOG.filter((r) => r.family === "custom");

  // Rails observed in traces but not in the catalog — never hide a real signal.
  const uncatalogued = useMemo(() => {
    const known = new Set(CATALOG.map((r) => r.guard_id));
    return (rollup?.guards ?? []).filter((g) => !known.has(g.guard_id));
  }, [rollup]);

  const coverage = useMemo(() => {
    const live = CATALOG.filter((r) => r.status === "live").length;
    const delegated = CATALOG.filter((r) => r.status === "delegated").length;
    const planned = CATALOG.filter((r) => r.status === "planned").length;
    const observed = CATALOG.filter((r) => (liveById.get(r.guard_id)?.runs ?? 0) > 0).length;
    return { live, delegated, planned, observed };
  }, [liveById]);

  const fired = rollup?.totals.fired ?? 0;
  const checks = rollup?.checks ?? 0;

  return (
    <div className="tv-root">
      <div className="tv-header">
        <div className="tv-title">
          <span>Genie</span> Voice · Guardrails
        </div>
        <div className="tv-header-spacer" />
        <div className="tv-filters">
          <button className="tv-btn ghost" onClick={() => (window.location.hash = "#/traces")}>
            Trace Explorer
          </button>
          <button className="tv-btn ghost" onClick={() => (window.location.hash = "#/")}>
            ← Home
          </button>
          <button className="tv-btn" onClick={load}>
            Refresh
          </button>
        </div>
      </div>

      <div className="gr-body">
        {loading && !rollup && <div className="tv-empty">Loading guardrail ledger…</div>}
        {err && <div className="tv-empty">Error: {err}</div>}

        {rollup && (
          <>
            <div className="gr-headline">
              <div className="gr-stat">
                <div className="gr-stat-value">{CATALOG.length}</div>
                <div className="gr-stat-label">rails in catalog</div>
                <div className="gr-stat-sub">
                  {coverage.live} live · {coverage.delegated} delegated · {coverage.planned} planned
                </div>
              </div>
              <div className="gr-stat">
                <div className="gr-stat-value">{checks}</div>
                <div className="gr-stat-label">checks recorded</div>
                <div className="gr-stat-sub">
                  across {rollup.turns_with_roster} turns · {coverage.observed} rails active
                </div>
              </div>
              <div className={`gr-stat${fired ? " is-hit" : " is-clean"}`}>
                <div className="gr-stat-value">{fired}</div>
                <div className="gr-stat-label">fired</div>
                <div className="gr-stat-sub">{fired ? "took action on a turn" : "nothing acted on a turn"}</div>
              </div>
              <div className="gr-stat">
                <div className="gr-stat-value">{rollup.totals.delegated ?? 0}</div>
                <div className="gr-stat-label">delegated to Qwen3-ASR</div>
                <div className="gr-stat-sub">the ASR model owns these for us</div>
              </div>
            </div>

            {checks === 0 && (
              <div className="tv-callout">
                No detections recorded yet — the catalog below is the full set of rails; the counts populate as
                voice turns run.
              </div>
            )}

            <div className="tv-section-title">Delegated to Qwen3-ASR</div>
            <div className="gr-guards">
              {qwenRails.map((r) => (
                <RailRow key={r.guard_id} def={r} live={liveById.get(r.guard_id)} />
              ))}
            </div>

            <div className="tv-section-title">Custom rails — this service</div>
            {CUSTOM_GROUPS.map((group) => {
              const rails = customRails.filter((r) => r.group === group);
              if (rails.length === 0) return null;
              return (
                <div key={group} className="gr-subgroup">
                  <div className="gr-subhead">{group}</div>
                  <div className="gr-guards">
                    {rails.map((r) => (
                      <RailRow key={r.guard_id} def={r} live={liveById.get(r.guard_id)} />
                    ))}
                  </div>
                </div>
              );
            })}

            {uncatalogued.length > 0 && (
              <>
                <div className="tv-section-title">Other rails observed in traces</div>
                <div className="gr-guards">
                  {uncatalogued.map((g) => (
                    <div className="gr-guard gr-rail is-live" key={g.guard_id}>
                      <div className="gr-guard-head">
                        <span className="gr-guard-name">{g.guard_id}</span>
                        <code className="gr-guard-id">{g.guard_id}</code>
                        {g.stage && <span className="gr-stage">{String(g.stage).replace(/_/g, " ")}</span>}
                        <span className="gr-guard-runs">
                          {g.runs} {g.runs === 1 ? "check" : "checks"}
                        </span>
                      </div>
                      <OutcomeBar outcomes={g.outcomes} runs={g.runs} />
                      <div className="gr-guard-foot">
                        {OUTCOME_ORDER.filter((o) => g.outcomes[o]).map((o) => (
                          <OutcomeBadge key={o} outcome={o} count={g.outcomes[o]} />
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </>
            )}

            {rollup.recent_fired.length > 0 && (
              <>
                <div className="tv-section-title">Turns where a guardrail acted</div>
                <div className="gr-fired">
                  {rollup.recent_fired.map((row, i) => (
                    <button
                      key={`${row.trace_id}-${row.guard_id}-${i}`}
                      className="gr-fired-row"
                      onClick={() => (window.location.hash = `#/traces?trace=${row.trace_id}`)}
                      title="Open this turn in the Trace Explorer"
                    >
                      <code className="gr-fired-guard">{row.guard_id}</code>
                      <span className="gr-fired-reason">{row.reason || "—"}</span>
                      <span className="gr-fired-meta">
                        {row.language || "—"} · turn {row.turn_id ?? "—"}
                      </span>
                    </button>
                  ))}
                </div>
              </>
            )}

            {Object.keys(rollup.by_language).length > 1 && (
              <>
                <div className="tv-section-title">By call language</div>
                <div className="gr-langs">
                  {Object.entries(rollup.by_language).map(([lang, outcomes]) => (
                    <div className="gr-lang" key={lang}>
                      <span className="gr-lang-code">{lang}</span>
                      {OUTCOME_ORDER.filter((o) => outcomes[o]).map((o) => (
                        <OutcomeBadge key={o} outcome={o} count={outcomes[o]} />
                      ))}
                    </div>
                  ))}
                </div>
              </>
            )}
          </>
        )}
      </div>
    </div>
  );
}
