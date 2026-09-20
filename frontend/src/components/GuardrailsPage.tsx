/**
 * Guardrails view driven entirely by config/guardrails.yaml through the API.
 * React owns presentation only; policy identity, status, assignment, and
 * deployment state are runtime data.
 */
import { useCallback, useEffect, useMemo, useState } from "react";

import {
  api,
  type GatewayInsights,
  type GuardFiredRow,
  type GuardRollup,
  type GuardRollupGuard,
  type PolicyCatalogEntry,
} from "../api/client";
import "../styles/traces.css";

const OUTCOMES = ["fired", "error", "passed", "delegated", "not_evaluated", "disabled"] as const;

function pct(part: number, whole: number): string {
  return whole ? `${Math.round((part / whole) * 100)}%` : "0%";
}

function OutcomeBar({ outcomes, runs }: { outcomes: Record<string, number>; runs: number }) {
  return (
    <div className="gr-bar">
      {OUTCOMES.filter((outcome) => outcomes[outcome]).map((outcome) => (
        <span
          key={outcome}
          className={`gr-bar-seg is-${outcome}`}
          style={{ width: `${Math.max(4, (outcomes[outcome] / runs) * 100)}%` }}
          title={`${outcome}: ${outcomes[outcome]} of ${runs}`}
        />
      ))}
    </div>
  );
}

function PolicyRow({ policy, live }: { policy: PolicyCatalogEntry; live?: GuardRollupGuard }) {
  const runs = live?.runs ?? 0;
  const fired = live?.outcomes.fired ?? 0;
  const owner =
    policy.owner === "gateway"
      ? "Unity AI Gateway"
      : policy.owner === "qwen"
        ? "Qwen"
        : "application";
  return (
    <div className={`gr-guard gr-rail is-${policy.status}`}>
      <div className="gr-guard-head">
        <span className="gr-guard-name">{policy.title}</span>
        <code className="gr-guard-id">{policy.guard_id}</code>
        <span className={`gr-owner${policy.owner === "gateway" ? " is-gateway" : policy.owner === "qwen" ? " is-qwen" : ""}`}>
          {owner}
        </span>
        <span className="gr-stage">{policy.stage}</span>
        <span className="gr-kind">{policy.kind}</span>
        <span className={`gr-status is-${policy.status}`}>{policy.status}</span>
        <span className="gr-guard-runs">
          {runs ? (
            <>
              {runs} {runs === 1 ? "check" : "checks"}
              {fired > 0 && <em> · {pct(fired, runs)} fired</em>}
            </>
          ) : policy.status === "planned" ? (
            <span className="gr-phase">{policy.phase}</span>
          ) : (
            <span className="gr-idle">no recorded turn yet</span>
          )}
        </span>
      </div>
      <div className="gr-guard-blurb">{policy.description}</div>
      {runs > 0 && live && (
        <>
          <OutcomeBar outcomes={live.outcomes} runs={runs} />
          <div className="gr-guard-foot">
            {OUTCOMES.filter((outcome) => live.outcomes[outcome]).map((outcome) => (
              <span key={outcome} className={`gr-outcome is-${outcome}`}>
                {outcome.replace("_", " ")} <strong>{live.outcomes[outcome]}</strong>
              </span>
            ))}
            {live.last_reason && <span className="gr-reason">latest: {live.last_reason}</span>}
          </div>
        </>
      )}
    </div>
  );
}

function interventionCopy(row: GuardFiredRow, policyTitle?: string) {
  const reason = (row.reason ?? "").toLowerCase();
  if (row.guard_id === "language_gate") {
    return {
      risk: "Language mismatch",
      decision: "Paused the turn",
      outcome: "Caller received the correct language-switch prompt",
    };
  }
  if (row.guard_id === "navigation.policy" && reason.includes("clarify")) {
    return {
      risk: "Ambiguous request",
      decision: "Withheld the action",
      outcome: "Caller was asked to clarify before anything changed",
    };
  }
  if (row.guard_id === "navigation.policy") {
    return {
      risk: "Unapproved capability",
      decision: "Stopped routing",
      outcome: "Conversation remained within an authorized workflow",
    };
  }
  if (row.guard_id === "sensitive_input_tiering") {
    return {
      risk: reason.includes("mask") ? "Contact data detected" : "Credential detected",
      decision: reason.includes("mask") ? "Masked before model use" : "Blocked transcript admission",
      outcome: "Raw sensitive data stayed out of conversation history and tools",
    };
  }
  if (row.guard_id === "tool_execution_policy") {
    return {
      risk: "Unsafe account change",
      decision: "Denied tool execution",
      outcome: "Customer state remained unchanged",
    };
  }
  if (row.guard_id.startsWith("gateway.")) {
    return {
      risk: "Risky model interaction",
      decision: "Gateway denied it",
      outcome: "Unsafe content never reached the next application boundary",
    };
  }
  return {
    risk: policyTitle ?? "Policy condition detected",
    decision: "Applied the configured guardrail",
    outcome: "The turn was changed before caller impact",
  };
}

export function GuardrailsPage() {
  const [rollup, setRollup] = useState<GuardRollup | null>(null);
  const [gateway, setGateway] = useState<GatewayInsights | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback((silent = false) => {
    if (!silent) setLoading(true);
    Promise.all([api.guardRollup(300), api.gatewayInsights()])
      .then(([guardData, gatewayData]) => {
        setRollup(guardData);
        setGateway(gatewayData);
        setError(null);
      })
      .catch((reason) => setError(reason instanceof Error ? reason.message : "Failed to load"))
      .finally(() => {
        if (!silent) setLoading(false);
      });
  }, []);

  useEffect(() => {
    load();
    const timer = window.setInterval(() => load(true), 15_000);
    return () => window.clearInterval(timer);
  }, [load]);

  const liveById = useMemo(
    () => new Map((rollup?.guards ?? []).map((guard) => [guard.guard_id, guard])),
    [rollup],
  );
  const catalog = gateway?.policy_manifest.catalog ?? [];
  const groups = useMemo(() => {
    const grouped = new Map<string, PolicyCatalogEntry[]>();
    for (const policy of catalog) {
      const key = `${policy.family}::${policy.group}`;
      grouped.set(key, [...(grouped.get(key) ?? []), policy]);
    }
    return [...grouped.entries()];
  }, [catalog]);
  const known = useMemo(() => new Set(catalog.map((policy) => policy.guard_id)), [catalog]);
  const uncatalogued = (rollup?.guards ?? []).filter((guard) => !known.has(guard.guard_id));
  const checks = rollup?.checks ?? 0;
  const fired = rollup?.totals.fired ?? 0;
  const errors = rollup?.totals.error ?? 0;
  const gatewayHealthy =
    (gateway?.services.length ?? 0) > 0 &&
    (gateway?.services ?? []).every((service) => service.policy_deployment_state === "configured");
  const boundariesAssigned = ["post_stt", "tool_execution", "pre_tts"].every(
    (boundary) => Boolean(gateway?.policy_manifest.assignments.boundaries[boundary]),
  );
  const protectionActive = gatewayHealthy && boundariesAssigned;
  const activeGroups = groups
    .map(([key, policies]) => [key, policies.filter((policy) => policy.status !== "planned")] as const)
    .filter(([, policies]) => policies.length > 0);
  const plannedPolicies = catalog.filter((policy) => policy.status === "planned");
  const firedFor = (...policyIds: string[]) =>
    policyIds.reduce((total, policyId) => total + (liveById.get(policyId)?.outcomes.fired ?? 0), 0);
  const outcomeGroups = [
    {
      key: "understanding",
      label: "Conversation steering",
      count: firedFor("language_gate", "navigation.policy"),
      description: "Ambiguous and wrong-language turns redirected",
    },
    {
      key: "privacy",
      label: "Privacy",
      count: firedFor("sensitive_input_tiering"),
      description: "Sensitive values blocked or masked",
    },
    {
      key: "actions",
      label: "Account safety",
      count: firedFor("tool_execution_policy"),
      description: "Unapproved mutations prevented",
    },
    {
      key: "delivery",
      label: "Speech delivery",
      count: firedFor("tool_markup_strip", "speech_output_boundary"),
      description: "Malformed tool markup kept out of synthesized audio",
    },
    {
      key: "gateway",
      label: "Model safety",
      count: firedFor("gateway.service_policy", "gateway.rate_limit", "gateway.request"),
      description: "Risky model interactions stopped",
    },
  ];
  const groupedOutcomes = outcomeGroups.reduce((total, group) => total + group.count, 0);
  const impactGroups = [
    ...outcomeGroups,
    ...(fired > groupedOutcomes
      ? [{
          key: "other",
          label: "Other protections",
          count: fired - groupedOutcomes,
          description: "Additional configured policy interventions",
        }]
      : []),
  ];
  const interventionRate = checks ? Math.round((fired / checks) * 1000) / 10 : 0;
  const journey = [
    {
      step: "Listen",
      boundary: "Post-STT",
      owner: "Application",
      promise: "Raw speech is safe to admit",
      proof: "Language checked · sensitive values tiered",
    },
    {
      step: "Reason",
      boundary: "AI Gateway",
      owner: "Databricks",
      promise: "Model traffic is governed",
      proof: `${gateway?.services.length ?? 0} service contracts verified`,
    },
    {
      step: "Act",
      boundary: "Tool execution",
      owner: "Application",
      promise: "Only approved changes run",
      proof: "Authorization · confirmation · state validation",
    },
    {
      step: "Speak",
      boundary: "Pre-TTS",
      owner: "Application",
      promise: "Only supported answers leave",
      proof: "Evidence required · tool markup removed",
    },
  ];

  return (
    <div className="tv-root">
      <div className="tv-header">
        <div className="tv-title"><span>Genie</span> Voice · Guardrails</div>
        <div className="tv-header-spacer" />
        <div className="tv-filters">
          <button className="tv-btn ghost" onClick={() => (window.location.hash = "#/traces")}>Trace Explorer</button>
          <button className="tv-btn ghost" onClick={() => (window.location.hash = "#/setup")}>Setup</button>
          <button className="tv-btn ghost" onClick={() => (window.location.hash = "#/")}>← Home</button>
          <button className="tv-btn" onClick={() => load()}>Refresh</button>
        </div>
      </div>

      <div className="gr-body">
        {loading && !rollup && <div className="tv-empty">Loading policy manifest and evidence…</div>}
        {error && <div className="tv-empty">Error: {error}</div>}
        {rollup && gateway && (
          <>
            <section className="gr-command">
              <div className="gr-command-posture">
                <div className={`gr-command-beacon ${protectionActive ? "is-ready" : "is-review"}`}>
                  <span>{protectionActive ? "✓" : "!"}</span>
                </div>
                <div>
                  <div className="gr-impact-eyebrow">Current protection</div>
                  <h1>{protectionActive ? "Voice journey secured" : "Boundary attention required"}</h1>
                  <p>
                    Four independent controls protect what the model alone cannot:
                    transcript admission, model traffic, customer actions, and spoken facts.
                  </p>
                </div>
              </div>
              <div className="gr-command-impact">
                <strong>{fired}</strong>
                <span>turn outcomes changed</span>
                <small>{interventionRate}% of {checks} evidenced decisions</small>
              </div>
            </section>

            <section className="gr-journey-section">
              <div className="gr-section-intro">
                <div>
                  <div className="gr-impact-eyebrow">End-to-end control</div>
                  <h2>One voice turn, four trust decisions</h2>
                </div>
                <span className={`gr-contract-pill ${gatewayHealthy ? "is-ready" : "is-review"}`}>
                  {gatewayHealthy ? "Gateway contract verified" : "Gateway drift"}
                </span>
              </div>
              <div className="gr-journey">
                {journey.map((stage, index) => (
                  <div className="gr-journey-stage" key={stage.step}>
                    <div className="gr-journey-top">
                      <span className="gr-journey-number">{index + 1}</span>
                      <span className="gr-journey-boundary">{stage.boundary}</span>
                    </div>
                    <strong>{stage.step}</strong>
                    <h3>{stage.promise}</h3>
                    <p>{stage.proof}</p>
                    <span className="gr-journey-owner">{stage.owner}</span>
                  </div>
                ))}
              </div>
            </section>

            <section className="gr-impact-board">
              <div className="gr-section-intro">
                <div>
                  <div className="gr-impact-eyebrow">Observed impact</div>
                  <h2>Where guardrails changed the outcome</h2>
                </div>
                <span className="gr-window-label">Current evidence window</span>
              </div>
              <div className="gr-impact-rows">
                {impactGroups.map((group) => (
                  <div className="gr-impact-row" key={group.key}>
                    <div className="gr-impact-row-label">
                      <strong>{group.label}</strong>
                      <span>{group.description}</span>
                    </div>
                    <div className="gr-impact-track">
                      <span style={{ width: `${fired ? Math.max(2, (group.count / fired) * 100) : 0}%` }} />
                    </div>
                    <strong className="gr-impact-row-count">{group.count}</strong>
                  </div>
                ))}
              </div>
            </section>

            <div className="gr-section-intro is-recent">
              <div>
                <div className="gr-impact-eyebrow">Protected moments</div>
                <h2>Risk → decision → customer outcome</h2>
              </div>
            </div>
            {rollup.recent_fired.length > 0 ? (
                <div className="gr-causal-list">
                  {rollup.recent_fired.slice(0, 5).map((row, index) => {
                    const copy = interventionCopy(row, catalog.find((policy) => policy.guard_id === row.guard_id)?.title);
                    return (
                    <button
                      key={`${row.trace_id}-${row.guard_id}-${index}`}
                      className="gr-causal-row"
                      disabled={!row.trace_id}
                      onClick={() => {
                        if (row.trace_id) window.location.hash = `#/traces?trace=${row.trace_id}`;
                      }}
                    >
                      <span className="gr-causal-cell is-risk">
                        <small>Risk detected</small>
                        <strong>{copy.risk}</strong>
                      </span>
                      <span className="gr-causal-arrow">→</span>
                      <span className="gr-causal-cell is-decision">
                        <small>Guardrail decision</small>
                        <strong>{copy.decision}</strong>
                      </span>
                      <span className="gr-causal-arrow">→</span>
                      <span className="gr-causal-cell is-outcome">
                        <small>Customer outcome</small>
                        <strong>{copy.outcome}</strong>
                      </span>
                      <span className="gr-causal-meta">
                        {row.language || "unknown"} · turn {row.turn_id ?? "—"}
                      </span>
                    </button>
                    );
                  })}
                </div>
            ) : (
              <div className="gr-quiet-state">No guardrail actions in the selected evidence window.</div>
            )}

            <details className="gr-details">
              <summary>Operator details</summary>
              <div className="gr-details-body">
                <div className="gr-detail-summary">
                  Manifest v{gateway.policy_manifest.version} · {catalog.length} policies ·{" "}
                  {rollup.turns_with_roster} traced turns · {rollup.standalone_events ?? 0} standalone events ·{" "}
                  {errors} evaluation errors
                </div>

                <div className="tv-section-title">Governed model services</div>
                <div className="gr-gateway-services">
                  {gateway.services.map((service) => {
                    const rate = service.rate_limits[0];
                    const traffic = service.traffic_7d;
                    return (
                      <div className="gr-gateway-service" key={service.service}>
                        <div className="gr-gateway-service-head">
                          <code>{service.service}</code>
                          <span>{service.roles.join(" · ")}</span>
                        </div>
                        <div className="gr-gateway-route">routes to {service.destination}</div>
                        <div className="gr-gateway-metrics">
                          <span><strong>{rate?.requests ?? "—"}</strong> QPM</span>
                          <span><strong>{rate?.tokens ?? "—"}</strong> TPM</span>
                          <span><strong>{traffic?.requests ?? 0}</strong> requests · 7d</span>
                          <span><strong>{traffic?.errors ?? 0}</strong> errors · 7d</span>
                          <span><strong>{traffic?.p95_latency_ms ?? "—"}</strong> ms p95</span>
                        </div>
                        <div className="gr-gateway-table">inference table: <code>{service.inference_table ?? "pending"}</code></div>
                        <div className="gr-gateway-policies">
                          {service.required_policies.map((policy) => (
                            <code key={policy.policy_id}>
                              {policy.policy_id} · {policy.phases.join("+")} · rank {policy.rank}
                              {" · "}{policy.mode ?? "enforce"}
                              {policy.action ? ` · ${policy.action}` : ""}
                              {policy.categories?.length ? ` · ${policy.categories.length} categories` : ""}
                              {" · "}{policy.deployment_state?.replace(/_/g, " ") ?? "unverified"}
                            </code>
                          ))}
                        </div>
                      </div>
                    );
                  })}
                </div>

                <div className="tv-section-title">Active policy evidence</div>
                {activeGroups.map(([key, policies]) => (
                  <div className="gr-subgroup" key={key}>
                    <div className="gr-subhead">{policies[0].group}</div>
                    <div className="gr-guards">
                      {policies.map((policy) => (
                        <PolicyRow key={policy.guard_id} policy={policy} live={liveById.get(policy.guard_id)} />
                      ))}
                    </div>
                  </div>
                ))}

                {uncatalogued.length > 0 && (
                  <div className="tv-callout">
                    Manifest drift: {uncatalogued.map((guard) => guard.guard_id).join(", ")}
                  </div>
                )}

                {plannedPolicies.length > 0 && (
                  <details className="gr-details is-nested">
                    <summary>Planned policies ({plannedPolicies.length})</summary>
                    <div className="gr-guards">
                      {plannedPolicies.map((policy) => (
                        <PolicyRow key={policy.guard_id} policy={policy} live={liveById.get(policy.guard_id)} />
                      ))}
                    </div>
                  </details>
                )}
              </div>
            </details>
          </>
        )}
      </div>
    </div>
  );
}
