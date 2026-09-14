/**
 * Guardrails view driven entirely by config/guardrails.yaml through the API.
 * React owns presentation only; policy identity, status, assignment, and
 * deployment state are runtime data.
 */
import { useCallback, useEffect, useMemo, useState } from "react";

import {
  api,
  type GatewayInsights,
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

  return (
    <div className="tv-root">
      <div className="tv-header">
        <div className="tv-title"><span>Genie</span> Voice · Guardrails</div>
        <div className="tv-header-spacer" />
        <div className="tv-filters">
          <button className="tv-btn ghost" onClick={() => (window.location.hash = "#/traces")}>Trace Explorer</button>
          <button className="tv-btn ghost" onClick={() => (window.location.hash = "#/")}>← Home</button>
          <button className="tv-btn" onClick={() => load()}>Refresh</button>
        </div>
      </div>

      <div className="gr-body">
        {loading && !rollup && <div className="tv-empty">Loading policy manifest and evidence…</div>}
        {error && <div className="tv-empty">Error: {error}</div>}
        {rollup && gateway && (
          <>
            <div className="gr-headline">
              <div className="gr-stat">
                <div className="gr-stat-value">{catalog.length}</div>
                <div className="gr-stat-label">manifest policies</div>
                <div className="gr-stat-sub">version {gateway.policy_manifest.version}</div>
              </div>
              <div className="gr-stat">
                <div className="gr-stat-value">{checks}</div>
                <div className="gr-stat-label">decisions recorded</div>
                <div className="gr-stat-sub">across {rollup.turns_with_roster} traced turns</div>
              </div>
              <div className={`gr-stat${fired ? " is-hit" : " is-clean"}`}>
                <div className="gr-stat-value">{fired}</div>
                <div className="gr-stat-label">acted</div>
                <div className="gr-stat-sub">blocked, changed, or declined</div>
              </div>
              <div className="gr-stat">
                <div className="gr-stat-value">{gateway.services.length}</div>
                <div className="gr-stat-label">governed FM services</div>
                <div className="gr-stat-sub">stable policy assignments</div>
              </div>
            </div>

            <div className="tv-section-title">Mandatory voice boundaries</div>
            <div className="gr-protection-grid is-flow">
              {Object.entries(gateway.policy_manifest.assignments.boundaries).map(([boundary, assignment]) => {
                const bundle = gateway.policy_manifest.bundles[assignment.bundle];
                return (
                  <div className="gr-protection-card is-enforced" key={boundary}>
                    <div className="gr-protection-head">
                      <span className="gr-guard-name">{bundle?.title ?? boundary}</span>
                      <span className="gr-stage">{boundary.replace("_", " ")}</span>
                      <span className="gr-protection-status is-enforced">enforced</span>
                    </div>
                    <div className="gr-guard-blurb">
                      {assignment.resources.join(", ")} · {(bundle?.policies ?? []).length} policies
                    </div>
                  </div>
                );
              })}
            </div>

            <div className="tv-section-title">Governed foundation-model services</div>
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
                          {" · "}{policy.deployment_state?.replace(/_/g, " ") ?? "unverified"}
                        </code>
                      ))}
                    </div>
                    <div className={`gr-protection-status is-${service.policy_deployment_state === "external_action_required" ? "ui-managed" : "enforced"}`}>
                      {service.policy_deployment_state.replace(/_/g, " ")}
                    </div>
                  </div>
                );
              })}
            </div>

            {groups.map(([key, policies]) => (
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
              <div className="gr-subgroup">
                <div className="gr-subhead">Uncatalogued decisions — manifest drift</div>
                <div className="tv-callout">
                  {uncatalogued.map((guard) => guard.guard_id).join(", ")}
                </div>
              </div>
            )}

            {rollup.recent_fired.length > 0 && (
              <>
                <div className="tv-section-title">Recent actions</div>
                <div className="gr-fired">
                  {rollup.recent_fired.map((row, index) => (
                    <button
                      key={`${row.trace_id}-${row.guard_id}-${index}`}
                      className="gr-fired-row"
                      onClick={() => (window.location.hash = `#/traces?trace=${row.trace_id}`)}
                    >
                      <code className="gr-fired-guard">{row.guard_id}</code>
                      <span className="gr-fired-reason">{row.reason || "—"}</span>
                      <span className="gr-fired-meta">{row.language || "unknown"} · turn {row.turn_id ?? "—"}</span>
                    </button>
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
