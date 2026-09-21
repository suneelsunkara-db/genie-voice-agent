import { CSSProperties, useCallback, useEffect, useState } from "react";
import {
  api,
  type GatewayInsights,
  type GatewayServiceInsight,
  type GuardRollup,
} from "../api/client";
import "../styles/traces.css";

const panel: CSSProperties = {
  border: "1px solid rgba(148,163,184,.22)",
  borderRadius: 12,
  padding: 16,
  background: "rgba(15,23,42,.56)",
};

const grid: CSSProperties = {
  display: "grid",
  gridTemplateColumns: "repeat(auto-fit,minmax(150px,1fr))",
  gap: 10,
};

function n(value: unknown): number {
  const parsed = Number(value ?? 0);
  return Number.isFinite(parsed) ? parsed : 0;
}

function fmt(value: unknown): string {
  const valueNumber = n(value);
  return new Intl.NumberFormat("en-US", { maximumFractionDigits: 1 }).format(valueNumber);
}

function pct(value: number, limit: number): string {
  if (!limit) return "unavailable";
  return `${Math.round((value / limit) * 1000) / 10}%`;
}

function statusColor(status?: string): string {
  if (status === "verified") return "#5ee6a8";
  if (status === "partial") return "#f7c96b";
  return "#ff8f8f";
}

function Metric({ label, value, detail }: { label: string; value: string; detail?: string }) {
  return (
    <div style={{ ...panel, padding: 12 }}>
      <div style={{ color: "#8fa4c2", fontSize: 11, textTransform: "uppercase", letterSpacing: ".07em" }}>{label}</div>
      <div style={{ color: "#f5f8ff", fontSize: 22, fontWeight: 750, marginTop: 5 }}>{value}</div>
      {detail && <div style={{ color: "#8fa4c2", fontSize: 11, marginTop: 3 }}>{detail}</div>}
    </div>
  );
}

function ServicePanel({ service }: { service: GatewayServiceInsight }) {
  const traffic = service.traffic_7d ?? {};
  const all = service.traffic_all_7d ?? {};
  const rate = service.rate_limits[0] ?? {};
  const rpmLimit = n(rate.requests);
  const tpmLimit = n(rate.tokens);
  const attached = service.attached_policies ?? [];

  return (
    <section style={panel}>
      <div style={{ display: "flex", gap: 12, alignItems: "flex-start", flexWrap: "wrap" }}>
        <div style={{ flex: 1, minWidth: 280 }}>
          <div style={{ color: "#8fa4c2", fontSize: 11, textTransform: "uppercase" }}>{service.key}</div>
          <code style={{ color: "#e8eef9", fontSize: 13 }}>{service.service}</code>
          <div style={{ color: "#9eb0c9", marginTop: 5 }}>
            destination <code>{service.destination}</code> · roles {service.roles.join(", ")}
          </div>
        </div>
        <strong style={{ color: statusColor(service.provenance?.status) }}>
          {service.provenance?.status ?? "unavailable"} provenance
        </strong>
      </div>

      <div style={{ ...grid, marginTop: 14 }}>
        <Metric label="Conversation requests · 7d" value={fmt(traffic.requests)} detail={`${fmt(all.requests)} all-service requests`} />
        <Metric label="Input tokens" value={fmt(traffic.input_tokens)} detail={`avg ${fmt(traffic.avg_input_tokens)} / request`} />
        <Metric label="Output tokens" value={fmt(traffic.output_tokens)} detail={`avg ${fmt(traffic.avg_output_tokens)} / request`} />
        <Metric label="Peak RPM" value={fmt(traffic.peak_rpm)} detail={`${pct(n(traffic.peak_rpm), rpmLimit)} of ${fmt(rpmLimit)} limit`} />
        <Metric label="Peak TPM" value={fmt(traffic.peak_tpm)} detail={`${pct(n(traffic.peak_tpm), tpmLimit)} of ${fmt(tpmLimit)} limit`} />
        <Metric label="P95 latency" value={`${fmt(traffic.p95_latency_ms)} ms`} detail={`P95 TTFT ${fmt(traffic.p95_ttft_ms)} ms`} />
        <Metric label="Errors / 429" value={`${fmt(traffic.errors)} / ${fmt(traffic.rate_limited)}`} detail={`${fmt(traffic.policy_envelopes)} policy envelopes`} />
        <Metric
          label="Trace matching"
          value={`${fmt(service.provenance?.matched_trace_ids)} matched`}
          detail={`${service.provenance?.unmatched_trace_ids?.length ?? 0} unmatched · ${fmt(service.provenance?.incomplete_requests)} incomplete`}
        />
      </div>

      <div style={{ marginTop: 14, overflowX: "auto" }}>
        <table className="tv-table">
          <thead>
            <tr>
              <th>Attached policy</th><th>Handler</th><th>Rank</th><th>Phase</th><th>Mode/action</th><th>Evaluator/categories</th>
            </tr>
          </thead>
          <tbody>
            {attached.map((policy, index) => {
              const options = policy.options ?? {};
              return (
                <tr key={`${policy.name}-${index}`}>
                  <td><code>{policy.name ?? "unnamed"}</code></td>
                  <td><code>{policy.handler ?? "—"}</code></td>
                  <td>{policy.rank ?? "—"}</td>
                  <td>{options.phases ?? "—"}</td>
                  <td>{options.dry_run === "true" ? "log" : "enforce"}{options.action ? ` / ${options.action}` : ""}</td>
                  <td>{options.model_service ?? options.categories ?? "—"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div style={{ color: "#8195b2", fontSize: 11, marginTop: 10 }}>
        inference table <code>{service.inference_table ?? "unavailable"}</code> · attachment state {service.policy_deployment_state}
      </div>
    </section>
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
      .then(([guards, insights]) => {
        setRollup(guards);
        setGateway(insights);
        setError(null);
      })
      .catch((reason) => setError(reason instanceof Error ? reason.message : "Failed to load"))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    load();
    const timer = window.setInterval(() => load(true), 15_000);
    return () => window.clearInterval(timer);
  }, [load]);

  const provenance = gateway?.provenance;
  const models = gateway?.model_inventory ?? [];
  const pages = gateway?.page_coverage ?? [];
  const events = gateway?.recent_events ?? [];
  const speech = gateway?.speech_endpoints ?? [];

  return (
    <div className="tv-page">
      <div className="tv-header">
        <div className="tv-title"><span>Genie</span> Voice · Model control plane</div>
        <div className="tv-header-spacer" />
        <div className="tv-filters">
          <button className="tv-btn ghost" onClick={() => (window.location.hash = "#/traces")}>Trace Explorer</button>
          <button className="tv-btn ghost" onClick={() => (window.location.hash = "#/setup")}>Setup</button>
          <button className="tv-btn ghost" onClick={() => (window.location.hash = "#/")}>← Home</button>
          <button className="tv-btn" onClick={() => load()}>Refresh</button>
        </div>
      </div>

      <main className="tv-body" style={{ display: "grid", gap: 18 }}>
        {loading && !gateway && <div className="tv-empty">Loading configuration and conversation evidence…</div>}
        {error && <div className="tv-empty">Error: {error}</div>}
        {gateway && rollup && (
          <>
            <section style={{ ...panel, borderColor: statusColor(provenance?.status) }}>
              <div style={{ display: "flex", justifyContent: "space-between", gap: 16, flexWrap: "wrap" }}>
                <div>
                  <div style={{ color: "#8fa4c2", fontSize: 11, textTransform: "uppercase" }}>Evidence scope</div>
                  <h1 style={{ margin: "5px 0 4px", color: "#f5f8ff", fontSize: 26 }}>
                    {provenance?.status ?? "unavailable"} conversation provenance
                  </h1>
                  <div style={{ color: "#9eb0c9" }}>
                    Only requests tagged as this app, conversation traffic, and the current app identity are counted below.
                  </div>
                </div>
                <div style={{ ...grid, minWidth: 420, flex: 1 }}>
                  <Metric label="Conversation traces" value={fmt(provenance?.conversation_trace_count)} />
                  <Metric label="Legacy traces excluded" value={fmt(provenance?.legacy_unclassified_trace_count)} />
                  <Metric label="Conversation checks" value={fmt(rollup.checks)} detail={`${fmt(rollup.turns_with_roster)} turns with roster`} />
                  <Metric label="Standalone events" value={fmt(rollup.standalone_events)} detail="Excluded from conversation KPIs" />
                </div>
              </div>
            </section>

            <section>
              <h2 style={{ color: "#f5f8ff", margin: "0 0 10px" }}>Unity AI Gateway services</h2>
              <div style={{ display: "grid", gap: 14 }}>
                {gateway.services.map((service) => <ServicePanel key={service.key} service={service} />)}
              </div>
            </section>

            <section style={panel}>
              <h2 style={{ color: "#f5f8ff", marginTop: 0 }}>Speech endpoint traffic from conversation traces</h2>
              <div style={grid}>
                {speech.map((item) => (
                  <Metric
                    key={`${item.endpoint}-${item.model_role}`}
                    label={item.model_role}
                    value={`${fmt(item.requests)} calls`}
                    detail={`${item.endpoint} · ${fmt(item.errors)} errors`}
                  />
                ))}
              </div>
              {speech.length === 0 && <div className="tv-empty">No newly tagged conversation speech calls are persisted yet.</div>}
            </section>

            <section style={panel}>
              <h2 style={{ color: "#f5f8ff", marginTop: 0 }}>Complete model and managed-service inventory</h2>
              <div style={{ overflowX: "auto" }}>
                <table className="tv-table">
                  <thead><tr><th>Model/service</th><th>Plane</th><th>Roles</th><th>Resource</th><th>Evidence source</th></tr></thead>
                  <tbody>
                    {models.map((item) => (
                      <tr key={item.id}>
                        <td><strong>{item.name}</strong><br /><code>{item.id}</code></td>
                        <td>{item.plane}</td>
                        <td>{item.roles.join(", ")}</td>
                        <td><code>{item.resource}</code></td>
                        <td>{item.telemetry}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>

            <section style={panel}>
              <h2 style={{ color: "#f5f8ff", marginTop: 0 }}>Page and ingress coverage</h2>
              <div style={{ overflowX: "auto" }}>
                <table className="tv-table">
                  <thead><tr><th>Surface</th><th>Profile</th><th>Traffic class</th><th>Models</th><th>Managed services/data</th></tr></thead>
                  <tbody>
                    {pages.map((item) => (
                      <tr key={item.surface}>
                        <td><code>{item.surface}</code></td>
                        <td>{item.profile ?? "—"}</td>
                        <td>{item.traffic_class}</td>
                        <td>{item.models.join(", ") || "none at page time"}</td>
                        <td>{item.managed_services.join(", ") || "—"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>

            <section style={panel}>
              <h2 style={{ color: "#f5f8ff", marginTop: 0 }}>Recent verified conversation Gateway events</h2>
              <div style={{ overflowX: "auto" }}>
                <table className="tv-table">
                  <thead><tr><th>Time</th><th>Service / role</th><th>Page / profile</th><th>Status</th><th>Latency</th><th>Trace</th></tr></thead>
                  <tbody>
                    {events.map((event, index) => (
                      <tr key={`${event.request_id}-${index}`}>
                        <td>{event.event_time ?? "—"}</td>
                        <td>{event.service_key} / {event.model_role ?? "unknown"}</td>
                        <td>{event.surface ?? "—"} / {event.profile ?? "—"}</td>
                        <td>{event.status_code ?? "—"}</td>
                        <td>{fmt(event.latency_ms)} ms</td>
                        <td>
                          {event.trace_id ? (
                            <button
                              className="tv-btn ghost"
                              disabled={!event.trace_matched}
                              onClick={() => (window.location.hash = `#/traces?trace=${event.trace_id}`)}
                            >
                              {event.trace_matched ? String(event.trace_id).slice(0, 10) : "unmatched"}
                            </button>
                          ) : "missing"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {events.length === 0 && <div className="tv-empty">No verified conversation Gateway events in the current seven-day window.</div>}
            </section>

            <section style={panel}>
              <h2 style={{ color: "#f5f8ff", marginTop: 0 }}>Standalone service and probe decisions</h2>
              <div style={{ color: "#9eb0c9", marginBottom: 10 }}>
                These records are deliberately outside conversation totals.
              </div>
              <div style={{ overflowX: "auto" }}>
                <table className="tv-table">
                  <thead><tr><th>Time</th><th>Context</th><th>Guard</th><th>Outcome</th><th>Phase</th><th>Resource</th></tr></thead>
                  <tbody>
                    {(rollup.standalone_recent ?? []).map((event, index) => (
                      <tr key={String(event.event_id ?? index)}>
                        <td>{String(event.occurred_at ?? "—")}</td>
                        <td>{String(event.context ?? "unknown")}</td>
                        <td><code>{String(event.guard_id ?? "unknown")}</code></td>
                        <td>{String(event.outcome ?? "unknown")}</td>
                        <td>{String(event.phase ?? "—")}</td>
                        <td><code>{String(event.resource ?? "—")}</code></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {(rollup.standalone_recent ?? []).length === 0 && <div className="tv-empty">No standalone events recorded.</div>}
            </section>
          </>
        )}
      </main>
    </div>
  );
}
