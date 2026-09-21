import { useCallback, useEffect, useState } from "react";
import {
  api,
  type GatewayInsights,
  type GatewayServiceInsight,
  type GuardRollup,
} from "../api/client";
import "../styles/traces.css";

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

function statusClass(status?: string): string {
  return status === "verified" ? "is-verified" : status === "partial" ? "is-partial" : "is-unavailable";
}

function Metric({ label, value, detail }: { label: string; value: string; detail?: string }) {
  return (
    <div className="gr-control-metric">
      <div className="gr-control-metric-label">{label}</div>
      <div className="gr-control-metric-value">{value}</div>
      {detail && <div className="gr-control-metric-detail">{detail}</div>}
    </div>
  );
}

type SpeechEndpoint = NonNullable<GatewayInsights["speech_endpoints"]>[number];

function SpeechModelCard({ model }: { model: SpeechEndpoint }) {
  const isTts = model.model_role === "tts";
  const inference = model.inference_table;
  const inferenceTable = inference?.enabled
    ? `${inference.catalog_name}.${inference.schema_name}.${inference.table_name_prefix}_payload`
    : null;
  return (
    <article className="gr-speech-model-card">
      <div className="gr-speech-model-head">
        <div>
          <div className="gr-control-eyebrow">{isTts ? "Text to speech" : "Speech to text"} · Gateway-enabled serving endpoint</div>
          <h3>{model.model_name}</h3>
        </div>
        <span className={`gr-control-status ${statusClass(model.provenance_status)}`}>
          <span className="gr-control-status-dot" />
          {model.provenance_status}
        </span>
      </div>
      <code className="gr-speech-endpoint" title={model.endpoint}>{model.endpoint}</code>
      <div className="gr-speech-metrics">
        <Metric label="Conversation calls" value={fmt(model.requests)} detail={`${fmt(model.trace_count)} matched traces`} />
        <Metric label="Errors" value={fmt(model.errors)} detail={model.errors ? "Inspect traces" : "No failed calls"} />
        <Metric label="Average latency" value={`${fmt(model.avg_latency_ms)} ms`} detail={`P95 ${fmt(model.p95_latency_ms)} ms`} />
        <Metric
          label={isTts ? "Time to first audio" : "Execution plane"}
          value={isTts ? `${fmt(model.avg_ttfb_ms)} ms` : "GPU endpoint"}
          detail={isTts ? `P95 ${fmt(model.p95_ttfb_ms)} ms · generation ${fmt(model.avg_generation_ms)} ms` : "Model Serving + AI Gateway"}
        />
      </div>
      <div className="gr-speech-coverage">
        <span><strong>Surfaces</strong> {model.surfaces.join(", ") || "unavailable"}</span>
        <span><strong>Profiles</strong> {model.profiles.join(", ") || "unavailable"}</span>
      </div>
      <div className="gr-speech-telemetry-note">
        <span>{inferenceTable ? "AI Gateway enabled" : "Trace-derived telemetry"}</span>
        <p>
          {inferenceTable
            ? <>Inference table <code>{inferenceTable}</code>. Rate limits, usage tracking, fallback, and chat guardrails are unsupported for ResponsesAgent endpoints.</>
            : <>Inference table is not configured. No Gateway telemetry is available for this endpoint.</>}
        </p>
      </div>
    </article>
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
    <article className="gr-control-service">
      <div className="gr-control-service-head">
        <div className="gr-control-service-identity">
          <div className="gr-control-eyebrow">{service.key} · Unity AI Gateway</div>
          <code className="gr-control-resource" title={service.service}>{service.service}</code>
          <div className="gr-control-subline">
            <span>Destination <code>{service.destination}</code></span>
            <span>Roles {service.roles.join(", ")}</span>
          </div>
        </div>
        <span className={`gr-control-status ${statusClass(service.provenance?.status)}`}>
          <span className="gr-control-status-dot" />
          {service.provenance?.status ?? "unavailable"} provenance
        </span>
      </div>

      <div className="gr-control-metrics">
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

      <div className="gr-control-table-wrap">
        <table className="gr-control-table gr-policy-table">
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
                  <td><code title={policy.name}>{policy.name ?? "unnamed"}</code></td>
                  <td><code title={policy.handler}>{policy.handler ?? "—"}</code></td>
                  <td>{policy.rank ?? "—"}</td>
                  <td>{options.phases ?? "—"}</td>
                  <td>{options.dry_run === "true" ? "log" : "enforce"}{options.action ? ` / ${options.action}` : ""}</td>
                  <td className="gr-control-wrap">{options.model_service ?? options.categories ?? "—"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="gr-control-footnote">
        Inference table <code>{service.inference_table ?? "unavailable"}</code>
        <span>Attachment state <strong>{service.policy_deployment_state}</strong></span>
      </div>
    </article>
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
  const speechModels = speech.filter((item) => item.model_role === "stt" || item.model_role === "tts");
  const unclassifiedSpeechCalls = speech
    .filter((item) => item.model_role !== "stt" && item.model_role !== "tts")
    .reduce((total, item) => total + item.requests, 0);

  return (
    <div className="tv-root gr-control-root">
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

      <main className="gr-control-body">
        {loading && !gateway && <div className="tv-empty">Loading configuration and conversation evidence…</div>}
        {error && <div className="gr-control-error">Unable to load control-plane evidence: {error}</div>}
        {gateway && rollup && (
          <>
            <section className={`gr-control-hero ${statusClass(provenance?.status)}`}>
              <div className="gr-control-hero-copy">
                <div className="gr-control-eyebrow">Conversation evidence scope</div>
                <div className="gr-control-hero-title">
                  <span className="gr-control-shield">✓</span>
                  <div>
                    <h1>{provenance?.status ?? "unavailable"} provenance</h1>
                    <p>
                      Metrics include only this app identity with <code>traffic_class=conversation</code>.
                      Warmups, probes, benchmarks, and legacy traffic remain excluded.
                    </p>
                  </div>
                </div>
              </div>
              <div className="gr-control-hero-metrics">
                <Metric label="Conversation traces" value={fmt(provenance?.conversation_trace_count)} />
                <Metric label="Legacy traces excluded" value={fmt(provenance?.legacy_unclassified_trace_count)} />
                <Metric label="Conversation checks" value={fmt(rollup.checks)} detail={`${fmt(rollup.turns_with_roster)} turns with roster`} />
                <Metric label="Standalone events" value={fmt(rollup.standalone_events)} detail="Outside conversation KPIs" />
              </div>
            </section>

            <section className="gr-gateway-plane">
              <div className="gr-gateway-plane-head">
                <div>
                  <div className="gr-control-eyebrow">Databricks governance plane</div>
                  <h2>AI Gateway</h2>
                  <p>
                    One control plane, shown by resource type because supported capabilities and telemetry differ.
                  </p>
                </div>
                <div className="gr-gateway-capability-summary">
                  <div>
                    <strong>{speechModels.length}</strong>
                    <span>Serving endpoints</span>
                    <small>Inference tables</small>
                  </div>
                  <div>
                    <strong>{gateway.services.length}</strong>
                    <span>Model services</span>
                    <small>Policies · limits · tokens · tables</small>
                  </div>
                </div>
              </div>

              <div className="gr-gateway-subsection">
                <div className="gr-gateway-subsection-head">
                  <div>
                    <div className="gr-control-eyebrow">Gateway-enabled custom models</div>
                    <h3>Serving endpoint Gateway</h3>
                    <p>Qwen3-ASR and VoxCPM2 remain GPU Model Serving endpoints with AI Gateway inference tables attached.</p>
                  </div>
                  <span className="gr-gateway-capability-badge">ResponsesAgent · inference tables supported</span>
                </div>
                {speechModels.length > 0 ? (
                  <div className="gr-speech-model-grid">
                    {speechModels.map((model) => (
                      <SpeechModelCard key={`${model.endpoint}-${model.model_role}`} model={model} />
                    ))}
                  </div>
                ) : (
                  <div className="gr-control-empty">No tagged conversation speech calls are persisted yet.</div>
                )}
                {unclassifiedSpeechCalls > 0 && (
                  <div className="gr-speech-unclassified">
                    {fmt(unclassifiedSpeechCalls)} legacy serving call(s) have no model role and remain excluded from these model cards.
                  </div>
                )}
              </div>

              <div className="gr-gateway-subsection">
                <div className="gr-gateway-subsection-head">
                  <div>
                    <div className="gr-control-eyebrow">Gateway-routed foundation models</div>
                    <h3>Unity Catalog model services</h3>
                    <p>App-owned routes expose conversation token usage, limits, attached policies, inference tables, and trace matching.</p>
                  </div>
                  <span className="gr-gateway-capability-badge">Trailing 7 days · conversation only</span>
                </div>
                <div className="gr-control-service-list">
                  {gateway.services.map((service) => <ServicePanel key={service.key} service={service} />)}
                </div>
              </div>
            </section>

            <section className="gr-control-panel">
              <div className="gr-control-section-head is-compact">
                <div>
                  <div className="gr-control-eyebrow">Trace-linked evidence</div>
                  <h2>Recent model-service Gateway events</h2>
                </div>
                <span className="gr-control-section-meta">{events.length} requests</span>
              </div>
              {events.length > 0 ? (
                <div className="gr-control-table-wrap">
                  <table className="gr-control-table gr-events-table">
                    <thead><tr><th>Time</th><th>Service / role</th><th>Surface</th><th>Status</th><th>Latency</th><th>Trace</th></tr></thead>
                    <tbody>
                      {events.map((event, index) => (
                        <tr key={`${event.request_id}-${index}`}>
                          <td>{event.event_time ? new Date(event.event_time).toLocaleString() : "—"}</td>
                          <td><strong>{event.service_key}</strong><span>{event.model_role ?? "unknown"}</span></td>
                          <td>{event.surface ?? "—"}<span>{event.profile ?? "—"}</span></td>
                          <td><span className={`gr-http-status ${n(event.status_code) < 400 ? "is-ok" : "is-error"}`}>{event.status_code ?? "—"}</span></td>
                          <td>{fmt(event.latency_ms)} ms</td>
                          <td>
                            {event.trace_id ? (
                              <button
                                className="gr-trace-link"
                                disabled={!event.trace_matched}
                                onClick={() => (window.location.hash = `#/traces?trace=${event.trace_id}`)}
                              >
                                {event.trace_matched ? String(event.trace_id).slice(0, 8) : "unmatched"}
                              </button>
                            ) : "missing"}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : <div className="gr-control-empty">No verified Gateway events in the current window.</div>}
            </section>

            <details className="gr-control-disclosure">
              <summary>
                <span>
                  <strong>Technical coverage</strong>
                  <small>Complete model inventory and every application ingress</small>
                </span>
                <span className="gr-control-disclosure-count">{models.length} models · {pages.length} surfaces</span>
              </summary>
              <div className="gr-control-disclosure-body">
                <section>
                  <h3>Model and managed-service inventory</h3>
                  <div className="gr-control-table-wrap">
                    <table className="gr-control-table gr-inventory-table">
                      <thead><tr><th>Model/service</th><th>Plane</th><th>Roles</th><th>Resource</th><th>Evidence source</th></tr></thead>
                      <tbody>
                        {models.map((item) => (
                          <tr key={item.id}>
                            <td><strong>{item.name}</strong><code>{item.id}</code></td>
                            <td>{item.plane}</td>
                            <td>{item.roles.join(", ")}</td>
                            <td><code title={item.resource}>{item.resource}</code></td>
                            <td>{item.telemetry}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </section>
                <section>
                  <h3>Page and ingress coverage</h3>
                  <div className="gr-surface-grid">
                    {pages.map((item) => (
                      <article className="gr-surface-card" key={item.surface}>
                        <div className="gr-surface-head">
                          <code>{item.surface}</code>
                          <span>{item.traffic_class}</span>
                        </div>
                        <div className="gr-surface-profile">{item.profile ?? "operator / read-only"}</div>
                        <dl>
                          <dt>Models</dt><dd>{item.models.join(", ") || "none at page time"}</dd>
                          <dt>Managed</dt><dd>{item.managed_services.join(", ") || "none"}</dd>
                        </dl>
                      </article>
                    ))}
                  </div>
                </section>
              </div>
            </details>

            <section className="gr-control-panel gr-standalone-panel">
              <div className="gr-control-section-head is-compact">
                <div>
                  <div className="gr-control-eyebrow">Excluded evidence plane</div>
                  <h2>Standalone service and probe decisions</h2>
                </div>
                <span className="gr-control-section-meta">{fmt(rollup.standalone_events)} events</span>
              </div>
              {(rollup.standalone_recent ?? []).length > 0 ? (
                <div className="gr-control-table-wrap">
                  <table className="gr-control-table">
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
              ) : (
                <div className="gr-control-empty is-inline">
                  <span>✓</span>
                  No standalone decisions recorded. This evidence plane remains outside conversation totals.
                </div>
              )}
            </section>
          </>
        )}
      </main>
    </div>
  );
}
