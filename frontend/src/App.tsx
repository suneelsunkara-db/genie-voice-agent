import { useEffect, useState } from "react";
import { api, CustomerWithIssue, StatusResponse } from "./api/client";
import { POLL_INTERVAL_MS } from "./config";
import { ASRBenchmarkPage } from "./components/ASRBenchmarkPage";
import { CallList } from "./components/CallList";
import databricksLogo from "./assets/databricks-logo.png";
import genieLogo from "./assets/genie-logo.png";

export default function App() {
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [customers, setCustomers] = useState<CustomerWithIssue[]>([]);
  const [customersLoading, setCustomersLoading] = useState(true);
  const [customersErr, setCustomersErr] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [page, setPage] = useState(() => window.location.hash || "#/");

  useEffect(() => {
    let active = true;
    const loadStatus = async () => {
      try {
        const s = await api.status();
        if (active) {
          setStatus(s);
          setError(null);
        }
      } catch (e) {
        if (active) setError(String(e));
      }
    };
    // Single source of truth for "customers with issues" - both the header pill
    // and the CallList read this, so we poll it once here instead of twice.
    const loadIssues = async () => {
      try {
        const issues = await api.customersWithIssues();
        if (!active) return;
        setCustomers(issues.customers ?? []);
        setCustomersErr(null);
      } catch (e) {
        if (active) setCustomersErr(e instanceof Error ? e.message : "failed");
      } finally {
        if (active) setCustomersLoading(false);
      }
    };
    const tick = () => {
      void loadStatus();
      void loadIssues();
    };
    tick();
    const id = setInterval(tick, POLL_INTERVAL_MS);
    return () => {
      active = false;
      clearInterval(id);
    };
  }, []);

  useEffect(() => {
    const onHashChange = () => setPage(window.location.hash || "#/");
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, []);

  const showBenchmark = page === "#/asr-benchmark";

  return (
    <div className="app">
      <header className="hero-shell">
        <div className="hero-topbar">
          <div className="hero-brand-row">
            <img className="hero-logo dbx-full" src={databricksLogo} alt="Databricks" />
            <img className="hero-logo genie-full" src={genieLogo} alt="Genie" />
            <span className="brand-chip voice">Voice Use Cases</span>
          </div>
          <div className="hero-meta">
            <a className="meta-pill nav-pill" href="#/">
              cockpit
            </a>
            <a className="meta-pill nav-pill" href="#/asr-benchmark">
              ASR benchmark
            </a>
            <div className="meta-pill">runtime: {status?.mode ?? "…"}</div>
            <div className="meta-pill">stt: {status?.stt_provider ?? "…"}</div>
            <div className="meta-pill">
              customers with issues:{" "}
              {customersLoading && !customers.length ? "…" : customers.length}
            </div>
          </div>
        </div>

        <div className="hero-content">
          <div className="eyebrow">Databricks Genie Voice Agent</div>
          <h1>Genie-Powered Voice Agent Experience</h1>
          <p>
            Voice conversations are transcribed by the configured STT provider and enriched with
            Databricks Genie over governed customer and billing context so agents can resolve calls
            faster.
          </p>
          <div className="hero-flow">
            <span className="flow-pill">Voice Input</span>
            <span className="flow-arrow">→</span>
            <span className="flow-pill">Genie Reasoning</span>
            <span className="flow-arrow">→</span>
            <span className="flow-pill">Agent Resolution</span>
          </div>
        </div>
      </header>

      {error && <div className="error">API error: {error} — is the backend running?</div>}

      {showBenchmark ? (
        <ASRBenchmarkPage />
      ) : (
        <section className="command-stage">
          <CallList
            calls={status?.call_states ?? []}
            sttProvider={status?.stt_provider ?? "deepgram"}
            customers={customers}
            customersLoading={customersLoading}
            customersErr={customersErr}
          />
        </section>
      )}
    </div>
  );
}
