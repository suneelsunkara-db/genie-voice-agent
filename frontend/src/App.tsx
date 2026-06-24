import { useEffect, useState } from "react";
import { api, StatusResponse } from "./api/client";
import { POLL_INTERVAL_MS } from "./config";
import { CallList } from "./components/CallList";
import databricksLogo from "./assets/databricks-logo.png";
import genieLogo from "./assets/genie-logo.png";

export default function App() {
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [issueCount, setIssueCount] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

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
    const loadIssues = async () => {
      try {
        const issues = await api.customersWithIssues();
        if (active) setIssueCount(issues.count);
      } catch {
        if (active) setIssueCount(null);
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
            <div className="meta-pill">runtime: {status?.mode ?? "…"}</div>
            <div className="meta-pill">
              customers with issues: {issueCount ?? "…"}
            </div>
          </div>
        </div>

        <div className="hero-content">
          <div className="eyebrow">Databricks Genie Voice Agent</div>
          <h1>Genie-Powered Voice Agent Experience</h1>
          <p>
            Voice conversations are transcribed with Deepgram and enriched with Databricks Genie
            over governed customer and billing context so agents can resolve calls faster.
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

      <section className="command-stage">
        <CallList calls={status?.call_states ?? []} />
      </section>
    </div>
  );
}
