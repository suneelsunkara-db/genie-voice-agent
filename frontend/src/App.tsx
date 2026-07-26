import { CSSProperties, useEffect, useState } from "react";
import { api, CustomerWithIssue, InteractionLanguage, StatusResponse } from "./api/client";
import { useUiLocale } from "./i18n";
import { POLL_INTERVAL_MS } from "./config";
import { ASRBenchmarkPage } from "./components/ASRBenchmarkPage";
import { CockpitPage } from "./components/CockpitPage";
import { TracesPage } from "./components/TracesPage";
import { VoiceBenchmarksPage } from "./components/VoiceBenchmarksPage";
import { SentientShell } from "./components/sentient/Sentient";

export default function App() {
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [customers, setCustomers] = useState<CustomerWithIssue[]>([]);
  const [customersLoading, setCustomersLoading] = useState(true);
  const [customersErr, setCustomersErr] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [page, setPage] = useState(() => window.location.hash || "#/");
  const [interactionLanguage, setInteractionLanguage] = useState<InteractionLanguage>("en-US");

  // Load (and cache) the pre-generated UI-copy bundle for the selected language;
  // re-renders the tree once it lands so uiCopy() picks up the localized chrome.
  useUiLocale(interactionLanguage);

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
  const showTraces = page === "#/traces";
  const showVoiceBenchmarks = page === "#/voice-benchmarks";

  useEffect(() => {
    // The UI language picker defaults to English and stays where the agent puts
    // it. Only correct the selection if it somehow isn't a supported language.
    const supported = status?.languages?.supported ?? [];
    const defaultLanguage = status?.languages?.default;
    if (supported.length > 0 && !supported.some((item) => item.code === interactionLanguage)) {
      setInteractionLanguage(defaultLanguage ?? supported[0].code);
    }
  }, [status?.languages, interactionLanguage]);

  // Full-screen tool surfaces (their own dark theme), rendered outside the
  // Sentient shell like dedicated tools.
  if (showTraces) {
    return <TracesPage />;
  }
  if (showVoiceBenchmarks) {
    return <VoiceBenchmarksPage />;
  }

  const navPill: CSSProperties = {
    padding: "6px 14px",
    borderRadius: 999,
    border: "1px solid rgba(110,168,254,0.5)",
    background: "rgba(20,28,48,0.75)",
    color: "#cfe0ff",
    fontSize: 12,
    fontWeight: 600,
    cursor: "pointer",
    backdropFilter: "blur(6px)",
  };

  return (
    <SentientShell>
      {error && <div className="error">API error: {error} — is the backend running?</div>}
      <div style={{ position: "fixed", top: 14, right: 16, zIndex: 50, display: "flex", gap: 8 }}>
        <button
          type="button"
          onClick={() => (window.location.hash = "#/voice-benchmarks")}
          title="See how the voice API scores across languages"
          style={navPill}
        >
          Benchmarks
        </button>
        <button
          type="button"
          onClick={() => (window.location.hash = "#/traces")}
          title="Open the voice trace explorer"
          style={navPill}
        >
          Traces
        </button>
      </div>
      {showBenchmark ? (
        <ASRBenchmarkPage />
      ) : (
        <CockpitPage
          status={status}
          customers={customers}
          customersLoading={customersLoading}
          customersErr={customersErr}
          interactionLanguage={interactionLanguage}
          onLanguageChange={setInteractionLanguage}
        />
      )}
    </SentientShell>
  );
}
