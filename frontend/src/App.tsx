import { useEffect, useState } from "react";
import { api, CustomerWithIssue, InteractionLanguage, StatusResponse } from "./api/client";
import { POLL_INTERVAL_MS } from "./config";
import { ASRBenchmarkPage } from "./components/ASRBenchmarkPage";
import { CockpitPage } from "./components/CockpitPage";
import { SentientShell } from "./components/sentient/Sentient";

export default function App() {
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [customers, setCustomers] = useState<CustomerWithIssue[]>([]);
  const [customersLoading, setCustomersLoading] = useState(true);
  const [customersErr, setCustomersErr] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [page, setPage] = useState(() => window.location.hash || "#/");
  const [interactionLanguage, setInteractionLanguage] = useState<InteractionLanguage>("en-US");

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

  useEffect(() => {
    const supported = status?.languages?.supported ?? [];
    const defaultLanguage = status?.languages?.default;
    if (defaultLanguage && interactionLanguage === "en-US" && defaultLanguage !== "en-US") {
      setInteractionLanguage(defaultLanguage);
      return;
    }
    if (supported.length > 0 && !supported.some((item) => item.code === interactionLanguage)) {
      setInteractionLanguage(defaultLanguage ?? supported[0].code);
    }
  }, [status?.languages, interactionLanguage]);

  return (
    <SentientShell>
      {error && <div className="error">API error: {error} — is the backend running?</div>}
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
