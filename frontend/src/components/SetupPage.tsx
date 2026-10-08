/**
 * Setup / readiness page.
 *
 * Open to any authenticated app user. It runs the backend `/readiness` checks
 * (config, warehouse, voice model endpoints, Genie spaces, Lakebase CDF/serving,
 * AI Gateway guardrails, Apps User Authorization) and shows one authoritative
 * checklist. Cheap idempotent fixes (e.g. snapshot the Lakebase reference cache)
 * run in-place; UI-only Databricks steps deep-link out and re-check. An explicit
 * billable action can reconcile scale-to-zero on the two App-managed voice
 * endpoints, with live progress. The banner flips to "Ready to use" only when
 * every check is green.
 */
import { CSSProperties, useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
  api,
  ReadinessCheck,
  ReadinessResponse,
  ReadinessStatus,
  VoiceServingStatus,
} from "../api/client";

const STATUS_COLOR: Record<ReadinessStatus, string> = {
  ok: "#4ade80",
  warn: "#fbbf24",
  fail: "#f87171",
};

const STATUS_LABEL: Record<ReadinessStatus, string> = {
  ok: "OK",
  warn: "ACTION",
  fail: "BLOCKED",
};

const CATEGORY_LABEL: Record<string, string> = {
  prereq: "Automated prerequisite validation",
  scripted: "Automated deployment validation",
  manual: "Manual configuration validation",
};

const page: CSSProperties = {
  minHeight: "100vh",
  background: "#0b1120",
  color: "#e5edff",
  padding: "32px 24px 64px",
  fontFamily: "Inter, system-ui, sans-serif",
};
const wrap: CSSProperties = { maxWidth: 860, margin: "0 auto" };
const card: CSSProperties = {
  border: "1px solid rgba(110,168,254,0.25)",
  borderRadius: 14,
  background: "rgba(17,25,45,0.7)",
  padding: 18,
  marginBottom: 12,
};
const pill: CSSProperties = {
  padding: "6px 14px",
  borderRadius: 999,
  border: "1px solid rgba(110,168,254,0.5)",
  background: "rgba(20,28,48,0.75)",
  color: "#cfe0ff",
  fontSize: 12,
  fontWeight: 600,
  cursor: "pointer",
};

function CheckRow({
  check,
  busy,
  onFix,
}: {
  check: ReadinessCheck;
  busy: boolean;
  onFix: (action: string) => void;
}) {
  return (
    <div style={card}>
      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        <span
          style={{
            fontSize: 11,
            fontWeight: 700,
            color: "#0b1120",
            background: STATUS_COLOR[check.status],
            borderRadius: 6,
            padding: "2px 8px",
            minWidth: 62,
            textAlign: "center",
          }}
        >
          {STATUS_LABEL[check.status]}
        </span>
        <strong style={{ fontSize: 15 }}>{check.title}</strong>
        <span style={{ marginLeft: "auto", fontSize: 11, opacity: 0.6 }}>
          {CATEGORY_LABEL[check.category] ?? check.category}
        </span>
      </div>
      <div style={{ fontSize: 13, opacity: 0.85, marginTop: 8, lineHeight: 1.5 }}>{check.detail}</div>
      {check.objects && check.objects.length > 0 && (
        <div
          style={{
            marginTop: 12,
            padding: "10px 12px",
            borderRadius: 9,
            background: "rgba(8,15,30,0.65)",
            border: "1px solid rgba(110,168,254,0.14)",
          }}
        >
          <div style={{ fontSize: 11, fontWeight: 700, color: "#9fbcf5", marginBottom: 6 }}>
            Exact objects checked
          </div>
          {check.objects.map((object) => (
            <div
              key={object}
              style={{
                fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
                fontSize: 11,
                lineHeight: 1.65,
                overflowWrap: "anywhere",
                color: "#d7e4ff",
              }}
            >
              {object}
            </div>
          ))}
        </div>
      )}
      {check.required_configuration && check.required_configuration.length > 0 && (
        <div style={{ marginTop: 12 }}>
          <strong style={{ fontSize: 12, color: "#c5a7ff" }}>Required configuration</strong>
          <ul style={{ fontSize: 12, lineHeight: 1.65, margin: "6px 0 0", paddingLeft: 20 }}>
            {check.required_configuration.map((item) => <li key={item}>{item}</li>)}
          </ul>
        </div>
      )}
      {check.workspace_actions && check.workspace_actions.length > 0 && (
        <div style={{ marginTop: 12 }}>
          <strong style={{ fontSize: 12, color: check.status === "ok" ? "#4ade80" : "#fbbf24" }}>
            {check.status === "ok"
              ? "Required workspace setup (verified — no action now)"
              : "Databricks workspace action required"}
          </strong>
          <ol style={{ fontSize: 12, lineHeight: 1.65, margin: "6px 0 0", paddingLeft: 22 }}>
            {check.workspace_actions.map((step) => <li key={step}>{step}</li>)}
          </ol>
        </div>
      )}
      {check.explanation && (
        <div style={{ fontSize: 12, opacity: 0.7, marginTop: 7, lineHeight: 1.5 }}>
          {check.explanation}
        </div>
      )}
      {check.resolution_steps && check.resolution_steps.length > 0 && (
        <div style={{ marginTop: 13, borderTop: "1px solid rgba(110,168,254,0.18)", paddingTop: 10 }}>
          <strong style={{ fontSize: 12 }}>How to fix</strong>
          <ol style={{ fontSize: 12, lineHeight: 1.65, margin: "6px 0 0", paddingLeft: 22 }}>
            {check.resolution_steps.map((step) => <li key={step}>{step}</li>)}
          </ol>
        </div>
      )}
      {check.technical_detail && (
        <details style={{ fontSize: 11, opacity: 0.72, marginTop: 10 }}>
          <summary style={{ cursor: "pointer" }}>Technical details for support / logs</summary>
          <pre style={{ whiteSpace: "pre-wrap", overflowWrap: "anywhere", marginBottom: 0 }}>
            {check.technical_detail}
          </pre>
        </details>
      )}
      {check.fix && (
        <div style={{ marginTop: 12 }}>
          {check.fix.kind === "auto" && check.fix.action ? (
            <button
              type="button"
              style={{ ...pill, opacity: busy ? 0.6 : 1 }}
              disabled={busy}
              onClick={() => onFix(check.fix!.action!)}
            >
              {busy ? "Working…" : check.fix.label}
            </button>
          ) : check.fix.kind === "link" && check.fix.href ? (
            <a href={check.fix.href} target="_blank" rel="noreferrer" style={{ ...pill, textDecoration: "none" }}>
              {check.fix.label} ↗
            </a>
          ) : (
            <span style={{ fontSize: 12, opacity: 0.7 }}>{check.fix.label}</span>
          )}
        </div>
      )}
    </div>
  );
}

export function SetupPage() {
  const [data, setData] = useState<ReadinessResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [busyAction, setBusyAction] = useState<string | null>(null);
  const [progress, setProgress] = useState(4);
  const [automatedOpen, setAutomatedOpen] = useState(false);
  const [manualOpen, setManualOpen] = useState(false);
  const [voiceServing, setVoiceServing] = useState<VoiceServingStatus | null>(null);
  const [voiceServingError, setVoiceServingError] = useState<string | null>(null);
  const [voiceActionBusy, setVoiceActionBusy] = useState(false);
  const [voiceActionTarget, setVoiceActionTarget] = useState<boolean | null>(null);
  const voiceWasCompliant = useRef(false);

  const load = useCallback(async () => {
    setLoading(true);
    setProgress(4);
    const timer = window.setInterval(() => {
      setProgress((value) => Math.min(92, value + Math.max(1, Math.round((92 - value) * 0.08))));
    }, 700);
    try {
      setData(await api.readiness());
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      window.clearInterval(timer);
      setProgress(100);
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const loadVoiceServing = useCallback(async () => {
    try {
      const status = await api.voiceServingStatus();
      setVoiceServing(status);
      setVoiceServingError(null);
    } catch (e) {
      setVoiceServingError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    void loadVoiceServing();
    const timer = window.setInterval(() => void loadVoiceServing(), 4_000);
    return () => window.clearInterval(timer);
  }, [loadVoiceServing]);

  useEffect(() => {
    if (voiceServing?.compliant && !voiceWasCompliant.current) {
      voiceWasCompliant.current = true;
      void load();
    } else if (voiceServing && !voiceServing.compliant) {
      voiceWasCompliant.current = false;
    }
  }, [voiceServing?.compliant, load]);

  useEffect(() => {
    if (!voiceServing || voiceActionTarget === null) return;
    const matching = voiceActionTarget
      ? voiceServing.metrics.scale_to_zero_enabled
      : voiceServing.metrics.scale_to_zero_disabled;
    if (
      matching === voiceServing.metrics.total
      && voiceServing.metrics.ready === voiceServing.metrics.total
      && !voiceServing.updating
      && voiceServing.metrics.failed === 0
    ) {
      setVoiceActionBusy(false);
      setVoiceActionTarget(null);
      void load();
    }
  }, [voiceServing, voiceActionTarget, load]);

  const setScaleToZero = useCallback(async (enabled: boolean) => {
    setVoiceActionBusy(true);
    setVoiceActionTarget(enabled);
    setVoiceServingError(null);
    try {
      const result = enabled
        ? await api.enableVoiceScaleToZero()
        : await api.disableVoiceScaleToZero();
      setVoiceServing(result.status);
    } catch (e) {
      setVoiceActionBusy(false);
      setVoiceActionTarget(null);
      setVoiceServingError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  const onFix = useCallback(async (action: string) => {
    setBusyAction(action);
    try {
      const res = await api.readinessFix(action);
      setData(res.readiness);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusyAction(null);
    }
  }, []);

  const ready = data?.ready ?? false;
  const summary = data?.summary;
  const automatedChecks = useMemo(
    () => data?.checks
      .filter((check) => check.category !== "manual")
      .sort((a, b) => Number(a.status === "ok") - Number(b.status === "ok")) ?? [],
    [data],
  );
  const manualChecks = useMemo(
    () => data?.checks
      .filter((check) => check.category === "manual")
      .sort((a, b) => Number(a.status === "ok") - Number(b.status === "ok")) ?? [],
    [data],
  );
  const automatedIssues = automatedChecks.filter((check) => check.status !== "ok").length;
  const manualIssues = manualChecks.filter((check) => check.status !== "ok").length;
  const voiceProgress = voiceServing && voiceActionTarget !== null && voiceServing.metrics.total > 0
    ? Math.round(100 * (
      voiceActionTarget
        ? voiceServing.metrics.scale_to_zero_enabled
        : voiceServing.metrics.scale_to_zero_disabled
    ) / voiceServing.metrics.total)
    : voiceServing?.progress ?? 0;

  useEffect(() => {
    if (!data) return;
    setAutomatedOpen(automatedIssues > 0);
    setManualOpen(manualIssues > 0);
  }, [data, automatedIssues, manualIssues]);

  const copyDiagnostics = useCallback(async () => {
    if (!data) return;
    const text = data.checks
      .map((check) => [
        `[${check.status.toUpperCase()}] ${check.id}: ${check.title}`,
        check.detail,
        ...(check.objects ?? []).map((object) => `Object: ${object}`),
        ...(check.required_configuration ?? []).map((item) => `Required: ${item}`),
        ...(check.workspace_actions ?? []).map((step, index) => `Workspace action ${index + 1}: ${step}`),
        check.technical_detail ? `Technical: ${check.technical_detail}` : "",
        ...(check.resolution_steps ?? []).map((step, index) => `${index + 1}. ${step}`),
      ].filter(Boolean).join("\n"))
      .join("\n\n");
    await navigator.clipboard.writeText(text);
  }, [data]);

  return (
    <div style={page}>
      <div style={wrap}>
        <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 18 }}>
          <button type="button" style={pill} onClick={() => (window.location.hash = "#/")}>
            ← Home
          </button>
          <h1 style={{ fontSize: 22, margin: 0 }}>Setup &amp; Readiness</h1>
          <button type="button" style={{ ...pill, marginLeft: "auto" }} onClick={() => void load()}>
            {loading ? "Checking…" : "Re-check"}
          </button>
          <button type="button" style={pill} disabled={!data} onClick={() => void copyDiagnostics()}>
            Copy diagnostics
          </button>
        </div>

        <div
          style={{
            ...card,
            borderColor: voiceServing?.compliant
              ? "rgba(74,222,128,0.5)"
              : "rgba(251,191,36,0.55)",
            background: voiceServing?.compliant
              ? "rgba(22,45,32,0.6)"
              : "rgba(69,43,12,0.62)",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <strong style={{ fontSize: 16 }}>Voice model serving</strong>
            <span style={{ marginLeft: "auto", color: voiceServing?.compliant ? "#4ade80" : "#fbbf24", fontSize: 12, fontWeight: 700 }}>
              {voiceServing?.compliant
                ? "WARM CONFIGURATION"
                : voiceServing?.updating
                  ? "UPDATING"
                  : "ACTION REQUIRED"}
            </span>
          </div>
          <p style={{ fontSize: 13, opacity: 0.85, lineHeight: 1.5, margin: "8px 0 12px" }}>
            Qwen3-ASR and VoxCPM2 must keep a GPU replica available. Disabling scale-to-zero is
            billable and prevents cold starts from swallowing the first transcription or spoken reply.
            Enabling it reduces idle cost, but the first call after an idle period may be delayed or silent.
          </p>

          {voiceServing && (
            <>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(3, minmax(0, 1fr))", gap: 10 }}>
                {[
                  ["Scale-to-zero disabled", `${voiceServing.metrics.scale_to_zero_disabled}/${voiceServing.metrics.total}`],
                  ["Endpoints ready", `${voiceServing.metrics.ready}/${voiceServing.metrics.total}`],
                  ["Update progress", `${voiceProgress}%`],
                ].map(([label, value]) => (
                  <div key={label} style={{ padding: "10px 12px", borderRadius: 9, background: "rgba(8,15,30,0.5)" }}>
                    <div style={{ fontSize: 11, opacity: 0.65 }}>{label}</div>
                    <div style={{ fontSize: 19, fontWeight: 750, marginTop: 3 }}>{value}</div>
                  </div>
                ))}
              </div>
              <div
                role="progressbar"
                aria-label="Voice endpoint update progress"
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuenow={voiceProgress}
                style={{ height: 8, borderRadius: 999, overflow: "hidden", background: "rgba(148,163,184,0.18)", marginTop: 12 }}
              >
                <div style={{
                  width: `${voiceProgress}%`,
                  height: "100%",
                  transition: "width 600ms ease",
                  background: voiceServing.compliant
                    ? "linear-gradient(90deg, #22c55e, #4ade80)"
                    : "linear-gradient(90deg, #f59e0b, #fbbf24)",
                }} />
              </div>
              <div style={{ marginTop: 12, display: "grid", gap: 7 }}>
                {voiceServing.endpoints.map((endpoint) => (
                  <div key={endpoint.name || endpoint.id} style={{ display: "flex", gap: 10, alignItems: "center", fontSize: 12 }}>
                    <strong style={{ minWidth: 44 }}>{endpoint.modality.toUpperCase()}</strong>
                    <span style={{ flex: 1, fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace", overflowWrap: "anywhere" }}>
                      {endpoint.name}
                    </span>
                    <span>{endpoint.scale_to_zero_enabled === false ? "always on" : endpoint.scale_to_zero_enabled === true ? "scale-to-zero" : "unknown"}</span>
                    <span>{endpoint.config_update && endpoint.config_update !== "NOT_UPDATING" ? endpoint.config_update : endpoint.ready || "unknown"}</span>
                  </div>
                ))}
              </div>
              <button
                type="button"
                style={{ ...pill, marginTop: 14, opacity: voiceActionBusy || voiceServing.updating ? 0.6 : 1 }}
                disabled={
                  voiceActionBusy
                  || voiceServing.updating
                  || (!voiceServing.compliant && !voiceServing.requires_action)
                }
                onClick={() => void setScaleToZero(voiceServing.compliant)}
              >
                {voiceServing.updating || voiceActionBusy
                  ? `${voiceActionTarget ? "Enabling" : "Disabling"} scale-to-zero… ${voiceProgress}%`
                  : voiceServing.compliant
                    ? "Enable scale-to-zero for both models"
                    : "Disable scale-to-zero for both models"}
              </button>
            </>
          )}
          {!voiceServing && !voiceServingError && (
            <div style={{ fontSize: 12, color: "#fbbf24" }}>Reading serving configuration…</div>
          )}
          {voiceServingError && (
            <div style={{ fontSize: 12, color: "#fca5a5", marginTop: 10 }}>
              {voiceServingError}
              <div style={{ opacity: 0.8, marginTop: 4 }}>
                Redeploy the App if its service principal does not yet have CAN_MANAGE on both voice endpoints.
              </div>
            </div>
          )}
        </div>

        <div
          style={{
            ...card,
            borderColor: ready ? "rgba(74,222,128,0.5)" : "rgba(251,191,36,0.4)",
            background: ready ? "rgba(22,45,32,0.6)" : "rgba(45,38,17,0.5)",
          }}
        >
          <div style={{ fontSize: 18, fontWeight: 700 }}>
            {loading && !data ? "⏳ Running readiness checks" : ready ? "✅ Ready to use" : "⏳ Not ready yet"}
          </div>
          <div style={{ fontSize: 13, opacity: 0.85, marginTop: 6 }}>
            {loading && !data
              ? "Running automated infrastructure probes and viewer-scoped manual validation."
              : ready
              ? "Every readiness check passed. The app is fully configured."
              : "Finish the ACTION / BLOCKED items below, then Re-check. Manual (UI) items open the workspace; installer/prereq items are fixed by re-running deploy_app.sh."}
          </div>
          <div
            role="progressbar"
            aria-label="Readiness validation progress"
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={progress}
            style={{
              height: 8,
              borderRadius: 999,
              overflow: "hidden",
              background: "rgba(148,163,184,0.18)",
              marginTop: 14,
            }}
          >
            <div
              style={{
                width: `${progress}%`,
                height: "100%",
                borderRadius: 999,
                transition: "width 500ms ease",
                background: ready
                  ? "linear-gradient(90deg, #22c55e, #4ade80)"
                  : "linear-gradient(90deg, #5b8cff, #a78bfa)",
              }}
            />
          </div>
          <div style={{ fontSize: 11, opacity: 0.7, marginTop: 6 }}>
            {loading
              ? `Deep validation in progress · ${progress}%`
              : summary
                ? `Validation complete · ${summary.ok}/${summary.total} passed`
                : "Waiting to start validation"}
          </div>
          {summary && (
            <div style={{ fontSize: 12, opacity: 0.7, marginTop: 8 }}>
              {summary.ok} ok · {summary.warn} action · {summary.fail} blocked · {summary.total} checks
            </div>
          )}
          {data?.viewer && (
            <div style={{ fontSize: 12, opacity: 0.72, marginTop: 7 }}>
              Viewer tested: {data.viewer.email || data.viewer.username || "identity header unavailable"}
              {" · "}
              User Authorization token: {data.viewer.user_authorization_token ? "present" : "missing"}
            </div>
          )}
          {loading && (
            <div style={{ fontSize: 12, color: "#fbbf24", marginTop: 10 }}>
              Running full validation: live TTS→STT, Gateway policies, viewer Genie, and Agent Mode.
              Cold endpoints can take several minutes; keep this page open.
            </div>
          )}
        </div>

        {error && (
          <div style={{ ...card, borderColor: "rgba(248,113,113,0.5)" }}>
            <strong>Could not load readiness:</strong> {error}
          </div>
        )}

        <details
          open={automatedOpen}
          onToggle={(event) => setAutomatedOpen(event.currentTarget.open)}
          style={{ marginTop: 24 }}
        >
          <summary style={{ cursor: "pointer", color: "#8ab4ff", fontSize: 17, fontWeight: 700 }}>
            1. Automated deploy checks · {automatedChecks.length - automatedIssues}/{automatedChecks.length} passed
            {automatedIssues > 0 ? ` · ${automatedIssues} need attention` : ""}
          </summary>
          <p style={{ fontSize: 12, opacity: 0.7, margin: "8px 2px 12px", lineHeight: 1.55 }}>
            Objects created, configured, granted, or populated by deploy_app.sh. This includes
            exact App service-principal grants, UC assets, jobs, models, endpoints, and serving data.
          </p>
          {automatedChecks.map((check) => (
            <CheckRow key={check.id} check={check} busy={busyAction !== null} onFix={onFix} />
          ))}
        </details>

        <details
          open={manualOpen}
          onToggle={(event) => setManualOpen(event.currentTarget.open)}
          style={{ marginTop: 28 }}
        >
          <summary style={{ cursor: "pointer", color: "#c5a7ff", fontSize: 17, fontWeight: 700 }}>
            2. Manual checks needed · {manualChecks.length - manualIssues}/{manualChecks.length} passed
            {manualIssues > 0 ? ` · ${manualIssues} need attention` : ""}
          </summary>
          <p style={{ fontSize: 12, opacity: 0.7, margin: "8px 2px 12px", lineHeight: 1.55 }}>
            UI-only configuration and viewer permissions: CDF publication, Gateway policy
            attachment, App authorization scopes and consent, App/Genie access, and Agent Mode Preview.
            Each check retains its exact required state and workspace action even when green.
          </p>
          {manualChecks.map((check) => (
            <CheckRow key={check.id} check={check} busy={busyAction !== null} onFix={onFix} />
          ))}
        </details>
      </div>
    </div>
  );
}
