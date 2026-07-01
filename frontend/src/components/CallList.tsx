import { ReactNode, useEffect, useMemo, useRef, useState } from "react";
import {
  AccountFacts,
  api,
  AssistPipelineStep,
  CallState,
  CustomerWithIssue,
  GenieResponse,
  LiveNudge,
  ResolutionEvent,
} from "../api/client";
import { WS_BASE_URL } from "../config";
import { intentLabel, PRIORITY_RANK, recommend } from "../guidance";
import {
  isSpeechCaptionSupported,
  MicRecordingSession,
  MicStreamSession,
  SpeechCaptionSession,
  startMicRecording,
  startMicStream,
  startSpeechCaption,
  VoiceUiState,
} from "../lib/micStream";
import databricksLogo from "../assets/databricks-logo.png";
import genieLogo from "../assets/genie-logo.png";

const SPOTLIGHT_CUSTOMER_ID = "CUST-4028";

function signalsOf(call: CallState) {
  const gold = (call.state?.gold ?? {}) as Record<string, any>;
  const live = (call.state?.live ?? {}) as Record<string, any>;
  return {
    intent: gold.primary_intent ?? live.primary_intent,
    disposition: gold.disposition,
    sentiment: gold.sentiment_label ?? live.sentiment_label,
    nba: gold.next_best_action ?? live.next_best_action,
    summary: gold.summary,
    invoice: gold.mentioned_invoice_id ?? live.mentioned_invoice_id,
    amount: gold.mentioned_amount ?? live.mentioned_amount,
  };
}

function customerPriority(c: CustomerWithIssue) {
  if (c.issue_status !== "closed") return "high" as const;
  if ((c.overdue_invoice_count ?? 0) > 0 || c.customer_status === "at_risk") return "medium" as const;
  return "low" as const;
}

export function CallList({
  calls,
  sttProvider,
  customers,
  customersLoading,
  customersErr,
}: {
  calls: CallState[];
  sttProvider: string;
  customers: CustomerWithIssue[];
  customersLoading: boolean;
  customersErr: string | null;
}) {
  const sortedCustomers = useMemo(() => {
    const base = [...customers].sort((a, b) => {
      const pa = PRIORITY_RANK[customerPriority(a)];
      const pb = PRIORITY_RANK[customerPriority(b)];
      if (pa !== pb) return pa - pb;
      return (b.overdue_amount ?? 0) - (a.overdue_amount ?? 0);
    });
    const spotlight = base.find((c) => c.customer_id === SPOTLIGHT_CUSTOMER_ID);
    if (!spotlight) return base;
    return [spotlight, ...base.filter((c) => c.customer_id !== SPOTLIGHT_CUSTOMER_ID)];
  }, [customers]);

  const callByCustomer = useMemo(() => {
    const map = new Map<string, CallState>();
    for (const c of calls) {
      if (c.customer_id && !map.has(c.customer_id)) map.set(c.customer_id, c);
    }
    return map;
  }, [calls]);

  const [selectedCustomerId, setSelectedCustomerId] = useState<string | null>(null);
  const [userPicked, setUserPicked] = useState(false);

  useEffect(() => {
    if (!sortedCustomers.length || userPicked) return;
    const defaultId =
      sortedCustomers.find((c) => c.customer_id === SPOTLIGHT_CUSTOMER_ID)?.customer_id ??
      sortedCustomers[0].customer_id;
    setSelectedCustomerId(defaultId);
  }, [sortedCustomers, userPicked]);

  const selectedCustomer =
    sortedCustomers.find((c) => c.customer_id === selectedCustomerId) ?? sortedCustomers[0] ?? null;
  const selectedCall = selectedCustomer?.call_id
    ? calls.find((c) => c.call_id === selectedCustomer.call_id) ??
      callByCustomer.get(selectedCustomer.customer_id) ??
      null
    : selectedCustomer
    ? callByCustomer.get(selectedCustomer.customer_id) ?? null
    : null;

  const [conversationByCall, setConversationByCall] = useState<
    Record<string, { text: string; speaker?: number }[]>
  >({});

  if (customersLoading && !customers.length) {
    return <p className="muted">Loading customers with issues…</p>;
  }

  if (!sortedCustomers.length) {
    return (
      <p className="muted">
        {customersErr
          ? `Unable to load customers: ${customersErr}`
          : "No customers with open issues found in account data."}
      </p>
    );
  }

  return (
    <div className="cc-layout">
      <aside className="cc-sidebar">
        <div className="cc-stack-brand">
          <img className="hero-logo dbx-full side" src={databricksLogo} alt="Databricks" />
          <img className="hero-logo genie-full side" src={genieLogo} alt="Genie" />
        </div>
        <div className="cc-sidebar-title">Customers with issues</div>
        <div className="cc-sidebar-sub">
          Billing risk, overdue exposure, and accounts needing agent assist
        </div>
        {sortedCustomers.map((c) => {
          const prio = customerPriority(c);
          const active = selectedCustomer?.customer_id === c.customer_id;
          const hasLiveCall = Boolean(c.call_id ?? callByCustomer.get(c.customer_id)?.call_id);
          return (
            <button
              key={c.customer_id}
              className={`cc-call-row cc-customer-row ${active ? "active" : ""} ${
                !hasLiveCall ? "cc-customer-muted" : ""
              }`}
              onClick={() => {
                setUserPicked(true);
                setSelectedCustomerId(c.customer_id);
              }}
            >
              <span className={`prio-dot p-${prio}`} />
              <span className="cc-call-main">
                <span className="cc-customer-name">{c.full_name ?? c.customer_id}</span>
                <span className="cc-call-id">{c.customer_id}</span>
                <span className="cc-call-intent">{c.rationale ?? intentLabel(c.primary_intent)}</span>
                {c.call_id && <span className="cc-customer-call">Call {c.call_id}</span>}
              </span>
              <span className={`badge sentiment cc-sentiment ${c.sentiment_label ?? "neutral"}`}>
                {c.sentiment_label ?? c.issue_status ?? "—"}
              </span>
            </button>
          );
        })}
      </aside>

      {selectedCall ? (
        <Cockpit
          call={selectedCall}
          customer={selectedCustomer}
          sttProvider={sttProvider}
          localTurns={conversationByCall[selectedCall.call_id] ?? []}
          onAppendLocalTurn={(turn) =>
            setConversationByCall((prev) => ({
              ...prev,
              [selectedCall.call_id]: [...(prev[selectedCall.call_id] ?? []), turn],
            }))
          }
          onResetLocalTurns={() =>
            setConversationByCall((prev) => ({
              ...prev,
              [selectedCall.call_id]: [],
            }))
          }
        />
      ) : (
        <div className="cc-main cc-empty-call">
          <div className="eyebrow">No live call</div>
          <h2>{selectedCustomer?.full_name ?? selectedCustomer?.customer_id}</h2>
          <p className="muted">
            This customer has an open account issue but is not in the active assist queue yet.
          </p>
          {selectedCustomer?.rationale && <p>{selectedCustomer.rationale}</p>}
        </div>
      )}
    </div>
  );
}

function Cockpit({
  call,
  customer,
  sttProvider,
  localTurns,
  onAppendLocalTurn,
  onResetLocalTurns,
}: {
  call: CallState;
  customer: CustomerWithIssue | null;
  sttProvider: string;
  localTurns: { text: string; speaker?: number }[];
  onAppendLocalTurn: (turn: { text: string; speaker?: number }) => void;
  onResetLocalTurns: () => void;
}) {
  const base = signalsOf(call);
  const [facts, setFacts] = useState<AccountFacts | null>(null);
  const [factErr, setFactErr] = useState<string | null>(null);
  const [live, setLive] = useState<Record<string, any> | null>(null);
  const [genieQuestion, setGenieQuestion] = useState("");
  const [genieResp, setGenieResp] = useState<GenieResponse | null>(null);
  const [genieShowSql, setGenieShowSql] = useState(false);
  const [genieLoading, setGenieLoading] = useState(false);
  const [genieErr, setGenieErr] = useState<string | null>(null);
  const [resolutionEvents, setResolutionEvents] = useState<ResolutionEvent[]>([]);
  const [assistMeta, setAssistMeta] = useState<LiveNudge | null>(null);
  const [resetBusy, setResetBusy] = useState(false);
  const [voiceUi, setVoiceUi] = useState<VoiceUiState>({ phase: "idle" });

  useEffect(() => {
    let active = true;
    setFacts(null);
    setFactErr(null);
    setLive(null);
    setGenieQuestion("");
    setGenieResp(null);
    setGenieShowSql(false);
    setGenieErr(null);
    setResolutionEvents([]);
    setAssistMeta(null);
    api
      .callAccount(call.call_id)
      .then((f) => active && setFacts(f))
      .catch((e) => active && setFactErr(e instanceof Error ? e.message : "failed"));
    api
      .resolutionEvents(call.call_id)
      .then((r) => active && setResolutionEvents(r.events ?? []))
      .catch(() => {});
    // Fire-and-forget: warm a Genie account insight off the live reply path so the
    // per-utterance agent reply can ground on it without paying Genie latency inline.
    api.prefetchGenieInsight(call.call_id).catch(() => {});
    return () => {
      active = false;
    };
  }, [call.call_id]);

  // Live simulated utterance overrides the call-level signals when present.
  const sentiment = live?.sentiment_label ?? base.sentiment;
  const nba = live?.next_best_action ?? base.nba;
  const intent = live?.primary_intent ?? base.intent;

  const cust = facts?.customer ?? {};
  const sum = facts?.summary ?? {};
  // Keep stream empty by default so the UI feels like a true live call surface.
  const utterances = localTurns;
  const hasAgentTurn = utterances.some((u) => (u.speaker ?? 0) === 0);
  const issueStatus = String(sum.issue_status ?? "open");
  const rec =
    issueStatus === "closed"
      ? {
          title: "Issue resolved — confirm and close warmly",
          detail:
            sum.resolution_note ??
            "Payment arrangement and waiver are applied. Confirm closure with the customer and offer brief follow-up help.",
          priority: "low" as const,
        }
      : hasAgentTurn
      ? recommend(nba, sentiment, facts)
      : {
          title: "Listening to customer context",
          detail:
            "Collecting customer request first. Recommended next action will appear right after the Genie-assisted agent response.",
          priority: "low" as const,
        };
  const overdueCount = Number(sum.overdue_invoice_count ?? 0);
  const overdueAmount = Number(sum.overdue_amount ?? 0);
  const riskLevel = overdueCount > 0 || !sum.autopay_enabled ? "elevated" : "stable";
  const suggestedQuestion =
    facts?.customer_id || call.customer_id
      ? `For customer ${facts?.customer_id ?? call.customer_id} on call ${
          call.call_id
        }, summarize account risk, overdue/declined payment context, and provide the best retention-safe next action for the agent.`
      : `Give a live assist summary for call ${call.call_id}, including likely intent and next best action.`;

  useEffect(() => {
    if (!genieQuestion) {
      setGenieQuestion(suggestedQuestion);
    }
  }, [suggestedQuestion, genieQuestion]);

  const refreshAssistData = () => {
    api
      .callAccount(call.call_id)
      .then((f) => setFacts(f))
      .catch(() => {});
    api
      .resolutionEvents(call.call_id)
      .then((r) => setResolutionEvents(r.events ?? []))
      .catch(() => {});
  };

  const askGenie = async (question: string, asFollowup = false) => {
    if (!question.trim()) return;
    setGenieLoading(true);
    setGenieErr(null);
    try {
      // Continue the same conversation for follow-ups so Genie retains context.
      const resp = await api.askGenie(
        question,
        asFollowup ? genieResp?.conversation_id : undefined
      );
      setGenieResp(resp);
      setGenieShowSql(false);
    } catch (e) {
      setGenieErr(e instanceof Error ? e.message : "Failed to query Genie");
    } finally {
      setGenieLoading(false);
    }
  };

  const resetScenario = async () => {
    setResetBusy(true);
    setFactErr(null);
    setResolutionEvents([]);
    setAssistMeta(null);
    try {
      await api.resetDemoSession(call.call_id);
      onResetLocalTurns();
      setLive(null);
      setVoiceUi({ phase: "idle" });
      setGenieResp(null);
      setGenieShowSql(false);
      const [f, r] = await Promise.all([api.callAccount(call.call_id), api.resolutionEvents(call.call_id)]);
      setFacts(f);
      setResolutionEvents(r.events ?? []);
    } catch (e) {
      setFactErr(e instanceof Error ? e.message : "reset failed");
    } finally {
      setResetBusy(false);
    }
  };

  return (
    <div className="cc-main">
      <div className="cc-top">
        <div>
          <div className="eyebrow">Active Customer</div>
          <div className="cust-name">{cust.full_name ?? call.customer_id ?? call.call_id}</div>
          <div className="cust-sub">
            {[cust.segment, cust.plan, cust.region].filter(Boolean).join(" · ") || "Customer profile loading"}
            {cust.tenure_months != null && <> · {cust.tenure_months} mo tenure</>}
          </div>
        </div>
        <div className="cust-status studio-status">
          <span className={`status-chip issue issue-${issueStatus}`}>issue: {issueStatus}</span>
          <span className={`status-chip ${riskLevel === "elevated" ? "st-at_risk" : "st-active"}`}>
            risk: {riskLevel}
          </span>
          {cust.status && (
            <span className={`status-chip st-${cust.status}`}>{cust.status}</span>
          )}
          {cust.monthly_charge != null && (
            <span className="cust-arpu">${cust.monthly_charge}/mo</span>
          )}
        </div>
      </div>
      {sum.resolution_note && (
        <div className="resolution-banner">
          {sum.resolution_note}
          {sum.resolved_at ? ` (resolved at ${sum.resolved_at})` : ""}
        </div>
      )}

      <div className="cc-grid">
        <div className="panel cc-conversation">
          <RecommendationCard rec={rec} intent={intent} sentiment={sentiment} />
          <div className="panel-title convo-title-row">
            <span>Conversation stream (voice to Genie to agent)</span>
            <button className="ghost mini" onClick={resetScenario} disabled={resetBusy}>
              {resetBusy ? "Resetting…" : "Reset scenario"}
            </button>
          </div>
          <div className="transcript cc-transcript">
            {utterances.length === 0 && voiceUi.phase === "idle" && (
              <div className="muted">No transcript captured.</div>
            )}
            {utterances.map((u, i) => {
              const isCustomer = (u.speaker ?? 0) === 1;
              return (
                <div key={i} className={`turn ${isCustomer ? "t-customer" : "t-agent"}`}>
                  <span className="turn-who">{isCustomer ? "Customer" : "Agent (Genie-assisted)"}</span>
                  <span className="turn-text">{u.text}</span>
                </div>
              );
            })}
            {voiceUi.phase === "speaking" && (
              <div className="turn t-customer turn-live">
                <span className="turn-who">Customer (speaking)</span>
                <span className="turn-text turn-placeholder">
                  {voiceUi.interimText?.trim() || "Listening…"}
                </span>
                <LiveWaveform level={voiceUi.micLevel ?? 0.2} active />
              </div>
            )}
            {voiceUi.phase === "transcribing" && (
              <div className="turn t-customer turn-live">
                <span className="turn-who">Customer</span>
                <span className="turn-text turn-placeholder transcribing">
                  {voiceUi.interimText?.trim() || voiceUi.processingLabel || "Transcribing your message…"}
                </span>
              </div>
            )}
            {voiceUi.phase === "agent_reply" && (
              <div className="turn t-agent turn-live">
                <span className="turn-who">Agent (Genie-assisted)</span>
                <span className="turn-text turn-placeholder transcribing">
                  {voiceUi.processingLabel || "Preparing Genie-assisted response…"}
                </span>
              </div>
            )}
          </div>
          <LiveAssist
            callId={call.call_id}
            customerId={String(facts?.customer_id ?? call.customer_id ?? "")}
            sttProvider={sttProvider}
            onNudge={(n) => {
              setLive(n.live);
              setAssistMeta(n);
              refreshAssistData();
            }}
            onLocalTurn={onAppendLocalTurn}
            onVoiceUiChange={setVoiceUi}
          />
          <AssistStatusPanel meta={assistMeta} />
        </div>

        <div className="panel cc-genie">
          <div className="panel-title">Databricks Genie live intelligence</div>
          <div className="genie-brand-note">
            Genie reads governed customer, invoice, payment, and call context to guide voice agents.
          </div>
          <div className="facts-grid cc-kpis">
            <Fact label="Open invoices" value={sum.open_invoice_count ?? 0} />
            <Fact
              label="Overdue"
              value={`${overdueCount} ($${overdueAmount})`}
              warn={overdueCount > 0}
            />
            <Fact
              label="Autopay"
              value={sum.autopay_enabled ? "on" : "off"}
              warn={!sum.autopay_enabled}
            />
            <Fact
              label="Declined pays"
              value={sum.recent_declined_payments ?? 0}
              warn={(sum.recent_declined_payments ?? 0) > 0}
            />
          </div>

          {factErr && <div className="account-error">unavailable: {factErr}</div>}
          {!facts && !factErr && <div className="muted">loading…</div>}

          {facts?.invoices && facts.invoices.length > 0 && (
            <table className="inv-table">
              <thead>
                <tr>
                  <th>Invoice</th>
                  <th>Period</th>
                  <th>Amount</th>
                  <th>Late fee</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {facts.invoices.slice(0, 4).map((inv) => (
                  <tr
                    key={inv.invoice_id}
                    className={
                      inv.status === "overdue"
                        ? "row-warn"
                        : inv.resolution_status === "closed" ||
                          String(inv.status) === "resolved" ||
                          (issueStatus === "closed" && String(inv.status) === "open")
                        ? "row-ok"
                        : ""
                    }
                  >
                    <td>{inv.invoice_id}</td>
                    <td>{inv.period}</td>
                    <td>${inv.amount}</td>
                    <td>${inv.late_fee}</td>
                    <td>{inv.status}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          <div className="resolution-timeline">
            <div className="panel-title">Resolution timeline</div>
            {resolutionEvents.length === 0 && <div className="muted">No resolution events yet.</div>}
            {resolutionEvents.map((ev) => (
              <div className="timeline-item" key={ev.event_id}>
                <div className="timeline-head">
                  <span className="timeline-type">{ev.event_type}</span>
                  <span className="timeline-status">{ev.issue_status ?? "open"}</span>
                </div>
                {ev.note && <div className="timeline-note">{ev.note}</div>}
              </div>
            ))}
          </div>

          <div className="genie-console cc-console">
            <label className="genie-label">Ask Genie for a real-time assist prompt</label>
            <textarea
              value={genieQuestion}
              onChange={(e) => setGenieQuestion(e.target.value)}
              className="genie-input-box"
              rows={3}
            />
            <div className="genie-actions">
              <button onClick={() => askGenie(genieQuestion)} disabled={genieLoading}>
                {genieLoading ? "Analyzing…" : "Run Genie Query"}
              </button>
              <button
                className="ghost"
                onClick={() => {
                  setGenieQuestion(suggestedQuestion);
                  askGenie(suggestedQuestion);
                }}
                disabled={genieLoading}
              >
                Refresh Assist
              </button>
            </div>
            {genieErr && <div className="account-error">{genieErr}</div>}
            {genieResp && (genieResp.answer || genieResp.description) && (
              <div className="genie-answer">
                {genieResp.answer ?? genieResp.description}
              </div>
            )}
            {genieResp?.description &&
              genieResp.answer &&
              genieResp.description !== genieResp.answer && (
                <div className="genie-hint">{genieResp.description}</div>
              )}
            {genieResp?.suggested_followups && genieResp.suggested_followups.length > 0 && (
              <div className="genie-followups">
                {genieResp.suggested_followups.map((f, i) => (
                  <button
                    key={i}
                    className="followup-chip"
                    disabled={genieLoading}
                    onClick={() => askGenie(f, true)}
                  >
                    {f}
                  </button>
                ))}
              </div>
            )}
            {!genieResp && (
              <div className="genie-hint">
                Designed around {customer?.full_name ?? "spotlight customers"}: ask for payment
                arrangement + late fee relief.
              </div>
            )}
            {genieResp?.sql && (
              <div className="genie-sql">
                <button className="sql-toggle" onClick={() => setGenieShowSql((v) => !v)}>
                  {genieShowSql ? "Hide query" : "Show query"}
                </button>
                {genieShowSql && <pre className="sql">{genieResp.sql}</pre>}
              </div>
            )}
          </div>
        </div>
      </div>

      <ResolutionJourneyStrip
        issueStatus={issueStatus}
        localTurns={utterances}
        assistMeta={assistMeta}
        voiceUi={voiceUi}
        live={live}
        facts={facts}
        intent={intent}
      />
    </div>
  );
}

function buildResolutionJourney({
  issueStatus,
  localTurns,
  assistMeta,
  voiceUi,
  live,
  facts,
  intent,
}: {
  issueStatus: string;
  localTurns: { text: string; speaker?: number }[];
  assistMeta: LiveNudge | null;
  voiceUi: VoiceUiState;
  live: Record<string, any> | null;
  facts: AccountFacts | null;
  intent?: string;
}): AssistPipelineStep[] {
  const stages: { key: string; label: string }[] = [
    { key: "describe", label: "Customer describes the issue" },
    { key: "understand", label: "Request understood" },
    { key: "review", label: "Account reviewed with Genie" },
    { key: "offer", label: "Resolution offered to customer" },
    { key: "apply", label: "Agreement applied to billing" },
    { key: "close", label: "Issue closed" },
  ];

  const hasCustomerTurn = localTurns.some((t) => (t.speaker ?? 0) === 1);
  const hasAgentTurn = localTurns.some((t) => (t.speaker ?? 0) === 0);
  const lastCustomer = [...localTurns].reverse().find((t) => (t.speaker ?? 0) === 1)?.text;
  const resolution = assistMeta?.resolution;
  const status = String(resolution?.status ?? issueStatus ?? "open");
  const actions = (resolution?.actions ?? {}) as Record<string, unknown>;
  const nudge = assistMeta?.live ?? live ?? {};
  const billing = assistMeta?.billing;
  const overdue = Number(facts?.summary?.overdue_amount ?? 0);
  const overdueCount = Number(facts?.summary?.overdue_invoice_count ?? 0);

  let doneThrough = -1;
  if (hasCustomerTurn || voiceUi.phase === "speaking" || voiceUi.phase === "transcribing") {
    doneThrough = 0;
  }
  if (assistMeta && hasCustomerTurn) doneThrough = 1;
  if (assistMeta?.agent_validation || assistMeta?.agent_reply) doneThrough = 2;
  if (hasAgentTurn || assistMeta?.agent_reply) doneThrough = 3;
  if (billing?.applied) doneThrough = 4;
  if (status === "closed") doneThrough = 5;

  if (status === "closed" && !hasCustomerTurn && !assistMeta) {
    doneThrough = 5;
  }

  let activeKey: string | null = null;
  const inProgressDetail: Record<string, string> = {};

  if (voiceUi.phase === "speaking") {
    activeKey = "describe";
    inProgressDetail.describe =
      voiceUi.interimText?.trim() || "Listening — customer is explaining the issue…";
  } else if (voiceUi.phase === "transcribing") {
    activeKey = "describe";
    inProgressDetail.describe =
      voiceUi.interimText?.trim() || "Capturing what the customer said…";
  } else if (voiceUi.phase === "agent_reply") {
    activeKey = assistMeta ? "offer" : "review";
    if (!assistMeta) {
      inProgressDetail.understand = "Understanding the customer's billing request…";
      inProgressDetail.review =
        "Genie is reviewing account facts and preparing the resolution offer for the agent…";
    }
  } else if (
    !billing?.applied &&
    status !== "closed" &&
    (String(nudge.customer_signal) === "confirm_proceed" || actions.pending_close)
  ) {
    activeKey = "apply";
    inProgressDetail.apply = "Applying the agreed payment arrangement and waiver to billing…";
  }

  const details: Record<string, string> = {};
  if (doneThrough >= 0) {
    details.describe =
      voiceUi.interimText?.trim() ||
      lastCustomer?.slice(0, 120) ||
      "Customer explains their billing concern on the call.";
  }
  if (doneThrough >= 1) {
    const plan = nudge.payment_plan_requested ? "payment plan" : null;
    const waiver = nudge.waiver_requested ? "late fee relief" : null;
    const extras = [plan, waiver].filter(Boolean).join(" + ");
    details.understand = extras
      ? `${intentLabel(intent)} — customer asked for ${extras}`
      : intentLabel(intent) || "Billing concern identified from the conversation.";
  }
  if (doneThrough >= 2) {
    if (overdueCount > 0) {
      details.review = `Genie confirmed ${overdueCount} overdue invoice(s) totaling $${overdue.toFixed(2)}.`;
    } else if (assistMeta?.agent_validation?.reply_available) {
      details.review = "Account facts checked against governed billing records.";
    } else {
      details.review = "Account context reviewed before offering next steps.";
    }
  }
  if (doneThrough >= 3) {
    if (actions.waiver_requested && actions.payment_plan_requested) {
      details.offer = "Agent proposed a payment arrangement and late fee waiver.";
    } else if (actions.waiver_requested) {
      details.offer = "Agent proposed late fee relief on the overdue balance.";
    } else if (actions.payment_plan_requested) {
      details.offer = "Agent proposed a payment arrangement.";
    } else {
      details.offer = "Agent shared next steps to resolve the billing issue.";
    }
  }
  if (doneThrough >= 4) {
    if (billing?.applied) {
      details.apply = `Billing updated (${String(billing.adjustment?.invoice_id ?? "invoice")}).`;
    } else if (billing && !billing.applied) {
      details.apply = `Billing not updated: ${billing.reason ?? "pending customer confirmation"}.`;
    } else if (status === "closed") {
      details.apply = "Payment arrangement and waiver recorded on the account.";
    }
  } else if (actions.waiver_requested || actions.payment_plan_requested) {
    details.apply = "Waiting for customer confirmation before updating billing.";
  }
  if (doneThrough >= 5) {
    details.close =
      resolution?.note ||
      facts?.summary?.resolution_note ||
      "Issue closed — customer informed that changes will appear on the next statement.";
  }

  if (doneThrough < 0 && voiceUi.phase === "idle" && !assistMeta) {
    return [
      {
        key: "waiting",
        label: "Awaiting customer",
        status: "pending",
        detail: "The resolution journey begins when the customer describes their issue.",
      },
    ];
  }

  return stages.map((stage, idx) => {
    const isActive = activeKey === stage.key;
    const isDone = !isActive && idx <= doneThrough;
    return {
      key: stage.key,
      label: stage.label,
      status: isActive ? "active" : isDone ? "done" : "pending",
      detail: (isActive && inProgressDetail[stage.key]) || details[stage.key],
    };
  });
}

function ResolutionJourneyStrip({
  issueStatus,
  localTurns,
  assistMeta,
  voiceUi,
  live,
  facts,
  intent,
}: {
  issueStatus: string;
  localTurns: { text: string; speaker?: number }[];
  assistMeta: LiveNudge | null;
  voiceUi: VoiceUiState;
  live: Record<string, any> | null;
  facts: AccountFacts | null;
  intent?: string;
}) {
  const steps = buildResolutionJourney({
    issueStatus,
    localTurns,
    assistMeta,
    voiceUi,
    live,
    facts,
    intent,
  });

  return (
    <div className="assist-pipeline-strip resolution-journey">
      <div className="assist-pipeline-head">
        <span className="panel-title">Issue resolution journey</span>
        <span className="assist-pipeline-total">Status: {issueStatus}</span>
      </div>
      <div className="assist-pipeline-track">
        {steps.map((step, idx) => (
          <div key={step.key} className={`assist-pipeline-step s-${step.status}`}>
            <div className="assist-pipeline-node">
              <span className="assist-pipeline-index">{idx + 1}</span>
            </div>
            <div className="assist-pipeline-copy">
              <div className="assist-pipeline-label">{step.label}</div>
              {step.detail && <div className="assist-pipeline-detail">{step.detail}</div>}
            </div>
            {idx < steps.length - 1 && <div className="assist-pipeline-connector" aria-hidden="true" />}
          </div>
        ))}
      </div>
    </div>
  );
}

function RecommendationCard({
  rec,
  intent,
  sentiment,
}: {
  rec: ReturnType<typeof recommend>;
  intent?: string;
  sentiment?: string;
}) {
  return (
    <div className={`rec-card r-${rec.priority} cc-recommendation`}>
      <div className="rec-top">
        <span className="rec-kicker">Recommended next action</span>
        <span className="rec-tags">
          <span className="chip">{intentLabel(intent)}</span>
          <span className={`badge sentiment ${sentiment ?? "neutral"}`}>{sentiment ?? "—"}</span>
        </span>
      </div>
      <div className="rec-title">{rec.title}</div>
      <div className="rec-detail">{rec.detail}</div>
    </div>
  );
}

function Fact({ label, value, warn }: { label: string; value: ReactNode; warn?: boolean }) {
  return (
    <div className={`fact ${warn ? "fact-warn" : ""}`}>
      <div className="fact-val">{value}</div>
      <div className="fact-label">{label}</div>
    </div>
  );
}

function LiveWaveform({ level, active }: { level: number; active: boolean }) {
  const bars = [0.35, 0.55, 0.75, 1, 0.8, 0.6, 0.45, 0.3];
  return (
    <div className={`wave-wrap ${active ? "wave-live" : ""}`} aria-hidden="true">
      {bars.map((weight, i) => (
        <span
          key={i}
          style={{ height: `${Math.max(4, Math.round(4 + level * 14 * weight))}px` }}
        />
      ))}
    </div>
  );
}

function AssistStatusPanel({ meta }: { meta: LiveNudge | null }) {
  if (!meta) return null;
  const validation = meta.agent_validation;
  const billing = meta.billing;
  const closeBlock = meta.close_block_reason;
  const resolutionStatus = meta.resolution?.status;
  const hasContent =
    validation || billing || closeBlock || (resolutionStatus && resolutionStatus !== "open");
  if (!hasContent) return null;

  return (
    <div className="assist-status-panel">
      {resolutionStatus && (
        <div className="assist-status-row">
          <span className="assist-status-label">Resolution</span>
          <span className={`status-chip issue issue-${resolutionStatus}`}>{resolutionStatus}</span>
        </div>
      )}
      {closeBlock && (
        <div className="assist-status-row warn">
          <span className="assist-status-label">Close blocked</span>
          <span>{closeBlock}</span>
        </div>
      )}
      {billing && (
        <div className="assist-status-row">
          <span className="assist-status-label">Billing</span>
          <span>
            {billing.applied
              ? `applied (${String(billing.adjustment?.invoice_id ?? "invoice")})`
              : `not applied: ${billing.reason ?? "unknown"}`}
          </span>
        </div>
      )}
      {validation && (
        <div className="assist-status-row">
          <span className="assist-status-label">Genie validation</span>
          <span>
            {validation.reply_available
              ? "reply validated"
              : validation.genie_error ?? "reply unavailable"}
            {validation.mismatches?.length
              ? ` · mismatches: ${validation.mismatches.join("; ")}`
              : ""}
            {validation.output_issues?.length
              ? ` · output: ${validation.output_issues.join("; ")}`
              : ""}
          </span>
        </div>
      )}
    </div>
  );
}

function LiveAssist({
  callId,
  sttProvider,
  onNudge,
  onLocalTurn,
  onVoiceUiChange,
}: {
  callId: string;
  customerId: string;
  sttProvider: string;
  onNudge: (n: LiveNudge) => void;
  onLocalTurn: (turn: { text: string; speaker?: number }) => void;
  onVoiceUiChange: (state: VoiceUiState) => void;
}) {
  const [text, setText] = useState("");
  const [speaker, setSpeaker] = useState<number>(1);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [recording, setRecording] = useState(false);
  const micSessionRef = useRef<MicStreamSession | MicRecordingSession | null>(null);
  const captionRef = useRef<SpeechCaptionSession | null>(null);
  const voiceRef = useRef({ interim: "", level: 0.15 });
  const voicePhaseRef = useRef<VoiceUiState["phase"]>("idle");
  const captionSupported = useMemo(() => isSpeechCaptionSupported(), []);

  const applyAgentReply = (nudge: LiveNudge) => {
    const reply = nudge.agent_reply?.trim();
    if (!reply) return;
    onLocalTurn({ text: reply, speaker: 0 });
  };

  const send = async () => {
    if (!text.trim()) return;
    const msg = text.trim();
    setBusy(true);
    setErr(null);
    try {
      onLocalTurn({ text: msg, speaker });
      if (speaker === 1) {
        onVoiceUiChange({
          phase: "agent_reply",
          source: "text",
          processingLabel: "Genie is preparing the agent response…",
        });
      }

      const n = await api.sendUtterance(callId, msg, speaker);
      onNudge(n);

      if (speaker === 1) {
        applyAgentReply(n);
        onVoiceUiChange({ phase: "idle" });
      }

      setText("");
    } catch (e) {
      onVoiceUiChange({ phase: "idle" });
      setErr(e instanceof Error ? e.message : "failed");
    } finally {
      setBusy(false);
    }
  };

  useEffect(() => {
    return () => {
      captionRef.current?.close();
      captionRef.current = null;
      micSessionRef.current?.close();
      micSessionRef.current = null;
      voicePhaseRef.current = "idle";
      onVoiceUiChange({ phase: "idle" });
    };
  }, [onVoiceUiChange]);

  const startMic = async () => {
    if (recording || busy) return;
    setErr(null);
    try {
      voiceRef.current = { interim: "", level: 0.15 };
      voicePhaseRef.current = "speaking";
      if (sttProvider === "databricks") {
        const session = await startMicRecording((level) => {
          voiceRef.current.level = level;
          if (voicePhaseRef.current !== "speaking") return;
          onVoiceUiChange({
            phase: "speaking",
            source: "mic",
            interimText: voiceRef.current.interim,
            micLevel: level,
          });
        });
        micSessionRef.current = session;
        // Best-effort on-device live caption so the audience can read along while
        // the Databricks model produces the authoritative transcript on stop.
        captionRef.current = startSpeechCaption((caption) => {
          voiceRef.current.interim = caption;
          if (voicePhaseRef.current !== "speaking") return;
          onVoiceUiChange({
            phase: "speaking",
            source: "mic",
            interimText: caption,
            micLevel: voiceRef.current.level,
          });
        });
        setRecording(true);
        onVoiceUiChange({
          phase: "speaking",
          source: "mic",
          interimText: "",
          processingLabel: "Listening…",
          micLevel: 0.15,
        });
        return;
      }

      const wsUrl = `${WS_BASE_URL}/calls/${callId}/mic-stream`;
      const session = await startMicStream(
        wsUrl,
        (transcript) => {
          if (transcript) voiceRef.current.interim = transcript;
          if (voicePhaseRef.current !== "speaking") return;
          onVoiceUiChange({
            phase: "speaking",
            source: "mic",
            interimText: voiceRef.current.interim,
            micLevel: voiceRef.current.level,
          });
        },
        (level) => {
          voiceRef.current.level = level;
          if (voicePhaseRef.current !== "speaking") return;
          onVoiceUiChange({
            phase: "speaking",
            source: "mic",
            interimText: voiceRef.current.interim,
            micLevel: level,
          });
        },
        (message) => setErr(message)
      );
      micSessionRef.current = session;
      setRecording(true);
      onVoiceUiChange({
        phase: "speaking",
        source: "mic",
        interimText: "",
        processingLabel: "Listening…",
        micLevel: 0.15,
      });
    } catch (e) {
      voicePhaseRef.current = "idle";
      onVoiceUiChange({ phase: "idle" });
      setErr(e instanceof Error ? e.message : "Unable to access microphone");
    }
  };

  const stopMic = async () => {
    const session = micSessionRef.current;
    if (!session || !recording) return;
    setRecording(false);
    setBusy(true);
    voicePhaseRef.current = "transcribing";
    onVoiceUiChange({
      phase: "transcribing",
      source: "mic",
      interimText: voiceRef.current.interim,
      processingLabel:
        sttProvider === "databricks"
          ? "Processing voice with Databricks model…"
          : "Processing voice with Deepgram…",
    });
    try {
      if (sttProvider === "databricks") {
        captionRef.current?.stop();
        captionRef.current = null;
        const recording = await (session as MicRecordingSession).stop();
        micSessionRef.current = null;
        const n = await api.transcribeMic(
          callId,
          recording.audioBase64,
          recording.mimeType,
          1
        );
        const textFromMic = String(n.transcript || "").trim();
        if (!textFromMic) throw new Error("No transcript returned from Databricks model");
        onLocalTurn({ text: textFromMic, speaker: 1 });
        voicePhaseRef.current = "agent_reply";
        onVoiceUiChange({
          phase: "agent_reply",
          source: "mic",
          processingLabel: "Genie is preparing the agent response…",
        });
        onNudge(n);
        applyAgentReply(n);
        voicePhaseRef.current = "idle";
        onVoiceUiChange({ phase: "idle" });
        return;
      }

      const textFromMic = (await (session as MicStreamSession).stop()).trim();
      micSessionRef.current = null;
      if (!textFromMic) throw new Error("No transcript returned from Deepgram");
      onLocalTurn({ text: textFromMic, speaker: 1 });
      voicePhaseRef.current = "agent_reply";
      onVoiceUiChange({
        phase: "agent_reply",
        source: "mic",
        processingLabel: "Genie is preparing the agent response…",
      });
      const n = await api.sendUtterance(callId, textFromMic, 1);
      onNudge(n);
      applyAgentReply(n);
      voicePhaseRef.current = "idle";
      onVoiceUiChange({ phase: "idle" });
    } catch (e) {
      captionRef.current?.close();
      captionRef.current = null;
      voicePhaseRef.current = "idle";
      onVoiceUiChange({ phase: "idle" });
      setErr(e instanceof Error ? e.message : "mic transcription failed");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="live-assist">
      <div className="live-input">
        <select
          value={speaker}
          onChange={(e) => setSpeaker(Number(e.target.value))}
          className="speaker-select"
        >
          <option value={1}>Customer</option>
          <option value={0}>Agent</option>
        </select>
        <input
          value={text}
          placeholder="Type the next customer/agent utterance..."
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && send()}
        />
        <button onClick={send} disabled={busy}>
          {busy ? "…" : "Send"}
        </button>
        <button onClick={recording ? () => void stopMic() : () => void startMic()} disabled={busy}>
          {recording ? "Stop Mic" : "Mic"}
        </button>
      </div>
      {sttProvider === "databricks" && (
        <div
          className={`caption-status ${captionSupported ? "ok" : "off"}`}
          title={
            captionSupported
              ? "Your browser supports a live on-screen caption while you speak. The Databricks model still produces the final transcript on stop."
              : "This browser has no Web Speech API, so the live caption is skipped. Recording and the Databricks transcript are unaffected. Try Chrome, Edge, or Safari."
          }
        >
          <span className="caption-dot" />
          {captionSupported
            ? "Live caption available · final transcript by Databricks model"
            : "Live caption unavailable in this browser · Databricks transcript on stop"}
        </div>
      )}
      {err && <div className="account-error">{err}</div>}
    </div>
  );
}
