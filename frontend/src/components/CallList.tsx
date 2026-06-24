import { ReactNode, useEffect, useMemo, useRef, useState } from "react";
import { AccountFacts, api, CallState, LiveNudge, ResolutionEvent } from "../api/client";
import { WS_BASE_URL } from "../config";
import { callPriority, intentLabel, PRIORITY_RANK, recommend } from "../guidance";
import { startMicStream, VoiceUiState } from "../lib/micStream";
import databricksLogo from "../assets/databricks-logo.png";
import genieLogo from "../assets/genie-logo.png";

const SPOTLIGHT_CUSTOMER = {
  id: "CUST-4028",
  name: "Omar Patel",
  rationale:
    "at_risk account with autopay off, overdue invoice exposure, and declined payments",
};

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

export function CallList({ calls }: { calls: CallState[] }) {
  const spotlightCall = useMemo(
    () => calls.find((c) => c.customer_id === SPOTLIGHT_CUSTOMER.id) ?? null,
    [calls]
  );

  const sorted = useMemo(() => {
    const base = [...calls].sort((a, b) => {
      const sa = signalsOf(a);
      const sb = signalsOf(b);
      const pa = callPriority(sa.nba, sa.disposition, sa.sentiment);
      const pb = callPriority(sb.nba, sb.disposition, sb.sentiment);
      return PRIORITY_RANK[pa] - PRIORITY_RANK[pb];
    });
    if (!spotlightCall) return base;
    return [
      spotlightCall,
      ...base.filter((c) => c.call_id !== spotlightCall.call_id),
    ];
  }, [calls, spotlightCall]);

  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [userPicked, setUserPicked] = useState(false);
  const [conversationByCall, setConversationByCall] = useState<
    Record<string, { text: string; speaker?: number }[]>
  >({});

  useEffect(() => {
    if (!sorted.length || userPicked) return;
    setSelectedId((spotlightCall ?? sorted[0]).call_id);
  }, [sorted, userPicked]);

  const selected =
    sorted.find((c) => c.call_id === selectedId) ?? sorted[0] ?? null;

  if (!calls.length) return <p className="muted">No calls processed yet…</p>;

  return (
    <div className="cc-layout">
      <aside className="cc-sidebar">
        <div className="cc-stack-brand">
          <img className="hero-logo dbx-full side" src={databricksLogo} alt="Databricks" />
          <img className="hero-logo genie-full side" src={genieLogo} alt="Genie" />
        </div>
        <div className="cc-sidebar-title">Live Call Queue</div>
        <div className="cc-sidebar-sub">Select a call to open the assist cockpit</div>
        <div className="cc-spotlight">
          <div className="cc-spotlight-label">Spotlight customer</div>
          <div className="cc-spotlight-name">
            {SPOTLIGHT_CUSTOMER.name} ({SPOTLIGHT_CUSTOMER.id})
          </div>
          <div className="cc-spotlight-call">
            CALL ID: {spotlightCall?.call_id ?? "not in current queue"}
          </div>
          {spotlightCall && selected?.call_id !== spotlightCall.call_id && (
            <button
              className="cc-spotlight-jump"
              onClick={() => {
                setUserPicked(true);
                setSelectedId(spotlightCall.call_id);
              }}
            >
              Switch to spotlight call
            </button>
          )}
          <div className="cc-spotlight-note">{SPOTLIGHT_CUSTOMER.rationale}</div>
        </div>
        {sorted.map((c) => {
          const s = signalsOf(c);
          const prio = callPriority(s.nba, s.disposition, s.sentiment);
          const active = selected?.call_id === c.call_id;
          return (
            <button
              key={c.call_id}
              className={`cc-call-row ${active ? "active" : ""}`}
              onClick={() => {
                setUserPicked(true);
                setSelectedId(c.call_id);
              }}
            >
              <span className={`prio-dot p-${prio}`} />
              <span className="cc-call-main">
                <span className="cc-call-id">{c.call_id}</span>
                <span className="cc-call-intent">{intentLabel(s.intent)}</span>
              </span>
              <span className={`badge sentiment cc-sentiment ${s.sentiment ?? "neutral"}`}>
                {s.sentiment ?? "—"}
              </span>
            </button>
          );
        })}
      </aside>

      {selected && (
        <Cockpit
          call={selected}
          localTurns={conversationByCall[selected.call_id] ?? []}
          onAppendLocalTurn={(turn) =>
            setConversationByCall((prev) => ({
              ...prev,
              [selected.call_id]: [...(prev[selected.call_id] ?? []), turn],
            }))
          }
          onResetLocalTurns={() =>
            setConversationByCall((prev) => ({
              ...prev,
              [selected.call_id]: [],
            }))
          }
        />
      )}
    </div>
  );
}

function Cockpit({
  call,
  localTurns,
  onAppendLocalTurn,
  onResetLocalTurns,
}: {
  call: CallState;
  localTurns: { text: string; speaker?: number }[];
  onAppendLocalTurn: (turn: { text: string; speaker?: number }) => void;
  onResetLocalTurns: () => void;
}) {
  const base = signalsOf(call);
  const [facts, setFacts] = useState<AccountFacts | null>(null);
  const [factErr, setFactErr] = useState<string | null>(null);
  const [live, setLive] = useState<Record<string, any> | null>(null);
  const [genieQuestion, setGenieQuestion] = useState("");
  const [genieAnswer, setGenieAnswer] = useState<string | null>(null);
  const [genieSql, setGenieSql] = useState<string | null>(null);
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
    setGenieAnswer(null);
    setGenieSql(null);
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

  const askGenie = async (question: string) => {
    if (!question.trim()) return;
    setGenieLoading(true);
    setGenieErr(null);
    try {
      const resp = await api.askGenie(question);
      setGenieAnswer(resp.answer ?? "No answer returned.");
      setGenieSql(resp.sql ?? null);
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
      setGenieAnswer(null);
      setGenieSql(null);
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
            {genieAnswer && <div className="genie-answer">{genieAnswer}</div>}
            {!genieAnswer && (
              <div className="genie-hint">
                Designed around {SPOTLIGHT_CUSTOMER.name}: ask for payment arrangement + late fee relief.
              </div>
            )}
            {genieSql && <pre className="sql">{genieSql}</pre>}
          </div>
        </div>
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
  customerId,
  onNudge,
  onLocalTurn,
  onVoiceUiChange,
}: {
  callId: string;
  customerId: string;
  onNudge: (n: LiveNudge) => void;
  onLocalTurn: (turn: { text: string; speaker?: number }) => void;
  onVoiceUiChange: (state: VoiceUiState) => void;
}) {
  const [text, setText] = useState("");
  const [speaker, setSpeaker] = useState<number>(1);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [recording, setRecording] = useState(false);
  const micSessionRef = useRef<Awaited<ReturnType<typeof startMicStream>> | null>(null);
  const voiceRef = useRef({ interim: "", level: 0.15 });
  const voicePhaseRef = useRef<VoiceUiState["phase"]>("idle");

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
      const wsUrl = `${WS_BASE_URL}/calls/${callId}/mic-stream`;
      const session = await startMicStream(
        wsUrl,
        (transcript) => {
          if (transcript) voiceRef.current.interim = transcript;
          if (voicePhaseRef.current !== "speaking") return;
          onVoiceUiChange({
            phase: "speaking",
            interimText: voiceRef.current.interim,
            micLevel: voiceRef.current.level,
          });
        },
        (level) => {
          voiceRef.current.level = level;
          if (voicePhaseRef.current !== "speaking") return;
          onVoiceUiChange({
            phase: "speaking",
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
      interimText: voiceRef.current.interim,
      processingLabel: "Processing voice with Deepgram…",
    });
    try {
      const textFromMic = (await session.stop()).trim();
      micSessionRef.current = null;
      if (!textFromMic) throw new Error("No transcript returned from Deepgram");
      onLocalTurn({ text: textFromMic, speaker: 1 });
      voicePhaseRef.current = "agent_reply";
      onVoiceUiChange({
        phase: "agent_reply",
        processingLabel: "Genie is preparing the agent response…",
      });
      const n = await api.sendUtterance(callId, textFromMic, 1);
      onNudge(n);
      applyAgentReply(n);
      voicePhaseRef.current = "idle";
      onVoiceUiChange({ phase: "idle" });
    } catch (e) {
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
      {err && <div className="account-error">{err}</div>}
    </div>
  );
}
