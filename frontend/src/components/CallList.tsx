import { ReactNode, useEffect, useMemo, useRef, useState } from "react";
import {
  AccountFacts,
  api,
  AssistPipelineStep,
  CallState,
  CustomerWithIssue,
  GenieResponse,
  InteractionLanguage,
  InteractionLanguageOption,
  INTERACTION_LANGUAGES,
  LiveNudge,
  ResolutionEvent,
} from "../api/client";
import { WS_BASE_URL } from "../config";
import { PRIORITY_RANK, recommend } from "../guidance";
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
import {
  localizedValue,
  localizeRationale,
  localizeResolutionNote,
  uiCopy,
} from "../i18n";
import databricksLogo from "../assets/databricks-logo.png";
import genieLogo from "../assets/genie-logo.png";

const SPOTLIGHT_CUSTOMER_ID = "CUST-4028";

type LocalTurn = { text: string; speaker?: number; language?: InteractionLanguage };

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

function localizedIntentLabel(language: InteractionLanguage, code?: string | null): string {
  if (!code) return "—";
  return localizedValue(language, code, "intent");
}

function localizedRecommendation(
  language: InteractionLanguage,
  nba: string | undefined,
  sentiment: string | undefined,
  facts?: AccountFacts | null
) {
  const base = recommend(nba, sentiment, facts);
  if (language !== "th-TH") return base;
  const overdue = facts?.invoices?.find((i) => i.status === "overdue");
  const amount = (value: unknown) => {
    const n = typeof value === "string" ? parseFloat(value) : (value as number);
    return Number.isFinite(n) ? `$${n.toFixed(2)}` : "$0.00";
  };
  switch (nba) {
    case "offer_fee_waiver":
      return {
        ...base,
        title: overdue
          ? `เสนอการยกเว้นค่าปรับ ${amount(overdue.late_fee)} สำหรับ ${overdue.invoice_id}`
          : "เสนอการยกเว้นค่าปรับ",
        detail: overdue
          ? `${overdue.invoice_id} (${overdue.period}) ค้างชำระ ${amount(overdue.amount)} + ค่าปรับ ${amount(
              overdue.late_fee
            )} ครบกำหนด ${overdue.due_date} การยกเว้นค่าปรับช่วยลดข้อโต้แย้งและรักษาประสบการณ์ลูกค้า`
          : "รับทราบค่าปรับและเสนอการยกเว้นแบบ goodwill หนึ่งครั้งเพื่อลดความตึงเครียด",
      };
    case "set_up_payment_plan":
      return {
        ...base,
        title: "ตั้งแผนชำระเงิน",
        detail: facts?.summary?.overdue_amount
          ? `เสนอแบ่งยอดค้างชำระ ${amount(facts.summary.overdue_amount)} เป็นงวด และเปิดชำระอัตโนมัติเพื่อลดค่าปรับในอนาคต`
          : "เสนอแบ่งยอดค้างชำระเป็นงวดเพื่อให้บัญชีกลับมาปกติ",
      };
    case "process_refund":
      return {
        ...base,
        title: "ดำเนินการคืนเงิน",
        detail: overdue
          ? `ตรวจสอบรายการที่โต้แย้งใน ${overdue.invoice_id} (${amount(overdue.amount)}) แล้วออกเงินคืนหรือเครดิตเข้าบัญชี`
          : "ตรวจสอบรายการที่โต้แย้งแล้วออกเงินคืนหรือเครดิตเข้าบัญชี",
      };
    case "escalate_retention_offer":
      return {
        ...base,
        title: "ส่งต่อพร้อมข้อเสนอรักษาลูกค้า",
        detail: "ลูกค้าถูกจัดว่ามีความเสี่ยง ให้ประสานทีม retention และเริ่มด้วยเครดิตหรือส่วนลดแพ็กเกจก่อนลูกค้าขอยกเลิก",
      };
    default:
      return {
        ...base,
        title: "ช่วยเหลือต่อได้ - ยังไม่ต้องยกระดับ",
        detail: "อารมณ์ลูกค้ายังคงที่และไม่พบความเสี่ยงด้านบิล ให้ตอบคำถาม ยืนยันขั้นตอนถัดไป และปิดบทสนทนาอย่างสุภาพ",
      };
  }
}

export function CallList({
  calls,
  sttProvider,
  customers,
  customersLoading,
  customersErr,
  languages,
  selectedLanguage,
  onLanguageChange,
}: {
  calls: CallState[];
  sttProvider: string;
  customers: CustomerWithIssue[];
  customersLoading: boolean;
  customersErr: string | null;
  languages?: {
    default?: InteractionLanguage;
    supported?: InteractionLanguageOption[];
  };
  selectedLanguage: InteractionLanguage;
  onLanguageChange: (language: InteractionLanguage) => void;
}) {
  const copy = uiCopy(selectedLanguage);
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
    Record<string, LocalTurn[]>
  >({});

  if (customersLoading && !customers.length) {
    return <p className="muted">{copy.loadingCustomers}</p>;
  }

  if (!sortedCustomers.length) {
    return (
      <p className="muted">
        {customersErr
          ? `${copy.unableToLoadCustomers}: ${customersErr}`
          : copy.noCustomers}
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
        <div className="cc-sidebar-title">{copy.sidebarTitle}</div>
        <div className="cc-sidebar-sub">{copy.sidebarSubtitle}</div>
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
                <span className="cc-call-intent">
                  {c.rationale
                    ? localizeRationale(selectedLanguage, c.rationale)
                    : localizedIntentLabel(selectedLanguage, c.primary_intent)}
                </span>
                {c.call_id && <span className="cc-customer-call">{copy.call} {c.call_id}</span>}
              </span>
              <span className={`badge sentiment cc-sentiment ${c.sentiment_label ?? "neutral"}`}>
                {localizedValue(selectedLanguage, c.sentiment_label ?? c.issue_status)}
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
          languageOptions={languages?.supported}
          defaultLanguage={languages?.default}
          selectedLanguage={selectedLanguage}
          onLanguageChange={onLanguageChange}
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
          <div className="eyebrow">{copy.noLiveCall}</div>
          <h2>{selectedCustomer?.full_name ?? selectedCustomer?.customer_id}</h2>
          <p className="muted">{copy.noLiveCallDetail}</p>
          {selectedCustomer?.rationale && <p>{localizeRationale(selectedLanguage, selectedCustomer.rationale)}</p>}
        </div>
      )}
    </div>
  );
}

function Cockpit({
  call,
  customer,
  sttProvider,
  languageOptions,
  defaultLanguage,
  selectedLanguage,
  onLanguageChange,
  localTurns,
  onAppendLocalTurn,
  onResetLocalTurns,
}: {
  call: CallState;
  customer: CustomerWithIssue | null;
  sttProvider: string;
  languageOptions?: InteractionLanguageOption[];
  defaultLanguage?: InteractionLanguage;
  selectedLanguage: InteractionLanguage;
  onLanguageChange: (language: InteractionLanguage) => void;
  localTurns: LocalTurn[];
  onAppendLocalTurn: (turn: LocalTurn) => void;
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
  const availableLanguages =
    languageOptions && languageOptions.length > 0 ? languageOptions : INTERACTION_LANGUAGES;
  const language = selectedLanguage;
  const copy = uiCopy(language);

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
    api.prefetchGenieInsight(call.call_id, language).catch(() => {});
    return () => {
      active = false;
    };
  }, [call.call_id, language]);

  useEffect(() => {
    const fallback = defaultLanguage ?? availableLanguages[0]?.code ?? "en-US";
    if (!availableLanguages.some((item) => item.code === language)) {
      onLanguageChange(fallback);
    }
  }, [availableLanguages, defaultLanguage, language, onLanguageChange]);

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
          title: copy.issueResolvedTitle,
          detail:
            sum.resolution_note ??
            copy.issueResolvedDetail,
          priority: "low" as const,
        }
      : hasAgentTurn
      ? localizedRecommendation(language, nba, sentiment, facts)
      : {
          title: copy.listeningTitle,
          detail: copy.listeningDetail,
          priority: "low" as const,
        };
  const overdueCount = Number(sum.overdue_invoice_count ?? 0);
  const overdueAmount = Number(sum.overdue_amount ?? 0);
  const riskLevel = overdueCount > 0 || !sum.autopay_enabled ? "elevated" : "stable";
  const suggestedQuestion =
    facts?.customer_id || call.customer_id
      ? copy.suggestedAssistQuestion(String(facts?.customer_id ?? call.customer_id), call.call_id)
      : copy.suggestedCallQuestion(call.call_id);

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
        asFollowup ? genieResp?.conversation_id : undefined,
        language
      );
      setGenieResp(resp);
      setGenieShowSql(false);
    } catch (e) {
      setGenieErr(e instanceof Error ? e.message : copy.replyUnavailable);
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
      setFactErr(e instanceof Error ? e.message : copy.unknown);
    } finally {
      setResetBusy(false);
    }
  };

  return (
    <div className="cc-main">
      <div className="cc-top">
        <div>
          <div className="eyebrow">{copy.activeCustomer}</div>
          <div className="cust-name">{cust.full_name ?? call.customer_id ?? call.call_id}</div>
          <div className="cust-sub">
            {[cust.segment, cust.plan, cust.region]
              .filter(Boolean)
              .map((item) => localizedValue(language, item, "profile"))
              .join(" · ") || copy.customerProfileLoading}
            {cust.tenure_months != null && <> · {copy.monthsTenure(Number(cust.tenure_months))}</>}
          </div>
        </div>
        <div className="cust-status studio-status">
          <label className="language-control">
            <span>{copy.interactionLanguage}</span>
            <select
              value={language}
              onChange={(e) => {
                onLanguageChange(e.target.value as InteractionLanguage);
                setGenieResp(null);
                setGenieShowSql(false);
              }}
              disabled={voiceUi.phase !== "idle"}
            >
              {availableLanguages.map((item) => (
                <option key={item.code} value={item.code}>
                  {item.label}
                </option>
              ))}
            </select>
          </label>
          <span className={`status-chip issue issue-${issueStatus}`}>
            {copy.issue}: {localizedValue(language, issueStatus)}
          </span>
          <span className={`status-chip ${riskLevel === "elevated" ? "st-at_risk" : "st-active"}`}>
            {copy.risk}: {localizedValue(language, riskLevel)}
          </span>
          {cust.status && (
            <span className={`status-chip st-${cust.status}`}>{localizedValue(language, cust.status)}</span>
          )}
          {cust.monthly_charge != null && (
            <span className="cust-arpu">${cust.monthly_charge}/{copy.perMonth}</span>
          )}
        </div>
      </div>
      {sum.resolution_note && (
        <div className="resolution-banner">
          {localizeResolutionNote(language, String(sum.resolution_note))}
          {sum.resolved_at ? ` (resolved at ${sum.resolved_at})` : ""}
        </div>
      )}

      <div className="cc-grid">
        <div className="panel cc-conversation">
          <RecommendationCard
            rec={rec}
            intent={intent}
            sentiment={sentiment}
            label={copy.recommendedNextAction}
            language={language}
          />
          <div className="panel-title convo-title-row">
            <span>{copy.conversationStream}</span>
            <button className="ghost mini" onClick={resetScenario} disabled={resetBusy}>
              {resetBusy ? copy.resetting : copy.resetScenario}
            </button>
          </div>
          <div className="transcript cc-transcript">
            {utterances.length === 0 && voiceUi.phase === "idle" && (
              <div className="muted">{copy.noTranscript}</div>
            )}
            {utterances.map((u, i) => {
              const isCustomer = (u.speaker ?? 0) === 1;
              return (
                <div key={i} className={`turn ${isCustomer ? "t-customer" : "t-agent"}`}>
                  <span className="turn-who">
                    {isCustomer ? copy.customer : copy.agentGenieAssisted}
                    {u.language && <span className="turn-language"> · {u.language}</span>}
                  </span>
                  <span className="turn-text">{u.text}</span>
                </div>
              );
            })}
            {voiceUi.phase === "speaking" && (
              <div className="turn t-customer turn-live">
                <span className="turn-who">{copy.customerSpeaking}</span>
                <span className="turn-text turn-placeholder">
                  {voiceUi.interimText?.trim() || copy.listening}
                </span>
                <LiveWaveform level={voiceUi.micLevel ?? 0.2} active />
              </div>
            )}
            {voiceUi.phase === "transcribing" && (
              <div className="turn t-customer turn-live">
                <span className="turn-who">{copy.customer}</span>
                <span className="turn-text turn-placeholder transcribing">
                  {voiceUi.interimText?.trim() || voiceUi.processingLabel || copy.transcribingMessage}
                </span>
              </div>
            )}
            {voiceUi.phase === "agent_reply" && (
              <div className="turn t-agent turn-live">
                <span className="turn-who">{copy.agentGenieAssisted}</span>
                <span className="turn-text turn-placeholder transcribing">
                  {voiceUi.processingLabel || copy.preparingGenieResponse}
                </span>
              </div>
            )}
          </div>
          <LiveAssist
            callId={call.call_id}
            customerId={String(facts?.customer_id ?? call.customer_id ?? "")}
            sttProvider={sttProvider}
            language={language}
            onNudge={(n) => {
              setLive(n.live);
              setAssistMeta(n);
              refreshAssistData();
            }}
            onLocalTurn={onAppendLocalTurn}
            onVoiceUiChange={setVoiceUi}
          />
          <AssistStatusPanel meta={assistMeta} language={language} />
        </div>

        <div className="panel cc-genie">
          <div className="panel-title">{copy.databricksGenieLive}</div>
          <div className="genie-brand-note">{copy.genieBrandNote}</div>
          <div className="facts-grid cc-kpis">
            <Fact label={copy.openInvoices} value={sum.open_invoice_count ?? 0} />
            <Fact
              label={copy.overdue}
              value={`${overdueCount} ($${overdueAmount})`}
              warn={overdueCount > 0}
            />
            <Fact
              label={copy.autopay}
              value={sum.autopay_enabled ? copy.on : copy.off}
              warn={!sum.autopay_enabled}
            />
            <Fact
              label={copy.declinedPays}
              value={sum.recent_declined_payments ?? 0}
              warn={(sum.recent_declined_payments ?? 0) > 0}
            />
          </div>

          {factErr && <div className="account-error">{copy.unavailable}: {factErr}</div>}
          {!facts && !factErr && <div className="muted">{copy.loading}</div>}

          {facts?.invoices && facts.invoices.length > 0 && (
            <table className="inv-table">
              <thead>
                <tr>
                  <th>{copy.invoice}</th>
                  <th>{copy.period}</th>
                  <th>{copy.amount}</th>
                  <th>{copy.lateFee}</th>
                  <th>{copy.status}</th>
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
                    <td>{localizedValue(language, inv.status)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          <div className="resolution-timeline">
            <div className="panel-title">{copy.resolutionTimeline}</div>
            {resolutionEvents.length === 0 && <div className="muted">{copy.noResolutionEvents}</div>}
            {resolutionEvents.map((ev) => (
              <div className="timeline-item" key={ev.event_id}>
                <div className="timeline-head">
                  <span className="timeline-type">{localizedValue(language, ev.event_type)}</span>
                  <span className="timeline-status">{localizedValue(language, ev.issue_status ?? "open")}</span>
                </div>
                {ev.note && <div className="timeline-note">{localizeResolutionNote(language, ev.note)}</div>}
              </div>
            ))}
          </div>

          <div className="genie-console cc-console">
            <label className="genie-label">{copy.askGenieLabel}</label>
            <textarea
              value={genieQuestion}
              onChange={(e) => setGenieQuestion(e.target.value)}
              className="genie-input-box"
              rows={3}
            />
            <div className="genie-actions">
              <button onClick={() => askGenie(genieQuestion)} disabled={genieLoading}>
                {genieLoading ? copy.analyzing : copy.runGenieQuery}
              </button>
              <button
                className="ghost"
                onClick={() => {
                  setGenieQuestion(suggestedQuestion);
                  askGenie(suggestedQuestion);
                }}
                disabled={genieLoading}
              >
                {copy.refreshAssist}
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
                {copy.designedAround(customer?.full_name ?? copy.spotlightCustomers)}
              </div>
            )}
            {genieResp?.sql && (
              <div className="genie-sql">
                <button className="sql-toggle" onClick={() => setGenieShowSql((v) => !v)}>
                  {genieShowSql ? copy.hideQuery : copy.showQuery}
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
        language={language}
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
  language,
}: {
  issueStatus: string;
  localTurns: LocalTurn[];
  assistMeta: LiveNudge | null;
  voiceUi: VoiceUiState;
  live: Record<string, any> | null;
  facts: AccountFacts | null;
  intent?: string;
  language: InteractionLanguage;
}): AssistPipelineStep[] {
  const copy = uiCopy(language);
  const stages: { key: string; label: string }[] = [
    { key: "describe", label: copy.stageDescribe },
    { key: "understand", label: copy.stageUnderstand },
    { key: "review", label: copy.stageReview },
    { key: "offer", label: copy.stageOffer },
    { key: "apply", label: copy.stageApply },
    { key: "close", label: copy.stageClose },
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
      voiceUi.interimText?.trim() || copy.describeInProgress;
  } else if (voiceUi.phase === "transcribing") {
    activeKey = "describe";
    inProgressDetail.describe =
      voiceUi.interimText?.trim() || copy.capturingSpeech;
  } else if (voiceUi.phase === "agent_reply") {
    activeKey = assistMeta ? "offer" : "review";
    if (!assistMeta) {
      inProgressDetail.understand = copy.understandingRequest;
      inProgressDetail.review = copy.reviewingFacts;
    }
  } else if (
    !billing?.applied &&
    status !== "closed" &&
    (String(nudge.customer_signal) === "confirm_proceed" || actions.pending_close)
  ) {
    activeKey = "apply";
    inProgressDetail.apply = copy.applyingAgreement;
  }

  const details: Record<string, string> = {};
  if (doneThrough >= 0) {
    details.describe =
      voiceUi.interimText?.trim() ||
      lastCustomer?.slice(0, 120) ||
      copy.customerExplains;
  }
  if (doneThrough >= 1) {
    const plan = nudge.payment_plan_requested ? copy.paymentPlan : null;
    const waiver = nudge.waiver_requested ? copy.lateFeeRelief : null;
    const extras = [plan, waiver].filter(Boolean).join(" + ");
    details.understand = extras
      ? `${localizedIntentLabel(language, intent)} - ${copy.customerAskedFor(extras)}`
      : localizedIntentLabel(language, intent) || copy.billingConcernIdentified;
  }
  if (doneThrough >= 2) {
    if (overdueCount > 0) {
      details.review = copy.genieConfirmedOverdue(overdueCount, `$${overdue.toFixed(2)}`);
    } else if (assistMeta?.agent_validation?.reply_available) {
      details.review = copy.accountFactsChecked;
    } else {
      details.review = copy.accountContextReviewed;
    }
  }
  if (doneThrough >= 3) {
    if (actions.waiver_requested && actions.payment_plan_requested) {
      details.offer = copy.proposedPlanAndWaiver;
    } else if (actions.waiver_requested) {
      details.offer = copy.proposedLateFeeRelief;
    } else if (actions.payment_plan_requested) {
      details.offer = copy.proposedPaymentPlan;
    } else {
      details.offer = copy.nextStepsShared;
    }
  }
  if (doneThrough >= 4) {
    if (billing?.applied) {
      details.apply = copy.billingUpdated(String(billing.adjustment?.invoice_id ?? copy.invoice));
    } else if (billing && !billing.applied) {
      details.apply = copy.billingNotUpdated(String(billing.reason ?? copy.waitingForConfirmation));
    } else if (status === "closed") {
      details.apply = copy.paymentArrangementRecorded;
    }
  } else if (actions.waiver_requested || actions.payment_plan_requested) {
    details.apply = copy.waitingForConfirmation;
  }
  if (doneThrough >= 5) {
    details.close =
      localizeResolutionNote(language, resolution?.note) ||
      localizeResolutionNote(language, facts?.summary?.resolution_note) ||
      copy.issueClosed;
  }

  if (doneThrough < 0 && voiceUi.phase === "idle" && !assistMeta) {
    return [
      {
        key: "waiting",
        label: copy.awaitingCustomer,
        status: "pending",
        detail: copy.awaitingCustomerDetail,
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
  language,
}: {
  issueStatus: string;
  localTurns: LocalTurn[];
  assistMeta: LiveNudge | null;
  voiceUi: VoiceUiState;
  live: Record<string, any> | null;
  facts: AccountFacts | null;
  intent?: string;
  language: InteractionLanguage;
}) {
  const copy = uiCopy(language);
  const steps = buildResolutionJourney({
    issueStatus,
    localTurns,
    assistMeta,
    voiceUi,
    live,
    facts,
    intent,
    language,
  });

  return (
    <div className="assist-pipeline-strip resolution-journey">
      <div className="assist-pipeline-head">
        <span className="panel-title">{copy.issueResolutionJourney}</span>
        <span className="assist-pipeline-total">{copy.status}: {localizedValue(language, issueStatus)}</span>
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
  label,
  language,
}: {
  rec: ReturnType<typeof recommend>;
  intent?: string;
  sentiment?: string;
  label: string;
  language: InteractionLanguage;
}) {
  return (
    <div className={`rec-card r-${rec.priority} cc-recommendation`}>
      <div className="rec-top">
        <span className="rec-kicker">{label}</span>
        <span className="rec-tags">
          <span className="chip">{localizedIntentLabel(language, intent)}</span>
          <span className={`badge sentiment ${sentiment ?? "neutral"}`}>
            {localizedValue(language, sentiment ?? "neutral")}
          </span>
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

function AssistStatusPanel({ meta, language }: { meta: LiveNudge | null; language: InteractionLanguage }) {
  const copy = uiCopy(language);
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
          <span className="assist-status-label">{copy.resolution}</span>
          <span className={`status-chip issue issue-${resolutionStatus}`}>
            {localizedValue(language, resolutionStatus)}
          </span>
        </div>
      )}
      {closeBlock && (
        <div className="assist-status-row warn">
          <span className="assist-status-label">{copy.closeBlocked}</span>
          <span>{localizedValue(language, closeBlock, "reason")}</span>
        </div>
      )}
      {billing && (
        <div className="assist-status-row">
          <span className="assist-status-label">{copy.billing}</span>
          <span>
            {billing.applied
              ? `${copy.applied} (${String(billing.adjustment?.invoice_id ?? copy.invoice)})`
              : `${copy.notApplied}: ${localizedValue(language, billing.reason ?? copy.unknown, "reason")}`}
          </span>
        </div>
      )}
      {validation && (
        <div className="assist-status-row">
          <span className="assist-status-label">{copy.genieValidation}</span>
          <span>
            {validation.reply_available
              ? copy.replyValidated
              : localizedValue(language, validation.genie_error ?? copy.replyUnavailable, "reason")}
            {validation.mismatches?.length
              ? ` · mismatches: ${validation.mismatches.join("; ")}`
              : ""}
            {validation.output_issues?.length
              ? ` · output: ${validation.output_issues.map((item) => localizedValue(language, item, "reason")).join("; ")}`
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
  language,
  onNudge,
  onLocalTurn,
  onVoiceUiChange,
}: {
  callId: string;
  customerId: string;
  sttProvider: string;
  language: InteractionLanguage;
  onNudge: (n: LiveNudge) => void;
  onLocalTurn: (turn: LocalTurn) => void;
  onVoiceUiChange: (state: VoiceUiState) => void;
}) {
  const copy = uiCopy(language);
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
      onLocalTurn({ text: msg, speaker, language });
      if (speaker === 1) {
        onVoiceUiChange({
          phase: "agent_reply",
          source: "text",
          processingLabel: copy.geniePreparing,
        });
      }

      const n = await api.sendUtterance(callId, msg, speaker, language);
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
        captionRef.current = startSpeechCaption(
          (caption) => {
            voiceRef.current.interim = caption;
            if (voicePhaseRef.current !== "speaking") return;
            onVoiceUiChange({
              phase: "speaking",
              source: "mic",
              interimText: caption,
              micLevel: voiceRef.current.level,
            });
          },
          undefined,
          language
        );
        setRecording(true);
        onVoiceUiChange({
          phase: "speaking",
          source: "mic",
          interimText: "",
          processingLabel: copy.listening,
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
        (message) => setErr(message),
        language
      );
      micSessionRef.current = session;
      setRecording(true);
      onVoiceUiChange({
        phase: "speaking",
        source: "mic",
        interimText: "",
          processingLabel: copy.listening,
        micLevel: 0.15,
      });
    } catch (e) {
      voicePhaseRef.current = "idle";
      onVoiceUiChange({ phase: "idle" });
      setErr(e instanceof Error ? e.message : copy.micAccessError);
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
          ? copy.processingDatabricks
          : copy.processingDeepgram,
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
          1,
          language
        );
        const textFromMic = String(n.transcript || "").trim();
        if (!textFromMic) throw new Error(copy.noDatabricksTranscript);
        onLocalTurn({ text: textFromMic, speaker: 1, language });
        voicePhaseRef.current = "agent_reply";
        onVoiceUiChange({
          phase: "agent_reply",
          source: "mic",
          processingLabel: copy.geniePreparing,
        });
        onNudge(n);
        applyAgentReply(n);
        voicePhaseRef.current = "idle";
        onVoiceUiChange({ phase: "idle" });
        return;
      }

      const textFromMic = (await (session as MicStreamSession).stop()).trim();
      micSessionRef.current = null;
      if (!textFromMic) throw new Error(copy.noDeepgramTranscript);
      onLocalTurn({ text: textFromMic, speaker: 1, language });
      voicePhaseRef.current = "agent_reply";
      onVoiceUiChange({
        phase: "agent_reply",
        source: "mic",
        processingLabel: copy.geniePreparing,
      });
      const n = await api.sendUtterance(callId, textFromMic, 1, language);
      onNudge(n);
      applyAgentReply(n);
      voicePhaseRef.current = "idle";
      onVoiceUiChange({ phase: "idle" });
    } catch (e) {
      captionRef.current?.close();
      captionRef.current = null;
      voicePhaseRef.current = "idle";
      onVoiceUiChange({ phase: "idle" });
      setErr(e instanceof Error ? e.message : copy.micTranscriptionFailed);
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
          <option value={1}>{copy.speakerCustomer}</option>
          <option value={0}>{copy.speakerAgent}</option>
        </select>
        <input
          value={text}
          placeholder={copy.utterancePlaceholder}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && send()}
        />
        <button onClick={send} disabled={busy}>
          {busy ? "…" : copy.send}
        </button>
        <button onClick={recording ? () => void stopMic() : () => void startMic()} disabled={busy}>
          {recording ? copy.stopMic : copy.mic}
        </button>
      </div>
      {sttProvider === "databricks" && (
        <div
          className={`caption-status ${captionSupported ? "ok" : "off"}`}
          title={
            captionSupported
              ? copy.captionAvailableTitle
              : copy.captionUnavailableTitle
          }
        >
          <span className="caption-dot" />
          {captionSupported
            ? copy.captionAvailable
            : copy.captionUnavailable}
        </div>
      )}
      {err && <div className="account-error">{err}</div>}
    </div>
  );
}
