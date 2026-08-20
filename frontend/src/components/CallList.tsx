import { ReactNode, useEffect, useReducer, useRef, useState } from "react";
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
import { API_BASE_URL, WS_BASE_URL } from "../config";
import { CustomerIssueTag } from "../lib/customerIssues";
import {
  VoiceUiState,
} from "../lib/micStream";
import {
  RealtimeVoiceSession,
  startRealtimeVoice,
} from "../lib/realtimeVoice";
import { useHalfDuplexVoice } from "../hooks/useHalfDuplexVoice";
import { getMe } from "../lib/me";
import { emptyConversation, turnReducer } from "../lib/turnState";
import {
  languageLabel,
  localizedValue,
  localizeResolutionNote,
  uiCopy,
} from "../i18n";
import { SentientHCol, SentientStep } from "./sentient/Sentient";
import { VoiceOrb, type VoiceOrbState } from "./VoiceOrb";

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

function localizedIntentLabel(language: InteractionLanguage, code?: string | null): string {
  if (!code) return "—";
  return localizedValue(language, code, "intent");
}

export function CockpitSession({
  call,
  customer,
  customerName = null,
  callLabel,
  issueTags = [],
  sttProvider,
  languageOptions,
  defaultLanguage,
  selectedLanguage,
  onLanguageChange,
  localTurns,
  onAppendLocalTurn,
  onUpdateLastCustomerTurn,
  onRemoveLastCustomerTurn,
  onResetLocalTurns,
  layout = "horizontal",
  panel = "all",
}: {
  call: CallState;
  customer: CustomerWithIssue | null;
  customerName?: string | null;
  callLabel?: string;
  issueTags?: CustomerIssueTag[];
  sttProvider: string;
  languageOptions?: InteractionLanguageOption[];
  defaultLanguage?: InteractionLanguage;
  selectedLanguage: InteractionLanguage;
  onLanguageChange: (language: InteractionLanguage) => void;
  localTurns: LocalTurn[];
  onAppendLocalTurn: (turn: LocalTurn) => void;
  onUpdateLastCustomerTurn: (turn: LocalTurn) => void;
  onRemoveLastCustomerTurn: () => void;
  onResetLocalTurns: () => void;
  layout?: "stacked" | "single" | "horizontal";
  panel?: "conversation" | "genie" | "resolution" | "all";
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
  const [detectedLanguage, setDetectedLanguage] = useState<string | null>(null);
  const [languageMismatch, setLanguageMismatch] = useState<{
    expected: string;
    detected: string;
  } | null>(null);
  const availableLanguages =
    languageOptions && languageOptions.length > 0 ? languageOptions : INTERACTION_LANGUAGES;
  const language = selectedLanguage;
  const copy = uiCopy(language);

  // A pending "you spoke X but the call is in Y" warning is about the language the
  // call USED to run in, so keeping it after a switch leaves the caller reading a
  // contradiction (the card assistant clears it the same way).
  useEffect(() => {
    setLanguageMismatch(null);
  }, [selectedLanguage]);

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
    setLanguageMismatch(null);
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
  const intent = live?.primary_intent ?? base.intent;

  const sum = facts?.summary ?? {};
  const issueStatus = String(assistMeta?.resolution?.status ?? sum.issue_status ?? "open");
  // Keep stream empty by default so the UI feels like a true live call surface.
  const utterances = localTurns;

  const overdueCount = Number(sum.overdue_invoice_count ?? 0);
  const overdueAmount = Number(sum.overdue_amount ?? 0);
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

  const hCompact = layout === "horizontal";
  const panelClass = hCompact ? "sentient-panel-inner" : "sentient-glass";

  const mismatchMessage = languageMismatch
    ? copy.languageMismatch(
        languageLabel(languageMismatch.expected, language),
        languageLabel(languageMismatch.detected, language)
      )
    : null;

  // Agent-initiated voice presence. Rendered at the TOP of the conversation
  // panel so the Genie orb + call controls sit up-front and aligned, matching
  // the card / knowledge / home surfaces (which all lead with the orb).
  const liveAssistEl = (
    <LiveAssist
      callId={call.call_id}
      customerId={String(facts?.customer_id ?? call.customer_id ?? "")}
      sttProvider={sttProvider}
      language={language}
      expectedLanguage={language}
      compact={hCompact}
      voiceUi={voiceUi}
      onReset={resetScenario}
      resetBusy={resetBusy}
      onLanguageMismatch={(expected, detected) =>
        setLanguageMismatch({ expected, detected })
      }
      onNudge={(n) => {
        setLive(n.live);
        setAssistMeta(n);
        if (n.billing?.applied && n.billing.adjustment) {
          const adj = n.billing.adjustment;
          setFacts((prev) => {
            if (!prev?.invoices) return prev;
            const invoices = prev.invoices.map((inv) =>
              inv.invoice_id === adj.invoice_id
                ? {
                    ...inv,
                    amount: String(adj.amount_after ?? inv.amount),
                    late_fee: String(adj.late_fee_after ?? inv.late_fee),
                    status: String(adj.status_after ?? inv.status),
                    resolution_status: "closed",
                  }
                : inv
            );
            const overdueInvoices = invoices.filter((inv) => inv.status === "overdue");
            return {
              ...prev,
              invoices,
              summary: {
                ...prev.summary,
                issue_status: n.resolution?.status ?? prev.summary?.issue_status,
                overdue_invoice_count: overdueInvoices.length,
                overdue_amount: overdueInvoices.reduce(
                  (total, inv) => total + Number(inv.amount ?? 0),
                  0
                ),
                resolution_note: n.resolution?.note ?? prev.summary?.resolution_note,
              },
            };
          });
        }
        refreshAssistData();
      }}
      onLocalTurn={onAppendLocalTurn}
      onUpdateLastCustomerTurn={onUpdateLastCustomerTurn}
      onRemoveLastCustomerTurn={onRemoveLastCustomerTurn}
      onClearTranscript={onResetLocalTurns}
      onVoiceUiChange={setVoiceUi}
      onLanguageDetected={(lang) => {
        // Show what STT heard, but never silently switch the agent's selection —
        // a mismatch surfaces via the warning banner instead so the agent stays
        // in control of the workspace language.
        setDetectedLanguage(lang);
      }}
      onAccountFacts={(newFacts) => setFacts(newFacts)}
    />
  );

  const conversationPanel = (
    <div className={`${panelClass} sentient-conversation`}>
          {liveAssistEl}
          {hCompact && customerName && (
            <div className="sentient-identity">
              <div className="sentient-identity-main">
                <span className="sentient-identity-name">{customerName}</span>
                {callLabel && <span className="sentient-identity-call">{callLabel}</span>}
              </div>
              {issueTags.length > 0 && (
                <div className="sentient-identity-tags">
                  {issueTags.map((tag) => (
                    <span
                      key={tag.id}
                      className={`sentient-identity-tag${tag.warn ? " is-warn" : ""}`}
                    >
                      {tag.label}
                    </span>
                  ))}
                </div>
              )}
            </div>
          )}
          {mismatchMessage && (
            <div className="sentient-lang-mismatch" role="alert">
              <span className="sentient-lang-mismatch-icon" aria-hidden="true">⚠</span>
              <span>{mismatchMessage}</span>
              <button
                className="sentient-lang-mismatch-switch"
                onClick={() => {
                  const base = languageMismatch!.detected.split("-")[0].toLowerCase();
                  const target = availableLanguages.find(
                    (item) => item.code.split("-")[0].toLowerCase() === base
                  );
                  if (target) onLanguageChange(target.code);
                  setLanguageMismatch(null);
                }}
              >
                {copy.interactionLanguage}
              </button>
            </div>
          )}
          {!hCompact && (
          <div className="sentient-kicker sentient-row-between">
            <span>
              {copy.conversationStream}
              {detectedLanguage && (
                <span className="sentient-lang-badge" title="Detected from caller's speech"> · {detectedLanguage}</span>
              )}
            </span>
          </div>
          )}
          {hCompact && (
            <div className="sentient-row-between sentient-conv-tools">
              <span className="sentient-muted-text">
                {copy.transcriptLabel}
                {detectedLanguage && (
                  <span className="sentient-lang-badge" title="Detected from caller's speech">
                    {" · "}
                    {detectedLanguage}
                  </span>
                )}
              </span>
            </div>
          )}
          <div className="sentient-transcript">
            {utterances.length === 0 && voiceUi.phase === "idle" && (
              <div className="sentient-muted-text">{copy.noTranscript}</div>
            )}
            {utterances.map((u, i) => {
              const isCustomer = (u.speaker ?? 0) === 1;
              const isLatest = i === utterances.length - 1 && voiceUi.phase === "idle";
              return (
                <div
                  key={i}
                  className={`sentient-turn ${isCustomer ? "is-customer" : "is-agent"}${isLatest ? " is-latest" : ""}`}
                >
                  <span className="sentient-turn-who">
                    {isCustomer ? copy.customer : copy.agentGenieAssisted}
                    {u.language && <span className="sentient-turn-lang"> · {u.language}</span>}
                  </span>
                  <span className="sentient-turn-text">{u.text}</span>
                </div>
              );
            })}
            {voiceUi.phase === "speaking" && (
              <div className="sentient-turn is-customer is-live">
                <span className="sentient-turn-who">{copy.customerSpeaking}</span>
                <span className="sentient-turn-text is-placeholder">
                  {voiceUi.interimText?.trim() || copy.listening}
                </span>
                <LiveWaveform level={voiceUi.micLevel ?? 0.2} active />
              </div>
            )}
            {voiceUi.phase === "transcribing" && (
              <div className="sentient-turn is-customer is-live">
                <span className="sentient-turn-who">{copy.customer}</span>
                <span className="sentient-turn-text is-placeholder">
                  {voiceUi.interimText?.trim() || voiceUi.processingLabel || copy.transcribingMessage}
                </span>
              </div>
            )}
            {voiceUi.phase === "agent_reply" && (
              <div className="sentient-turn is-agent is-live">
                <span className="sentient-turn-who">{copy.agentGenieAssisted}</span>
                <span className="sentient-turn-text is-placeholder">
                  {voiceUi.processingLabel || copy.preparingGenieResponse}
                </span>
              </div>
            )}
          </div>
          <AssistStatusPanel meta={assistMeta} language={language} />
    </div>
  );

  const geniePanel = (
    <div className={`${panelClass} sentient-genie`}>
          <div className={hCompact ? "sentient-genie-scroll" : undefined}>
          {!hCompact && (
            <>
          <div className="sentient-kicker">{copy.databricksGenieLive}</div>
          <p className="sentient-muted-text">{copy.genieBrandNote}</p>
            </>
          )}
          <div className="sentient-stat-grid">
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

          {factErr && <div className="sentient-alert">{copy.unavailable}: {factErr}</div>}
          {!facts && !factErr && <div className="sentient-muted-text">{copy.loading}</div>}

          {facts?.invoices && facts.invoices.length > 0 && (
            <table className="sentient-table">
              <thead>
                <tr>
                  <th>{copy.invoice}</th>
                  {!hCompact && <th>{copy.period}</th>}
                  <th>{copy.amount}</th>
                  {facts.invoices.some((inv) => Number(inv.late_fee) > 0) && (
                    <th>{copy.lateFee}</th>
                  )}
                  <th>{copy.status}</th>
                </tr>
              </thead>
              <tbody>
                {facts.invoices.map((inv) => (
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
                    {!hCompact && <td>{inv.period}</td>}
                    <td>${inv.amount}</td>
                    {facts.invoices!.some((row) => Number(row.late_fee) > 0) && (
                      <td>{Number(inv.late_fee) > 0 ? `$${inv.late_fee}` : "—"}</td>
                    )}
                    <td>{localizedValue(language, inv.status)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          {!hCompact && (
          <div className="sentient-timeline">
            <div className="sentient-kicker">{copy.resolutionTimeline}</div>
            {resolutionEvents.length === 0 && <div className="sentient-muted-text">{copy.noResolutionEvents}</div>}
            {resolutionEvents.map((ev, index) => (
              <div
                className={`sentient-timeline-item${index === 0 ? " is-latest" : ""}`}
                key={ev.event_id}
              >
                <div className="sentient-timeline-head">
                  <span>{localizedValue(language, ev.event_type)}</span>
                  <span>{localizedValue(language, ev.issue_status ?? "open")}</span>
                </div>
                {ev.note && (
                  <div className="sentient-timeline-note">
                    {localizeResolutionNote(language, ev.note, ev.issue_status)}
                  </div>
                )}
              </div>
            ))}
          </div>
          )}
          </div>

          <div className={`sentient-genie-console${genieLoading ? " is-loading" : ""}${genieResp ? " has-answer" : ""}`}>
            <label className="sentient-field-label">{copy.askGenieLabel}</label>
            <textarea
              value={genieQuestion}
              onChange={(e) => setGenieQuestion(e.target.value)}
              className="sentient-textarea"
              rows={hCompact ? 2 : 3}
            />
            <div className="sentient-actions">
              <button className="sentient-btn" onClick={() => askGenie(genieQuestion)} disabled={genieLoading}>
                {genieLoading ? copy.analyzing : copy.runGenieQuery}
              </button>
              <button
                className="sentient-btn-ghost"
                onClick={() => {
                  setGenieQuestion(suggestedQuestion);
                  askGenie(suggestedQuestion);
                }}
                disabled={genieLoading}
              >
                {copy.refreshAssist}
              </button>
            </div>
            {genieErr && <div className="sentient-alert">{genieErr}</div>}
            {genieResp && (genieResp.answer || genieResp.description) && (
              <div className="sentient-answer is-active">
                {genieResp.answer ?? genieResp.description}
              </div>
            )}
            {genieResp?.description &&
              genieResp.answer &&
              genieResp.description !== genieResp.answer && (
                <div className="sentient-muted-text">{genieResp.description}</div>
              )}
            {genieResp?.suggested_followups && genieResp.suggested_followups.length > 0 && (
              <div className="sentient-chips">
                {genieResp.suggested_followups.map((f, i) => (
                  <button
                    key={i}
                    className="sentient-chip"
                    disabled={genieLoading}
                    onClick={() => askGenie(f, true)}
                  >
                    {f}
                  </button>
                ))}
              </div>
            )}
            {!genieResp && !hCompact && (
              <div className="sentient-muted-text">
                {copy.designedAround(customer?.full_name ?? copy.spotlightCustomers)}
              </div>
            )}
            {genieResp?.sql && (
              <div className="sentient-sql">
                <button className="sentient-btn-ghost sentient-btn-sm" onClick={() => setGenieShowSql((v) => !v)}>
                  {genieShowSql ? copy.hideQuery : copy.showQuery}
                </button>
                {genieShowSql && <pre className="sentient-sql-pre">{genieResp.sql}</pre>}
              </div>
            )}
          </div>
    </div>
  );

  const resolutionPanel = (
    <div className={`${panelClass} sentient-resolution`}>
      <ResolutionJourneyStrip
        issueStatus={issueStatus}
        localTurns={utterances}
        assistMeta={assistMeta}
        voiceUi={voiceUi}
        live={live}
        facts={facts}
        intent={intent}
        language={language}
        compact={hCompact}
      />
    </div>
  );

  if (layout === "horizontal") {
    return (
      <div className="sentient-h-session">
        <SentientHCol step={2} title={copy.onCallTitle} description={copy.onCallDesc}>
          {conversationPanel}
        </SentientHCol>
        <SentientHCol step={3} title={copy.genieColTitle} description={copy.genieColDesc}>
          {geniePanel}
        </SentientHCol>
        <SentientHCol step={4} title={copy.resolutionColTitle} description={copy.resolutionColDesc}>
          {resolutionPanel}
        </SentientHCol>
      </div>
    );
  }

  if (layout === "stacked") {
    return (
      <div className="sentient-session-stack">
        <SentientStep
          step={2}
          title="What's happening on the call?"
          description="Listen, transcribe, and respond. Use the mic or type the next utterance."
        >
          {conversationPanel}
        </SentientStep>
        <SentientStep
          step={3}
          title="What does Genie recommend?"
          description="Review account facts, invoices, and run a live Genie assist query."
        >
          {geniePanel}
        </SentientStep>
        <SentientStep
          step={4}
          title="Where are we in resolution?"
          description="Track the issue journey from first utterance through close."
        >
          {resolutionPanel}
        </SentientStep>
      </div>
    );
  }

  return (
    <div className="sentient-session">
      {(panel === "all" || panel === "conversation") && conversationPanel}
      {(panel === "all" || panel === "genie") && geniePanel}
      {(panel === "all" || panel === "resolution") && resolutionPanel}
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
      localizeResolutionNote(language, resolution?.note, status) ||
      localizeResolutionNote(language, facts?.summary?.resolution_note, status) ||
      copy.issueClosed;
  }

  if (doneThrough < 0 && voiceUi.phase === "idle" && !assistMeta) {
    return [
      {
        key: "waiting",
        label: copy.awaitingCustomer,
        status: "active",
        detail: copy.awaitingCustomerDetail,
      },
    ];
  }

  return stages.map((stage, idx) => {
    const isActive = activeKey === stage.key;
    const isDone = !isActive && idx <= doneThrough;
    let stepStatus: "active" | "done" | "pending" | "complete" = isActive
      ? "active"
      : isDone
        ? "done"
        : "pending";
    if (status === "closed" && stage.key === "close" && !activeKey) {
      stepStatus = "complete";
    }
    return {
      key: stage.key,
      label: stage.label,
      status: stepStatus,
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
  compact = false,
  horizontal = false,
}: {
  issueStatus: string;
  localTurns: LocalTurn[];
  assistMeta: LiveNudge | null;
  voiceUi: VoiceUiState;
  live: Record<string, any> | null;
  facts: AccountFacts | null;
  intent?: string;
  language: InteractionLanguage;
  compact?: boolean;
  horizontal?: boolean;
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
    <div className={`sentient-journey${horizontal ? " is-horizontal" : ""}`}>
      {!compact && (
      <div className="sentient-journey-head">
        <span className="sentient-kicker">{copy.issueResolutionJourney}</span>
        <span className="sentient-muted-text">{copy.status}: {localizedValue(language, issueStatus)}</span>
      </div>
      )}
      {compact && (
        <div
          className={`sentient-journey-status${
            issueStatus === "closed" ? " is-ok" : issueStatus === "open" ? " is-active" : " is-warn"
          }`}
        >
          {copy.status}: <strong>{localizedValue(language, issueStatus)}</strong>
        </div>
      )}
      <div className="sentient-journey-track">
        {steps.map((step, idx) => (
          <div key={step.key} className={`sentient-journey-step is-${step.status}`}>
            <div className="sentient-journey-node">
              <span>{idx + 1}</span>
            </div>
            <div className="sentient-journey-copy">
              <div className="sentient-journey-label">{step.label}</div>
              {step.detail && <div className="sentient-journey-detail">{step.detail}</div>}
            </div>
            {idx < steps.length - 1 && <div className="sentient-journey-connector" aria-hidden="true" />}
          </div>
        ))}
      </div>
    </div>
  );
}

function Fact({ label, value, warn }: { label: string; value: ReactNode; warn?: boolean }) {
  return (
    <div className={`sentient-stat${warn ? " sentient-stat-warn" : ""}`}>
      <div className="sentient-stat-value">{value}</div>
      <div className="sentient-stat-label">{label}</div>
    </div>
  );
}

function LiveWaveform({ level, active }: { level: number; active: boolean }) {
  const bars = [0.35, 0.55, 0.75, 1, 0.8, 0.6, 0.45, 0.3];
  return (
    <div className={`sentient-wave${active ? " is-live" : ""}`} aria-hidden="true">
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
    <div className="sentient-status">
      {resolutionStatus && (
        <div className={`sentient-status-row${resolutionStatus === "closed" ? " is-ok" : " is-active"}`}>
          <span className="sentient-status-label">{copy.resolution}</span>
          <span>{localizedValue(language, resolutionStatus)}</span>
        </div>
      )}
      {closeBlock && (
        <div className="sentient-status-row is-warn">
          <span className="sentient-status-label">{copy.closeBlocked}</span>
          <span>{localizedValue(language, closeBlock, "reason")}</span>
        </div>
      )}
      {billing && (
        <div className={`sentient-status-row${billing.applied ? " is-ok" : " is-active"}`}>
          <span className="sentient-status-label">{copy.billing}</span>
          <span>
            {billing.applied
              ? `${copy.applied} (${String(billing.adjustment?.invoice_id ?? copy.invoice)})`
              : `${copy.notApplied}: ${localizedValue(language, billing.reason ?? copy.unknown, "reason")}`}
          </span>
        </div>
      )}
      {validation && (
        <div className={`sentient-status-row${validation.reply_available ? " is-ok" : " is-warn"}`}>
          <span className="sentient-status-label">{copy.genieValidation}</span>
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
  customerId,
  language,
  expectedLanguage,
  voiceUi,
  onReset,
  resetBusy,
  onNudge,
  onLocalTurn,
  onUpdateLastCustomerTurn: _onUpdateLastCustomerTurn,
  onRemoveLastCustomerTurn: _onRemoveLastCustomerTurn,
  onClearTranscript,
  onVoiceUiChange,
  onLanguageDetected,
  onLanguageMismatch,
  onAccountFacts,
}: {
  callId: string;
  customerId: string;
  sttProvider: string;
  language: InteractionLanguage;
  expectedLanguage?: InteractionLanguage;
  compact?: boolean;
  voiceUi: VoiceUiState;
  onReset?: () => void;
  resetBusy?: boolean;
  onNudge: (n: LiveNudge) => void;
  onLocalTurn: (turn: LocalTurn) => void;
  onUpdateLastCustomerTurn: (turn: LocalTurn) => void;
  onRemoveLastCustomerTurn: () => void;
  /** Drop the on-screen turns (used when a language switch restarts the call). */
  onClearTranscript: () => void;
  onVoiceUiChange: (state: VoiceUiState) => void;
  onLanguageDetected?: (lang: string) => void;
  onLanguageMismatch?: (expected: string, detected: string) => void;
  onAccountFacts?: (facts: any) => void;
}) {
  const [, dispatchTurn] = useReducer(turnReducer, undefined, emptyConversation);
  const copy = uiCopy(language);
  const [err, setErr] = useState<string | null>(null);
  const [inCall, setInCall] = useState(false);
  const rtSessionRef = useRef<RealtimeVoiceSession | null>(null);
  // True while a session is opening (getUserMedia + WS) but before rtSessionRef is
  // assigned, so a tap on the orb during that window can't start a second session.
  const startingRef = useRef(false);
  const voicePhaseRef = useRef<VoiceUiState["phase"]>("idle");
  // Last mic level, kept in a ref so frequent onLevel updates don't churn state.
  const micLevelRef = useRef(0.15);
  // Latest live-caption text, kept in a ref so the frequent onLevel updates can
  // carry it forward. onVoiceUiChange REPLACES the state object, so if onLevel
  // omitted interimText it would blank the caption between interim events and the
  // text would flicker in/out while the caller speaks.
  const interimTextRef = useRef("");
  // The language THIS session was opened with. The picker's value can change
  // mid-call, and the session can't follow it without being reopened, so the two
  // are tracked separately.
  const callLanguageRef = useRef<string>(expectedLanguage ?? language);

  // Shared half-duplex plumbing (playback queue + mic gating), identical to the
  // card assistant: while the agent's TTS plays we mute the mic so it doesn't loop
  // back in and get re-transcribed, then resume once playback drains. On the final
  // chunk we flip the UI to "speaking".
  const {
    playbackRef,
    resetPlayback,
    gateMic,
    ungateMicAfter,
    handleResponseAudio,
    handlePlaybackStop,
    interrupt,
    switchLanguage,
    teardownPlayback,
  } = useHalfDuplexVoice({
    sessionRef: rtSessionRef,
    onFinal: () => {
      voicePhaseRef.current = "speaking";
      onVoiceUiChange({ phase: "speaking", source: "mic", micLevel: 0.15 });
    },
    callLanguageRef,
    isCallLive: () => rtSessionRef.current !== null,
    closeSession: () => {
      // Deliberately no playback teardown here: the queue was already flushed,
      // and closing/recreating the AudioContext mid-switch is what Chrome
      // suspends (which garbles the first audio of the new session). startVoice
      // installs a fresh queue via resetPlayback.
      rtSessionRef.current?.close();
      rtSessionRef.current = null;
      startingRef.current = false;
      interimTextRef.current = "";
      setInCall(false);
    },
    reopenSession: (nextLanguage) => startVoice(nextLanguage),
    // Only the transcript: the account panels are data (localized live from the
    // catalog), while these turns are speech frozen in the language just left.
    clearConversation: onClearTranscript,
  });

  // Opening greeting, generated in the call language by the backend (cached per
  // base language). Same design as the card assistant: the agent speaks first, so
  // it (a) opens the call warmly and (b) LOCKS a clean voice reference for the
  // whole call from a curated line instead of freezing whatever the first live
  // answer happened to sound like. Returns "" if serving is down — we then just
  // open the mic instead of speaking a fake English line.
  const greetingCacheRef = useRef<Map<string, string>>(new Map());
  const fetchGreeting = async (lang: string): Promise<string> => {
    const key = (lang || "en").split("-")[0];
    const cached = greetingCacheRef.current.get(key);
    if (cached !== undefined) return cached;
    try {
      // Greet the signed-in Databricks user by name (nameless when anonymous).
      const me = await getMe();
      const nameQ = me.name ? `&name=${encodeURIComponent(me.name)}` : "";
      const r = await fetch(`${API_BASE_URL}/calls/greeting?language=${encodeURIComponent(lang)}${nameQ}`);
      const data = (await r.json()) as { text?: string };
      const t = typeof data.text === "string" ? data.text : "";
      greetingCacheRef.current.set(key, t);
      return t;
    } catch {
      return "";
    }
  };
  // Speak agent text through the SAME voice session (synthesize turn) so it shares
  // the session's cloned voice — identical to the card's speakViaTTS. Audio returns
  // as response.audio events, which handleResponseAudio plays and half-duplex
  // mic-gates; onFinal flips the UI back to listening.
  const speakGreeting = (textToSpeak: string) => {
    const session = rtSessionRef.current;
    if (!session || !textToSpeak.trim()) return;
    playbackRef.current?.flush();
    gateMic();
    onLocalTurn({ text: textToSpeak, speaker: 0 });
    voicePhaseRef.current = "agent_reply";
    onVoiceUiChange({ phase: "agent_reply", source: "mic", processingLabel: textToSpeak });
    session.synthesize(textToSpeak, callLanguageRef.current);
  };

  useEffect(() => {
    return () => {
      teardownPlayback();
      rtSessionRef.current?.close();
      rtSessionRef.current = null;
      voicePhaseRef.current = "idle";
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // `langOverride` is how a language switch reopens the call: the picker's new
  // value arrives here directly rather than via a prop that may not have
  // re-rendered yet. Guarded on refs only (not `inCall`, whose setState hasn't
  // flushed when a restart closes and reopens in the same tick).
  const startVoice = async (langOverride?: string) => {
    if (rtSessionRef.current || startingRef.current) return;
    startingRef.current = true;
    setErr(null);
    const callLanguage = langOverride ?? expectedLanguage ?? language;
    callLanguageRef.current = callLanguage;

    try {
      voicePhaseRef.current = "speaking";
      onVoiceUiChange({ phase: "speaking", source: "mic", micLevel: 0.15 });

      resetPlayback();
      // Warm the greeting cache while the socket connects so it's ready to speak
      // the instant the session opens.
      void fetchGreeting(callLanguage);

      const session = await startRealtimeVoice(WS_BASE_URL, callId, customerId, {
        onLanguageMismatch: (expected, detected) => {
          onLanguageMismatch?.(expected, detected);
        },
        onLevel: (level) => {
          micLevelRef.current = level;
          if (voicePhaseRef.current === "speaking") {
            onVoiceUiChange({
              phase: "speaking",
              source: "mic",
              micLevel: level,
              interimText: interimTextRef.current || undefined,
            });
          }
        },
        onSessionReady: (_sessionId, lang) => {
          if (lang && lang !== "auto") {
            onLanguageDetected?.(lang);
          }
          // Agent speaks first: fetch + speak the greeting (mic stays paused via
          // startMicPaused until it finishes). The async fetch also ensures
          // rtSessionRef is assigned by the time we synthesize. If no greeting is
          // available, just open the mic so the caller can still speak.
          void (async () => {
            const text = await fetchGreeting(callLanguage);
            if (text) {
              speakGreeting(text);
            } else {
              voicePhaseRef.current = "speaking";
              onVoiceUiChange({ phase: "speaking", source: "mic", micLevel: 0.15 });
              ungateMicAfter(0);
            }
          })();
        },
        onSpeechStarted: () => {
          voicePhaseRef.current = "speaking";
          interimTextRef.current = "";
          onVoiceUiChange({ phase: "speaking", source: "mic", processingLabel: copy.listening });
        },
        onInterimTranscript: (text) => {
          // Live on-device caption (framework): show words as the caller speaks in
          // the EXISTING interim slot. Ignore the empty clear on final so it can't
          // stomp the agent_reply phase that onTranscript sets right before it.
          if (!text) return;
          voicePhaseRef.current = "speaking";
          interimTextRef.current = text;
          onVoiceUiChange({
            phase: "speaking",
            source: "mic",
            interimText: text,
            micLevel: micLevelRef.current,
          });
        },
        onTranscript: (transcriptText, lang, turnId) => {
          dispatchTurn({ type: "user_transcript", turnId, text: transcriptText });
          if (lang) {
            onLanguageDetected?.(lang);
          }
          interimTextRef.current = "";
          onLocalTurn({ text: transcriptText, speaker: 1, language: lang as InteractionLanguage || language });
          voicePhaseRef.current = "agent_reply";
          onVoiceUiChange({ phase: "agent_reply", source: "mic", processingLabel: copy.geniePreparing });
        },
        onResponseText: (responseText, turnId) => {
          dispatchTurn({ type: "response_text", turnId, text: responseText });
          onLocalTurn({ text: responseText, speaker: 0 });
        },
        onTurnEvent: ({ turnId, seq, kind, payload }) => {
          dispatchTurn({ type: "turn_event", turnId, seq, kind, payload });
        },
        onResponseAudio: (pcmB64, sampleRate, final, _turnId, meta) => {
          // Half-duplex playback + mic gating (shared with the card assistant):
          // mute the mic while the agent's audio plays so it isn't captured and
          // re-transcribed, enqueue, and resume the mic after turn completion.
          handleResponseAudio(pcmB64, sampleRate, final, meta);
        },
        onPlaybackStop: (_turnId, speechEpoch, reason) => {
          handlePlaybackStop(speechEpoch, reason);
        },
        onToolCalled: (name, result) => {
          if (name === "lookup_account" && result && typeof result === "object") {
            onAccountFacts?.(result);
          }
          // A successful billing action closes the issue. Map the tool result
          // into the same billing/resolution shape the text path returns so the
          // resolution journey advances to "close" and the invoice updates
          // immediately (the backend also persists this; refreshAssistData
          // reconciles on the next fetch).
          const billingResult =
            name === "apply_billing_action" &&
            result &&
            typeof result === "object" &&
            (result as Record<string, unknown>).applied
              ? (result as Record<string, unknown>)
              : null;
          if (billingResult) {
            onNudge({
              tool_name: name,
              tool_result: result,
              billing: { applied: true, adjustment: billingResult },
              resolution: { status: "closed" },
            } as unknown as LiveNudge);
          } else {
            onNudge({ tool_name: name, tool_result: result } as unknown as LiveNudge);
          }
        },
        onError: (code, message) => {
          setErr(message);
          if (code === "ws_closed") {
            // Connection dropped — fully end the call
            rtSessionRef.current = null;
            teardownPlayback();
            setInCall(false);
            voicePhaseRef.current = "idle";
            interimTextRef.current = "";
            onVoiceUiChange({ phase: "idle" });
          } else if (voicePhaseRef.current !== "idle") {
            voicePhaseRef.current = "speaking";
            onVoiceUiChange({ phase: "speaking", source: "mic", micLevel: 0.15 });
          }
        },
      }, callLanguage, { profile: "billing", startMicPaused: true });
      rtSessionRef.current = session;
      setInCall(true);
    } catch (e) {
      voicePhaseRef.current = "idle";
      onVoiceUiChange({ phase: "idle" });
      setErr(e instanceof Error ? e.message : "Microphone access denied");
    } finally {
      startingRef.current = false;
    }
  };

  const endCall = () => {
    rtSessionRef.current?.close();
    rtSessionRef.current = null;
    teardownPlayback();
    setInCall(false);
    voicePhaseRef.current = "idle";
    interimTextRef.current = "";
    onVoiceUiChange({ phase: "idle" });
  };

  // Agent-initiated: open the call automatically on mount so Genie greets the
  // caller without a "Start Call" tap — the same behavior the button used to
  // trigger. Browsers require a prior user gesture to play audio; arriving from
  // the Home concierge (same document via hash routing) carries that activation
  // so the greeting plays immediately. Scheduling via a cleared timeout keeps
  // React StrictMode's dev double-mount from opening two mics/sessions.
  useEffect(() => {
    const t = window.setTimeout(() => void startVoice(), 0);
    return () => window.clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Follow the language picker mid-call. The language is negotiated once in the
  // session.start payload, so the framework reopens the session in the new
  // language (which also re-greets in it) instead of leaving the caller with an
  // English-speaking agent behind a translated UI.
  useEffect(() => {
    void switchLanguage(expectedLanguage ?? language);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [expectedLanguage, language, switchLanguage]);

  // Tapping the Genie orb: interrupt while the agent speaks; otherwise start or
  // resume audio if the browser suspended the context.
  const onOrbTap = () => {
    if (inCall && (voiceUi.phase === "agent_reply" || voiceUi.phase === "speaking")) {
      interrupt();
      return;
    }
    if (inCall) {
      playbackRef.current?.resume();
      return;
    }
    void startVoice();
  };

  // Barge-in via Escape while the agent is speaking.
  useEffect(() => {
    if (!inCall || (voiceUi.phase !== "agent_reply" && voiceUi.phase !== "speaking")) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") interrupt();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [inCall, voiceUi.phase, interrupt]);

  const orbState: VoiceOrbState = !inCall
    ? "idle"
    : voiceUi.phase === "agent_reply" || voiceUi.phase === "speaking"
      ? "speaking"
      : voiceUi.phase === "transcribing"
        ? "thinking"
        : "listening";

  return (
    <div className="sentient-voicebar">
      <VoiceOrb
        state={orbState}
        level={voiceUi.micLevel ?? 0.15}
        size="96px"
        onClick={onOrbTap}
        ariaLabel={inCall ? copy.liveListeningHint : copy.startCall}
      />
      <div className="sentient-voicebar-status">
        {inCall ? copy.liveListeningHint : copy.startCall}
      </div>
      <div className="sentient-voicebar-actions">
        {inCall && (
          <button
            className="sentient-btn sentient-btn-mic is-recording"
            onClick={endCall}
          >
            {copy.endCall}
          </button>
        )}
        {onReset && (
          <button
            className="sentient-btn-ghost sentient-btn-sm"
            onClick={onReset}
            disabled={resetBusy}
          >
            {resetBusy ? copy.resetting : copy.resetScenario}
          </button>
        )}
      </div>
      {err && <div className="sentient-alert">{err}</div>}
    </div>
  );
}
