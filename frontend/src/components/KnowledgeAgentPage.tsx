import { useCallback, useEffect, useMemo, useReducer, useRef, useState } from "react";
import { RealtimeVoiceSession, startRealtimeVoice } from "../lib/realtimeVoice";
import { useHalfDuplexVoice } from "../hooks/useHalfDuplexVoice";
import { getMe } from "../lib/me";
import { emptyConversation, turnReducer } from "../lib/turnState";
import {
  DEFAULT_LANGUAGE_OPTIONS,
  Lang,
  fetchSupportedLanguages,
} from "../lib/languages";
import { LanguageBar } from "./LanguageBar";
import { BrandLockup } from "./BrandLockup";
import { VoiceBackdrop } from "./VoiceBackdrop";
import { VoiceOrb } from "./VoiceOrb";
import {
  AnalysisResult,
  StructuredAnswer,
} from "./StructuredAnswer";
import { API_BASE_URL, WS_BASE_URL } from "../config";
import { getAppLanguage } from "../lib/appLanguage";
import {
  ProgressStep,
  formatElapsed,
  readProgressSteps,
} from "../lib/workspaceProgress";
import "../styles/knowledge.css";

/**
 * Databricks Knowledge Agent — cited platform Q&A by voice (Tier 1).
 *
 * The "knowledge" realtime profile greets the signed-in user and answers
 * questions about the Databricks platform from a governed corpus, backed by the
 * `knowledge_search` tool. `/knowledge/corpus` lists the same entries the tool
 * searches, so the topic tiles on screen and the spoken answers can never
 * disagree about what the agent knows.
 *
 * CITE-OR-SILENCE made visible: every `knowledge_search` result carries its own
 * citation, and this page renders those citations beside the answer. When the
 * search comes back empty the agent says so instead of improvising, and the
 * sources panel stays empty — the UI shows exactly what grounded the answer.
 *
 * Uses the FRAMEWORK primitives like every other voice surface: the shared voice
 * stack (startRealtimeVoice + useHalfDuplexVoice), the shared config-driven
 * language bar, and the shared greeting mechanism. SEAM: back the corpus with
 * Databricks Vector Search over real docs without touching this component.
 */

/**
 * One published question. Every card is a LIVE Genie One question: asking it runs
 * a round-trip against the caller's own governed workspace, so there is only a
 * ``preview`` of what asking does — the answer itself arrives from that turn.
 */
type Topic = {
  id: string;
  category: string;
  topic: string;
  question: string;
  source: string;
  lane: string;
  preview: string;
};

type AgentState = "idle" | "greeting" | "listening" | "thinking" | "speaking";

/**
 * What actually grounded the answer, read off the runtime's ``evidence.available``
 * event. Domain-neutral on purpose: the same shape covers a Genie One table and a
 * docs lookup, so the panel never has to know which tool ran.
 */
type AnswerEvidence = {
  tool: string;
  source: string;
  citations: string[];
  columns: string[];
  rowCount: number;
  queryResults: AnalysisResult[];
  errorCode: string | null;
};

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" ? (value as Record<string, unknown>) : null;
}

function readEvidence(payload: Record<string, unknown>): AnswerEvidence | null {
  const evidence = asRecord(payload.evidence);
  if (!evidence) return null;
  const table = asRecord(evidence.table);
  const prose = asRecord(evidence.prose);
  const error = asRecord(evidence.error);
  const meta = asRecord(evidence.meta);
  const list = (value: unknown): string[] =>
    Array.isArray(value) ? value.map((v) => String(v)) : [];
  const queryResults: AnalysisResult[] = Array.isArray(meta?.query_results)
    ? meta.query_results.flatMap((value) => {
        const result = asRecord(value);
        if (!result || !Array.isArray(result.columns) || !Array.isArray(result.rows)) {
          return [];
        }
        const columns = result.columns.flatMap((value) => {
          const column = asRecord(value);
          return column && typeof column.name === "string"
            ? [{ name: column.name, typeText: String(column.type_text ?? "") }]
            : [];
        });
        if (!columns.length) return [];
        return [{
          itemId: String(result.item_id ?? ""),
          sql: typeof result.sql === "string" ? result.sql : null,
          columns,
          rows: result.rows.filter(Array.isArray),
          totalRowCount:
            typeof result.total_row_count === "number"
              ? result.total_row_count
              : result.rows.length,
          truncated: result.truncated === true,
        }];
      })
    : [];
  return {
    tool: typeof payload.name === "string" ? payload.name : "",
    source: typeof evidence.source === "string" ? evidence.source : "",
    // The query itself is shown once, beside the result it produced, under the full
    // answer's "View SQL" — not here, where it would repeat the same statement.
    citations: [...list(table?.citations), ...list(prose?.citations)],
    columns: list(table?.columns),
    rowCount: Array.isArray(table?.rows) ? table.rows.length : 0,
    queryResults,
    errorCode: typeof error?.code === "string" ? error.code : null,
  };
}

export function KnowledgeAgentPage() {
  const [, dispatchTurn] = useReducer(turnReducer, undefined, emptyConversation);
  const [phase, setPhase] = useState<"idle" | "connecting" | "live">("idle");
  const [agentState, setAgentState] = useState<AgentState>("idle");
  const [topics, setTopics] = useState<Topic[]>([]);
  const [categories, setCategories] = useState<string[]>([]);
  // The card the user tapped to read (and, when live, hear) its cited answer.
  const [openId, setOpenId] = useState<string | null>(null);
  const [langOptions, setLangOptions] = useState<Lang[]>(DEFAULT_LANGUAGE_OPTIONS);
  // Seeded from the home-page choice; the picker here is locked (chosen once, on #/).
  const [callLanguage, setCallLanguage] = useState<string>(() => getAppLanguage());
  const [caption, setCaption] = useState("");
  const [micLevel, setMicLevel] = useState(0);
  const [speakLevel, setSpeakLevel] = useState(0);
  // The live answer, on screen while it is being spoken. Filled from the same turn
  // the voice is reading, so the panel and the audio can never disagree.
  const [askedQuestion, setAskedQuestion] = useState("");
  const [answerText, setAnswerText] = useState("");
  const [answerProgress, setAnswerProgress] = useState("");
  const [progressSteps, setProgressSteps] = useState<ProgressStep[]>([]);
  // Wall clock for the wait. Set when the question is transcribed, cleared the
  // moment there is an answer, so the timer measures the wait and nothing else.
  const [waitStartedAt, setWaitStartedAt] = useState<number | null>(null);
  const [waitNow, setWaitNow] = useState(() => Date.now());
  const [fullAnswer, setFullAnswer] = useState("");
  const [localizationPending, setLocalizationPending] = useState(false);
  const [answerEvidence, setAnswerEvidence] = useState<AnswerEvidence[]>([]);
  const [answerFailure, setAnswerFailure] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const sessionRef = useRef<RealtimeVoiceSession | null>(null);
  const callIdRef = useRef<string>("");
  const callLanguageRef = useRef<string>(getAppLanguage());
  const greetingRef = useRef<Map<string, string>>(new Map());

  const {
    playbackRef,
    resetPlayback,
    gateMic,
    ungateMicAfter,
    handleResponseAudio,
    handlePlaybackStop,
    interimText,
    handleInterimTranscript,
    interrupt,
    switchLanguage,
    teardownPlayback,
  } = useHalfDuplexVoice({
    sessionRef,
    onMicResume: () => setAgentState("listening"),
    onSpeaking: () => setAgentState("speaking"),
    callLanguageRef,
    isCallLive: () => phase !== "idle",
    closeSession: () => {
      sessionRef.current?.close();
      sessionRef.current = null;
      setAgentState("greeting");
      setPhase("connecting");
    },
    reopenSession: (nextLanguage) => openSession(nextLanguage),
    // This page shows one rolling caption rather than a transcript, so the stale
    // line to drop is just that.
    clearConversation: () => setCaption(""),
  });

  useEffect(() => {
    if (agentState !== "speaking") {
      setSpeakLevel(0);
      return;
    }
    let raf = 0;
    const tick = () => {
      setSpeakLevel(playbackRef.current?.getLevel() ?? 0);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [agentState, playbackRef]);

  useEffect(() => {
    let active = true;
    void (async () => {
      try {
        const r = await fetch(`${API_BASE_URL}/knowledge/corpus`);
        const data = (await r.json()) as { topics?: Topic[]; categories?: string[] };
        if (active) {
          setTopics(data.topics ?? []);
          setCategories(data.categories ?? []);
        }
      } catch {
        /* tiles just stay empty */
      }
    })();
    void (async () => {
      const langs = await fetchSupportedLanguages();
      if (active && langs.length) setLangOptions(langs);
    })();
    return () => {
      active = false;
    };
  }, []);

  const fetchGreeting = useCallback(async (language: string): Promise<string> => {
    const cached = greetingRef.current.get(language);
    if (cached !== undefined) return cached;
    try {
      // Greet the signed-in Databricks user by name (nameless when anonymous).
      const me = await getMe();
      const nameQ = me.name ? `&name=${encodeURIComponent(me.name)}` : "";
      const r = await fetch(
        `${API_BASE_URL}/knowledge/greeting?language=${encodeURIComponent(language)}${nameQ}`
      );
      const data = (await r.json()) as { text?: string };
      const text = typeof data.text === "string" ? data.text : "";
      greetingRef.current.set(language, text);
      return text;
    } catch {
      return "";
    }
  }, []);

  const speakViaTTS = useCallback(
    (text: string) => {
      const session = sessionRef.current;
      if (!session || !text.trim()) return;
      playbackRef.current?.flush();
      gateMic();
      setAgentState("speaking");
      session.synthesize(text, callLanguageRef.current);
    },
    [gateMic, playbackRef]
  );

  const teardown = useCallback(() => {
    teardownPlayback();
    sessionRef.current?.close();
    sessionRef.current = null;
  }, [teardownPlayback]);

  useEffect(() => () => teardown(), [teardown]);

  const openSession = useCallback(
    async (language: string) => {
      const session = await startRealtimeVoice(
        WS_BASE_URL,
        callIdRef.current,
        "guest",
        {
          onSessionReady: () => {
            setPhase("live");
            setAgentState("greeting");
            void (async () => {
              const greeting = await fetchGreeting(callLanguageRef.current);
              if (greeting) {
                setCaption(greeting);
                speakViaTTS(greeting);
              } else {
                ungateMicAfter(0);
                setAgentState("listening");
              }
            })();
          },
          onLevel: (level) => setMicLevel(level),
          onInterimTranscript: (text) => handleInterimTranscript(text),
          onSpeechStarted: () => setAgentState("listening"),
          onTurnStarted: () => setAgentState((s) => (s === "speaking" ? s : "thinking")),
          onTranscript: (text, _language, turnId) => {
            dispatchTurn({ type: "user_transcript", turnId, text });
            if (!text.trim()) return;
            // A new question retires the previous answer, so the panel never shows
            // last turn's evidence under this turn's question.
            setCaption(text);
            setAskedQuestion(text);
            setAnswerText("");
            setAnswerProgress("");
            setProgressSteps([]);
            setWaitStartedAt(Date.now());
            setFullAnswer("");
            setLocalizationPending(false);
            setAnswerEvidence([]);
            setAnswerFailure(null);
          },
          onResponseText: (text, turnId) => {
            dispatchTurn({ type: "response_text", turnId, text });
          },
          onTurnEvent: ({ turnId, seq, kind, payload }) => {
            dispatchTurn({ type: "turn_event", turnId, seq, kind, payload });
            if (kind === "answer.render.started") {
              if (typeof payload.question === "string") setAskedQuestion(payload.question);
              if (typeof payload.summary === "string") setAnswerText(payload.summary);
              setFullAnswer(typeof payload.report === "string" ? payload.report : "");
              setLocalizationPending(payload.localization_pending === true);
              setAnswerProgress("");
              setWaitStartedAt(null);
            } else if (kind === "answer.render.delta") {
              if (typeof payload.delta === "string") {
                setFullAnswer((previous) => previous + payload.delta);
              }
            } else if (kind === "answer.render.completed") {
              setLocalizationPending(false);
            } else if (kind === "speech.committed") {
              // The words the voice is about to read. Showing these (rather than the
              // model's display prose) is what keeps the page and the audio identical.
              if (typeof payload.text === "string" && payload.text.trim()) {
                setAnswerText(payload.text);
                setAnswerProgress("");
                setWaitStartedAt(null);
              }
            } else if (kind === "speech.progress") {
              // Exact localized words being spoken during long governed work.
              if (typeof payload.text === "string" && payload.text.trim()) {
                setAnswerProgress(payload.text);
              }
              const steps = readProgressSteps(payload.steps);
              if (steps.length) setProgressSteps(steps);
            } else if (kind === "tool.progress") {
              // Upstream confirmed it is still active. Its own steps say what is
              // running; keep any richer spoken phrase already shown, and never
              // stand in for the answer with an invented partial one.
              const steps = readProgressSteps(payload.steps);
              if (steps.length) setProgressSteps(steps);
              setAnswerProgress((previous) =>
                previous || "I’m reviewing the relevant business information…"
              );
            } else if (kind === "evidence.available") {
              const evidence = readEvidence(payload);
              if (evidence) setAnswerEvidence((prev) => [...prev, evidence]);
            } else if (kind === "turn.failed") {
              const code = typeof payload.code === "string" ? payload.code : "failed";
              setAnswerFailure(code);
              setWaitStartedAt(null);
            }
          },
          onResponseAudio: (pcmB64, sampleRate, final, _turnId, meta) => {
            handleResponseAudio(pcmB64, sampleRate, final, meta);
          },
          onPlaybackStop: (_turnId, speechEpoch, reason) => {
            handlePlaybackStop(speechEpoch, reason);
          },
          onError: (code, message) => {
            if (code === "ws_closed") {
              setError(message);
              teardown();
              setPhase("idle");
              setAgentState("idle");
            } else {
              setError(`${code}: ${message}`);
            }
          },
        },
        language,
        { profile: "knowledge", startMicPaused: true }
      );
      sessionRef.current = session;
    },
    [
      fetchGreeting,
      speakViaTTS,
      ungateMicAfter,
      handleInterimTranscript,
      handleResponseAudio,
      handlePlaybackStop,
      teardown,
    ]
  );

  const startCall = useCallback(async () => {
    if (phase !== "idle") return;
    setAnswerFailure(null);
    setError(null);
    setPhase("connecting");
    setAgentState("greeting");
    callIdRef.current = `knowledge-${Date.now()}`;
    resetPlayback();
    void fetchGreeting(callLanguageRef.current);
    try {
      await openSession(callLanguageRef.current);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Microphone access denied");
      setPhase("idle");
      setAgentState("idle");
    }
  }, [phase, resetPlayback, fetchGreeting, openSession]);

  // Auto-connect on mount so Genie greets the moment the page opens — same as the
  // card assistant. The browser gesture that arrived with us from Home unlocks
  // audio; a cold refresh falls back to tapping the orb. The cleared timeout keeps
  // React StrictMode's dev double-mount from opening two sessions.
  useEffect(() => {
    const t = window.setTimeout(() => void startCall(), 0);
    return () => window.clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const changeLanguage = useCallback(
    async (code: string) => {
      if (code === callLanguageRef.current) return;
      setCallLanguage(code);
      // Mic gating / teardown / reopen ordering lives in the shared voice hook, so
      // every use case moves a live call to a new language identically.
      try {
        await switchLanguage(code);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
        setPhase("idle");
        setAgentState("idle");
      }
    },
    [switchLanguage]
  );

  const endCall = useCallback(() => {
    teardown();
    setPhase("idle");
    setAgentState("idle");
    setCaption("");
    setAskedQuestion("");
    setAnswerText("");
    setAnswerProgress("");
    setProgressSteps([]);
    setWaitStartedAt(null);
    setFullAnswer("");
    setLocalizationPending(false);
    setAnswerEvidence([]);
    setAnswerFailure(null);
  }, [teardown]);

  // A governed workspace read runs for minutes, so the wait gets a visible clock.
  // It ticks only while something is actually pending.
  useEffect(() => {
    if (waitStartedAt === null) return;
    setWaitNow(Date.now());
    const id = window.setInterval(() => setWaitNow(Date.now()), 1000);
    return () => window.clearInterval(id);
  }, [waitStartedAt]);

  // Group topics into their categories, preserving the corpus's declared order
  // (falling back to first-seen order for any category the payload omits).
  const groups = useMemo(() => {
    const byCat = new Map<string, Topic[]>();
    for (const t of topics) {
      const list = byCat.get(t.category);
      if (list) list.push(t);
      else byCat.set(t.category, [t]);
    }
    const ordered = [
      ...categories.filter((c) => byCat.has(c)),
      ...[...byCat.keys()].filter((c) => !categories.includes(c)),
    ];
    return ordered.map((category) => ({ category, items: byCat.get(category) ?? [] }));
  }, [topics, categories]);

  const analysisResults = useMemo(() => {
    const seen = new Set<string>();
    return answerEvidence.flatMap((evidence) =>
      evidence.queryResults.filter((result) => {
        const key = result.itemId || `${result.sql}-${result.rows.length}`;
        if (seen.has(key)) return false;
        seen.add(key);
        return true;
      })
    );
  }, [answerEvidence]);

  // Tapping a question shows what asking it will run. It deliberately does NOT
  // touch the answer panel: that panel only ever shows what Genie One actually
  // returned for a question you asked out loud.
  const onTopicClick = useCallback((t: Topic) => {
    setOpenId((prev) => (prev === t.id ? null : t.id));
  }, []);

  const answerStatus = answerFailure
    ? null
    : answerText
      ? agentState === "speaking"
        ? "Speaking now"
        : "Answered"
      : "Asking Genie One…";

  const waitElapsed =
    waitStartedAt === null ? null : formatElapsed(waitNow - waitStartedAt);

  const orbLevel = agentState === "speaking" ? speakLevel : agentState === "listening" ? micLevel : 0;
  const liveCaption = interimText.trim() || caption;

  // Barge-in via Escape / orb tap while the agent is speaking.
  useEffect(() => {
    if (agentState !== "speaking") return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") interrupt();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [agentState, interrupt]);

  const onOrbTap = () => {
    if (agentState === "speaking") {
      interrupt();
      return;
    }
    if (phase === "idle") void startCall();
    else playbackRef.current?.resume();
  };

  return (
    <div className="kb-root">
      <VoiceBackdrop />
      <div className="gv-content">
        <header className="kb-top">
          <BrandLockup product="Knowledge Agent" />
          <button type="button" className="kb-back" onClick={() => (window.location.hash = "#/")}>
            ← Home
          </button>
        </header>

        <div className="kb-body">
          {/* Voice rail */}
          <aside className="kb-rail">
            <div className="kb-orb">
              <VoiceOrb
                state={agentState}
                level={orbLevel}
                size="120px"
                onClick={onOrbTap}
                ariaLabel={agentState === "speaking" ? "Interrupt Genie" : "Genie"}
              />
            </div>
            <div className={`kb-status kb-status-${agentState}`}>
              {phase === "idle" && "Tap the orb to start"}
              {agentState === "greeting" && "Connecting…"}
              {agentState === "listening" && "Listening…"}
              {agentState === "thinking" && "Asking your governed workspace…"}
              {agentState === "speaking" && "Genie is speaking…"}
            </div>
            {liveCaption && phase !== "idle" && <p className="kb-caption">{liveCaption}</p>}
            <div className="kb-langwrap">
              <LanguageBar
                value={callLanguage}
                options={langOptions}
                onChange={(c) => void changeLanguage(c)}
                disabled
              />
            </div>
            {/* The call starts on load, so there is no "start" CTA to press. Ending
                is still explicit, and the orb re-opens a session after a cold
                refresh where the browser withheld the audio gesture. */}
            <button type="button" className="kb-end" onClick={endCall}>
              End call
            </button>
            {error && <p className="kb-error">{error}</p>}
            <p className="kb-disclaimer">
              Every question here runs live against your own Databricks workspace through
              Genie One, under your permissions — so the answer describes what you can
              actually reach. Anything Genie cannot cite, it says so instead of guessing.
            </p>
          </aside>

          {/* Knowledge canvas */}
          <main className="kb-canvas">
            <section className="kb-head">
              <div>
                <div className="kb-head-title">What you can ask Genie One</div>
                <div className="kb-head-sub">
                  {topics.length
                    ? `${topics.length} questions across ${groups.length} areas · every one runs live against your own governed workspace`
                    : "Loading questions…"}
                </div>
              </div>
              <div className="kb-head-badge">Genie One · live</div>
            </section>

            {/* The answer, on screen while it is spoken. */}
            <section className={`kb-answer${askedQuestion ? " is-active" : ""}`}>
              <div className="kb-section-title">
                Answer
                {askedQuestion && answerStatus && (
                  <span className="kb-answer-status">{answerStatus}</span>
                )}
              </div>
              {askedQuestion ? (
                <>
                  <p className="kb-asked">“{askedQuestion}”</p>
                  {answerFailure ? (
                    <p className="kb-error">
                      {answerFailure === "timeout"
                        ? "Genie One took too long to answer. Ask again, or narrow the question."
                        : `Genie One could not complete that turn (${answerFailure}).`}
                    </p>
                  ) : answerText ? (
                    <>
                      <p className="kb-answer-text">{answerText}</p>
                      {(fullAnswer || localizationPending) && (
                        <div className="kb-full-answer">
                          <div className="kb-evidence-title">Full answer</div>
                          {fullAnswer ? (
                            <StructuredAnswer
                              markdown={fullAnswer}
                              results={analysisResults}
                            />
                          ) : (
                            <p className="kb-answer-pending">
                              Translating the full answer…
                            </p>
                          )}
                          {fullAnswer && localizationPending && (
                            <p className="kb-answer-pending">
                              Translating the full answer…
                            </p>
                          )}
                        </div>
                      )}
                    </>
                  ) : (
                    <div className="kb-progress">
                      <div className="kb-progress-head">
                        <span className="kb-progress-spinner" aria-hidden="true" />
                        <span className="kb-progress-label">
                          Analyzing your request
                        </span>
                        {waitElapsed && (
                          <span className="kb-progress-clock">{waitElapsed}</span>
                        )}
                      </div>
                      {/* The words being spoken right now, so the screen and the
                          audio say the same thing. */}
                      <p className="kb-progress-said">
                        {answerProgress || "I’m working through the relevant information…"}
                      </p>
                      {progressSteps.length > 0 && (
                        <ol className="kb-progress-steps">
                          {progressSteps.map((step, i) => (
                            <li
                              className={`kb-progress-step is-${step.status}`}
                              key={`${i}-${step.label}`}
                            >
                              <span className="kb-progress-dot" aria-hidden="true" />
                              <span className="kb-progress-step-label">{step.label}</span>
                            </li>
                          ))}
                        </ol>
                      )}
                    </div>
                  )}
                  {answerEvidence.length > 0 && (
                    <div className="kb-evidence">
                      <div className="kb-evidence-title">What grounded this</div>
                      {answerEvidence.map((e, i) => (
                        <div className="kb-evidence-item" key={`${e.tool}-${i}`}>
                          <div className="kb-evidence-head">
                            <span className="kb-evidence-source">{e.source || e.tool}</span>
                            {e.errorCode ? (
                              <span className="kb-evidence-flag">{e.errorCode}</span>
                            ) : (
                              e.rowCount > 0 && (
                                <span className="kb-evidence-shape">
                                  {e.rowCount} {e.rowCount === 1 ? "row" : "rows"} ·{" "}
                                  {e.columns.length}{" "}
                                  {e.columns.length === 1 ? "column" : "columns"}
                                </span>
                              )
                            )}
                          </div>
                          {e.citations.length > 0 && (
                            <div className="kb-evidence-cites">{e.citations.join(" · ")}</div>
                          )}
                        </div>
                      ))}
                    </div>
                  )}
                </>
              ) : (
                <p className="kb-answer-pending">
                  Ask any question below out loud. The answer appears here as Genie speaks
                  it, with the governed sources it came from.
                </p>
              )}
            </section>

            {groups.map(({ category, items }) => (
              <section key={category} className="kb-cat">
                <div className="kb-cat-title">{category}</div>
                <div className="kb-topics">
                  {items.map((t) => {
                    const isOpen = openId === t.id;
                    return (
                      <button
                        type="button"
                        key={t.id}
                        className={`kb-topic${isOpen ? " is-open" : ""}`}
                        onClick={() => onTopicClick(t)}
                        aria-expanded={isOpen}
                      >
                        {/* The question itself is the headline: this page publishes
                            questions to ask, not topics to browse. */}
                        <div className="kb-topic-name">
                          <span>{t.question}</span>
                          <span className="kb-topic-toggle" aria-hidden="true">
                            {isOpen ? "−" : "+"}
                          </span>
                        </div>
                        {isOpen && <div className="kb-topic-answer">{t.preview}</div>}
                        <div className="kb-topic-foot">
                          <span className="kb-topic-cite">{t.source}</span>
                        </div>
                      </button>
                    );
                  })}
                </div>
              </section>
            ))}
          </main>
        </div>
      </div>
    </div>
  );
}
