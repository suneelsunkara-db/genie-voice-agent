import { ReactElement, useCallback, useEffect, useReducer, useRef, useState } from "react";
import { RealtimeVoiceSession, startRealtimeVoice } from "../lib/realtimeVoice";
import { useHalfDuplexVoice } from "../hooks/useHalfDuplexVoice";
import { getMe } from "../lib/me";
import { useAppLanguage } from "../lib/appLanguage";
import { AppVoice, useAppVoice } from "../lib/appVoice";
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
import { API_BASE_URL, WS_BASE_URL } from "../config";
import "../styles/home.css";

/**
 * Landing page for "Genie Assisted Voice" (a Databricks demo).
 *
 * A voice concierge (the "concierge" realtime profile) greets the signed-in user
 * by name, gives a short spoken overview, and asks which experience to open. The
 * user answers BY VOICE; the LLM calls `select_industry`, which arrives as
 * `tool.called` and drives hash navigation:
 *   telco -> #/telco (billing), fsi -> #/card, knowledge -> #/knowledge.
 *
 * Everything routes through the FRAMEWORK: the shared voice stack
 * (startRealtimeVoice + useHalfDuplexVoice), the shared config-driven language
 * bar (lib/languages + LanguageBar), and the shared greeting mechanism. The
 * sample-question + FSI-context panels are illustrative concept visuals.
 */

type Industry = {
  id: "telco" | "fsi" | "knowledge";
  hash: string;
  title: string;
  tag: string;
  blurb: string;
  cues: string[];
  Icon: () => ReactElement;
};

const INDUSTRIES: Industry[] = [
  {
    id: "telco",
    hash: "#/telco",
    title: "Telco",
    tag: "Billing Support",
    blurb: "Resolve charges, waive fees, and set up payment plans on a live call.",
    cues: ["“Telco”", "“billing”", "“my phone bill”"],
    Icon: SignalIcon,
  },
  {
    id: "fsi",
    hash: "#/card",
    title: "Financial Services",
    tag: "Credit-Card Assistant",
    blurb: "Understand statements and rewards, with deep “why” reasoning on demand.",
    cues: ["“Financial services”", "“credit card”", "“my statement”"],
    Icon: CardIcon,
  },
  {
    id: "knowledge",
    hash: "#/knowledge",
    title: "Databricks Knowledge Agent",
    tag: "Cited Platform Q&A",
    blurb: "Ask how the platform works and hear an answer grounded in governed docs.",
    cues: ["“Knowledge”", "“Databricks”", "“the platform”"],
    Icon: KnowledgeIcon,
  },
];

// Three questions a caller can actually ask, one per surface — the concrete
// version of "what is this for?". Illustrative copy; each surface answers live.
type SampleQuestion = {
  id: string;
  surface: string;
  question: string;
  detail: string;
  Icon: () => ReactElement;
};

const SAMPLE_QUESTIONS: SampleQuestion[] = [
  {
    id: "telco",
    surface: "Telco",
    question: "“Can you waive the late fee on my account?”",
    detail: "Checks eligibility against policy, then applies the waiver on the call.",
    Icon: SignalIcon,
  },
  {
    id: "fsi",
    surface: "Financial Services",
    question: "“Which spend earned me the most rewards?”",
    detail: "Ranks the cycle's categories from the statement and names the top driver.",
    Icon: CardIcon,
  },
  {
    id: "knowledge",
    surface: "Knowledge Agent",
    question: "“What is Unity Catalog for?”",
    detail: "Answers from the governed knowledge base and attributes the source.",
    Icon: KnowledgeIcon,
  },
];

// A concrete illustrative "why" investigation — this is what DEEP reasoning does
// that a scripted bot can't: pull governed history, compare, isolate drivers,
// and explain with evidence. (Illustrative content; the live version runs on the
// card deep-dive page.)
const REASONING_QUESTION = "“Why is my card statement higher this month?”";
const REASONING_HOPS = [
  "Pulls six months of statement history",
  "Compares charges line-by-line against the usual baseline",
  "Isolates what actually changed this cycle",
];
const REASONING_CONCLUSION =
  "$41 of the $47 increase is a one-time annual fee — next cycle returns to normal.";
// Last-resort route fallback if the confirmation TTS stream never reaches a
// final audio chunk. Normal routing is driven by `onFinal` after audio drains.
const ROUTE_FALLBACK_MS = 30_000;

type AgentState = "idle" | "greeting" | "listening" | "thinking" | "speaking";

export function HomePage() {
  const [, dispatchTurn] = useReducer(turnReducer, undefined, emptyConversation);
  const [phase, setPhase] = useState<"idle" | "connecting" | "live">("idle");
  const [agentState, setAgentState] = useState<AgentState>("idle");
  const [userName, setUserName] = useState<string>("");
  const [langOptions, setLangOptions] = useState<Lang[]>(DEFAULT_LANGUAGE_OPTIONS);
  const [caption, setCaption] = useState<string>("");
  const [chosen, setChosen] = useState<Industry["id"] | null>(null);
  const [micLevel, setMicLevel] = useState(0);
  const [speakLevel, setSpeakLevel] = useState(0);
  const [error, setError] = useState<string | null>(null);
  // The one app language, chosen here on the home page and inherited (locked) by
  // every use-case page. The concierge greets AND listens in this language.
  const [appLanguage, setAppLanguage] = useAppLanguage();
  // One persisted voice choice. startRealtimeVoice reads it automatically, so
  // every destination page inherits the same speaker without per-page wiring.
  const [appVoice, setAppVoice] = useAppVoice();

  const sessionRef = useRef<RealtimeVoiceSession | null>(null);
  const callIdRef = useRef<string>("");
  const callLanguageRef = useRef<string>(appLanguage);
  // Greeting cache keyed by language (one model call per language, like card).
  const greetingRef = useRef<Map<string, string>>(new Map());
  const navTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  // Hash to route to once the agent's spoken confirmation finishes. Armed by the
  // select_industry tool result; consumed on the confirmation's final audio.
  const pendingNavRef = useRef<string | null>(null);
  // Latest "response finished speaking" handler; kept in a ref so the hook's
  // onFinal stays stable while always seeing fresh state (goTo/playbackRef).
  const onFinalRef = useRef<() => void>(() => {});

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
    teardownPlayback,
  } = useHalfDuplexVoice({
    sessionRef,
    onMicResume: () => setAgentState("listening"),
    onSpeaking: () => setAgentState("speaking"),
    onFinal: () => onFinalRef.current(),
  });

  // Speaking-orb amplitude from the actual TTS playback.
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

  // Prefetch the signed-in user + the config-driven supported language set.
  useEffect(() => {
    let active = true;
    void getMe().then((me) => {
      if (active && me.name) setUserName(me.name);
    });
    void (async () => {
      const langs = await fetchSupportedLanguages();
      if (active && langs.length) setLangOptions(langs);
    })();
    return () => {
      active = false;
    };
  }, []);

  const fetchGreeting = useCallback(
    async (language: string): Promise<string> => {
      const cached = greetingRef.current.get(language);
      if (cached !== undefined) return cached;
      try {
        const r = await fetch(
          `${API_BASE_URL}/concierge/greeting?language=${encodeURIComponent(language)}&name=${encodeURIComponent(userName)}`
        );
        const data = (await r.json()) as { text?: string };
        const text = typeof data.text === "string" ? data.text : "";
        greetingRef.current.set(language, text);
        return text;
      } catch {
        return "";
      }
    },
    [userName]
  );

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
    if (navTimerRef.current) {
      clearTimeout(navTimerRef.current);
      navTimerRef.current = null;
    }
    pendingNavRef.current = null;
    teardownPlayback();
    sessionRef.current?.close();
    sessionRef.current = null;
  }, [teardownPlayback]);

  useEffect(() => () => teardown(), [teardown]);

  const goTo = useCallback(
    (hash: string) => {
      teardown();
      window.location.hash = hash;
    },
    [teardown]
  );

  const navigateForIndustry = useCallback(
    (industry: Industry["id"]) => {
      const target = INDUSTRIES.find((i) => i.id === industry);
      if (!target) return;
      setChosen(industry);
      // Confirm-then-route: arm the target and let the agent's spoken
      // confirmation finish before the page swaps (handled in onFinal, below).
      // The timer here is deliberately long: it is ONLY a last-resort escape if
      // confirmation TTS stalls, never something that can race a normal TTS turn.
      pendingNavRef.current = target.hash;
      if (navTimerRef.current) clearTimeout(navTimerRef.current);
      navTimerRef.current = setTimeout(() => goTo(target.hash), ROUTE_FALLBACK_MS);
    },
    [goTo]
  );

  // When a response finishes streaming: if a route is pending (the caller just
  // picked an industry), navigate once the confirmation has actually drained
  // from the speakers — so we never cut it off mid-sentence.
  // Navigate only after turn_final (half-duplex onFinal). Mid-turn segment_final
  // must not route away while progressive speech is still in flight.
  onFinalRef.current = () => {
    const hash = pendingNavRef.current;
    if (!hash) return;
    pendingNavRef.current = null;
    if (navTimerRef.current) clearTimeout(navTimerRef.current);
    const wait = (playbackRef.current?.msUntilIdle() ?? 0) + 250;
    navTimerRef.current = setTimeout(() => goTo(hash), wait);
  };

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
            if (text.trim()) setCaption(text);
          },
          onResponseText: (text, turnId) => {
            dispatchTurn({ type: "response_text", turnId, text });
            if (text.trim()) setCaption(text);
          },
          onTurnEvent: ({ turnId, seq, kind, payload }) => {
            dispatchTurn({ type: "turn_event", turnId, seq, kind, payload });
          },
          onResponseAudio: (pcmB64, sampleRate, final, _turnId, meta) => {
            handleResponseAudio(pcmB64, sampleRate, final, meta);
          },
          onPlaybackStop: (_turnId, speechEpoch, reason) => {
            handlePlaybackStop(speechEpoch, reason);
          },
          onToolCalled: (name, result) => {
            const r = result && typeof result === "object" ? (result as Record<string, unknown>) : {};
            if (name === "select_industry" && typeof r.industry === "string") {
              navigateForIndustry(r.industry as Industry["id"]);
            }
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
        // Pin STT to the chosen app language so short destination replies are
        // transcribed in that language before typed semantic navigation.
        { profile: "concierge", startMicPaused: true, sttLanguage: language }
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
      navigateForIndustry,
      teardown,
    ]
  );

  const startCall = useCallback(async () => {
    if (phase !== "idle") return;
    setError(null);
    setChosen(null);
    setPhase("connecting");
    setAgentState("greeting");
    callIdRef.current = `home-${Date.now()}`;
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

  // Pick the app language from the dropdown. Records it globally (every use-case
  // page inherits it) and, if a concierge call is already live, restarts the
  // session in the new language so the greeting + STT follow immediately (the
  // session's language is fixed once opened).
  const changeLanguage = useCallback(
    (code: string) => {
      if (!code || code === callLanguageRef.current) return;
      setAppLanguage(code);
      callLanguageRef.current = code;
      greetingRef.current.clear();
      setCaption("");
      if (phase !== "idle") {
        teardown();
        setChosen(null);
        setError(null);
        setPhase("connecting");
        setAgentState("greeting");
        callIdRef.current = `home-${Date.now()}`;
        resetPlayback();
        void openSession(code).catch((e) => {
          setError(e instanceof Error ? e.message : String(e));
          setPhase("idle");
          setAgentState("idle");
        });
      }
    },
    [phase, setAppLanguage, teardown, resetPlayback, openSession]
  );

  const changeVoice = useCallback(
    (voice: AppVoice) => {
      if (voice === appVoice) return;
      setAppVoice(voice);
      // A session's reference is immutable once speech begins. Restart a live
      // concierge session so the next greeting already uses the selected voice;
      // all later pages read the same persisted choice at session.start.
      if (phase !== "idle") {
        teardown();
        setChosen(null);
        setCaption("");
        setError(null);
        setPhase("connecting");
        setAgentState("greeting");
        callIdRef.current = `home-${Date.now()}`;
        resetPlayback();
        void openSession(callLanguageRef.current).catch((e) => {
          setError(e instanceof Error ? e.message : String(e));
          setPhase("idle");
          setAgentState("idle");
        });
      }
    },
    [appVoice, setAppVoice, phase, teardown, resetPlayback, openSession]
  );

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

  const onOrbTap = useCallback(() => {
    if (agentState === "speaking") {
      interrupt();
      return;
    }
    if (phase === "idle") void startCall();
  }, [agentState, phase, interrupt, startCall]);

  return (
    <div className="home-root">
      <VoiceBackdrop />

      <header className="home-top">
        <BrandLockup product="Assisted Voice" />
        <div className="home-topright">
          <div className="home-voice-choice" role="group" aria-label="Agent voice">
            <span className="home-voice-label">Voice</span>
            <div className="home-voice-toggle">
              {(["female", "male"] as const).map((voice) => (
                <button
                  key={voice}
                  type="button"
                  className={appVoice === voice ? "is-active" : ""}
                  aria-pressed={appVoice === voice}
                  onClick={() => changeVoice(voice)}
                >
                  {voice === "female" ? "Female" : "Male"}
                </button>
              ))}
            </div>
          </div>
          {/* The single source of truth for language: chosen here, inherited and
              locked on every use-case page. Changing it re-greets in the new
              language (and restarts a live concierge call). */}
          <LanguageBar
            value={appLanguage}
            options={langOptions}
            onChange={changeLanguage}
            label="Language"
          />
          <nav className="home-topnav">
            <a href="#/voice-benchmarks">Benchmarks</a>
            <a href="#/traces">Traces</a>
            <a href="#/guardrails">Guardrails</a>
          </nav>
        </div>
      </header>

      <main className="home-main">
        <section className="home-hero">
          <VoiceOrb
            state={agentState}
            level={orbLevel}
            size="clamp(74px, 11vh, 112px)"
            onClick={onOrbTap}
            disabled={phase !== "idle" && agentState !== "speaking"}
            ariaLabel={
              agentState === "speaking"
                ? "Interrupt Genie"
                : phase === "idle"
                  ? "Talk to Genie"
                  : "Genie"
            }
          />

          <h1 className="home-title">
            {userName ? (
              <>Welcome back, {userName}.</>
            ) : (
              <>Welcome to Databricks Genie Assisted Voice.</>
            )}
          </h1>
          <p className="home-sub">
            One voice platform, {langOptions.length}{" "}
            {langOptions.length === 1 ? "language" : "languages"}, three agents — powered by the
            Databricks Genie ontology and deep reasoning.
          </p>

          <div className="home-controls">
            {phase === "idle" ? (
              <button type="button" className="home-cta" onClick={() => void startCall()}>
                Talk to Genie
              </button>
            ) : (
              <div className={`home-status home-status-${agentState}`}>
                {agentState === "greeting" && "Genie is connecting…"}
                {agentState === "listening" &&
                  "Listening — say Telco, Financial Services, or Knowledge"}
                {agentState === "thinking" && "Thinking…"}
                {agentState === "speaking" && "Genie is speaking…"}
                {agentState === "idle" && "Ready"}
              </div>
            )}
          </div>

          {liveCaption && phase !== "idle" && <p className="home-caption">{liveCaption}</p>}
          {error && <p className="home-error">{error}</p>}
        </section>

        <section className="home-industries">
          {INDUSTRIES.map((ind) => (
            <button
              key={ind.id}
              type="button"
              className={`home-card home-card-${ind.id}${chosen === ind.id ? " is-chosen" : ""}`}
              onClick={() => goTo(ind.hash)}
            >
              <div className="home-card-head">
                <span className="home-card-icon" aria-hidden="true">
                  <ind.Icon />
                </span>
                <div className="home-card-tag">{ind.tag}</div>
              </div>
              <div className="home-card-title">{ind.title}</div>
              <div className="home-card-blurb">{ind.blurb}</div>
              <div className="home-card-cues">
                {ind.cues.map((c) => (
                  <span key={c}>{c}</span>
                ))}
              </div>
              <div className="home-card-go">Open →</div>
            </button>
          ))}
        </section>

        <section className="home-genie">
          <SampleQuestionsPanel />
          <ReasoningPanel />
        </section>
      </main>

      <footer className="home-foot">
        Powered by Databricks Genie · Unity Catalog governed · Realtime API built on OSS voice models
      </footer>
    </div>
  );
}

function SampleQuestionsPanel() {
  return (
    <div className="home-panel">
      <div className="home-panel-title">Ask Genie</div>
      <div className="home-panel-sub">
        Say it out loud — each surface answers from its own governed data.
      </div>
      <div className="home-samples">
        {SAMPLE_QUESTIONS.map((s, i) => (
          <div
            key={s.id}
            className={`home-sample home-sample-${s.id}`}
            style={{ animationDelay: `${0.15 + i * 0.18}s` }}
          >
            <div className="home-sample-head">
              <span className="home-sample-icon" aria-hidden="true">
                <s.Icon />
              </span>
              <span className="home-sample-surface">{s.surface}</span>
            </div>
            <div className="home-sample-q">{s.question}</div>
            <div className="home-sample-detail">{s.detail}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

function ReasoningPanel() {
  return (
    <div className="home-panel">
      <div className="home-panel-title">FSI context</div>
      <div className="home-panel-sub">
        Beyond quick answers — Genie investigates the “why”, grounded in your own data.
      </div>
      <div className="home-dr">
        <div className="home-dr-q">{REASONING_QUESTION}</div>
        <ol className="home-dr-hops">
          {REASONING_HOPS.map((hop, i) => (
            <li key={hop} style={{ animationDelay: `${0.4 + i * 0.7}s` }}>
              <span className="home-dr-dot" />
              {hop}
            </li>
          ))}
        </ol>
        <div className="home-dr-answer" style={{ animationDelay: `${0.4 + REASONING_HOPS.length * 0.7}s` }}>
          {REASONING_CONCLUSION}
        </div>
        <div className="home-dr-foot">Grounded in governed data · with citations</div>
      </div>
    </div>
  );
}

/* ---- surface icons ------------------------------------------------------- *
 * Inline single-stroke glyphs rather than an icon dependency: they inherit
 * `currentColor` from the card, so each surface tints its own icon and the set
 * stays visually uniform across the home page. */

function SignalIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round">
      <path d="M12 20v-7" />
      <path d="M8.4 10.4a5 5 0 0 1 7.2 0" />
      <path d="M5.6 7.4a9 9 0 0 1 12.8 0" />
      <circle cx="12" cy="12.6" r="1.3" fill="currentColor" stroke="none" />
    </svg>
  );
}

function CardIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round">
      <rect x="3" y="6" width="18" height="12" rx="2.5" />
      <path d="M3 10.5h18" />
      <path d="M6.5 14.5h4" />
    </svg>
  );
}

function KnowledgeIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round">
      <path d="M4 5.5A2 2 0 0 1 6 3.5h5V20H6a2 2 0 0 0-2 2z" />
      <path d="M11 3.5h7a2 2 0 0 1 2 2V16" />
      <circle cx="17.5" cy="19" r="2.5" />
      <path d="M20 16v.5a2.5 2.5 0 0 1-2.5 2.5" />
    </svg>
  );
}
