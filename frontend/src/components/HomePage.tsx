import { useCallback, useEffect, useRef, useState } from "react";
import { RealtimeVoiceSession, startRealtimeVoice } from "../lib/realtimeVoice";
import { useHalfDuplexVoice } from "../hooks/useHalfDuplexVoice";
import { getMe } from "../lib/me";
import {
  DEFAULT_LANGUAGE,
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
 * by name, gives a short spoken overview, and asks which industry to open. The
 * user answers BY VOICE; the LLM calls `select_industry`, which arrives as
 * `tool.called` and drives hash navigation:
 *   telco -> #/telco (billing), fsi -> #/card, healthcare -> #/hls.
 *
 * Everything routes through the FRAMEWORK: the shared voice stack
 * (startRealtimeVoice + useHalfDuplexVoice), the shared config-driven language
 * bar (lib/languages + LanguageBar), and the shared greeting mechanism. The
 * Genie ontology + deep-reasoning panels are illustrative concept visuals.
 */

type Industry = {
  id: "telco" | "fsi" | "healthcare";
  hash: string;
  title: string;
  tag: string;
  blurb: string;
  cues: string[];
};

const INDUSTRIES: Industry[] = [
  {
    id: "telco",
    hash: "#/telco",
    title: "Telco",
    tag: "Billing Support",
    blurb: "Resolve charges, waive fees, and set up payment plans on a live call.",
    cues: ["“Telco”", "“billing”", "“my phone bill”"],
  },
  {
    id: "fsi",
    hash: "#/card",
    title: "Financial Services",
    tag: "Credit-Card Assistant",
    blurb: "Understand statements and rewards, with deep “why” reasoning on demand.",
    cues: ["“Financial services”", "“credit card”", "“my statement”"],
  },
  {
    id: "healthcare",
    hash: "#/hls",
    title: "Healthcare",
    tag: "Care & Claims",
    blurb: "Explain claims, coverage, and visit summaries in plain language.",
    cues: ["“Healthcare”", "“my claim”", "“coverage”"],
  },
];

const ONTOLOGY_NODES = [
  { id: "customer", label: "Customer", x: 50, y: 18 },
  { id: "account", label: "Account", x: 18, y: 46 },
  { id: "statement", label: "Statement", x: 50, y: 50 },
  { id: "claim", label: "Claim", x: 82, y: 46 },
  { id: "charge", label: "Charge", x: 34, y: 82 },
  { id: "reward", label: "Reward", x: 66, y: 82 },
];
const ONTOLOGY_EDGES: Array<[string, string]> = [
  ["customer", "account"],
  ["customer", "statement"],
  ["customer", "claim"],
  ["account", "charge"],
  ["statement", "charge"],
  ["statement", "reward"],
];

// A concrete illustrative "why" investigation — this is what DEEP reasoning does
// that a scripted bot can't: pull governed history, compare, isolate drivers,
// and explain with evidence. (Illustrative content; the live version runs on the
// card deep-dive page.)
const REASONING_QUESTION = "“Why is my bill higher this month?”";
const REASONING_HOPS = [
  "Pulls 6 months of billing history",
  "Compares charges line-by-line vs. the usual baseline",
  "Isolates what actually changed this cycle",
];
const REASONING_CONCLUSION =
  "$41 of the $47 increase is a one-time device fee — next cycle returns to normal.";
// Last-resort route fallback if the confirmation TTS stream never reaches a
// final audio chunk. Normal routing is driven by `onFinal` after audio drains.
const ROUTE_FALLBACK_MS = 30_000;

type AgentState = "idle" | "greeting" | "listening" | "thinking" | "speaking";

export function HomePage() {
  const [phase, setPhase] = useState<"idle" | "connecting" | "live">("idle");
  const [agentState, setAgentState] = useState<AgentState>("idle");
  const [userName, setUserName] = useState<string>("");
  const [langOptions, setLangOptions] = useState<Lang[]>(DEFAULT_LANGUAGE_OPTIONS);
  const [caption, setCaption] = useState<string>("");
  const [chosen, setChosen] = useState<Industry["id"] | null>(null);
  const [micLevel, setMicLevel] = useState(0);
  const [speakLevel, setSpeakLevel] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const sessionRef = useRef<RealtimeVoiceSession | null>(null);
  const callIdRef = useRef<string>("");
  const callLanguageRef = useRef<string>(DEFAULT_LANGUAGE);
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
    interimText,
    handleInterimTranscript,
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
          onTranscript: (text) => {
            if (text.trim()) setCaption(text);
          },
          onResponseText: (text) => {
            if (text.trim()) setCaption(text);
          },
          onResponseAudio: (pcmB64, sampleRate, final) => {
            handleResponseAudio(pcmB64, sampleRate, final);
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
        // Home concierge is English-only (picker disabled). Pin STT to English
        // so short replies ("Telco") can't be mis-detected as another language
        // and dropped by the mismatch gate before the router runs.
        { profile: "concierge", startMicPaused: true, sttLanguage: DEFAULT_LANGUAGE }
      );
      sessionRef.current = session;
    },
    [
      fetchGreeting,
      speakViaTTS,
      ungateMicAfter,
      handleInterimTranscript,
      handleResponseAudio,
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

  const orbLevel = agentState === "speaking" ? speakLevel : agentState === "listening" ? micLevel : 0;
  const liveCaption = interimText.trim() || caption;

  return (
    <div className="home-root">
      <VoiceBackdrop />

      <header className="home-top">
        <BrandLockup product="Assisted Voice" />
        <div className="home-topright">
          {/* Home concierge is English-only; language selection lives on the
              use-case pages (billing / card / healthcare). Shown but disabled so
              the capability is visible without being changeable here. */}
          <LanguageBar
            value={DEFAULT_LANGUAGE}
            options={langOptions}
            onChange={() => {}}
            label=""
            disabled
          />
          <nav className="home-topnav">
            <a href="#/voice-benchmarks">Benchmarks</a>
            <a href="#/traces">Traces</a>
          </nav>
        </div>
      </header>

      <main className="home-main">
        <section className="home-hero">
          <VoiceOrb
            state={agentState}
            level={orbLevel}
            size="clamp(74px, 11vh, 112px)"
            onClick={phase === "idle" ? () => void startCall() : undefined}
            disabled={phase !== "idle"}
            ariaLabel={phase === "idle" ? "Talk to Genie" : "Genie"}
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
            {langOptions.length === 1 ? "language" : "languages"}, three industries — powered by the
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
                {agentState === "listening" && "Listening — say Telco, Financial Services, or Healthcare"}
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
              <div className="home-card-tag">{ind.tag}</div>
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
          <OntologyPanel />
          <ReasoningPanel />
        </section>
      </main>

      <footer className="home-foot">
        Powered by Databricks Genie · Unity Catalog governed · Realtime API built on OSS voice models
      </footer>
    </div>
  );
}

function OntologyPanel() {
  return (
    <div className="home-panel">
      <div className="home-panel-title">Genie Ontology</div>
      <div className="home-panel-sub">
        A governed semantic model — the entities and relationships Genie understands.
      </div>
      <svg className="home-ontology" viewBox="0 0 100 100" preserveAspectRatio="xMidYMid meet">
        {ONTOLOGY_EDGES.map(([a, b], i) => {
          const na = ONTOLOGY_NODES.find((n) => n.id === a)!;
          const nb = ONTOLOGY_NODES.find((n) => n.id === b)!;
          return (
            <line
              key={i}
              x1={na.x}
              y1={na.y}
              x2={nb.x}
              y2={nb.y}
              className="home-ontology-edge"
              style={{ animationDelay: `${i * 0.25}s` }}
            />
          );
        })}
        {ONTOLOGY_NODES.map((n) => (
          <g key={n.id} className="home-ontology-node">
            <circle cx={n.x} cy={n.y} r={3.4} />
            <text x={n.x} y={n.y - 5} textAnchor="middle">
              {n.label}
            </text>
          </g>
        ))}
      </svg>
    </div>
  );
}

function ReasoningPanel() {
  return (
    <div className="home-panel">
      <div className="home-panel-title">Genie Deep Reasoning</div>
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
