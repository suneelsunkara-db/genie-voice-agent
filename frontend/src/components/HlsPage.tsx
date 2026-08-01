import { useCallback, useEffect, useRef, useState } from "react";
import { RealtimeVoiceSession, startRealtimeVoice } from "../lib/realtimeVoice";
import { useHalfDuplexVoice } from "../hooks/useHalfDuplexVoice";
import { getMe } from "../lib/me";
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
import { getAppLanguage } from "../lib/appLanguage";
import "../styles/hls.css";

/**
 * Healthcare voice assistant — "MetroCare Health" (Tier 1).
 *
 * A real, navigable healthcare surface: the "hls" realtime profile greets the
 * member and answers questions about claims, coverage, and their last visit,
 * backed by MOCK clinical data (`/hls/summary`, shared with the voice tool).
 *
 * Uses the FRAMEWORK primitives like every other voice surface: the shared voice
 * stack (startRealtimeVoice + useHalfDuplexVoice), the shared config-driven
 * language bar, and the shared greeting mechanism. SEAM: swap `/hls/summary` +
 * the `health_summary` tool for a real Lakebase/Genie backend (mirroring the card
 * assistant) without touching this component's contract.
 */

type Claim = {
  date: string;
  provider: string;
  type: string;
  billed: number;
  plan_paid: number;
  you_owe: number;
  status: string;
};
type Summary = {
  member: { name: string; plan: string; member_id: string };
  coverage: {
    deductible: number;
    deductible_met: number;
    out_of_pocket_max: number;
    out_of_pocket_met: number;
    primary_care_copay: number;
    specialist_copay: number;
  };
  recent_claims: Claim[];
  last_visit: { date: string; provider: string; summary: string };
};

type AgentState = "idle" | "greeting" | "listening" | "thinking" | "speaking";
const money = (n: number) => `$${n.toLocaleString()}`;

export function HlsPage() {
  const [phase, setPhase] = useState<"idle" | "connecting" | "live">("idle");
  const [agentState, setAgentState] = useState<AgentState>("idle");
  const [summary, setSummary] = useState<Summary | null>(null);
  const [langOptions, setLangOptions] = useState<Lang[]>(DEFAULT_LANGUAGE_OPTIONS);
  // Seeded from the home-page choice; the picker here is locked (chosen once, on #/).
  const [callLanguage, setCallLanguage] = useState<string>(() => getAppLanguage());
  const [caption, setCaption] = useState("");
  const [micLevel, setMicLevel] = useState(0);
  const [speakLevel, setSpeakLevel] = useState(0);
  const [pulseClaims, setPulseClaims] = useState(false);
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
    interimText,
    handleInterimTranscript,
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
        const r = await fetch(`${API_BASE_URL}/hls/summary`);
        const data = (await r.json()) as Summary;
        if (active) setSummary(data);
      } catch {
        /* cards just stay empty */
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
        `${API_BASE_URL}/hls/greeting?language=${encodeURIComponent(language)}${nameQ}`
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
        "member",
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
          onToolCalled: (name) => {
            if (name === "health_summary") {
              setPulseClaims(true);
              setTimeout(() => setPulseClaims(false), 2400);
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
        { profile: "hls", startMicPaused: true }
      );
      sessionRef.current = session;
    },
    [
      fetchGreeting,
      speakViaTTS,
      ungateMicAfter,
      handleInterimTranscript,
      handleResponseAudio,
      teardown,
    ]
  );

  const startCall = useCallback(async () => {
    if (phase !== "idle") return;
    setError(null);
    setPhase("connecting");
    setAgentState("greeting");
    callIdRef.current = `hls-${Date.now()}`;
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
  }, [teardown]);

  const orbLevel = agentState === "speaking" ? speakLevel : agentState === "listening" ? micLevel : 0;
  const liveCaption = interimText.trim() || caption;
  const cov = summary?.coverage;

  return (
    <div className="hls-root">
      <VoiceBackdrop />
      <div className="gv-content">
      <header className="hls-top">
        <BrandLockup product="MetroCare Health" />
        <button type="button" className="hls-back" onClick={() => (window.location.hash = "#/")}>
          ← Home
        </button>
      </header>

      <div className="hls-body">
        {/* Voice rail */}
        <aside className="hls-rail">
          <div className="hls-orb">
            <VoiceOrb state={agentState} level={orbLevel} size="120px" ariaLabel="Genie" />
          </div>
          <div className={`hls-status hls-status-${agentState}`}>
            {phase === "idle" && "Ready when you are"}
            {agentState === "greeting" && "Connecting…"}
            {agentState === "listening" && "Listening…"}
            {agentState === "thinking" && "Looking that up…"}
            {agentState === "speaking" && "Genie is speaking…"}
          </div>
          {liveCaption && phase !== "idle" && <p className="hls-caption">{liveCaption}</p>}
          <div className="hls-langwrap">
            <LanguageBar
              value={callLanguage}
              options={langOptions}
              onChange={(c) => void changeLanguage(c)}
              disabled
            />
          </div>
          {phase === "idle" ? (
            <button type="button" className="hls-cta" onClick={() => void startCall()}>
              Talk to Genie
            </button>
          ) : (
            <button type="button" className="hls-end" onClick={endCall}>
              End call
            </button>
          )}
          {error && <p className="hls-error">{error}</p>}
          <p className="hls-disclaimer">
            Genie explains claims and coverage — it is not a doctor and does not give
            medical advice.
          </p>
        </aside>

        {/* Clinical canvas */}
        <main className="hls-canvas">
          <section className="hls-member">
            <div>
              <div className="hls-member-name">{summary?.member.name ?? "Member"}</div>
              <div className="hls-member-plan">{summary?.member.plan ?? "—"}</div>
            </div>
            <div className="hls-member-id">{summary?.member.member_id ?? ""}</div>
          </section>

          <section className="hls-grid">
            <div className="hls-tile">
              <div className="hls-tile-title">Deductible</div>
              {cov && (
                <>
                  <div className="hls-bar">
                    <span style={{ width: `${(cov.deductible_met / cov.deductible) * 100}%` }} />
                  </div>
                  <div className="hls-tile-sub">
                    {money(cov.deductible_met)} of {money(cov.deductible)} met
                  </div>
                </>
              )}
            </div>
            <div className="hls-tile">
              <div className="hls-tile-title">Out-of-pocket</div>
              {cov && (
                <>
                  <div className="hls-bar">
                    <span
                      style={{ width: `${(cov.out_of_pocket_met / cov.out_of_pocket_max) * 100}%` }}
                    />
                  </div>
                  <div className="hls-tile-sub">
                    {money(cov.out_of_pocket_met)} of {money(cov.out_of_pocket_max)} met
                  </div>
                </>
              )}
            </div>
            <div className="hls-tile">
              <div className="hls-tile-title">Copays</div>
              {cov && (
                <div className="hls-copays">
                  <div>
                    <strong>{money(cov.primary_care_copay)}</strong> primary care
                  </div>
                  <div>
                    <strong>{money(cov.specialist_copay)}</strong> specialist
                  </div>
                </div>
              )}
            </div>
          </section>

          <section className={`hls-claims${pulseClaims ? " is-pulse" : ""}`}>
            <div className="hls-section-title">Recent claims</div>
            <table>
              <thead>
                <tr>
                  <th>Date</th>
                  <th>Provider</th>
                  <th>Type</th>
                  <th>Plan paid</th>
                  <th>You owe</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {(summary?.recent_claims ?? []).map((c, i) => (
                  <tr key={i}>
                    <td>{c.date}</td>
                    <td>{c.provider}</td>
                    <td>{c.type}</td>
                    <td>{money(c.plan_paid)}</td>
                    <td className="hls-owe">{money(c.you_owe)}</td>
                    <td>
                      <span className={`hls-pill hls-pill-${c.status.toLowerCase()}`}>{c.status}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>

          {summary?.last_visit && (
            <section className="hls-visit">
              <div className="hls-section-title">Last visit</div>
              <div className="hls-visit-meta">
                {summary.last_visit.date} · {summary.last_visit.provider}
              </div>
              <p>{summary.last_visit.summary}</p>
            </section>
          )}
        </main>
      </div>
      </div>
    </div>
  );
}
