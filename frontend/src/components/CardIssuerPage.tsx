import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  DeepDiveReport,
  RealtimeVoiceSession,
  startRealtimeVoice,
  streamDeepDive,
} from "../lib/realtimeVoice";
import { useHalfDuplexVoice } from "../hooks/useHalfDuplexVoice";
import { getMe, useMe } from "../lib/me";
import { BrandLockup } from "./BrandLockup";
import { VoiceBackdrop } from "./VoiceBackdrop";
import { VoiceOrb } from "./VoiceOrb";
import {
  colorFor,
  computeRewardsWaterfall,
  computeSpikeWaterfall,
  cycleTotals,
  num,
  RewardRecord,
  SpendingRecord,
  WaterfallStep,
} from "../lib/cardAnalytics";
import { API_BASE_URL, WS_BASE_URL } from "../config";
import { getAppLanguage } from "../lib/appLanguage";
import { languageLabel, uiCopy, useUiLocale, type UiCopy } from "../i18n";
import "../styles/card.css";

/**
 * Agent-initiated, voice-first credit-card assistant ("Genie Agent").
 *
 * Layout is a CONVERGED cockpit: a compact voice rail (left) carries the human
 * presence — orb, status, transcript — while a large analytics canvas (right)
 * is where the two Genie lanes become visible:
 *   - FAST lane  (Conversation API): the Customer-360 strip, the 12-month expense
 *     trend, and the fast-fact ticker populate the instant the call connects.
 *   - DEEP lane  (Agent Mode): the "why" investigation runs a Plan → Query →
 *     Compute → Explain pipeline with streamed reasoning + SQL, climaxing in a
 *     spike / rewards-leakage waterfall computed from the cardholder's own data.
 *
 * It consumes the three working APIs WITHOUT changing them (realtime voice, Genie
 * Conversation, Genie Agent Mode). The opening greeting + the deep-dive summary
 * are spoken by the TTS API alone (no server-first turn needed).
 */

const AGENT_NAME = "Genie Agent";
// Shown while the DEEP lane (Genie Agent Mode) is actively investigating, to make
// the shift from the fast conversational agent to the deep reasoning agent visible.
const DEEP_AGENT_NAME = "Genie Deep Reasoning Agent";
// The client stall-watchdog is single-sourced from the SERVER: the deep-dive SSE
// leads with a `meta` event carrying the server's read timeout, and we wait that
// + a small buffer so the client can never give up before the server does. This
// fallback only applies until that meta arrives (it should be the first event);
// it is set above the server default (420s) for the same reason.
const DEEP_DIVE_FALLBACK_TIMEOUT_MS = 435_000;
const DEEP_DIVE_WATCHDOG_BUFFER_MS = 15_000;
// How long to hold the stream open after the report for the translated text. A
// measured Hindi swap landed 20s after the spoken summary, so this is a leak guard
// with margin, not a budget: if it never lands, the English report simply stays.
const DEEP_DIVE_LOCALIZE_GRACE_MS = 90_000;
const BRAND = "EveryCard";

// The caller picks the call language. We pin it as the EXPECTED language (same
// mechanism billing support uses) so STT auto-detect flipping to another language
// mid-call is GATED — the off-language turn is dropped and the agent re-prompts, instead
// of the reply silently switching languages. Changing the picker restarts the
// session in the new language (session config is immutable after session.start).
type Lang = { code: string; label: string };
const DEFAULT_LANGUAGE = "en-US";
// The picker offers ONLY the config-driven supported set fetched from the backend
// (GET /card/languages — Qwen3-ASR ∩ VoxCPM2, ~24, the same catalog the billing
// cockpit uses). Before that lands we show just the guaranteed baseline (English)
// rather than a divergent hardcoded list that could offer an unsupported language.
// Native labels come from Intl.DisplayNames, so there is no hand-maintained names.
const DEFAULT_LANGUAGE_OPTIONS: Lang[] = [{ code: DEFAULT_LANGUAGE, label: "English" }];

function langBase(code: string | null | undefined): string {
  // Primary ISO-639 subtag (matches the backend's split("-")[0]). NOT a 2-char
  // slice — that collapses 3-letter bases like "fil"→"fi" (Filipino→Finnish) and
  // "yue"→"yu", which both round-trip in the supported set.
  return (code || "").split("-", 1)[0].toLowerCase();
}
// Native endonym for the picker (e.g. "hi-IN" -> "हिन्दी"), via Intl.DisplayNames.
function langLabel(code: string | null | undefined): string {
  if (!code) return "English";
  return languageLabel(code, code);
}
// Flag emoji from the BCP-47 region subtag ("en-US" -> 🇺🇸); a globe when none.
function flagFor(code: string | null | undefined): string {
  const region = (code || "").split("-").find((p, i) => i > 0 && /^[A-Za-z]{2}$/.test(p));
  if (!region) return "🌐";
  return region
    .toUpperCase()
    .replace(/./g, (c) => String.fromCodePoint(127397 + c.charCodeAt(0)));
}
// Map a detected language ("hi", "hi-IN") to a supported picker code, if any.
function supportedFromDetected(
  detected: string | null | undefined,
  options: Lang[],
): string | null {
  const base = langBase(detected);
  return options.find((l) => langBase(l.code) === base)?.code ?? null;
}

type Persona = { customerId: string; name: string; blurb: string };

const PERSONAS: Persona[] = [
  { customerId: "CH-0001", name: "Suneel Sunkara", blurb: "Primary cardholder" },
];

const USE_CASES: Record<string, { label: string; tagline: string; icon: string }> = {
  statement_insights: {
    label: "Statement Insights",
    tagline: "Why your expenses changed this cycle.",
    icon: "▲",
  },
  rewards_optimizer: {
    label: "Rewards Optimizer",
    tagline: "Points being left on the table.",
    icon: "★",
  },
};

// The 4-stage deep-dive pipeline (mirrors how Genie Agent Mode actually works).
const pipelineStages = (copy: UiCopy): string[] => [
  copy.deepStagePlan,
  copy.deepStageQuery,
  copy.deepStageCompute,
  copy.deepStageExplain,
];

type CardProfile = {
  cardholder: Record<string, unknown> | null;
  recent_statements: Record<string, unknown>[];
  spending_by_category: SpendingRecord[];
  rewards_ledger: RewardRecord[];
  summary: Record<string, unknown>;
  found: boolean;
};

type GenieFact = { id: number; question: string | null; answer: string; rows?: unknown[][]; columns?: string[] };

type Investigation = {
  id: string;
  /** What Genie was asked (English, built by the LLM so the SQL planning is sound). */
  question: string;
  /** What the CALLER said, in their own words — this is what the panel shows. */
  spokenQuestion: string | null;
  useCase: string | null;
  status: "running" | "done" | "error";
  startedAt: number;
  elapsed: number;
  steps: { kind: "reasoning" | "sql"; text: string; key: number }[];
  report?: DeepDiveReport;
  errorMessage?: string;
};

type Turn = { role: "agent" | "customer"; text: string; turnId: number; key: number };
type Phase = "idle" | "connecting" | "live";
type AgentState = "greeting" | "listening" | "thinking" | "speaking";

function money(n: unknown): string {
  const v = Number(n);
  if (!Number.isFinite(v)) return "—";
  return v.toLocaleString(undefined, { style: "currency", currency: "USD", maximumFractionDigits: 0 });
}

function pts(n: unknown): string {
  return num(n).toLocaleString();
}

function mmss(total: number): string {
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m}:${String(s).padStart(2, "0")}`;
}

// ===========================================================================
// Main component
// ===========================================================================
export function CardIssuerPage() {
  // Agent-initiated: land straight in the live call (no intro hero) and connect,
  // so arriving from the Home concierge drops the caller directly into the call.
  const [phase, setPhase] = useState<Phase>("connecting");
  const [agentState, setAgentState] = useState<AgentState>("greeting");
  const [persona] = useState<Persona>(PERSONAS[0]);
  // The signed-in Databricks user — shown on screen and spoken by name so the
  // whole call is addressed to the real caller (demo persona is the fallback).
  const me = useMe();
  const displayPersona = useMemo<Persona>(
    () => (me.name ? { ...persona, name: me.name } : persona),
    [me.name, persona]
  );
  const [turns, setTurns] = useState<Turn[]>([]);
  const [selectedUseCase, setSelectedUseCase] = useState<string | null>(null);
  const [facts, setFacts] = useState<Record<string, unknown> | null>(null);
  const [genieFacts, setGenieFacts] = useState<GenieFact[]>([]);
  const [micLevel, setMicLevel] = useState(0);
  const [speakLevel, setSpeakLevel] = useState(0);
  const [callSeconds, setCallSeconds] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [profile, setProfile] = useState<CardProfile | null>(null);
  const [investigations, setInvestigations] = useState<Map<string, Investigation>>(new Map());
  const [langMismatch, setLangMismatch] = useState<{ expected: string; detected: string } | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  // Seeded from the home-page choice so the greeting opens in the right language;
  // the picker here is locked (language is chosen once, on #/).
  const [callLanguage, setCallLanguage] = useState<string>(() => getAppLanguage());
  // Full end-to-end supported set from the backend (config-driven, ~24), not a
  // hardcoded subset. Shows only the English baseline until the fetch resolves.
  const [langOptions, setLangOptions] = useState<Lang[]>(DEFAULT_LANGUAGE_OPTIONS);
  // This page owns its own call language (App's cockpit language is separate), so
  // it loads its own locale bundle; every uiCopy(callLanguage) below then renders
  // translated chrome once it arrives.
  useUiLocale(callLanguage);

  const sessionRef = useRef<RealtimeVoiceSession | null>(null);
  const turnKeyRef = useRef(0);
  const factKeyRef = useRef(0);
  // Every in-flight deep dive registers its cancel fn here so teardown / a
  // language restart can stop them all (a single ref would leak concurrent dives).
  const deepDiveCancelsRef = useRef<Set<() => void>>(new Set());
  const transcriptEndRef = useRef<HTMLDivElement>(null);
  // Stable call id (so a mid-call language restart keeps the same traced call) and
  // the live call language, both read from inside session callbacks.
  const callIdRef = useRef<string | null>(null);
  const callLanguageRef = useRef<string>(getAppLanguage());
  // Count of in-flight Agent-Mode deep dives. While > 0 the mic stays CLOSED for
  // the whole investigation window (tens of seconds) so speaker audio / ambient
  // noise can't create phantom turns that surface as "random" TTS afterward.
  const runningDeepDivesRef = useRef(0);
  // The caller's most recent sentence, read from inside session callbacks so a deep
  // dive can echo what they actually asked.
  const lastCustomerTextRef = useRef<string>("");
  // Opening greeting text, generated in-language by the backend (no hardcoded
  // per-language table). Cached per base-language so a mid-call language restart
  // reuses it and the open is never blocked on a second round-trip.
  const greetingCacheRef = useRef<Map<string, string>>(new Map());

  // Shared half-duplex plumbing (playback queue + mic gating), identical to the
  // billing cockpit. The mic stays closed for the whole Agent-Mode window and the
  // orb/state flip to speaking/listening via these callbacks.
  const {
    playbackRef,
    micGatedRef,
    resetPlayback,
    gateMic,
    ungateMicAfter,
    handleResponseAudio,
    interimText,
    handleInterimTranscript,
    interrupt,
    switchLanguage,
    teardownPlayback,
  } = useHalfDuplexVoice({
    sessionRef,
    shouldStayGated: () => runningDeepDivesRef.current > 0,
    onMicResume: () => setAgentState("listening"),
    onSpeaking: () => setAgentState("speaking"),
    callLanguageRef,
    isCallLive: () => phase !== "idle",
    closeSession: () => {
      // Abandon in-flight Agent-Mode runs: their reports would arrive in the
      // language we just left, and their cancels belong to the old session.
      deepDiveCancelsRef.current.forEach((c) => c());
      deepDiveCancelsRef.current.clear();
      runningDeepDivesRef.current = 0;
      sessionRef.current?.close();
      sessionRef.current = null;
      setSessionId(null);
      setAgentState("greeting");
      setPhase("connecting");
    },
    reopenSession: (nextLanguage) => openSession(nextLanguage),
    // The transcript only. The 360 canvas is data with catalog-localized labels,
    // so it re-renders in the new language on its own; investigations already
    // completed are left alone rather than silently deleted.
    clearConversation: () => setTurns([]),
  });

  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [turns]);

  // Load the FULL supported language set (config-driven, ~24) so the picker isn't
  // limited to a hardcoded handful. Native labels via Intl.DisplayNames.
  useEffect(() => {
    let cancelled = false;
    fetch(`${API_BASE_URL}/card/languages`)
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => {
        const raw = data?.options as Array<{ code: string }> | undefined;
        if (cancelled || !raw?.length) return;
        setLangOptions(raw.map((o) => ({ code: o.code, label: langLabel(o.code) })));
      })
      .catch(() => {
        /* keep the English-only baseline */
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (phase !== "live") return;
    const id = window.setInterval(() => setCallSeconds((s) => s + 1), 1000);
    return () => window.clearInterval(id);
  }, [phase]);

  // Drive the orb from the ACTUAL TTS waveform while the agent speaks.
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
  }, [agentState]);

  const teardown = useCallback(() => {
    deepDiveCancelsRef.current.forEach((c) => c());
    deepDiveCancelsRef.current.clear();
    runningDeepDivesRef.current = 0;
    sessionRef.current?.close();
    sessionRef.current = null;
    // Stops scheduled buffers immediately (no lingering audio after End) + clears
    // the mic-resume timer and releases the audio context.
    teardownPlayback();
  }, [teardownPlayback]);

  // Speak agent text (greeting / deep-dive summary) through the SAME voice
  // session via a synthesize turn, so it shares the session's locked voice
  // reference — the caller hears ONE consistent voice for the whole call rather
  // than a separate TTS-socket timbre. Audio returns as response.audio events,
  // which handleResponseAudio plays and half-duplex mic-gates. We flush first so
  // any stray/echo PCM can't play right before the clean line.
  const speakViaTTS = useCallback((text: string) => {
    const session = sessionRef.current;
    if (!session || !text.trim()) return;
    playbackRef.current?.flush();
    gateMic();
    setAgentState("speaking");
    session.synthesize(text, callLanguageRef.current);
  }, [gateMic, playbackRef]);

  // Fetch the opening greeting generated in `language` by the backend (cached per
  // base-language). Returns "" if serving is unavailable — the caller then just
  // opens the mic rather than speaking a fake English line.
  const fetchGreeting = useCallback(async (language: string): Promise<string> => {
    const key = langBase(language);
    const cached = greetingCacheRef.current.get(key);
    if (cached !== undefined) return cached;
    // Greet the signed-in Databricks user by name; fall back to the demo persona
    // only when there's no forwarded identity (e.g. anonymous local dev).
    const me = await getMe();
    const first = me.name || persona.name.split(" ")[0];
    try {
      const r = await fetch(
        `${API_BASE_URL}/card/greeting?language=${encodeURIComponent(language)}&name=${encodeURIComponent(first)}`
      );
      const data = (await r.json()) as { text?: string };
      const text = typeof data.text === "string" ? data.text : "";
      greetingCacheRef.current.set(key, text);
      return text;
    } catch {
      return "";
    }
  }, [persona]);


  const runDeepDive = useCallback(
    (question: string, useCase: string | null) => {
      const id = `inv-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
      let stepKey = 0;
      const newInv: Investigation = {
        id, question, useCase,
        // Echo the caller's own sentence rather than the English question the LLM
        // wrote for Genie, which on a non-English call is the one line in the panel
        // they didn't say and can't read.
        spokenQuestion: lastCustomerTextRef.current || null,
        status: "running", startedAt: Date.now(), elapsed: 0, steps: [],
      };
      setInvestigations((prev) => new Map(prev).set(id, newInv));
      // Keep the mic CLOSED for the whole Agent-Mode run (can be tens of seconds).
      runningDeepDivesRef.current += 1;
      gateMic();

      // Each dive settles EXACTLY once (report | error | done-without-report |
      // watchdog timeout). Without this a stalled Agent-Mode stream — heartbeats
      // keep the EventSource open but no `report` ever lands — left the box spinning
      // forever and the mic gated, so follow-up questions "got stuck".
      let settled = false;
      let cancel = () => {};
      let watchdog = 0;
      const closeStream = () => {
        window.clearTimeout(watchdog);
        cancel();
        deepDiveCancelsRef.current.delete(cancel);
      };
      // `keepStream` settles the VOICE while leaving the stream open for the
      // translated report, which lands a few seconds after the spoken summary.
      const settle = (spoken: string | null, keepStream = false) => {
        if (settled) return;
        settled = true;
        if (keepStream) {
          // Nothing else can rescue this dive now, so bound the extra wait rather
          // than leaking an EventSource if the swap event never comes.
          window.clearTimeout(watchdog);
          watchdog = window.setTimeout(closeStream, DEEP_DIVE_LOCALIZE_GRACE_MS);
        } else {
          closeStream();
        }
        runningDeepDivesRef.current = Math.max(0, runningDeepDivesRef.current - 1);
        // speakViaTTS re-opens the mic when it finishes (and only if no other deep
        // dive is still running); if there's nothing to speak, re-open directly.
        if (spoken) void speakViaTTS(spoken);
        else ungateMicAfter(300);
      };
      const failInv = (message: string) => {
        setInvestigations((prev) => {
          const m = new Map(prev);
          const inv = m.get(id);
          if (inv && inv.status === "running") {
            m.set(id, { ...inv, status: "error", errorMessage: message, elapsed: Date.now() - inv.startedAt });
          }
          return m;
        });
      };

      const armWatchdog = (ms: number) => {
        window.clearTimeout(watchdog);
        watchdog = window.setTimeout(() => {
          if (settled) return;
          failInv("This is taking longer than usual — please ask the question again.");
          settle(null);
        }, ms);
      };

      cancel = streamDeepDive(API_BASE_URL, question, useCase, {
        // Re-arm the watchdog off the server's single-source timeout (+ buffer)
        // so client and server can't disagree on when a run has stalled.
        onMeta: (timeoutMs) => armWatchdog(timeoutMs + DEEP_DIVE_WATCHDOG_BUFFER_MS),
        onStep: (step, text) => {
          if (settled || !text.trim()) return;
          stepKey += 1;
          const key = stepKey;
          setInvestigations((prev) => {
            const m = new Map(prev);
            const inv = m.get(id);
            if (inv && inv.status === "running") {
              m.set(id, { ...inv, steps: [...inv.steps, { kind: step, text, key }], elapsed: Date.now() - inv.startedAt });
            }
            return m;
          });
        },
        onReport: (report) => {
          if (settled) return;
          setInvestigations((prev) => {
            const m = new Map(prev);
            const inv = m.get(id);
            if (inv) m.set(id, { ...inv, status: "done", report, elapsed: Date.now() - inv.startedAt });
            return m;
          });
          // The spoken "why" is the backend LLM summary ONLY — it names the cause
          // in the caller's language. If it's unavailable we stay silent (the full
          // report is on screen) rather than speak an English-only heuristic.
          settle(report.spokenSummary || null, report.localizationPending);
        },
        // Streamed translation chunks: append in order so the localized report
        // paints progressively in the panel (which opened empty — no English
        // flash). Each chunk is activity, so push back the grace watchdog.
        onReportLocalizedDelta: (delta) => {
          if (!delta) return;
          window.clearTimeout(watchdog);
          watchdog = window.setTimeout(closeStream, DEEP_DIVE_LOCALIZE_GRACE_MS);
          setInvestigations((prev) => {
            const m = new Map(prev);
            const inv = m.get(id);
            if (inv?.report) {
              m.set(id, {
                ...inv,
                report: { ...inv.report, report: inv.report.report + delta },
              });
            }
            return m;
          });
        },
        // The authoritative full text in the caller's language: replace whatever
        // the deltas built (correcting any loss) and clear the pending flag.
        onReportLocalized: (text) => {
          if (text.trim()) {
            setInvestigations((prev) => {
              const m = new Map(prev);
              const inv = m.get(id);
              if (inv?.report) {
                m.set(id, {
                  ...inv,
                  report: { ...inv.report, report: text, localizationPending: false },
                });
              }
              return m;
            });
          }
          closeStream();
        },
        onError: (message) => {
          if (settled) return;
          failInv(message);
          settle(null);
        },
        // A `done` with no preceding `report` means the stream ended empty — resolve
        // the box instead of leaving it running.
        onDone: () => {
          if (settled) return;
          failInv("The investigation ended without a result. Please ask again.");
          settle(null);
        },
      }, {
        callId: callIdRef.current,
        sessionId: sessionRef.current?.sessionId ?? null,
        customerId: persona.customerId,
        language: callLanguageRef.current,
      });
      deepDiveCancelsRef.current.add(cancel);

      // Fallback watchdog until the server's `meta` timeout arrives (first event);
      // onMeta re-arms it to the server value + buffer.
      armWatchdog(DEEP_DIVE_FALLBACK_TIMEOUT_MS);
    },
    [persona, speakViaTTS, gateMic, ungateMicAfter]
  );

  useEffect(() => () => teardown(), [teardown]);

  const endCall = useCallback(() => {
    teardown();
    setPhase("idle");
    setAgentState("greeting");
    setMicLevel(0);
    setCallSeconds(0);
  }, [teardown]);

  const pushTurn = useCallback((role: "agent" | "customer", text: string, turnId: number) => {
    turnKeyRef.current += 1;
    setTurns((prev) => [...prev, { role, text, turnId, key: turnKeyRef.current }]);
  }, []);

  // Open (or re-open) the realtime voice session in `language`. Shared by the
  // initial connect and the mid-call language switch; reuses callIdRef so a
  // restart stays the SAME traced call.
  const openSession = useCallback(async (language: string) => {
    callLanguageRef.current = language;
    if (!callIdRef.current) callIdRef.current = `card-${persona.customerId}-${Date.now()}`;
    micGatedRef.current = false;
    resetPlayback();

    const session = await startRealtimeVoice(
      WS_BASE_URL,
      callIdRef.current,
      persona.customerId,
      {
        onSessionReady: (sid) => {
          setSessionId(sid ?? null);
          setPhase("live");
          setAgentState("greeting");
          void (async () => {
            const greeting = await fetchGreeting(language);
            if (greeting) {
              pushTurn("agent", greeting, 0);
              speakViaTTS(greeting);
            } else {
              // No greeting available (serving down) — open the mic so the caller
              // can still speak, instead of hanging on a silent "greeting" state.
              ungateMicAfter(0);
              setAgentState("listening");
            }
          })();
        },
        onLevel: (level) => setMicLevel(level),
        // Live on-device caption (framework): words appear as the caller speaks;
        // the shared hook holds the text and clears it when transcript.final lands.
        onInterimTranscript: (text) => handleInterimTranscript(text),
        onSpeechStarted: () => setAgentState("listening"),
        onTurnStarted: () => setAgentState((s) => (s === "speaking" ? s : "thinking")),
        onTranscript: (text, _language, turnId) => {
          // A valid (on-language) transcript arrived — clear any prior warning.
          setLangMismatch(null);
          if (text.trim()) {
            lastCustomerTextRef.current = text.trim();
            pushTurn("customer", text, turnId);
          }
        },
        onResponseText: (text, turnId) => {
          if (text.trim()) pushTurn("agent", text, turnId);
        },
        // Language gate fired: STT heard a different language than the call's.
        // The backend dropped the turn and re-prompts — surface a banner + switch.
        onLanguageMismatch: (expected, detected) => {
          setLangMismatch({ expected, detected });
        },
        onResponseAudio: (pcmB64, sampleRate, final) => {
          // Half-duplex playback + mic gating (shared with billing): gate once for
          // the response, enqueue, and schedule mic resume when the final chunk lands.
          handleResponseAudio(pcmB64, sampleRate, final);
        },
        onToolCalled: (name, result) => {
          const r = result && typeof result === "object" ? (result as Record<string, unknown>) : {};
          if (name === "card_account_facts" && r.found) {
            setFacts(r);
          } else if (name === "select_use_case" && typeof r.use_case === "string") {
            setSelectedUseCase(r.use_case);
          } else if (name === "ask_card_genie" && typeof r.answer === "string") {
            factKeyRef.current += 1;
            const id = factKeyRef.current;
            setGenieFacts((prev) => [
              { id, question: null, answer: r.answer as string, rows: r.rows as unknown[][], columns: r.columns as string[] },
              ...prev,
            ].slice(0, 6));
          } else if (name === "start_deep_dive" && typeof r.question === "string") {
            runDeepDive(r.question, (r.use_case as string) ?? null);
          }
        },
        onError: (code, message) => {
          if (code === "ws_closed") {
            setError(message);
            endCall();
          } else {
            setError(`${code}: ${message}`);
          }
        },
      },
      language,
      { profile: "card", startMicPaused: true }
    );
    sessionRef.current = session;
  }, [persona, endCall, pushTurn, speakViaTTS, runDeepDive, gateMic, ungateMicAfter, fetchGreeting, handleInterimTranscript]);

  const startCall = useCallback(async () => {
    setError(null);
    setTurns([]);
    setSelectedUseCase(null);
    setFacts(null);
    setGenieFacts([]);
    setInvestigations(new Map());
    setLangMismatch(null);
    setSessionId(null);
    setCallSeconds(0);
    setAgentState("greeting");
    setPhase("connecting");
    runningDeepDivesRef.current = 0;
    callIdRef.current = `card-${persona.customerId}-${Date.now()}`;

    // Fast lane: fetch the 360 profile immediately so the canvas is never blank.
    fetch(`${API_BASE_URL}/card/profile/${persona.customerId}`)
      .then((r) => r.json())
      .then((data: CardProfile) => setProfile(data))
      .catch(() => { /* graphs will just not render */ });

    // Prefetch the in-language greeting in parallel so it's cached before the WS
    // session is ready — the agent speaks immediately instead of after a round-trip.
    void fetchGreeting(callLanguageRef.current);

    try {
      await openSession(callLanguageRef.current);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      teardown();
      setPhase("idle");
    }
  }, [persona, openSession, teardown, fetchGreeting]);

  // Auto-connect on mount so the call begins the moment the page opens (the
  // browser gesture that arrived with us from Home unlocks audio; a cold refresh
  // falls back to tapping the orb). The cleared timeout keeps React StrictMode's
  // dev double-mount from opening two sessions.
  useEffect(() => {
    const t = window.setTimeout(() => void startCall(), 0);
    return () => window.clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Hang up → back to the Home concierge (where the call was initiated).
  const hangUp = useCallback(() => {
    teardown();
    window.location.hash = "#/";
  }, [teardown]);

  // Tapping the Genie orb: interrupt while the agent speaks; otherwise (re)start
  // the call, or resume audio if the browser suspended it on a cold-load auto-start.
  const onOrbTap = useCallback(() => {
    if (agentState === "speaking") {
      interrupt();
      return;
    }
    if (phase !== "live") {
      void startCall();
      return;
    }
    playbackRef.current?.resume();
  }, [agentState, phase, interrupt, startCall, playbackRef]);

  // Change the call language. Pre-call: just record it. Mid-call: restart the
  // voice session in the new language (session config is immutable after start),
  // keeping the transcript / profile / investigations already on screen.
  const changeLanguage = useCallback(async (code: string) => {
    if (code === callLanguageRef.current) return;
    setCallLanguage(code);
    setLangMismatch(null);
    // The framework owns the mic gating, teardown and reopen ordering (shared with
    // the billing cockpit) so both pages move a live call the same way.
    try {
      await switchLanguage(code);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setPhase("idle");
    }
  }, [switchLanguage]);

  return (
    <div className="card-page">
      <VoiceBackdrop />
      <div className="gv-content">
      <button className="card-back" onClick={() => (window.location.hash = "#/")}>
        ← Cockpit
      </button>

      {error && <div className="card-error">{error}</div>}

      <LiveCall
        persona={displayPersona}
        phase={phase}
        agentState={agentState}
        micLevel={micLevel}
        speakLevel={speakLevel}
        callSeconds={callSeconds}
        turns={turns}
        interimText={interimText}
        selectedUseCase={selectedUseCase}
        facts={facts}
        genieFacts={genieFacts}
        profile={profile}
        investigations={investigations}
        langMismatch={langMismatch}
        sessionId={sessionId}
        callLanguage={callLanguage}
        langOptions={langOptions}
        onLanguage={(c) => void changeLanguage(c)}
        onInterrupt={interrupt}
        onOrbTap={onOrbTap}
        onEnd={hangUp}
        transcriptEndRef={transcriptEndRef}
      />
      </div>
    </div>
  );
}

/* Language picker — pre-call sets the language; in-call it restarts the session. */
function LanguagePicker({
  value, options, onChange, label, disabled,
}: {
  value: string;
  options: Lang[];
  onChange: (code: string) => void;
  label?: string;
  disabled?: boolean;
}) {
  return (
    <label className="card-langpick">
      {label && (
        <span className="card-langpick-lbl">
          {label} · {options.length} languages
        </span>
      )}
      <select
        className="card-langpick-select"
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(e.target.value)}
      >
        {options.map((l) => (
          <option key={l.code} value={l.code}>
            {flagFor(l.code)} {l.label}
          </option>
        ))}
      </select>
    </label>
  );
}

/* --------------------------------------------------------------------------- */
/* Live call: voice rail (left) + analytics canvas (right)                     */
/* --------------------------------------------------------------------------- */
function LiveCall({
  persona, phase, agentState, micLevel, speakLevel, callSeconds,
  turns, interimText, selectedUseCase, facts, genieFacts, profile, investigations,
  langMismatch, sessionId, callLanguage, langOptions, onLanguage, onInterrupt,
  onOrbTap, onEnd, transcriptEndRef,
}: {
  persona: Persona;
  phase: Phase;
  agentState: AgentState;
  micLevel: number;
  speakLevel: number;
  callSeconds: number;
  turns: Turn[];
  interimText: string;
  selectedUseCase: string | null;
  facts: Record<string, unknown> | null;
  genieFacts: GenieFact[];
  profile: CardProfile | null;
  investigations: Map<string, Investigation>;
  langMismatch: { expected: string; detected: string } | null;
  sessionId: string | null;
  callLanguage: string;
  langOptions: Lang[];
  onLanguage: (code: string) => void;
  onInterrupt: () => void;
  onOrbTap: () => void;
  onEnd: () => void;
  transcriptEndRef: React.RefObject<HTMLDivElement>;
}) {
  // Barge-in via Escape while the agent is speaking.
  useEffect(() => {
    if (agentState !== "speaking") return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onInterrupt();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [agentState, onInterrupt]);
  const runningCount = [...investigations.values()].filter((i) => i.status === "running").length;
  const connecting = phase === "connecting";
  const statusText = connecting
    ? `Connecting you to ${AGENT_NAME}…`
    : agentState === "speaking"
      ? `${AGENT_NAME} is speaking`
      : runningCount > 0
        ? `${DEEP_AGENT_NAME} is analyzing…`
        : agentState === "thinking"
          ? `${AGENT_NAME} is thinking…`
          : agentState === "greeting"
            ? `${AGENT_NAME} is joining…`
            : "Listening…";

  return (
    <div className="card-live">
      {/* ---- Voice rail ---- */}
      <aside className="card-rail">
        <div style={{ marginBottom: 12, display: "flex", justifyContent: "center" }}>
          <BrandLockup />
        </div>
        <div className="card-callmeta">
          <div className="card-callwho">
            <span className="card-avatar-mini">{AGENT_NAME[0]}</span>
            <div>
              <div className="card-callwho-name">{AGENT_NAME}</div>
              <div className="card-callwho-role">{BRAND} assistant</div>
            </div>
          </div>
          <div className="card-calltimer">{connecting ? "•••" : mmss(callSeconds)}</div>
          <button className="card-end" onClick={onEnd}>End</button>
        </div>

        <div className="card-railtools">
          <LanguagePicker value={callLanguage} options={langOptions} onChange={onLanguage} disabled />
        </div>

        <div className="card-orbwrap">
          <button
            type="button"
            className="card-orbbtn"
            onClick={onOrbTap}
            aria-label={
              agentState === "speaking"
                ? `Interrupt ${AGENT_NAME}`
                : phase !== "live"
                  ? `Call ${AGENT_NAME}`
                  : "Assistant"
            }
          >
            <Avatar
              size={148}
              state={connecting ? "greeting" : agentState}
              level={agentState === "speaking" ? speakLevel : agentState === "listening" ? micLevel : 0}
              ring
            />
          </button>
          <div className={`card-status card-status-${connecting ? "greeting" : agentState}`}>
            {agentState === "thinking" && !connecting ? (
              <span className="card-dots">{statusText}<i /><i /><i /></span>
            ) : statusText}
          </div>
          {agentState === "speaking" ? (
            <button className="card-interrupt" onClick={onInterrupt}>
              Tap to interrupt <span className="card-interrupt-key">Esc</span>
            </button>
          ) : (
            <div className="card-callee">Call with {persona.name.split(" ")[0]}</div>
          )}
        </div>

        {runningCount > 0 && (
          <div className="card-agents-badge card-fade-in">
            <span className="card-agents-badge-dot" />
            <span className="card-agents-badge-text">
              {runningCount === 1
                ? `${DEEP_AGENT_NAME} working`
                : `${runningCount} ${DEEP_AGENT_NAME}s working`}
            </span>
          </div>
        )}

        {langMismatch && (
          <div className="card-langwarn card-fade-in">
            <div>
              Heard <b>{langLabel(langMismatch.detected)}</b>, but this call is in{" "}
              <b>{langLabel(langMismatch.expected)}</b>. {AGENT_NAME} kept going in{" "}
              {langLabel(langMismatch.expected)}.
            </div>
            {supportedFromDetected(langMismatch.detected, langOptions) && (
              <button
                className="card-langwarn-switch"
                onClick={() =>
                  onLanguage(supportedFromDetected(langMismatch.detected, langOptions) as string)
                }
              >
                Switch to {langLabel(langMismatch.detected)}
              </button>
            )}
          </div>
        )}

        <div className="card-usecases">
          {Object.entries(USE_CASES).map(([key, uc]) => (
            <div key={key} className={`card-usecase ${selectedUseCase === key ? "is-selected" : ""}`}>
              <span className="card-usecase-icon">{uc.icon}</span>
              <div>
                <div className="card-usecase-label">{uc.label}</div>
                <div className="card-usecase-tag">{uc.tagline}</div>
              </div>
            </div>
          ))}
        </div>

        <div className="card-transcript">
          {turns.length === 0 && (
            <div className="card-transcript-empty">{AGENT_NAME} is about to say hello…</div>
          )}
          {turns.map((t) => (
            <div key={t.key} className={`card-turn card-turn-${t.role}`}>
              <span className="card-turn-who">
                {t.role === "agent" ? AGENT_NAME : persona.name.split(" ")[0]}
              </span>
              <span className="card-turn-text">{t.text}</span>
            </div>
          ))}
          {interimText.trim() && agentState === "listening" && (
            <div className="card-turn card-turn-customer is-interim">
              <span className="card-turn-who">{persona.name.split(" ")[0]}</span>
              <span className="card-turn-text is-interim">{interimText}</span>
            </div>
          )}
          <div ref={transcriptEndRef} />
        </div>

        {sessionId && (
          <a className="card-tracechip" href="#/traces" title="This call's turns are traced">
            <span className="card-tracechip-dot" /> traced · {sessionId.slice(0, 8)}
          </a>
        )}
      </aside>

      {/* ---- Analytics canvas ---- */}
      <main className="card-canvas">
        <AnalyticsCanvas
          persona={persona}
          selectedUseCase={selectedUseCase}
          facts={facts}
          genieFacts={genieFacts}
          profile={profile}
          investigations={investigations}
          language={callLanguage}
        />
      </main>
    </div>
  );
}

/* --------------------------------------------------------------------------- */
/* Analytics canvas                                                            */
/* --------------------------------------------------------------------------- */
function AnalyticsCanvas({
  persona, selectedUseCase, facts, genieFacts, profile, investigations, language,
}: {
  persona: Persona;
  selectedUseCase: string | null;
  facts: Record<string, unknown> | null;
  genieFacts: GenieFact[];
  profile: CardProfile | null;
  investigations: Map<string, Investigation>;
  language: string;
}) {
  const copy = uiCopy(language);
  const summary = ((facts?.summary as Record<string, unknown>) ?? profile?.summary ?? {}) as Record<string, unknown>;
  const invItems = [...investigations.values()].sort((a, b) => b.startedAt - a.startedAt);

  const spike = useMemo(
    () => (profile ? computeSpikeWaterfall(profile.spending_by_category) : null),
    [profile]
  );
  const rewards = useMemo(
    () => (profile ? computeRewardsWaterfall(profile.rewards_ledger) : null),
    [profile]
  );

  if (!profile) {
    return (
      <div className="card-canvas-loading">
        <div className="card-shimmer" />
        <div className="card-shimmer" />
        <div className="card-shimmer card-shimmer-lg" />
        <div className="card-canvas-hint">Pulling {persona.name.split(" ")[0]}’s account…</div>
      </div>
    );
  }

  const showRewards = selectedUseCase === "rewards_optimizer";
  const showSpike = selectedUseCase === "statement_insights" || (!selectedUseCase && !!spike);

  return (
    <div className="card-canvas-scroll">
      <Customer360Strip cardholder={profile.cardholder} summary={summary} />

      <div className="card-canvas-grid">
        <HeroMetric summary={summary} spike={spike} rewards={rewards} selected={selectedUseCase} />
        <ExpenseTrendChart spending={profile.spending_by_category} summary={summary} />
      </div>

      {showSpike && spike && <SpikeWaterfall data={spike} />}
      {showRewards && rewards && <RewardsWaterfall data={rewards} pointsBalance={summary.points_balance} />}

      {invItems.length > 0 && (
        <DeepDivePipeline
          items={invItems}
          copy={copy}
          showAgentProse={langBase(language) === "en"}
        />
      )}

      {genieFacts.length > 0 && <GenieFastTicker facts={genieFacts} />}
    </div>
  );
}

/* --------------------------------------------------------------------------- */
/* Customer 360 strip                                                          */
/* --------------------------------------------------------------------------- */
function Customer360Strip({
  cardholder, summary,
}: {
  cardholder: Record<string, unknown> | null;
  summary: Record<string, unknown>;
}) {
  const ch = cardholder ?? {};
  const limit = num(summary.credit_limit ?? ch.credit_limit);
  const balance = num(summary.new_balance);
  const util = limit > 0 ? Math.min(1, balance / limit) : 0;
  const name = String(ch.full_name ?? "Cardholder");
  const product = String(ch.primary_product_id ?? "Card");
  const status = String(summary.status ?? ch.status ?? "active");
  const tenure = num(ch.tenure_months);
  const apr = num(ch.apr_pct);

  return (
    <section className="card-360 card-fade-in">
      <div className="card-360-id">
        <div className="card-360-avatar">{name.slice(0, 1)}</div>
        <div>
          <div className="card-360-name">{name}</div>
          <div className="card-360-sub">
            {product} · <span className={`card-360-status is-${status.toLowerCase()}`}>{status}</span>
            {tenure > 0 && ` · ${tenure >= 12 ? `${Math.floor(tenure / 12)}y ${tenure % 12}m` : `${tenure}m`} member`}
          </div>
        </div>
      </div>
      <div className="card-360-metrics">
        <UtilizationGauge util={util} balance={balance} limit={limit} />
        <div className="card-360-stat">
          <div className="card-360-stat-val">{pts(summary.points_balance ?? ch.points_balance)}</div>
          <div className="card-360-stat-lbl">Points balance</div>
        </div>
        <div className="card-360-stat">
          <div className="card-360-stat-val">{money(summary.min_payment)}</div>
          <div className="card-360-stat-lbl">Min due{summary.due_date ? ` · ${String(summary.due_date)}` : ""}</div>
        </div>
        <div className="card-360-stat">
          <div className="card-360-stat-val">{apr > 0 ? `${apr.toFixed(2)}%` : "—"}</div>
          <div className="card-360-stat-lbl">APR</div>
        </div>
      </div>
    </section>
  );
}

function UtilizationGauge({ util, balance, limit }: { util: number; balance: number; limit: number }) {
  const r = 26;
  const c = 2 * Math.PI * r;
  const dash = c * util;
  const hot = util > 0.5;
  const stroke = util > 0.7 ? "#f87171" : util > 0.4 ? "#f59e0b" : "#34d399";
  return (
    <div className="card-gauge">
      <svg viewBox="0 0 64 64" className="card-gauge-svg">
        <circle cx="32" cy="32" r={r} className="card-gauge-track" />
        <circle
          cx="32" cy="32" r={r}
          className="card-gauge-fill"
          stroke={stroke}
          strokeDasharray={`${dash} ${c - dash}`}
          strokeDashoffset={c * 0.25}
          transform="rotate(-90 32 32)"
        />
        <text x="32" y="30" className="card-gauge-pct">{Math.round(util * 100)}%</text>
        <text x="32" y="42" className="card-gauge-cap">used</text>
      </svg>
      <div className="card-gauge-meta">
        <div className={`card-gauge-bal ${hot ? "is-hot" : ""}`}>{money(balance)}</div>
        <div className="card-gauge-lbl">of {money(limit)} limit</div>
      </div>
    </div>
  );
}

/* --------------------------------------------------------------------------- */
/* Hero metric — the headline the voice narrates                               */
/* --------------------------------------------------------------------------- */
function HeroMetric({
  summary, spike, rewards, selected,
}: {
  summary: Record<string, unknown>;
  spike: ReturnType<typeof computeSpikeWaterfall>;
  rewards: ReturnType<typeof computeRewardsWaterfall>;
  selected: string | null;
}) {
  if (selected === "rewards_optimizer" && rewards) {
    const pct = rewards.possible > 0 ? Math.round((rewards.gap / rewards.possible) * 100) : 0;
    return (
      <section className="card-hero-metric is-rewards card-fade-in">
        <div className="card-hero-metric-lbl">Rewards left on the table</div>
        <div className="card-hero-metric-val">{pts(rewards.gap)} pts</div>
        <div className="card-hero-metric-sub">
          You kept {pts(rewards.earned)} of {pts(rewards.possible)} points you qualified for
          {pct > 0 ? ` — that's ${pct}% slipping away.` : "."}
        </div>
      </section>
    );
  }
  const thisExp = num(summary.this_month_expenses);
  const avgExp = num(summary.avg_monthly_expenses);
  const change = num(summary.expense_change ?? (spike ? spike.increase : 0));
  const up = change > 0;
  const pct = avgExp > 0 ? Math.round((change / avgExp) * 100) : 0;
  return (
    <section className={`card-hero-metric ${up ? "is-up" : ""} card-fade-in`}>
      <div className="card-hero-metric-lbl">This month’s expenses</div>
      <div className="card-hero-metric-val">{money(thisExp)}</div>
      <div className="card-hero-metric-sub">
        {up ? (
          <>
            <span className="card-delta up">▲ {money(change)}{pct ? ` (+${pct}%)` : ""}</span> vs your
            typical {money(avgExp)}
          </>
        ) : (
          <>In line with your typical {money(avgExp)}</>
        )}
      </div>
    </section>
  );
}

/* --------------------------------------------------------------------------- */
/* Expense trend — 12 months, baseline line, spike annotation, hover tooltip   */
/* --------------------------------------------------------------------------- */
function ExpenseTrendChart({
  spending, summary,
}: {
  spending: SpendingRecord[];
  summary: Record<string, unknown>;
}) {
  const [hover, setHover] = useState<number | null>(null);
  const data = useMemo(() => cycleTotals(spending), [spending]);
  if (data.length === 0) return null;

  const max = Math.max(...data.map((d) => d.total), 1);
  const avg = data.length > 1
    ? data.slice(0, -1).reduce((s, d) => s + d.total, 0) / (data.length - 1)
    : num(summary.avg_monthly_expenses);

  const W = 520, H = 190, padL = 8, padR = 8, padB = 26, padT = 16;
  const plotW = W - padL - padR;
  const plotH = H - padB - padT;
  const n = data.length;
  const slot = plotW / n;
  const barW = Math.min(30, slot * 0.6);
  const y = (v: number) => padT + plotH - (v / max) * plotH;
  const avgY = y(avg);
  const lastIdx = n - 1;

  return (
    <section className="card-chart card-fade-in">
      <div className="card-panel-title">
        Monthly expenses <span className="card-panel-note">last {n} cycles</span>
      </div>
      <div className="card-chart-wrap" onMouseLeave={() => setHover(null)}>
        <svg viewBox={`0 0 ${W} ${H}`} className="card-chart-svg" preserveAspectRatio="none">
          {/* baseline (typical month) */}
          <line x1={padL} x2={W - padR} y1={avgY} y2={avgY} className="card-chart-avg" />
          <text x={W - padR} y={avgY - 5} textAnchor="end" className="card-chart-avg-lbl">
            typical {money(avg)}
          </text>
          {data.map((d, i) => {
            const cx = padL + slot * i + slot / 2;
            const bh = (d.total / max) * plotH;
            const by = padT + plotH - bh;
            const isLast = i === lastIdx;
            const isHover = hover === i;
            const spiked = isLast && d.total > avg * 1.4;
            return (
              <g key={d.cycle} onMouseEnter={() => setHover(i)}>
                <rect
                  x={cx - slot / 2} y={padT} width={slot} height={plotH}
                  fill="transparent"
                />
                <rect
                  x={cx - barW / 2} y={by} width={barW} height={bh} rx={4}
                  className={`card-bar ${isLast ? "is-last" : ""} ${isHover ? "is-hover" : ""}`}
                  fill={spiked ? "#f59e0b" : isLast ? "#fbbf24" : "#334155"}
                  opacity={hover !== null && !isHover ? 0.55 : 1}
                />
                <text x={cx} y={H - 8} textAnchor="middle" className="card-chart-label">
                  {d.label}
                </text>
                {spiked && (
                  <text x={cx} y={by - 6} textAnchor="middle" className="card-chart-spike-tag">
                    +{Math.round(((d.total - avg) / avg) * 100)}%
                  </text>
                )}
              </g>
            );
          })}
        </svg>
        {hover !== null && (
          <div
            className="card-chart-tip"
            style={{ left: `${((hover + 0.5) / n) * 100}%` }}
          >
            <div className="card-chart-tip-cycle">{data[hover].label}</div>
            <div className="card-chart-tip-total">{money(data[hover].total)}</div>
            {data[hover].topCat && (
              <div className="card-chart-tip-cat">
                <span className="card-legend-dot" style={{ background: colorFor(data[hover].topCat) }} />
                top: {data[hover].topCat}
              </div>
            )}
          </div>
        )}
      </div>
    </section>
  );
}

/* --------------------------------------------------------------------------- */
/* Waterfall (shared)                                                          */
/* --------------------------------------------------------------------------- */
function Waterfall({ steps, unit }: { steps: WaterfallStep[]; unit: "money" | "pts" }) {
  const fmt = unit === "money" ? money : (v: unknown) => `${pts(v)} pts`;
  // Running totals to place floating bars.
  const maxVal = Math.max(
    steps[0]?.delta ?? 0,
    steps[steps.length - 1]?.delta ?? 0,
    ...steps.reduce<number[]>((acc, s, i) => {
      const prev = i === 0 ? 0 : acc[i - 1];
      const running = s.kind === "base" || s.kind === "final" ? s.delta : prev + s.delta;
      acc.push(running);
      return acc;
    }, [])
  );
  let running = 0;
  return (
    <div className="card-waterfall">
      {steps.map((s, i) => {
        let base: number, top: number, val: number;
        if (s.kind === "base") {
          base = 0; top = s.delta; running = s.delta; val = s.delta;
        } else if (s.kind === "final") {
          base = 0; top = s.delta; val = s.delta; running = s.delta;
        } else {
          const start = running;
          running += s.delta;
          base = Math.min(start, running);
          top = Math.max(start, running);
          val = s.delta;
        }
        const bottomPct = (base / maxVal) * 100;
        const heightPct = Math.max(((top - base) / maxVal) * 100, 1.5);
        return (
          <div key={i} className={`card-wf-col card-wf-${s.kind}`}>
            <div className="card-wf-track">
              <div
                className="card-wf-bar"
                style={{ bottom: `${bottomPct}%`, height: `${heightPct}%`, background: s.color }}
              >
                <span className="card-wf-val">
                  {s.kind === "down" ? "−" : s.kind === "up" ? "+" : ""}
                  {fmt(Math.abs(val))}
                </span>
              </div>
            </div>
            <div className="card-wf-lbl">{s.label}</div>
          </div>
        );
      })}
    </div>
  );
}

function SpikeWaterfall({ data }: { data: NonNullable<ReturnType<typeof computeSpikeWaterfall>> }) {
  return (
    <section className="card-panel card-climax card-fade-in">
      <div className="card-panel-title">
        What drove the increase
        <span className="card-panel-badge is-fast">Genie · fast lane</span>
      </div>
      <Waterfall steps={data.steps} unit="money" />
      <div className="card-climax-caption">
        {money(data.baseline)} typical → {money(data.current)} this month.
        The bars show which categories pushed it up.
      </div>
    </section>
  );
}

function RewardsWaterfall({
  data, pointsBalance,
}: {
  data: NonNullable<ReturnType<typeof computeRewardsWaterfall>>;
  pointsBalance: unknown;
}) {
  return (
    <section className="card-panel card-climax card-fade-in">
      <div className="card-panel-title">
        Where your points are leaking
        <span className="card-panel-badge is-fast">Genie · fast lane</span>
      </div>
      <Waterfall steps={data.steps} unit="pts" />
      <div className="card-climax-caption">
        You have {pts(pointsBalance)} points banked, but qualified for {pts(data.possible)} this cycle and
        kept {pts(data.earned)}. Closing the gap is worth {pts(data.gap)} pts.
      </div>
    </section>
  );
}

/* --------------------------------------------------------------------------- */
/* Deep-dive pipeline — Plan → Query → Compute → Explain (Genie Agent Mode)    */
/* --------------------------------------------------------------------------- */
function stageOf(inv: Investigation): number {
  if (inv.status === "done" || inv.report) return 3;
  const hasSql = inv.steps.some((s) => s.kind === "sql");
  const hasReasoning = inv.steps.some((s) => s.kind === "reasoning");
  if (hasSql) return 2;
  if (hasReasoning) return 1;
  return 0;
}

function DeepDivePipeline({
  items,
  copy,
  showAgentProse,
}: {
  items: Investigation[];
  copy: UiCopy;
  showAgentProse: boolean;
}) {
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    if (!items.some((i) => i.status === "running")) return;
    const id = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(id);
  }, [items]);

  return (
    <section className="card-panel card-fade-in">
      <div className="card-panel-title">
        {DEEP_AGENT_NAME}
        <span className="card-panel-badge is-deep">Genie · Agent Mode</span>
      </div>
      {items.map((inv) => (
        <InvestigationCard
          key={inv.id}
          inv={inv}
          now={now}
          copy={copy}
          showAgentProse={showAgentProse}
        />
      ))}
    </section>
  );
}

function InvestigationCard({
  inv,
  now,
  copy,
  showAgentProse,
}: {
  inv: Investigation;
  now: number;
  copy: UiCopy;
  showAgentProse: boolean;
}) {
  const elapsed = inv.status === "running" ? now - inv.startedAt : inv.elapsed;
  const stage = stageOf(inv);
  // Show Genie's natural-language REASONING (business language), not raw SQL.
  // The SQL is provenance — it lives in the report's opt-in "Show the data" drawer.
  // Genie reasons in English; on a non-English call its raw text would be the one
  // thing in the panel the caller can't read, so we narrate the stage instead and
  // let the pacing (one line per real step) carry the progress.
  const reasoningSteps = inv.steps.filter((s) => s.kind === "reasoning").slice(-4);
  const queryCount = inv.steps.filter((s) => s.kind === "sql").length;
  const lastIsQuery = inv.steps.length > 0 && inv.steps[inv.steps.length - 1].kind === "sql";

  return (
    <div className={`card-inv card-inv-${inv.status}`}>
      <div className="card-inv-header">
        <span className={`card-inv-dot card-inv-dot-${inv.status}`} />
        <span className="card-inv-q">
          “{inv.spokenQuestion ?? inv.question.replace(/^For cardholder [^:]+:\s*/i, "")}”
        </span>
        <span className="card-inv-time">{Math.round(elapsed / 1000)}s</span>
      </div>

      <div className="card-pipeline">
        {pipelineStages(copy).map((label, i) => {
          const state = inv.status === "error"
            ? (i <= stage ? "error" : "todo")
            : i < stage || inv.status === "done"
              ? "done"
              : i === stage && inv.status === "running"
                ? "active"
                : "todo";
          return (
            <div key={label} className={`card-pl-node is-${state}`}>
              <span className="card-pl-dot">{state === "done" ? "✓" : i + 1}</span>
              <span className="card-pl-lbl">{label}</span>
            </div>
          );
        })}
      </div>

      {inv.status === "running" && (
        <div className="card-inv-live">
          {reasoningSteps.length > 0 || queryCount > 0 ? (
            <ol className="card-steps">
              {reasoningSteps.map((s) => (
                <li key={s.key} className="card-step card-step-reasoning card-fade-in">
                  <span className="card-step-icon">✦</span>
                  <span className="card-step-text">
                    {showAgentProse ? s.text : copy.deepReasoningStep}
                  </span>
                </li>
              ))}
              {queryCount > 0 && (
                <li className="card-step card-step-query">
                  <span className="card-step-icon">▤</span>
                  <span className="card-step-text">
                    {lastIsQuery ? copy.deepQuerying : copy.deepPulledQueries(String(queryCount))}
                  </span>
                </li>
              )}
            </ol>
          ) : (
            <div className="card-inv-hint">{copy.deepHint}</div>
          )}
          <div className="card-step card-step-live">
            <span className="card-dots">{copy.deepWorking(DEEP_AGENT_NAME)}<i /><i /><i /></span>
          </div>
        </div>
      )}

      {inv.status === "done" && inv.report && <DeepDiveReportView report={inv.report} copy={copy} />}
      {inv.status === "error" && (
        <div className="card-inv-error">{copy.deepFailed(inv.errorMessage ?? "")}</div>
      )}
    </div>
  );
}

function DeepDiveReportView({ report, copy }: { report: DeepDiveReport; copy: UiCopy }) {
  const tables = useMemo(() => report.tables ?? [], [report.tables]);
  return (
    <div className="card-report card-fade-in">
      {report.report ? (
        <p className="card-report-text">{report.report}</p>
      ) : (
        // Non-English reports open EMPTY and stream in the caller's language, so an
        // empty body while pending means "translation on its way", not "no summary".
        <p className="card-inv-hint">
          {report.localizationPending ? copy.deepTranslating : copy.deepNoSummary(report.status)}
        </p>
      )}
      {/* While chunks are still arriving, keep the hint under the growing text so
          the caller knows the localized report isn't complete yet. */}
      {report.report && report.localizationPending && (
        <p className="card-inv-hint">{copy.deepTranslating}</p>
      )}
      {tables.map((tbl, i) => (
        <ReportTable key={i} table={tbl} />
      ))}
      {report.sql?.length > 0 && (
        <details className="card-report-sql">
          <summary>{copy.deepShowData(String(report.sql.length))}</summary>
          {report.sql.map((q, i) => (
            <pre key={i}>{q}</pre>
          ))}
        </details>
      )}
    </div>
  );
}

function ReportTable({ table }: { table: Record<string, unknown> }) {
  const columns = (table.columns as string[]) ?? [];
  const rows = (table.preview_rows as unknown[][]) ?? [];
  if (columns.length === 0 && rows.length === 0) return null;
  return (
    <div className="card-report-table">
      <table>
        <thead>
          <tr>{columns.map((c, i) => <th key={i}>{c}</th>)}</tr>
        </thead>
        <tbody>
          {rows.slice(0, 12).map((row, i) => (
            <tr key={i}>
              {(Array.isArray(row) ? row : [row]).map((cell, j) => (
                <td key={j}>{String(cell)}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* --------------------------------------------------------------------------- */
/* Genie fast-fact ticker (Conversation API lane)                              */
/* --------------------------------------------------------------------------- */
function GenieFastTicker({ facts }: { facts: GenieFact[] }) {
  return (
    <section className="card-panel card-fade-in">
      <div className="card-panel-title">
        Quick answers
        <span className="card-panel-badge is-fast">Genie · fast lane</span>
      </div>
      <div className="card-ticker">
        {facts.map((f) => (
          <details key={f.id} className="card-ticker-item card-fade-in">
            <summary>{f.answer}</summary>
            {f.columns && f.rows && f.rows.length > 0 && (
              <ReportTable table={{ columns: f.columns, preview_rows: f.rows }} />
            )}
          </details>
        ))}
      </div>
    </section>
  );
}

/* --------------------------------------------------------------------------- */
/* Avatar / voice orb                                                          */
/* --------------------------------------------------------------------------- */
function Avatar({
  size, state, level = 0,
}: {
  size: number;
  state: AgentState;
  level?: number;
  // `ring` is accepted for call-site compatibility; the shared Genie orb always
  // carries its own halo rings, so it's intentionally unused.
  ring?: boolean;
}) {
  return <VoiceOrb state={state} level={level} size={`${size}px`} ariaLabel={AGENT_NAME} />;
}
