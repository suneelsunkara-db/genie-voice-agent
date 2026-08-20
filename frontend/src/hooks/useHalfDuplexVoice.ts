import { useCallback, useRef, useState, type MutableRefObject, type RefObject } from "react";

import {
  AudioPlaybackQueue,
  decodePcmChunk,
  type RealtimeVoiceSession,
} from "../lib/realtimeVoice";

/**
 * Shared half-duplex voice plumbing for every voice page (billing cockpit + card
 * assistant). Encapsulates the ONE pattern both pages implemented independently:
 *
 *   - an {@link AudioPlaybackQueue} for gapless agent-audio playback,
 *   - a single mic "gate" so the agent's own TTS from the speakers never loops
 *     back into the mic and gets re-transcribed as a phantom customer turn, and
 *   - mid-call language switching, which has to reopen the session: the language
 *     rides the session.start payload, so changing a picker cannot move a call
 *     that is already connected.
 *
 * The mic is closed while the agent speaks and re-opened once the queued audio
 * drains (plus a short tail so the speaker's decay doesn't leak into the first
 * captured frame). Page-specific UI reactions are injected as callbacks so the
 * timing/echo logic itself lives in exactly one place.
 */
export interface UseHalfDuplexVoiceOptions {
  /** The live voice session (mic pause/resume target). */
  sessionRef: RefObject<RealtimeVoiceSession | null>;
  /**
   * Optional pin for the playback {@link AudioContext} sample rate.
   * Default: omit (device-native clock). Do not pass 24000 — VoxCPM2 content is
   * 48 kHz and a lower context permanently truncates it.
   */
  playbackSampleRate?: number;
  /** Extra silence after the queue drains before re-opening the mic. Defaults to 350ms. */
  resumeTailMs?: number;
  /** If it returns true when the resume timer fires, the mic STAYS closed (e.g. a
   *  deep dive is still running). */
  shouldStayGated?: () => boolean;
  /** Called when the mic is actually re-opened (page can flip UI to "listening"). */
  onMicResume?: () => void;
  /** Called for each agent-audio chunk (page can flip UI to "speaking"). */
  onSpeaking?: () => void;
  /** Called once when the final chunk of a response arrives (turn completion). */
  onFinal?: () => void;
  /** Ref holding the language the LIVE session is running in. {@link HalfDuplexVoice.switchLanguage}
   *  updates it before reopening, so page code (greeting fetch, synthesize calls)
   *  always reads the language the session actually has rather than a stale prop. */
  callLanguageRef?: MutableRefObject<string>;
  /** True while a call is live. When idle, picking a language only records it —
   *  there is no session to reopen. */
  isCallLive?: () => boolean;
  /** Close the live session AND clear the page's session refs/flags, so the
   *  following {@link UseHalfDuplexVoiceOptions.reopenSession} isn't rejected by a
   *  "already in a call" guard. */
  closeSession?: () => void;
  /** Open a fresh session in `language`; the page supplies its own callbacks. */
  reopenSession?: (language: string) => Promise<void>;
  /** Drop the on-screen conversation before reopening.
   *
   *  A language switch restarts the call and the agent re-greets in the new
   *  language, so anything already on screen was spoken in the language the
   *  caller just left. Leaving it there reads as a bug — an English opener
   *  sitting above a Hindi greeting looks like the switch only half worked. */
  clearConversation?: () => void;
}

export interface ResponseAudioMeta {
  segmentFinal?: boolean;
  turnFinal?: boolean;
  /** Server speech generation; drop chunks older than the last playback.stop. */
  speechEpoch?: number;
}

export interface HalfDuplexVoice {
  playbackRef: MutableRefObject<AudioPlaybackQueue | null>;
  micGatedRef: MutableRefObject<boolean>;
  /** Fresh playback queue (flush+close any prior one). Call at session start. */
  resetPlayback: () => AudioPlaybackQueue;
  /** Close the mic once (idempotent) and cancel any pending re-open. */
  gateMic: () => void;
  /** Re-open the mic after `ms`, unless `shouldStayGated()` is true then. */
  ungateMicAfter: (ms: number) => void;
  /**
   * Handle one `response.audio` chunk: gate → enqueue → schedule resume on turn
   * completion. Progressive: pass ``meta.turnFinal`` / ``meta.segmentFinal``;
   * legacy callers pass only ``final`` (treated as turn completion).
   */
  handleResponseAudio: (
    pcmB64: string,
    sampleRate: number,
    final: boolean,
    meta?: ResponseAudioMeta
  ) => void;
  /** Server playback.stop. ``reason: "inject"`` flushes only (mic stays gated for
   *  the replacement speech); ``barge_in`` (or omitted) may reopen the mic. */
  handlePlaybackStop: (speechEpoch?: number, reason?: string) => void;
  /** Live on-device interim transcript (framework browser caption). "" when idle
   *  or once the turn is finalized. Any voice page renders this the same way. */
  interimText: string;
  /** Wire to the session's `onInterimTranscript`; updates {@link interimText}. */
  handleInterimTranscript: (text: string) => void;
  /** Barge-in: stop playback, notify server, re-open the mic now. */
  interrupt: () => void;
  /** Move a call to `language`. Pre-call this just records the choice; mid-call it
   *  reopens the session, because the language is negotiated once at session.start. */
  switchLanguage: (language: string) => Promise<void>;
  /** Clear timers, flush + close the playback queue (call on End/unmount). */
  teardownPlayback: () => void;
}

export function useHalfDuplexVoice(options: UseHalfDuplexVoiceOptions): HalfDuplexVoice {
  const playbackRef = useRef<AudioPlaybackQueue | null>(null);
  const micGatedRef = useRef(false);
  const resumeTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  // Server speech_epoch accepted after the last playback.stop / inject. Chunks
  // stamped with an older epoch are dropped so late pre-inject audio cannot
  // overlap the deep-dive summary.
  const acceptedSpeechEpochRef = useRef<number | null>(null);
  // Live caption preview text (framework-driven). Held here so every voice page
  // renders it identically without re-implementing the state per UI.
  const [interimText, setInterimText] = useState("");
  const handleInterimTranscript = useCallback((text: string) => setInterimText(text), []);
  // Latest options in a ref so the returned callbacks stay referentially stable
  // (safe to list in effect/callback deps) while always seeing fresh callbacks.
  const optsRef = useRef(options);
  optsRef.current = options;

  const clearResumeTimer = useCallback(() => {
    if (resumeTimerRef.current) {
      clearTimeout(resumeTimerRef.current);
      resumeTimerRef.current = null;
    }
  }, []);

  const resetPlayback = useCallback(() => {
    playbackRef.current?.flush();
    playbackRef.current?.close();
    acceptedSpeechEpochRef.current = null;
    // Device-native AudioContext unless a caller explicitly pins a rate (tests).
    const queue = new AudioPlaybackQueue(optsRef.current.playbackSampleRate);
    playbackRef.current = queue;
    return queue;
  }, []);

  const gateMic = useCallback(() => {
    micGatedRef.current = true;
    optsRef.current.sessionRef.current?.pauseMic();
    clearResumeTimer();
  }, [clearResumeTimer]);

  const ungateMicAfter = useCallback((ms: number) => {
    clearResumeTimer();
    resumeTimerRef.current = setTimeout(() => {
      resumeTimerRef.current = null;
      if (optsRef.current.shouldStayGated?.()) return;
      micGatedRef.current = false;
      optsRef.current.sessionRef.current?.resumeMic();
      optsRef.current.onMicResume?.();
    }, ms);
  }, [clearResumeTimer]);

  const handleResponseAudio = useCallback((
    pcmB64: string,
    sampleRate: number,
    final: boolean,
    meta?: ResponseAudioMeta
  ) => {
    // Drop superseded speech (pre-inject / pre-barge-in chunks still on the wire).
    if (
      meta?.speechEpoch !== undefined &&
      acceptedSpeechEpochRef.current !== null &&
      meta.speechEpoch < acceptedSpeechEpochRef.current
    ) {
      return;
    }
    if (meta?.speechEpoch !== undefined) {
      acceptedSpeechEpochRef.current = meta.speechEpoch;
    }
    // Gate the mic ONCE for the whole response (not per chunk) so pause/resume
    // can't flap while the agent is mid-sentence.
    if (!micGatedRef.current) gateMic();
    optsRef.current.onSpeaking?.();
    const { samples } = decodePcmChunk(pcmB64, sampleRate);
    playbackRef.current?.enqueue(samples, sampleRate);
    // turn_final (or legacy final) completes the turn and reopens mic.
    // segment_final alone does not — progressive mid-turn speech stays gated
    // while work continues unless the page opts into awaiting_user later.
    const turnDone = meta?.turnFinal === true || (meta?.turnFinal !== false && final);
    if (turnDone) {
      optsRef.current.onFinal?.();
      const tail = optsRef.current.resumeTailMs ?? 350;
      ungateMicAfter((playbackRef.current?.msUntilIdle() ?? 0) + tail);
    }
  }, [gateMic, ungateMicAfter]);

  const handlePlaybackStop = useCallback((speechEpoch?: number, reason?: string) => {
    playbackRef.current?.flush();
    if (typeof speechEpoch === "number") {
      acceptedSpeechEpochRef.current = speechEpoch;
    }
    clearResumeTimer();
    setInterimText("");
    // Same-turn inject: agent is about to speak again — keep mic gated.
    if (reason === "inject") {
      micGatedRef.current = true;
      optsRef.current.sessionRef.current?.pauseMic();
      return;
    }
    if (optsRef.current.shouldStayGated?.()) {
      micGatedRef.current = true;
      optsRef.current.sessionRef.current?.pauseMic();
      return;
    }
    micGatedRef.current = false;
    optsRef.current.sessionRef.current?.resumeMic();
    optsRef.current.onMicResume?.();
  }, [clearResumeTimer]);

  const interrupt = useCallback(() => {
    if (!playbackRef.current && !optsRef.current.sessionRef.current) return;
    playbackRef.current?.flush();
    clearResumeTimer();
    setInterimText("");
    // Notify server FIRST so late audio is cancelled, then reopen mic locally.
    optsRef.current.sessionRef.current?.bargeIn();
    if (optsRef.current.shouldStayGated?.()) {
      micGatedRef.current = true;
      optsRef.current.sessionRef.current?.pauseMic();
      return;
    }
    micGatedRef.current = false;
    optsRef.current.sessionRef.current?.resumeMic();
    optsRef.current.onMicResume?.();
  }, [clearResumeTimer]);

  // Serializes restarts: a caller clicking through three languages must not leave
  // three sessions racing to open, with the last one to connect deciding the call.
  const switchingRef = useRef(false);

  const switchLanguage = useCallback(async (language: string) => {
    const o = optsRef.current;
    if (!language || language === o.callLanguageRef?.current) return;
    if (o.callLanguageRef) o.callLanguageRef.current = language;
    // Pre-call the picker only records a choice — the language is sent in the
    // session.start payload, so there is nothing live to move yet.
    if (!o.isCallLive?.()) return;
    if (switchingRef.current) return;
    switchingRef.current = true;
    try {
      // Silence the old session first: its queued audio is in the previous
      // language, and an open mic during the swap would be transcribed against a
      // session that is about to disappear.
      gateMic();
      playbackRef.current?.flush();
      o.closeSession?.();
      // Before the re-greet, not after: the new greeting must not land underneath
      // turns from the old language, and clearing after it arrives would wipe it.
      setInterimText("");
      o.clearConversation?.();
      await o.reopenSession?.(language);
    } finally {
      switchingRef.current = false;
    }
  }, [gateMic]);

  const teardownPlayback = useCallback(() => {
    clearResumeTimer();
    micGatedRef.current = false;
    setInterimText("");
    playbackRef.current?.flush();
    playbackRef.current?.close();
    playbackRef.current = null;
  }, [clearResumeTimer]);

  return {
    playbackRef,
    micGatedRef,
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
  };
}
