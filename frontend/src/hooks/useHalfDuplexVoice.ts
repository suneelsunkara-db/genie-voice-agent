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
 *   - an {@link AudioPlaybackQueue} for gapless agent-audio playback, and
 *   - a single mic "gate" so the agent's own TTS from the speakers never loops
 *     back into the mic and gets re-transcribed as a phantom customer turn.
 *
 * The mic is closed while the agent speaks and re-opened once the queued audio
 * drains (plus a short tail so the speaker's decay doesn't leak into the first
 * captured frame). Page-specific UI reactions are injected as callbacks so the
 * timing/echo logic itself lives in exactly one place.
 */
export interface UseHalfDuplexVoiceOptions {
  /** The live voice session (mic pause/resume target). */
  sessionRef: RefObject<RealtimeVoiceSession | null>;
  /** Playback queue sample rate (Hz). Defaults to 24000. */
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
  /** Called once when the final chunk of a response arrives. */
  onFinal?: () => void;
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
  /** Handle one `response.audio` chunk: gate → enqueue → schedule resume on final. */
  handleResponseAudio: (pcmB64: string, sampleRate: number, final: boolean) => void;
  /** Live on-device interim transcript (framework browser caption). "" when idle
   *  or once the turn is finalized. Any voice page renders this the same way. */
  interimText: string;
  /** Wire to the session's `onInterimTranscript`; updates {@link interimText}. */
  handleInterimTranscript: (text: string) => void;
  /** Barge-in: stop playback immediately and re-open the mic now. */
  interrupt: () => void;
  /** Clear timers, flush + close the playback queue (call on End/unmount). */
  teardownPlayback: () => void;
}

export function useHalfDuplexVoice(options: UseHalfDuplexVoiceOptions): HalfDuplexVoice {
  const playbackRef = useRef<AudioPlaybackQueue | null>(null);
  const micGatedRef = useRef(false);
  const resumeTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
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
    const queue = new AudioPlaybackQueue(optsRef.current.playbackSampleRate ?? 24_000);
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

  const handleResponseAudio = useCallback((pcmB64: string, sampleRate: number, final: boolean) => {
    // Gate the mic ONCE for the whole response (not per chunk) so pause/resume
    // can't flap while the agent is mid-sentence.
    if (!micGatedRef.current) gateMic();
    optsRef.current.onSpeaking?.();
    const { samples } = decodePcmChunk(pcmB64, sampleRate);
    playbackRef.current?.enqueue(samples, sampleRate);
    if (final) {
      optsRef.current.onFinal?.();
      const tail = optsRef.current.resumeTailMs ?? 350;
      ungateMicAfter((playbackRef.current?.msUntilIdle() ?? 0) + tail);
    }
  }, [gateMic, ungateMicAfter]);

  const interrupt = useCallback(() => {
    if (!playbackRef.current) return;
    playbackRef.current.flush();
    clearResumeTimer();
    micGatedRef.current = false;
    setInterimText("");
    optsRef.current.sessionRef.current?.resumeMic();
    optsRef.current.onMicResume?.();
  }, [clearResumeTimer]);

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
    interimText,
    handleInterimTranscript,
    interrupt,
    teardownPlayback,
  };
}
