/**
 * Realtime Voice client for the contact-center cockpit.
 *
 * Replaces the Deepgram WS + Databricks batch STT paths with a single
 * WebSocket to the realtime API (speech-llm-toolassist-speech). The server
 * handles STT → LLM (with tool calls) → TTS in one session, auto-detecting
 * the caller's language per turn.
 *
 * Usage in LiveAssist:
 *   const session = await startRealtimeVoice(wsBaseUrl, callId, customerId, callbacks);
 *   // user speaks... audio frames stream to server
 *   // callbacks fire as turn progresses: transcript → response → audio chunks
 *   session.endTurn();   // signal end of speech (or rely on server VAD)
 *   session.close();     // tear down session + mic
 */

import {
  isSpeechCaptionSupported,
  startSpeechCaption,
  type SpeechCaptionSession,
} from "./micStream";

/** Result of an async Genie Agent-mode "why" investigation (card profile). */
export interface DeepDiveReport {
  useCase: string | null;
  status: string;
  report: string;
  // LLM-generated short spoken "why" (names the cause); the client speaks this
  // instead of reading the full report. Absent -> client falls back to a heuristic.
  spokenSummary: string | null;
  tables: Array<Record<string, unknown>>;
  sql: string[];
  reasoning: string[];
  error: unknown;
  /** True while a translation of `report` is still on its way (see onReportLocalized). */
  localizationPending: boolean;
}

export interface RealtimeVoiceCallbacks {
  onSpeechStarted?: (turnId: number) => void;
  onTranscript?: (text: string, language: string, turnId: number) => void;
  onResponseText?: (text: string, turnId: number) => void;
  onResponseAudio?: (pcmB64: string, sampleRate: number, final: boolean, turnId: number) => void;
  onTurnStarted?: (turnId: number) => void;
  onTurnDone?: (turnId: number) => void;
  onToolCalled?: (name: string, result: unknown, turnId: number) => void;
  onLanguageMismatch?: (expected: string, detected: string, turnId: number) => void;
  onError?: (code: string, message: string) => void;
  onSessionReady?: (sessionId: string, language: string) => void;
  onLevel?: (level: number) => void;
  /**
   * Live, on-device interim transcript (framework-managed browser caption).
   *
   * Fires with partial words AS the caller speaks so the UI can show text with
   * zero server latency, and with "" when the turn's authoritative
   * ``transcript.final`` arrives. Display-only — the server transcript is always
   * the source of truth. No-op where the browser lacks Web Speech support.
   */
  onInterimTranscript?: (text: string, turnId: number) => void;
}

export interface StartRealtimeVoiceOptions {
  /** Assistant profile. Omit for the telco cockpit; "card" for the card issuer. */
  profile?: string;
  /** Start with mic paused (agent-initiated greeting flows). The caller must
   *  explicitly resumeMic() after the greeting finishes playing. */
  startMicPaused?: boolean;
  /** Live on-device caption preview (browser Web Speech API) while the caller
   *  speaks, surfaced via ``onInterimTranscript``. Default true; set false to opt
   *  out. Automatically a no-op where the browser lacks Web Speech support. */
  liveCaption?: boolean;
  /**
   * Pin STT to this BCP-47 language instead of auto-detecting. Use ONLY for
   * single-language surfaces (e.g. the English-only home concierge whose picker
   * is disabled): auto-detect mis-tags short words ("Telco" -> Hindi) and the
   * mismatch gate then drops the turn. Omit to auto-detect (multilingual pages).
   */
  sttLanguage?: string;
}

export interface RealtimeVoiceSession {
  /** Signal end of user speech for the current turn (server VAD also auto-finalizes). */
  endTurn: () => void;
  /**
   * Speak agent text through THIS session (no STT turn). The audio streams back
   * as normal response.audio events, so it shares the session's locked voice
   * reference — used for the agent-initiated greeting and deep-dive summary so
   * the whole call keeps ONE consistent voice.
   */
  synthesize: (text: string, language?: string) => void;
  /** Pause mic streaming (stop sending audio) without closing the session. */
  pauseMic: () => void;
  /** Resume mic streaming after a pause. */
  resumeMic: () => void;
  /** Gracefully close the session and release mic/ws resources. */
  close: () => void;
  /** True while the WebSocket is connected and session is active. */
  readonly connected: boolean;
  /** True while mic is actively streaming audio to the server. */
  readonly micActive: boolean;
  /** The session ID assigned by the server (available after onSessionReady). */
  readonly sessionId: string | null;
  /** Last detected language from STT. */
  readonly detectedLanguage: string | null;
}

const TARGET_SAMPLE_RATE = 16_000;

function floatTo16BitPCM(input: Float32Array): ArrayBuffer {
  const output = new Int16Array(input.length);
  for (let i = 0; i < input.length; i++) {
    const s = Math.max(-1, Math.min(1, input[i]));
    output[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return output.buffer;
}

/**
 * Start a realtime voice session against the speech-llm-toolassist-speech endpoint.
 *
 * Opens the mic, connects the WebSocket, sends session.start, and begins
 * streaming PCM audio. The server auto-detects language, runs STT → LLM → TTS,
 * and streams back transcript/response/audio events. The session stays open
 * for multiple turns until close() is called.
 */
export async function startRealtimeVoice(
  wsBaseUrl: string,
  callId: string,
  customerId: string,
  callbacks: RealtimeVoiceCallbacks,
  expectedLanguage?: string,
  options?: StartRealtimeVoiceOptions
): Promise<RealtimeVoiceSession> {
  // Capture RAW audio for ASR. The browser echo-canceller "locks onto" the TTS
  // playing through the speakers and then gates the caller's own voice into
  // fragmented near-silence (observed: turn 1 clean, turns 2+ degrade to ~7%
  // voiced once the agent has spoken). Noise suppression + auto-gain likewise
  // distort speech for the STT model. We handle echo ourselves via half-duplex
  // mic muting during playback, so all three browser DSP stages are disabled.
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
  });

  // Framework-level live caption (shared by EVERY voice use case — no per-UI
  // wiring). The server's STT is whole-utterance (no partials), so text can only
  // appear after the caller stops + one inference. To show words WHILE the caller
  // speaks, we run the browser's on-device recognizer purely as a preview and
  // surface it via onInterimTranscript; the server's transcript.final stays
  // authoritative. Best-effort: a no-op when unsupported or opted out. Its
  // pause/resume is driven by the same mic gate used for half-duplex playback so
  // it never transcribes the agent's own TTS.
  let caption: SpeechCaptionSession | null = null;
  let captionTurnId = 0;
  const wantCaption =
    (options?.liveCaption ?? true) && !!callbacks.onInterimTranscript && isSpeechCaptionSupported();
  if (wantCaption) {
    caption = startSpeechCaption(
      (text) => callbacks.onInterimTranscript?.(text, captionTurnId),
      undefined,
      expectedLanguage || "en-US"
    );
    // Agent-initiated flows open with the mic paused (greeting plays first); keep
    // the caption paused until the mic is resumed so it doesn't hear the greeting.
    if (options?.startMicPaused) caption?.pause();
  }

  // Request a 16 kHz context; modern Chrome/Edge/Safari honor this and hand back
  // high-quality (browser-resampled) 16 kHz audio. But some browsers/devices
  // IGNORE the option and run the context at the hardware rate (typically
  // 48 kHz). Previously we still declared 16 kHz to the server, so on those
  // browsers the PCM was 48 kHz mislabeled as 16 kHz — STT then received 3x-fast
  // audio and mis-transcribed. We therefore read the ACTUAL context rate and
  // report it truthfully below; the server resamples to 16 kHz where it needs a
  // fixed rate (endpointing). We do NOT hand-roll a streaming downsample here —
  // naive linear decimation aliases sibilants into the speech band (see
  // micStream.resampleTo16k).
  const audioContext = new AudioContext({ sampleRate: TARGET_SAMPLE_RATE });
  const actualSampleRate = Math.round(audioContext.sampleRate);
  const source = audioContext.createMediaStreamSource(stream);
  const analyser = audioContext.createAnalyser();
  analyser.fftSize = 256;
  source.connect(analyser);

  const processor = audioContext.createScriptProcessor(4096, 1, 1);
  const silentGain = audioContext.createGain();
  silentGain.gain.value = 0;
  source.connect(processor);
  processor.connect(silentGain);
  silentGain.connect(audioContext.destination);

  // Mic level meter
  const levelData = new Uint8Array(analyser.frequencyBinCount);
  let levelRaf = 0;
  const tickLevel = () => {
    analyser.getByteFrequencyData(levelData);
    const avg = levelData.reduce((sum, v) => sum + v, 0) / levelData.length;
    callbacks.onLevel?.(Math.min(1, avg / 128));
    levelRaf = requestAnimationFrame(tickLevel);
  };
  levelRaf = requestAnimationFrame(tickLevel);

  // WebSocket to realtime API
  const wsUrl = `${wsBaseUrl}/realtime/v1/speech-llm-toolassist-speech`;
  const ws = new WebSocket(wsUrl);
  ws.binaryType = "arraybuffer";

  let sessionId: string | null = null;
  let detectedLanguage: string | null = null;
  let closed = false;

  const cleanup = () => {
    closed = true;
    caption?.close();
    caption = null;
    cancelAnimationFrame(levelRaf);
    processor.onaudioprocess = null;
    processor.disconnect();
    source.disconnect();
    analyser.disconnect();
    stream.getTracks().forEach((t) => t.stop());
    void audioContext.close();
    if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
      ws.close();
    }
  };

  // Wait for WS open, then send session.start
  await new Promise<void>((resolve, reject) => {
    ws.onopen = () => {
      ws.send(
        JSON.stringify({
          type: "session.start",
          // Auto-detect for multilingual surfaces; pin (sttLanguage) for
          // single-language ones so STT can't mis-tag short words and get the
          // turn dropped by the mismatch gate.
          language: options?.sttLanguage || "auto",
          sample_rate_hz: actualSampleRate,
          encoding: "pcm_s16le",
          call_id: callId,
          customer_id: customerId,
          // STT still auto-detects; this is only used to warn on a mismatch
          // between what the agent selected and what the caller actually speaks.
          ...(expectedLanguage ? { expected_language: expectedLanguage } : {}),
          // "card" selects the agent-initiated credit-card assistant; omitted for telco.
          ...(options?.profile ? { profile: options.profile } : {}),
        })
      );
      resolve();
    };
    ws.onerror = () => {
      cleanup();
      reject(new Error("Realtime voice WebSocket connection failed"));
    };
  });

  // Handle incoming events from the realtime API
  ws.onmessage = (ev) => {
    if (closed) return;
    try {
      const msg = JSON.parse(String(ev.data));
      switch (msg.type) {
        case "session.ready":
          sessionId = msg.session_id ?? null;
          detectedLanguage = msg.language ?? null;
          callbacks.onSessionReady?.(msg.session_id, msg.language);
          break;
        case "speech.started":
          captionTurnId = typeof msg.turn_id === "number" ? msg.turn_id : captionTurnId;
          callbacks.onSpeechStarted?.(msg.turn_id);
          break;
        case "turn.started":
          callbacks.onTurnStarted?.(msg.turn_id);
          break;
        case "transcript.final":
          detectedLanguage = msg.language ?? detectedLanguage;
          callbacks.onTranscript?.(msg.text, msg.language, msg.turn_id);
          // Authoritative transcript arrived: drop the preview and clear the UI's
          // interim so it doesn't linger next to the finalized turn.
          caption?.reset();
          callbacks.onInterimTranscript?.("", msg.turn_id);
          break;
        case "response.text":
          callbacks.onResponseText?.(msg.text, msg.turn_id);
          break;
        case "response.audio":
          callbacks.onResponseAudio?.(
            msg.audio_b64,
            msg.sample_rate_hz,
            msg.final ?? false,
            msg.turn_id
          );
          break;
        case "tool.called":
          callbacks.onToolCalled?.(msg.name, msg.result, msg.turn_id);
          break;
        case "language.mismatch":
          callbacks.onLanguageMismatch?.(msg.expected, msg.detected, msg.turn_id);
          break;
        case "error":
          callbacks.onError?.(msg.code, msg.message);
          break;
      }
    } catch {
      // ignore malformed frames
    }
  };

  ws.onclose = () => {
    if (!closed) {
      cleanup();
      callbacks.onError?.("ws_closed", "Connection lost. Start a new call to reconnect.");
    }
  };

  // Stream mic audio as binary PCM frames
  let micPaused = options?.startMicPaused ?? false;
  processor.onaudioprocess = (event) => {
    if (closed || micPaused || ws.readyState !== WebSocket.OPEN) return;
    const channel = event.inputBuffer.getChannelData(0);
    ws.send(new Uint8Array(floatTo16BitPCM(channel)));
  };

  return {
    endTurn: () => {
      if (!closed && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "audio.end" }));
      }
    },
    synthesize: (text: string, language?: string) => {
      if (!closed && ws.readyState === WebSocket.OPEN && text.trim()) {
        ws.send(JSON.stringify({ type: "synthesize", text, language: language || undefined }));
      }
    },
    pauseMic: () => {
      micPaused = true;
      // Pause the on-device caption in lockstep so it can't transcribe the
      // agent's TTS coming from the speakers (same half-duplex discipline as mic).
      caption?.pause();
    },
    resumeMic: () => {
      micPaused = false;
      caption?.resume();
    },
    close: () => {
      if (!closed) {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ type: "audio.end" }));
          ws.send(JSON.stringify({ type: "session.stop" }));
        }
        cleanup();
      }
    },
    get connected() {
      return !closed && ws.readyState === WebSocket.OPEN;
    },
    get micActive() {
      return !closed && !micPaused;
    },
    get sessionId() {
      return sessionId;
    },
    get detectedLanguage() {
      return detectedLanguage;
    },
  };
}

/**
 * Decode a base64 PCM audio chunk and schedule it for playback.
 *
 * Each response.audio event from the server carries a base64-encoded PCM s16le
 * chunk. This utility decodes it to Float32 and returns the samples + metadata
 * so the caller can queue them into an AudioContext for gapless playback.
 */
export function decodePcmChunk(
  b64: string,
  sampleRate: number
): { samples: Float32Array; sampleRate: number } {
  const raw = atob(b64);
  const bytes = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
  const int16 = new Int16Array(bytes.buffer);
  const float32 = new Float32Array(int16.length);
  for (let i = 0; i < int16.length; i++) float32[i] = int16[i] / 32768.0;
  return { samples: float32, sampleRate };
}

/**
 * Simple audio playback queue for TTS response chunks.
 *
 * Buffers decoded PCM chunks and plays them back gaplessly through the Web
 * Audio API. Handles the streaming nature of the response (chunks arrive over
 * time) and ensures smooth playback without gaps or overlaps.
 */
export class AudioPlaybackQueue {
  private ctx: AudioContext;
  private analyser: AnalyserNode;
  private levelBuf: Uint8Array;
  private nextStartTime: number = 0;
  private playing: boolean = false;
  // Currently scheduled/playing sources, so flush() can stop them on the SAME
  // context instead of tearing the context down (see flush()).
  private sources: Set<AudioBufferSourceNode> = new Set();

  constructor(sampleRate: number = 24_000) {
    this.ctx = new AudioContext({ sampleRate });
    this.analyser = this.makeAnalyser();
    this.levelBuf = new Uint8Array(this.analyser.fftSize);
  }

  private makeAnalyser(): AnalyserNode {
    // Route TTS through an analyser so callers can drive a speaking orb from the
    // ACTUAL audio amplitude (true lip-sync feel) rather than a synthetic wobble.
    const analyser = this.ctx.createAnalyser();
    analyser.fftSize = 1024;
    analyser.smoothingTimeConstant = 0.6;
    analyser.connect(this.ctx.destination);
    return analyser;
  }

  enqueue(samples: Float32Array, sampleRate: number): void {
    // Agent-initiated speech (the card greeting / deep-dive summary) is enqueued
    // from a WS callback, not a user gesture, so the context can be "suspended".
    // Resume it here (idempotent, no-op if already running) — otherwise scheduled
    // sources pile up silently and then all fire at once when the context later
    // resumes, which sounds like badly garbled/overlapping audio.
    if (this.ctx.state === "suspended") void this.ctx.resume();

    const buffer = this.ctx.createBuffer(1, samples.length, sampleRate);
    buffer.copyToChannel(samples as unknown as Float32Array<ArrayBuffer>, 0);
    const source = this.ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(this.analyser);

    const now = this.ctx.currentTime;
    const startAt = Math.max(now, this.nextStartTime);
    source.start(startAt);
    this.nextStartTime = startAt + buffer.duration;
    this.playing = true;
    this.sources.add(source);

    source.onended = () => {
      this.sources.delete(source);
      if (this.ctx.currentTime >= this.nextStartTime - 0.01) {
        this.playing = false;
      }
    };
  }

  /**
   * Current output loudness in [0, 1], derived from the live waveform (RMS with
   * a mild curve so quiet speech still nudges the orb). Returns 0 when idle.
   */
  getLevel(): number {
    if (!this.playing) return 0;
    this.analyser.getByteTimeDomainData(this.levelBuf as unknown as Uint8Array<ArrayBuffer>);
    let sumSq = 0;
    for (let i = 0; i < this.levelBuf.length; i++) {
      const v = (this.levelBuf[i] - 128) / 128; // center silence at 0
      sumSq += v * v;
    }
    const rms = Math.sqrt(sumSq / this.levelBuf.length);
    return Math.min(1, Math.pow(rms * 2.2, 0.8));
  }

  /** Milliseconds until all currently-queued audio finishes playing (0 if idle). */
  msUntilIdle(): number {
    return Math.max(0, (this.nextStartTime - this.ctx.currentTime) * 1000);
  }

  /** Stop all queued audio immediately. */
  flush(): void {
    // Stop the scheduled sources on the EXISTING context rather than closing and
    // recreating it. Recreating an AudioContext outside a user gesture (this is
    // called before the agent's greeting / each deep-dive summary, from a WS
    // callback) yields a "suspended" context, so the freshly-enqueued audio
    // stacks up and later plays all at once — the "totally messed up" garble.
    // Reusing the original (running) context keeps playback clean.
    for (const source of this.sources) {
      source.onended = null;
      try {
        source.stop();
      } catch {
        /* already stopped/ended */
      }
      source.disconnect();
    }
    this.sources.clear();
    this.nextStartTime = 0;
    this.playing = false;
    if (this.ctx.state === "suspended") void this.ctx.resume();
  }

  get isPlaying(): boolean {
    return this.playing;
  }

  /** Resume the audio context. Call from a real user gesture (e.g. tapping the
   *  Genie orb) to unblock playback when the browser suspended the context
   *  because the call auto-started without a prior interaction. */
  resume(): void {
    if (this.ctx.state === "suspended") void this.ctx.resume();
  }

  close(): void {
    void this.ctx.close();
  }
}

/**
 * Speak server-synthesized text through the EXISTING TTS API, into a shared
 * playback queue. Used for the agent's opening greeting and the deep-dive spoken
 * summary — both of which are plain TTS, so they need no change to the STT→LLM→TTS
 * voice engine. Resolves once playback has been fully enqueued.
 */
export function synthesizeToPlayback(
  wsBaseUrl: string,
  text: string,
  language: string | undefined,
  playback: AudioPlaybackQueue,
  opts?: { onStart?: () => void }
): Promise<void> {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(`${wsBaseUrl}/realtime/v1/text-to-speech`);
    ws.binaryType = "arraybuffer";
    let started = false;
    let doneTimer = 0;
    const finish = () => {
      try {
        ws.close();
      } catch {
        /* already closing */
      }
      resolve();
    };
    ws.onopen = () => {
      ws.send(
        JSON.stringify({
          type: "session.start",
          language: language || "en-US",
          sample_rate_hz: 24_000,
          encoding: "pcm_s16le",
        })
      );
    };
    ws.onmessage = (ev) => {
      let msg: Record<string, unknown>;
      try {
        msg = JSON.parse(String(ev.data));
      } catch {
        return;
      }
      switch (msg.type) {
        case "session.ready":
          ws.send(JSON.stringify({ type: "synthesize", text, language: language || undefined }));
          break;
        case "response.audio": {
          if (!started) {
            started = true;
            opts?.onStart?.();
          }
          const { samples } = decodePcmChunk(msg.audio_b64 as string, msg.sample_rate_hz as number);
          playback.enqueue(samples, msg.sample_rate_hz as number);
          if (msg.final) {
            if (doneTimer) window.clearTimeout(doneTimer);
            doneTimer = window.setTimeout(finish, playback.msUntilIdle() + 200);
          }
          break;
        }
        case "error":
          try {
            ws.close();
          } catch {
            /* already closing */
          }
          reject(new Error(String(msg.message ?? "TTS failed")));
          break;
      }
    };
    ws.onerror = () => reject(new Error("TTS connection failed"));
    ws.onclose = () => {
      if (!started) resolve();
    };
  });
}

export interface DeepDiveHandlers {
  /** Server's single-source timeout for this run (ms); use to size the client watchdog. */
  onMeta?: (timeoutMs: number) => void;
  onStep?: (step: "reasoning" | "sql", text: string) => void;
  onReport?: (report: DeepDiveReport) => void;
  /**
   * A chunk of the translated report, streamed into the (initially empty) panel
   * while `localizationPending` so the caller reads their own language from the
   * first token — never a flash of English. Append each delta in order.
   */
  onReportLocalizedDelta?: (delta: string, language: string) => void;
  /**
   * The FULL report text in the caller's language, arriving after the streamed
   * deltas (or as the sole payload if the endpoint can't stream). Only sent when
   * the report announced `localizationPending`; use it as the authoritative final
   * text and clear the pending flag.
   */
  onReportLocalized?: (report: string, language: string) => void;
  onError?: (message: string) => void;
  onDone?: () => void;
}

/**
 * Stream a Genie Agent-mode "why" investigation from the app backend's SSE proxy
 * (`GET /card/deepdive`). This calls the Genie Agent Mode API directly through the
 * proxy — entirely OFF the voice WebSocket — surfacing reasoning steps + SQL live
 * and the final report. Returns a cancel function.
 */
export interface DeepDiveMeta {
  callId?: string | null;
  sessionId?: string | null;
  customerId?: string | null;
  language?: string | null;
}

export function streamDeepDive(
  apiBaseUrl: string,
  question: string,
  useCase: string | null,
  handlers: DeepDiveHandlers,
  meta?: DeepDiveMeta
): () => void {
  const params = new URLSearchParams({ question });
  if (useCase) params.set("use_case", useCase);
  // Link the deep-dive trace to the originating voice call (Trace Explorer).
  if (meta?.callId) params.set("call_id", meta.callId);
  if (meta?.sessionId) params.set("session_id", meta.sessionId);
  if (meta?.customerId) params.set("customer_id", meta.customerId);
  if (meta?.language) params.set("language", meta.language);
  const es = new EventSource(`${apiBaseUrl}/card/deepdive?${params.toString()}`);
  let settled = false;
  const stop = () => {
    settled = true;
    es.close();
  };
  es.onmessage = (e) => {
    let ev: Record<string, unknown>;
    try {
      ev = JSON.parse(e.data);
    } catch {
      return;
    }
    switch (ev.kind) {
      case "meta":
        if (typeof ev.timeout_ms === "number") handlers.onMeta?.(ev.timeout_ms);
        break;
      case "reasoning":
        handlers.onStep?.("reasoning", String(ev.text ?? ""));
        break;
      case "sql":
        handlers.onStep?.("sql", String(ev.sql ?? ""));
        break;
      case "report":
        handlers.onReport?.({
          useCase: (ev.use_case as string) ?? useCase ?? null,
          status: (ev.status as string) ?? "unknown",
          report: (ev.report as string) ?? "",
          spokenSummary: (ev.spoken_summary as string) ?? null,
          tables: (ev.tables as Array<Record<string, unknown>>) ?? [],
          sql: (ev.sql as string[]) ?? [],
          reasoning: (ev.reasoning as string[]) ?? [],
          error: null,
          localizationPending: ev.localization_pending === true,
        });
        break;
      case "report_localized_delta":
        handlers.onReportLocalizedDelta?.(
          String(ev.delta ?? ""),
          String(ev.report_language ?? "")
        );
        break;
      case "report_localized":
        handlers.onReportLocalized?.(
          String(ev.report ?? ""),
          String(ev.report_language ?? "")
        );
        break;
      case "error":
        handlers.onError?.(
          String((ev.error as { message?: string } | undefined)?.message ?? "Investigation failed")
        );
        break;
      case "done":
        stop();
        handlers.onDone?.();
        break;
    }
  };
  es.onerror = () => {
    if (!settled) {
      stop();
      handlers.onError?.("Deep-dive connection lost.");
      handlers.onDone?.();
    }
  };
  return stop;
}
