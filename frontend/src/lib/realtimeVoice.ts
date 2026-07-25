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
}

export interface RealtimeVoiceSession {
  /** Signal end of user speech for the current turn (server VAD also auto-finalizes). */
  endTurn: () => void;
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
  expectedLanguage?: string
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

  const audioContext = new AudioContext({ sampleRate: TARGET_SAMPLE_RATE });
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
          language: "auto",
          sample_rate_hz: TARGET_SAMPLE_RATE,
          encoding: "pcm_s16le",
          call_id: callId,
          customer_id: customerId,
          // STT still auto-detects; this is only used to warn on a mismatch
          // between what the agent selected and what the caller actually speaks.
          ...(expectedLanguage ? { expected_language: expectedLanguage } : {}),
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
          callbacks.onSpeechStarted?.(msg.turn_id);
          break;
        case "turn.started":
          callbacks.onTurnStarted?.(msg.turn_id);
          break;
        case "transcript.final":
          detectedLanguage = msg.language ?? detectedLanguage;
          callbacks.onTranscript?.(msg.text, msg.language, msg.turn_id);
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
  let micPaused = false;
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
    pauseMic: () => {
      micPaused = true;
    },
    resumeMic: () => {
      micPaused = false;
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
  private nextStartTime: number = 0;
  private playing: boolean = false;

  constructor(sampleRate: number = 24_000) {
    this.ctx = new AudioContext({ sampleRate });
  }

  enqueue(samples: Float32Array, sampleRate: number): void {
    const buffer = this.ctx.createBuffer(1, samples.length, sampleRate);
    buffer.copyToChannel(samples as unknown as Float32Array<ArrayBuffer>, 0);
    const source = this.ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(this.ctx.destination);

    const now = this.ctx.currentTime;
    const startAt = Math.max(now, this.nextStartTime);
    source.start(startAt);
    this.nextStartTime = startAt + buffer.duration;
    this.playing = true;

    source.onended = () => {
      if (this.ctx.currentTime >= this.nextStartTime - 0.01) {
        this.playing = false;
      }
    };
  }

  /** Milliseconds until all currently-queued audio finishes playing (0 if idle). */
  msUntilIdle(): number {
    return Math.max(0, (this.nextStartTime - this.ctx.currentTime) * 1000);
  }

  /** Stop all queued audio immediately. */
  flush(): void {
    this.nextStartTime = 0;
    this.playing = false;
    void this.ctx.close();
    this.ctx = new AudioContext({ sampleRate: this.ctx.sampleRate });
  }

  get isPlaying(): boolean {
    return this.playing;
  }

  close(): void {
    void this.ctx.close();
  }
}
