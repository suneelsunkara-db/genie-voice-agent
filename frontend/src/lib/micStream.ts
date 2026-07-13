import type { InteractionLanguage } from "../api/client";

export type VoiceUiPhase = "idle" | "speaking" | "transcribing" | "agent_reply";

export type VoiceInputSource = "mic" | "text";
export type SpeechRecognitionLanguage = "en-US" | "th-TH" | "id-ID" | "zh-CN";

export function speechRecognitionLanguage(language: InteractionLanguage): SpeechRecognitionLanguage {
  if (language.startsWith("zh-CN")) return "zh-CN";
  return language as SpeechRecognitionLanguage;
}

export interface VoiceUiState {
  phase: VoiceUiPhase;
  source?: VoiceInputSource;
  interimText?: string;
  processingLabel?: string;
  micLevel?: number;
}

export interface MicStreamSession {
  stop: () => Promise<string>;
  close: () => void;
}

export interface MicRecordingSession {
  stop: () => Promise<{ audioBase64: string; mimeType: string }>;
  close: () => void;
}

export interface SpeechCaptionSession {
  stop: () => void;
  close: () => void;
}

/** True when the browser exposes the Web Speech API used for the live caption. */
export function isSpeechCaptionSupported(): boolean {
  return (
    typeof window !== "undefined" &&
    !!((window as any).SpeechRecognition || (window as any).webkitSpeechRecognition)
  );
}

/**
 * Best-effort, on-device live caption using the browser's Web Speech API.
 *
 * The Databricks Whisper serving endpoint is batch (it transcribes a whole clip
 * on stop), so it cannot stream words while you speak. To let an audience read
 * along, we run the browser's built-in recognizer purely for a live preview; the
 * authoritative transcript still comes from the Databricks model once the clip is
 * sent. Returns null when the browser has no Web Speech support (caption is then
 * simply skipped - recording/transcription are unaffected).
 */
export function startSpeechCaption(
  onText: (text: string) => void,
  onUnavailable?: () => void,
  language: SpeechRecognitionLanguage = "en-US"
): SpeechCaptionSession | null {
  const Ctor =
    (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
  if (!Ctor) {
    onUnavailable?.();
    return null;
  }
  const rec = new Ctor();
  rec.lang = language;
  rec.continuous = true;
  rec.interimResults = true;

  let finalText = "";
  let stopped = false;

  rec.onresult = (event: any) => {
    let interim = "";
    for (let i = event.resultIndex; i < event.results.length; i += 1) {
      const res = event.results[i];
      const chunk = String(res[0]?.transcript ?? "");
      if (res.isFinal) finalText = (finalText ? `${finalText} ` : "") + chunk.trim();
      else interim += chunk;
    }
    const display = `${finalText} ${interim}`.trim();
    if (display) onText(display);
  };
  // The recognizer ends itself on a pause; restart so the caption keeps flowing
  // for the whole utterance until we explicitly stop it.
  rec.onend = () => {
    if (!stopped) {
      try {
        rec.start();
      } catch {
        /* already starting */
      }
    }
  };
  rec.onerror = () => {
    /* caption is best-effort; ignore (e.g. no-speech / network) */
  };

  try {
    rec.start();
  } catch {
    onUnavailable?.();
    return null;
  }

  return {
    stop: () => {
      stopped = true;
      try {
        rec.stop();
      } catch {
        /* noop */
      }
    },
    close: () => {
      stopped = true;
      try {
        rec.abort();
      } catch {
        /* noop */
      }
    },
  };
}

/** Deepgram streaming returns one segment per is_final; accumulate for long speech. */
export function mergeStreamingTranscript(
  committed: string,
  interim: string,
  text: string,
  isFinal: boolean
): { committed: string; interim: string; display: string } {
  const chunk = text.trim();
  if (!chunk) {
    const display = committed + (interim ? (committed ? " " : "") + interim : "");
    return { committed, interim, display: display.trim() };
  }
  if (isFinal) {
    const nextCommitted = committed ? `${committed} ${chunk}` : chunk;
    return { committed: nextCommitted, interim: "", display: nextCommitted };
  }
  const nextInterim = chunk;
  const display = committed
    ? `${committed} ${nextInterim}`
    : nextInterim;
  return { committed, interim: nextInterim, display: display.trim() };
}

function floatTo16BitPCM(input: Float32Array): ArrayBuffer {
  const output = new Int16Array(input.length);
  for (let i = 0; i < input.length; i += 1) {
    const s = Math.max(-1, Math.min(1, input[i]));
    output[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return output.buffer;
}

export async function startMicStream(
  wsUrl: string,
  onTranscript: (text: string) => void,
  onLevel: (level: number) => void,
  onError: (message: string) => void,
  language: SpeechRecognitionLanguage = "en-US"
): Promise<MicStreamSession> {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const audioContext = new AudioContext();
  const sampleRate = audioContext.sampleRate;
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

  const url = new URL(wsUrl, window.location.href);
  url.searchParams.set("sample_rate", String(Math.round(sampleRate)));
  url.searchParams.set("language", language);
  const ws = new WebSocket(url.toString());
  ws.binaryType = "arraybuffer";

  let committed = "";
  let interim = "";
  let ended = false;
  let levelRaf = 0;
  let resolveStop: ((text: string) => void) | null = null;

  const fullTranscript = () => {
    const tail = interim.trim();
    const head = committed.trim();
    if (!head) return tail;
    if (!tail) return head;
    return `${head} ${tail}`;
  };

  const levelData = new Uint8Array(analyser.frequencyBinCount);
  const tickLevel = () => {
    analyser.getByteFrequencyData(levelData);
    const avg = levelData.reduce((sum, v) => sum + v, 0) / levelData.length;
    onLevel(Math.min(1, avg / 128));
    levelRaf = requestAnimationFrame(tickLevel);
  };
  levelRaf = requestAnimationFrame(tickLevel);

  const cleanupAudio = () => {
    cancelAnimationFrame(levelRaf);
    processor.disconnect();
    source.disconnect();
    analyser.disconnect();
    stream.getTracks().forEach((t) => t.stop());
    void audioContext.close();
  };

  const waitOpen = new Promise<void>((resolve, reject) => {
    ws.onopen = () => resolve();
    ws.onerror = () => reject(new Error("Mic stream WebSocket failed"));
  });

  ws.onmessage = (ev) => {
    try {
      const msg = JSON.parse(String(ev.data));
      if (msg.type === "transcript") {
        const merged = mergeStreamingTranscript(
          committed,
          interim,
          String(msg.transcript || ""),
          Boolean(msg.is_final)
        );
        committed = merged.committed;
        interim = merged.interim;
        if (merged.display) onTranscript(merged.display);
      } else if (msg.type === "error") {
        onError(String(msg.message || "Deepgram stream error"));
      } else if (msg.type === "stream_end" && resolveStop) {
        ended = true;
        resolveStop(fullTranscript());
        resolveStop = null;
      }
    } catch {
      // ignore malformed frames
    }
  };

  processor.onaudioprocess = (event) => {
    if (ws.readyState !== WebSocket.OPEN || ended) return;
    const channel = event.inputBuffer.getChannelData(0);
    ws.send(floatTo16BitPCM(channel));
  };

  await waitOpen;

  return {
    stop: () =>
      new Promise<string>((resolve) => {
        resolveStop = resolve;
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ type: "stop" }));
        } else {
          cleanupAudio();
          resolve(fullTranscript());
          return;
        }
        window.setTimeout(() => {
          if (!resolveStop) return;
          ended = true;
          resolveStop(fullTranscript());
          resolveStop = null;
          cleanupAudio();
          if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) ws.close();
        }, 8000);
      }).then((text) => {
        cleanupAudio();
        if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) ws.close();
        return text;
      }),
    close: () => {
      ended = true;
      cleanupAudio();
      if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) ws.close();
      if (resolveStop) {
        resolveStop(fullTranscript());
        resolveStop = null;
      }
    },
  };
}

function blobToBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(reader.error ?? new Error("Failed to read audio blob"));
    reader.onload = () => {
      const result = String(reader.result || "");
      resolve(result.includes(",") ? result.split(",")[1] : result);
    };
    reader.readAsDataURL(blob);
  });
}

/** Target rate for the Databricks ASR endpoints (SenseVoice/Paraformer/Qwen3/Whisper). */
const ASR_TARGET_SAMPLE_RATE = 16000;

function concatFloat32(chunks: Float32Array[]): Float32Array {
  const total = chunks.reduce((n, a) => n + a.length, 0);
  const merged = new Float32Array(total);
  let offset = 0;
  for (const a of chunks) {
    merged.set(a, offset);
    offset += a.length;
  }
  return merged;
}

/**
 * High-quality resample to 16 kHz using the browser's native (anti-aliased)
 * resampler via OfflineAudioContext. Naive linear decimation aliases high-
 * frequency consonant energy (sibilants) into the speech band and wrecks ASR,
 * so we never hand-roll the downsample.
 */
async function resampleTo16k(samples: Float32Array, inputRate: number): Promise<Float32Array> {
  if (inputRate === ASR_TARGET_SAMPLE_RATE) return samples;
  const outLength = Math.max(1, Math.round((samples.length * ASR_TARGET_SAMPLE_RATE) / inputRate));
  const OfflineCtx =
    (window as any).OfflineAudioContext || (window as any).webkitOfflineAudioContext;
  const offline = new OfflineCtx(1, outLength, ASR_TARGET_SAMPLE_RATE);
  const srcBuffer = offline.createBuffer(1, samples.length, inputRate);
  srcBuffer.copyToChannel(samples, 0);
  const src = offline.createBufferSource();
  src.buffer = srcBuffer;
  src.connect(offline.destination);
  src.start(0);
  const rendered = await offline.startRendering();
  return rendered.getChannelData(0).slice();
}

/** Encode mono Float32 PCM as a 16-bit little-endian WAV blob. */
function encodeWavPcm16(samples: Float32Array, sampleRate: number): Blob {
  const pcm = new Int16Array(floatTo16BitPCM(samples));
  const buffer = new ArrayBuffer(44 + pcm.length * 2);
  const view = new DataView(buffer);
  const writeStr = (offset: number, str: string) => {
    for (let i = 0; i < str.length; i += 1) view.setUint8(offset + i, str.charCodeAt(i));
  };
  const dataBytes = pcm.length * 2;
  writeStr(0, "RIFF");
  view.setUint32(4, 36 + dataBytes, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true); // PCM chunk size
  view.setUint16(20, 1, true); // audio format = PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); // byte rate
  view.setUint16(32, 2, true); // block align
  view.setUint16(34, 16, true); // bits per sample
  writeStr(36, "data");
  view.setUint32(40, dataBytes, true);
  new Int16Array(buffer, 44).set(pcm);
  return new Blob([buffer], { type: "audio/wav" });
}

/**
 * Capture mic audio and produce a 16 kHz mono PCM WAV client-side.
 *
 * We deliberately avoid MediaRecorder/WebM-Opus: the Databricks ASR serving
 * containers decode via torchaudio/ffmpeg, and MediaRecorder's headerless,
 * 48 kHz Opus stream was being mis-decoded (long prompts collapsed to a few
 * garbled characters). Sending the same 16 kHz mono WAV the endpoints already
 * transcribe correctly (FLEURS holdout) removes that decode/resample variable.
 * Resampling runs incrementally so stop-time latency stays ~0.
 */
export async function startMicRecording(
  onLevel: (level: number) => void
): Promise<MicRecordingSession> {
  // Disable the browser's voice-processing: echo cancellation in particular
  // treats any audio coming from the speakers as "echo" and cancels it, which
  // wrecks capture when the source is speaker playback (e.g. TTS from another
  // window) — the mic then records fragmented near-silence. Noise suppression
  // and auto-gain also distort speech for ASR, so we send raw audio.
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      echoCancellation: false,
      noiseSuppression: false,
      autoGainControl: false,
    },
  });
  // Capture at the mic's NATIVE rate. Do NOT force the live AudioContext to
  // 16 kHz: when the browser honors that hint it must resample the 48 kHz mic
  // stream inside the live graph feeding a main-thread ScriptProcessor, which
  // glitches and drops buffers (captured audio comes back fragmented, ~silence
  // with sporadic bursts, and ASR returns garbage). We downsample the complete
  // signal to 16 kHz offline (anti-aliased) at stop() instead.
  const audioContext = new AudioContext();
  const inputRate = audioContext.sampleRate;
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

  const captured: Float32Array[] = [];
  let levelRaf = 0;
  let stopped = false;

  processor.onaudioprocess = (event) => {
    if (stopped) return;
    // Copy: the underlying buffer is reused by the audio thread.
    captured.push(Float32Array.from(event.inputBuffer.getChannelData(0)));
  };

  const levelData = new Uint8Array(analyser.frequencyBinCount);
  const tickLevel = () => {
    analyser.getByteFrequencyData(levelData);
    const avg = levelData.reduce((sum, v) => sum + v, 0) / levelData.length;
    onLevel(Math.min(1, avg / 128));
    levelRaf = requestAnimationFrame(tickLevel);
  };

  const cleanup = () => {
    cancelAnimationFrame(levelRaf);
    processor.onaudioprocess = null;
    processor.disconnect();
    source.disconnect();
    analyser.disconnect();
    stream.getTracks().forEach((t) => t.stop());
    void audioContext.close();
  };

  levelRaf = requestAnimationFrame(tickLevel);

  return {
    stop: () =>
      new Promise<{ audioBase64: string; mimeType: string }>((resolve, reject) => {
        if (stopped) {
          reject(new Error("Mic recording already stopped"));
          return;
        }
        stopped = true;
        const raw = concatFloat32(captured);
        cleanup();
        if (!raw.length) {
          reject(new Error("No audio captured"));
          return;
        }
        resampleTo16k(raw, inputRate)
          .then((samples) => {
            const blob = encodeWavPcm16(samples, ASR_TARGET_SAMPLE_RATE);
            return blobToBase64(blob);
          })
          .then((audioBase64) => resolve({ audioBase64, mimeType: "audio/wav" }))
          .catch(reject);
      }),
    close: () => {
      stopped = true;
      cleanup();
    },
  };
}
