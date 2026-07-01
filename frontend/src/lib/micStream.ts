export type VoiceUiPhase = "idle" | "speaking" | "transcribing" | "agent_reply";

export type VoiceInputSource = "mic" | "text";

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
  onUnavailable?: () => void
): SpeechCaptionSession | null {
  const Ctor =
    (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
  if (!Ctor) {
    onUnavailable?.();
    return null;
  }
  const rec = new Ctor();
  rec.lang = "en-US";
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
  onError: (message: string) => void
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

  const ws = new WebSocket(`${wsUrl}?sample_rate=${Math.round(sampleRate)}`);
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

function preferredMimeType(): string {
  const candidates = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4"];
  return candidates.find((mime) => MediaRecorder.isTypeSupported(mime)) ?? "";
}

export async function startMicRecording(
  onLevel: (level: number) => void
): Promise<MicRecordingSession> {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const audioContext = new AudioContext();
  const source = audioContext.createMediaStreamSource(stream);
  const analyser = audioContext.createAnalyser();
  analyser.fftSize = 256;
  source.connect(analyser);

  const mimeType = preferredMimeType();
  const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
  const chunks: BlobPart[] = [];
  let levelRaf = 0;
  let stopped = false;

  const levelData = new Uint8Array(analyser.frequencyBinCount);
  const tickLevel = () => {
    analyser.getByteFrequencyData(levelData);
    const avg = levelData.reduce((sum, v) => sum + v, 0) / levelData.length;
    onLevel(Math.min(1, avg / 128));
    levelRaf = requestAnimationFrame(tickLevel);
  };

  const cleanup = () => {
    cancelAnimationFrame(levelRaf);
    source.disconnect();
    analyser.disconnect();
    stream.getTracks().forEach((t) => t.stop());
    void audioContext.close();
  };

  recorder.ondataavailable = (event) => {
    if (event.data.size > 0) chunks.push(event.data);
  };
  recorder.start();
  levelRaf = requestAnimationFrame(tickLevel);

  return {
    stop: () =>
      new Promise<{ audioBase64: string; mimeType: string }>((resolve, reject) => {
        if (stopped) {
          reject(new Error("Mic recording already stopped"));
          return;
        }
        stopped = true;
        recorder.onerror = () => {
          cleanup();
          reject(new Error("Mic recording failed"));
        };
        recorder.onstop = () => {
          cleanup();
          const blob = new Blob(chunks, { type: recorder.mimeType || mimeType || "audio/webm" });
          blobToBase64(blob)
            .then((audioBase64) =>
              resolve({ audioBase64, mimeType: blob.type || "audio/webm" })
            )
            .catch(reject);
        };
        recorder.stop();
      }),
    close: () => {
      if (!stopped && recorder.state !== "inactive") {
        stopped = true;
        recorder.stop();
      }
      cleanup();
    },
  };
}
