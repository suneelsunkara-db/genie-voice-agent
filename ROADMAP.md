# Roadmap

Future and optional work. Shipped functionality is documented in
[`README.md`](README.md). Items here are **not** missing core features — they are
hardening, validation, and optional integrations.

Status legend: 🟢 done · 🟡 in progress · ⚪ planned · 🔵 optional / conditional

---

## Realtime Voice API

### Shipped 🟢

- Standalone FastAPI WebSocket API (`realtime_api/`), decoupled from the
  contact-center app and from the UI.
- STT → LLM → TTS loop over Databricks Model Serving: **Qwen3-ASR** (STT),
  **Qwen3-Next** (LLM, temperature + tool calling), **VoxCPM2** (TTS).
- Streaming TTS (`predict_stream`) — ~3.5–4.5× faster time-to-first-audio vs
  full-sentence synthesis.
- Server-side VAD/endpointing, turn IDs + stale-result suppression, opt-in barge-in.
- Auto language detection; end-to-end language set computed as STT ∩ TTS (currently
  **24**) and exposed at `GET /v1/languages`.
- Verification mode (echo transcript + detected language).
- Per-endpoint latency (`stt_ms`, `llm_ms`, `tts_first_ms`) surfaced to the UI.
- Browser test UI (`realtime_test_ui/`) with client + server latency panels and the
  supported-languages line on page load.

### Planned ⚪

- **Multilingual validation sweep** — only 4 of 24 languages (en, th, id, zh) are
  round-trip validated end to end; the rest are model-*claimed*. Run a TTS→STT
  round-trip across all 24 before promising them to users.
- **Test coverage gaps** — add WebSocket tests for VAD/endpointing, barge-in, turn
  cancellation, malformed frames, and endpoint-failure handling; add tests for the
  per-stage timing events (`stt_ms`/`llm_ms`/`tts_first_ms`) and `GET /v1/languages`.
- **Endpoint smoke + GPU placement** — confirm the STT/TTS agent endpoints run on the
  expected GPU, and measure cold-start vs warm latency.
- **Concurrency / load validation** — verify behaviour under multiple simultaneous
  WebSocket sessions (turn isolation, latency degradation, endpoint throughput).
- **Latency** — streaming TTS time-to-first-audio floor is ~1.5–1.7s on GPU_MEDIUM.
  Lower requires a larger GPU or a faster TTS. No formal **p95** measured yet.
- **p95 latency harness** — a repeatable script that drives representative turns and
  reports p50/p95 per stage and end to end, to check against the realtime budget.
- **Observability** — request IDs, model version, and selected route in structured
  logs (turn timings and errors already logged).

### Housekeeping 🟢

- Consolidated all component READMEs into the root [`README.md`](README.md); retired
  the stale `realtime_api/TASKS.md` in favour of this roadmap.

### Promotion gates ⚪

- p95 latency within the agreed realtime budget.
- Human multilingual quality review.
- Shadow-traffic review before route activation.
- Confirmed GPU inference for GPU endpoints.

> Note: per-language endpoint routing is intentionally **not** planned — the design
> uses single **multilingual** STT/TTS endpoints with auto-detect, so no routing
> table is required.

---

## Telephony codec conversion 🔵 (optional — only for phone/PSTN calls)

**Status: not implemented, by design.** The Realtime Voice API today accepts only
`pcm_s16le` (raw 16-bit linear PCM) — exactly what a browser's Web Audio API
produces. Telephony codec conversion is only needed if the assistant answers **actual
phone calls** (call-center / IVR / contact-center over SIP/PSTN), not browser mics.

### What it is

Phone/VoIP audio does not arrive as linear PCM. It arrives as:

- **G.711 μ-law / A-law** — classic PSTN codec, 8-bit companded samples at **8 kHz**
  (what a SIP trunk from Twilio / Genesys / Asterisk sends).
- **Opus / G.722** — wideband codecs used by WebRTC and modern VoIP, in RTP packets.

A telephony ingress layer would need to:

1. **Decode** the codec (e.g. μ-law byte → 16-bit linear sample).
2. **Resample** 8 kHz → 16 kHz for Qwen3-ASR.
3. On the return path, **resample** TTS output (VoxCPM2 emits 48 kHz) down to 8 kHz
   and **re-encode** to μ-law for playback down the line.
4. Handle **RTP framing / jitter buffering / packet loss** for live SIP/RTP streams.

### Current vs. telephony

| | Browser (now) | Telephony (future) |
|---|---|---|
| Codec | Linear PCM (`pcm_s16le`) | G.711 μ-law/A-law, Opus |
| Sample rate in | 16 kHz | 8 kHz |
| Transport | WebSocket, raw bytes | SIP/RTP (or Twilio Media Streams over WS) |
| Conversion needed | None | Decode + resample both directions |

### Approach when needed

Add a small transcoding shim **in front of** the existing WebSocket (a telephony
ingress adapter), so the core pipeline keeps receiving 16 kHz PCM unchanged:

- Terminate the telephony transport (e.g. Twilio Media Streams over WS, or a
  SIP/RTP gateway) and decode μ-law@8k → linear PCM.
- Upsample to 16 kHz before the STT stage; downsample + μ-law-encode the TTS audio.
- Keep the protocol/contracts of `WS /v1/realtime/voice` untouched.

This keeps telephony an **edge concern** rather than leaking codec logic into the
STT/LLM/TTS orchestration.

---

## ML / ASR pipeline ⚪

- Migrate EN finetuned Whisper registration off the legacy `scripts/asr/05_register*`
  bridge into `scripts/ml_asr/` (see README → ML ASR pipeline).
- Decommission the legacy EN-centric bake-off (`scripts/asr/`) once multilingual
  coverage fully supersedes it.
