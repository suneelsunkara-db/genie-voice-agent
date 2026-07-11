# Voice Telephony Integration — Risks & ASR Model Evaluation

**Status:** Evaluation / decision doc
**Scope:** What it takes to make `genie-voice-agent` place or receive a real phone
call (PSTN) and hold a spoken conversation that drives the existing `/assist`
resolution + billing flow.
**Companion docs:** `ASR_MODEL_BUILD_SPEC.md` · `ARCHITECTURE.md` · `PRD.md`

> TL;DR: the architecture (SIP trunk → PBX/orchestrator → audio bridge → STT →
> `/assist` → TTS) is sound, but **no single layer is a drop-in**. Five concrete
> risks below must each be adapted + tested. The **highest-severity** risk is
> ASR quality on **8 kHz telephony audio** — and the current multilingual model
> routing (Qwen3-ASR for `id`/`zh`) is **known to fail on telephony-band audio**.

---

## 1. The 5 real risks

### Risk 1 — ASR does not match phone audio (8 kHz telephony band)  🔴 High

**Problem.** Phone calls deliver **G.711 mono @ 8 kHz**. The current STT models
are trained for **16 kHz** wideband audio. Two failure modes:

- **Whisper family** (`en`, `th` routes): pretrained on 16 kHz. Raw 8 kHz causes
  an acoustic mismatch → substantial WER degradation. Must **upsample 8 kHz → 16
  kHz** *and* ideally fine-tune on telephony-band audio.
- **LLM-based ASR** (`id`, `zh` routes use **Qwen3-ASR**): newer models like
  Qwen3-ASR and Parakeet **frequently fail or produce catastrophic errors on 8
  kHz telephony**. `faster-whisper` is reported as the *only* family robust with
  G.711/G.729 codecs.

**Evidence.**
- Whisper on telephony without adaptation is poor: base Whisper-Small scored
  **WER 0.888** on Swedish telephony vs **0.170** after telephony fine-tuning
  ([WMRNORDIC](https://huggingface.co/WMRNORDIC/whisper-swedish-telephonic)).
- Multilingual models "degrade to a prohibitively high 40.94% WER when exposed to
  real-world telephony audio" ([arXiv 2512.16401](https://arxiv.org/html/2512.16401v3)).
- Telephony 8 kHz robustness by family: **faster-whisper ✅ Works; Cohere
  Transcribe ❌ Fails; Parakeet TDT ❌ Fails; Qwen3-ASR ❌ Unusable (catastrophic
  errors)** ([STT telephony evaluation](https://github.com/alfonsodg/concurrent-faster-qwen3-server/blob/main/docs/TTS_STT_EVALUATION.md)).

**Impact on this app.** Wrong transcripts break the business-critical entities the
whole pipeline depends on: invoice IDs, dollar amounts, waiver/payment-plan
phrases, and confirmation ("yes, go ahead") — which gate resolution + billing
close.

**Mitigation.** See Section 3 (model recommendations). Minimum: upsample to 16
kHz, switch `id`/`zh` off Qwen3-ASR for telephony, fine-tune Whisper on
telephony-band data (extends the existing LoRA pipeline in `ASR_MODEL_BUILD_SPEC`).

---

### Risk 2 — TTS output format is not phone-playable  🟠 Medium-High

**Problem.** `ElevenLabsTTS.synthesize()` returns **`mp3_44100_128`** (MP3 @ 44.1
kHz). Telephony needs **8 kHz PCM / µ-law**. Also the adapter is:
- gated behind `get_settings().is_live` **and** the `ELEVENLABS_API_KEY` env var, and
- **not wired to any API route** (nothing ever calls it).

**Impact.** Even with a working call, the agent cannot be *heard* until TTS audio
is transcoded to the telephony format and exposed via an endpoint/bridge.

**Mitigation.** Request PCM output, transcode to 8 kHz µ-law in the bridge, and
wire `synthesize()` into the call path. (Same wiring is needed for a browser
voice agent, minus the 8 kHz transcode.)

---

### Risk 3 — `/assist` assumes call state already exists  🟠 Medium

**Problem.** `post_assist()` reads `serving().get_call_state(call_id)` and pulls
`customer_id` from it. A phone call has **no pre-seeded `call_id` / `customer_id`**,
so account facts resolve to `None` and the reply degrades (no validated metrics,
no billing close).

**Impact.** Without seeding, the agent cannot ground on the caller's overdue
invoice / late fee — defeating the demo.

**Mitigation.** On call start: create Lakebase call state, map the dialed/calling
number → `customer_id` / `call_id`, then call `POST /genie-insight` to warm
context — before the first utterance.

---

### Risk 4 — External PBX/VM → Databricks authentication  🟠 Medium

**Problem.** The app authenticates as the user (U2M, local) or the app service
principal (on Databricks Apps). A **standalone PBX/VM** (Asterisk/FreeSWITCH +
bridge) calling FM + Lakebase + Genie needs **its own OAuth/PAT identity and
grants** (UC + Lakebase + Genie), which is not set up today.

**Mitigation.** Provision a service principal (or PAT) for the bridge host and
grant it the same access the app SP has (`infra/apps/grant_app_sp.py` pattern).

---

### Risk 5 — End-to-end per-turn latency  🟡 Medium-Low

**Problem.** Per customer turn: (batch) STT + FM enrich + FM reply + optional
Genie validation + TTS + PSTN leg. This can total **several seconds**.

**Impact.** Acceptable for a push-to-talk demo; awkward for natural, interruptible
conversation over a phone.

**Mitigation.** Stream where possible (streaming STT, sentence-wise TTS), keep
Genie off the per-utterance path (already the design), and measure before
promising real-time UX.

---

## 2. Does Whisper match phone audio? — Direct answer

**No — not out of the box.** Whisper (and the current LoRA/`pathumma` Whisper
routes) is a **16 kHz wideband** model. On 8 kHz telephony it needs, in order:

1. **Upsample 8 kHz → 16 kHz** (mandatory — Whisper's front-end expects 16 kHz).
2. **Telephony-band fine-tuning** to recover accuracy lost to codec/band limits.

The current `ASR_MODEL_BUILD_SPEC.md` targets **16 kHz, mic-quality, push-to-talk
utterances** — it explicitly does not cover telephony conditions. So the existing
fine-tune is not validated for phone audio.

**Worse:** the `id-ID` and `zh-CN` routes use **Qwen3-ASR**, which is reported
**unusable on 8 kHz telephony** (catastrophic errors). This is the single most
important config-level risk for a multilingual phone deployment.

### Current model routing vs telephony fitness

| Language | Current endpoint (config.yaml) | Family | 8 kHz telephony fitness |
|---|---|---|---|
| `en-US` | `voice_asr_en_finetuned_whisper_lora` | Whisper (LoRA) | ⚠️ Needs upsample + telephony fine-tune |
| `th-TH` | `..._th_oss_pathumma_whisper_large_v3` | Whisper large-v3 | ⚠️ Needs upsample + telephony fine-tune |
| `id-ID` | `..._id_oss_qwen3_asr_0_6b` | Qwen3-ASR | 🔴 Known to fail on 8 kHz |
| `zh-CN` | `..._zh_oss_qwen3_asr_0_6b` | Qwen3-ASR | 🔴 Known to fail on 8 kHz |

---

## 3. Better models for multilingual + telephony

Your target languages are **English, Thai, Indonesian, Chinese** — this rules out
some otherwise-strong options.

### Option A (recommended, self-hosted OSS): telephony-tuned faster-whisper

- **`faster-whisper large-v3` / `large-v3-turbo`** — the **only** open family
  reported robust with G.711/G.729 (8 kHz) codecs, and covers **99 languages**
  incl. Thai/Indonesian/Chinese ([telephony eval](https://github.com/alfonsodg/concurrent-faster-qwen3-server/blob/main/docs/TTS_STT_EVALUATION.md)).
- Apply your **existing LoRA pipeline**, but train/augment on **8 kHz-upsampled,
  telephony-band** audio (add codec + band-limit augmentation).
- Keeps one model family across all four languages → simpler ops.

### Option B (fastest to a working call): Deepgram Nova-3

- Already integrated (`providers/stt/deepgram.py`), **telephony-grade / native 8
  kHz**, diarization, redaction, multilingual incl. Thai/Indonesian/Chinese.
- Not OSS and not "on Databricks," but removes Risk 1 immediately and is the
  documented `M6` live-vendor path in `PRD.md`.
- Pragmatic recommendation: **use Deepgram for the phone leg first**, migrate to
  telephony-tuned Whisper once quality is proven.

### Not recommended for this app's phone path

| Model | Why not (for your case) |
|---|---|
| **Qwen3-ASR 0.6B/1.7B** (current `id`/`zh`) | Catastrophic errors on 8 kHz telephony |
| **Parakeet TDT v3** | Fails on 8 kHz; also English/European focus, weak on Thai/Indonesian/Chinese |
| **Canary-1B-v2** | Strong multilingual but **25 European languages only** — no Thai/Indonesian/Chinese; telephony behavior unproven |
| **Cohere Transcribe 2B** | Fails on 8 kHz telephony |

### Recommendation summary

| Priority | Choice |
|---|---|
| Quickest reliable phone demo | **Deepgram Nova-3** for all four languages |
| OSS / on-Databricks, telephony-robust | **faster-whisper large-v3(-turbo)**, LoRA-fine-tuned on 8 kHz-upsampled telephony audio |
| Immediate config fix regardless | **Move `id`/`zh` off Qwen3-ASR** for any telephony path |

---

## 3b. Deep research — per-language OSS models (recommended routing)

There is **no single OSS model** that is best across English + Thai + Indonesian +
Chinese *and* telephony. The strongest approach is **best-of-breed per language**,
which your existing `providers.stt.options.databricks.routes` block already
supports. All 16 kHz models still require **8 kHz→16 kHz upsampling** on the phone
path; Paraformer offers native 8 kHz telephony variants.

### English (`en-US`)

| Model | License | Notes |
|---|---|---|
| **faster-whisper large-v3 / -turbo**, telephony-LoRA | MIT | Only OSS family robust on G.711 8 kHz; reuse your LoRA pipeline with telephony augmentation |
| Deepgram Nova-3 (non-OSS) | — | Fastest reliable fallback for the phone leg |
| ~~Parakeet TDT~~ | — | ❌ fails on 8 kHz telephony |

### Thai (`th-TH`) — upgrade from current `pathumma` route

| Model | License | Notes |
|---|---|---|
| **Typhoon ASR Real-Time** (scb-10x) | Apache/MIT | **Streaming** FastConformer-Transducer, CER ~0.098, ~4097× RT — best fit for live phone; weaker on Thai-English code-switching |
| **Typhoon Whisper Large-v3 / Turbo** | MIT | SOTA Thai offline (utterance), ~10–11k hrs Thai |
| **Thonburian Whisper large-v3** | MIT | WER 6.59%, explicitly robust on **financial-domain** + noisy audio (fits billing) |

Sources: [Typhoon ASR Real-Time](https://opentyphoon.ai/blog/en/typhoon-asr-realtime-release), [Typhoon Whisper](https://huggingface.co/typhoon-ai/typhoon-whisper-large-v3), [Thonburian](https://huggingface.co/biodatlab/whisper-th-large-v3-combined).

### Indonesian (`id-ID`) — 🔴 replace Qwen3-ASR

| Model | License | Notes |
|---|---|---|
| **cahya/whisper-large-id** | MIT | WER ~6.25% (CommonVoice); Whisper family → 8 kHz-tunable |
| **cahya/whisper-medium-id** | MIT | WER ~3.83% CV / 9.74% FLEURS; lighter |
| **Bagus/whisper-small-id** | Apache 2.0 | 5.9% WER, ONNX (edge/browser) |

Sources: [cahya/whisper-large-id](https://huggingface.co/cahya/whisper-large-id), [cahya/whisper-medium-id](https://huggingface.co/cahya/whisper-medium-id), [Indonesian ASR study](https://arxiv.org/html/2410.08828v1).

### Chinese (`zh-CN`) — 🔴 replace Qwen3-ASR

| Model | License | Notes |
|---|---|---|
| **FunASR SenseVoice-Small** | MIT | Chinese CER **7.81%** vs Whisper-large-v3 ~20%; non-autoregressive, **built-in FSMN-VAD**, 170× RT GPU / 17× CPU |
| **FunASR Paraformer** (incl. **8 kHz telephony variant**) | MIT | Streaming + native **8 kHz** models → strongest telephony fit for Chinese |

Sources: [FunASR](https://github.com/modelscope/FunASR), [FunASR vs Whisper benchmark](https://www.funasr.com/en/blog/funasr-vs-whisper-benchmark.html), [SenseVoice](https://github.com/FunAudioLLM/SenseVoice).

### Why not "one multilingual model"?

- **SenseVoice** covers only **zh/en/ja/ko/yue** — **no Thai, no Indonesian**.
- **Canary-1B-v2** covers **European languages only** — none of your SEA/CJK set.
- **Qwen3-ASR** covers 52 languages but is **unusable on 8 kHz telephony**.
- **Whisper** covers 99 languages but needs telephony fine-tuning per language.

→ Route per language; keep one *toolkit* per family where possible (Whisper for
en/th/id, FunASR for zh) to limit serving sprawl.

### Recommended telephony routing (replaces current `routes:`)

| Lang | Current | Recommended (telephony) |
|---|---|---|
| `en-US` | Whisper LoRA | faster-whisper large-v3 + telephony LoRA (or Deepgram) |
| `th-TH` | Pathumma Whisper | Typhoon ASR Real-Time (live) / Typhoon or Thonburian Whisper (utterance) |
| `id-ID` | **Qwen3-ASR 🔴** | cahya/whisper-large-id |
| `zh-CN` | **Qwen3-ASR 🔴** | FunASR SenseVoice-Small or Paraformer-8k |

---

## 4. Decision checklist before building the phone path

- [ ] Pick STT for telephony: Deepgram (fast) or telephony-tuned faster-whisper (OSS).
- [ ] Add 8 kHz→16 kHz upsampling in the audio bridge (if Whisper family).
- [ ] Replace Qwen3-ASR routes for `id`/`zh` on the phone path.
- [ ] Extend `ASR_MODEL_BUILD_SPEC` gold set with telephony-band clips per language.
- [ ] Request PCM from ElevenLabs + transcode to 8 kHz µ-law; wire `synthesize()` to a route.
- [ ] Seed Lakebase call state + phone→customer mapping on call start.
- [ ] Provision bridge-host identity + UC/Lakebase/Genie grants.
- [ ] Measure per-turn latency on a real call before claiming real-time UX.

---

## 5. Sources

- Whisper telephony degradation & fine-tuning: [WMRNORDIC whisper-swedish-telephonic](https://huggingface.co/WMRNORDIC/whisper-swedish-telephonic), [Diabolocom ASR fine-tuning](https://www.diabolocom.com/research/fine-tuning-asr-focus-on-whisper/), [arXiv 2512.16401](https://arxiv.org/html/2512.16401v3), [startelelogic whisper-medium-ccaas](https://huggingface.co/startelelogic/whisper-medium-ccaas)
- Telephony 8 kHz robustness by model family: [STT/TTS evaluation](https://github.com/alfonsodg/concurrent-faster-qwen3-server/blob/main/docs/TTS_STT_EVALUATION.md)
- Multilingual ASR benchmarks: [Canary-1B-v2 & Parakeet](https://arxiv.org/html/2509.14128v2), [Qwen3-ASR Technical Report](https://arxiv.org/pdf/2601.21337), [Soniqo benchmarks](https://soniqo.audio/benchmarks)
- Thai OSS ASR: [Typhoon ASR Real-Time](https://opentyphoon.ai/blog/en/typhoon-asr-realtime-release), [Typhoon Whisper Large-v3](https://huggingface.co/typhoon-ai/typhoon-whisper-large-v3), [Thonburian Whisper](https://github.com/biodatlab/thonburian-whisper)
- Indonesian OSS ASR: [cahya/whisper-large-id](https://huggingface.co/cahya/whisper-large-id), [cahya/whisper-medium-id](https://huggingface.co/cahya/whisper-medium-id), [Bagus whisper-small-id](https://huggingface.co/cmaree/Bagus-whisper-small-id-onnx), [Indonesian ASR study](https://arxiv.org/html/2410.08828v1)
- Chinese OSS ASR: [FunASR](https://github.com/modelscope/FunASR), [FunASR vs Whisper](https://www.funasr.com/en/blog/funasr-vs-whisper-benchmark.html), [SenseVoice](https://github.com/FunAudioLLM/SenseVoice)
