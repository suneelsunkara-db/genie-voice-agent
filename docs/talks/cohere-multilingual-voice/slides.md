# Building Voice Agents for Asian Languages — Applied Learnings

Qwen3-ASR-1.7B and VoxCPM2 on one serving path

**Deck:** [Google Slides](https://docs.google.com/presentation/d/1k2krQZrYTw74O4rezTKHhEbfMTEbH4R4KTdYKhLdsnU/edit)

Cohere Labs · Open Science  
Suneel Sunkara · Databricks

Main deck: 15 content slides plus the title.

**Story:** Multilingual requirement → language challenges → Qwen3-ASR → VoxCPM2 → model evolution → why the pair → model-serving architecture → ontology → tool calling → multilingual evaluation → findings → deployment lessons → runtime guardrails.

**Visual system:** Inter · warm off-white / ink navy / coral accent · direct labels · no decorative dashboards.

---

## Slide 1 — Title

**Building Voice Agents for Asian Languages — Applied Learnings**

Focal Asian languages: Thai · Indonesian · Mandarin · Hindi · Japanese · Filipino  
Reference baseline: English · Full benchmark: 24 languages

---

## Slide 2 — The multilingual requirement: one voice agent must work across Asian languages

THE REQUIREMENT

- Speak without choosing a language first
- Preserve names, numbers, and English terms mixed into local speech
- Retrieve information or perform the requested action
- Reply in the same language, with the same voice and a short delay

One shared pipeline must handle these language differences without requiring a separate system for each language.

---

## Slide 3 — Language challenges: Asian languages impose distinct constraints on a shared pipeline

TONE & PRONUNCIATION  
Thai and Mandarin are tonal — pitch decides the word. A small acoustic slip changes the word itself, not just its spelling.

WORD BOUNDARIES & SCRIPTS  
Thai, Mandarin, and Japanese write without spaces; Japanese mixes several scripts. Filipino packs tense into affixes: *nagbayad* (paid) vs *magbabayad* (will pay). With no clean word unit, we score characters.

CODE-SWITCHING  
English names, product terms, and ID numbers appear inside local sentences. The recognizer must transcribe both languages within a single utterance.

DATA AVAILABILITY  
English has abundant labeled speech. Conversational and business-specific data is far scarcer across the six focal Asian languages.

These differences set the requirements for model selection and evaluation.

---

## Slide 4 (deck eyebrow 03) — Qwen3-ASR: an LLM decoder for speech recognition

Audio features are decoded by a language model with textual context—not by an acoustic decoder alone.

MODEL EVOLUTION

- **wav2vec 2.0 (2020):** self-supervised masked prediction learns representations from unlabeled audio; a separately fine-tuned CTC head maps them to text.
- **Whisper (2022):** an encoder–decoder Transformer trained on 680k hours of weakly supervised audio maps speech features directly to autoregressive text tokens.
- **Qwen3-ASR (2026):** a 300M AuT encoder and projector feed Qwen3-1.7B. Post-training from Qwen3-Omni brings language-model context into transcription.

INSIDE QWEN3-ASR

16 kHz audio → 128-dimensional Fbank → AuT encoder (8× downsample; 12.5 Hz) → projector → Qwen3-1.7B → language ID + transcript

MECHANISM & TRAINING

- AuT pretraining uses ~40M hours of pseudo-labeled speech, mostly Chinese and English.
- Qwen3-Omni pretraining adds 3T multimodal tokens; ASR fine-tuning adds multilingual, context-biasing, and streaming-enhancement data.
- The LLM decoder can use context to resolve ambiguous acoustics, but plausible language can also introduce substitutions; entity accuracy still requires direct evaluation.

STREAMING & ALIGNMENT

- FastConformer reduces encoder cost with 8× convolutional subsampling and optional local attention.
- Qwen uses dynamic 1–8 second attention windows in AuT, followed by an LLM decoder; streaming is currently available only with the vLLM backend.
- This endpoint waits for a completed utterance.
- The separate non-autoregressive aligner supports 11 languages and up to five minutes. It timestamps known text–audio pairs; it cannot repair recognition and is not deployed here.

Sources: [Qwen3-ASR report](https://arxiv.org/abs/2601.21337) · [model card](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) · [wav2vec 2.0](https://arxiv.org/abs/2006.11477) · [Whisper](https://arxiv.org/abs/2212.04356) · [FastConformer](https://arxiv.org/abs/2305.05084) · [deployed wrapper](https://github.com/suneelsunkara-db/genie-voice-agent/blob/main/scripts/ml_asr/realtime_stt_agent.py)

---

## Slide 5 (deck eyebrow 04) — VoxCPM2: semantic planning, acoustic rendering

Tokenizer-free means no external speech-codec vocabulary—not an absence of quantization inside the model.

WHY TOKENIZER-FREE?

- **Discrete-codec TTS:** neural codecs turn speech into token IDs. This gives a language model a stable vocabulary, but quantization can discard fine acoustic detail; many systems then require a separate diffusion decoder.
- **VoxCPM2’s nuance:** it removes the external speech tokenizer, not quantization altogether. It generates continuous AudioVAE latents while an internal differentiable FSQ bottleneck creates a semi-discrete semantic/prosodic skeleton.

HIERARCHICAL GENERATION

Past audio latents → LocEnc → compact acoustic history  
Text + history → TSLM (MiniCPM-4-1B) → FSQ semantic/prosodic skeleton  
Skeleton + RALM residual detail → LocDiT denoising → AudioVAE decoder → 48 kHz waveform

WHAT EACH STAGE DOES

- TSLM predicts the content-and-prosody plan.
- FSQ regularizes its hidden state into a semi-discrete skeleton without an external codec vocabulary.
- RALM restores residual acoustic detail omitted by the skeleton.
- LocDiT uses flow-matching diffusion to generate each continuous latent patch; AudioVAE V2 decodes it to 48 kHz.

CAPABILITIES & DEPLOYMENT

- 2B parameters; 2M+ training hours; 30 languages; 48 kHz output.
- Voice design creates a voice from text; voice cloning preserves a speaker from audio. They are different modes.
- This endpoint uses reference cloning and streaming at six diffusion steps / CFG 2.0.
- Published RTX 4090 real-time factors are hardware-specific. The model card reports variation across languages and instability on long or highly expressive inputs.

Sources: [VoxCPM2 model card](https://huggingface.co/openbmb/VoxCPM2) · [VoxCPM paper](https://arxiv.org/abs/2509.24650) · [official repository](https://github.com/OpenBMB/VoxCPM) · [deployed wrapper](https://github.com/suneelsunkara-db/genie-voice-agent/blob/main/scripts/ml_asr/realtime_tts_agent.py)

---

## Slide 6 (deck eyebrow 05) — Model evolution: two parallel shifts in speech modeling

_Visualization: parallel STT/TTS timeline widget (`charts/evolution_timeline.png`), four eras (Statistical → Early Deep Learning → End-to-End → Foundation Models)._

SPEECH-TO-TEXT — GMM–HMM (1990s) → DNN–HMM (2011) → Deep Speech / CTC (2014) → Listen-Attend-Spell (2016) → wav2vec 2.0 + Conformer (2020) → Whisper (2022) → FastConformer (2023) → **Qwen3-ASR-1.7B (2026)** → Qwen3.5-Omni (2026)

TEXT-TO-SPEECH — Unit-selection / concatenative (1990s) → statistical parametric (2013) → WaveNet (2016) → Tacotron 2 (2017) → FastSpeech (2019) → VITS (2021) → VALL-E / AudioLM (2022–23) → GPT-4o Voice + Moshi (2024) → **VoxCPM2 (2026)**

Coral hero nodes mark the two models deployed in this study. Horizontal position shows progression within each era, not a linear time scale. Full citations in speaker notes.

---

## Slide 7 (deck eyebrow 06) — The models: why these two models fit the experiment

RECOGNITION · Qwen3-ASR-1.7B

- Transcribes speech in about 30 languages
- Also reports which language it recognized
- Accepts roughly 30 seconds of audio per request
- Can use spelling hints and be adapted with new data

SYNTHESIS · VoxCPM2

- Generates speech in about 30 languages
- Copies a voice from a short reference recording
- Streams the reply in small audio segments
- Runs inside the same controlled environment

The models overlap on 24 languages. Detailed analysis uses the seven-language focal scope defined on slide 1.

---

## Slide 8 (deck eyebrow 07) — Model serving architecture: Two voice models around a tool-calling LLM

_Visualization: three-plane architecture diagram (`charts/serving_architecture.png`)._

VOICE SERVING PLANE — the two models we deploy ourselves

- Qwen3-ASR-1.7B (STT) and VoxCPM2 (TTS) are each packaged as an MLflow ResponsesAgent, registered in Unity Catalog under the `candidate` alias, and deployed as separate GPU Model Serving endpoints (`realtime_voice_stt_qwen3_asr_1_7b`, `realtime_voice_tts_voxcpm2`).

REASONING PLANE — the LLM in the middle

- Speech becomes a transcript, which goes to a Databricks foundation-model endpoint (`qwen3-next-80b`). A tool-calling loop (≤3 iterations) decides what information to fetch, then produces the response text sent back to VoxCPM2.

DATA & ONTOLOGY PLANE — where tool calls gather governed business data

- Tools reach the **Genie semantic layer** (Unity Catalog tables + instructions + entity matching that turn natural language into governed SQL — this is what the UI calls the "business ontology"), **Lakebase** for sub-millisecond account and billing facts, and **Genie One** for governed workspace answers.

Governance: Unity Catalog governs the self-registered STT/TTS endpoints and the Genie-served data; Lakebase changes flow back into UC history.

---

## Slide 9 (deck eyebrow 08) — Why ontology: grounding spoken requests in a governed semantic layer

A correct transcript still lacks the entities, definitions, and relationships an answer requires.

_Visualization: ontology knowledge graph (`charts/ontology_graph.png`) — business entities (Customer, Account, Invoice, Billing Cycle, Adjustment, Plan, Payment, Usage) linked by named relationships. The spoken request "Why did my bill increase?" enters on the left; a coral resolution path highlights the customer, invoice, prior billing cycle, and the adjustment that changed the total._

WHAT THE SEMANTIC LAYER ADDS (bottom strip of the diagram)

- Definitions — certified metrics and term meanings
- Governance — permission-aware, account-scoped access
- Sources — governed Unity Catalog tables
- Continuity — carried across follow-up questions

Grounding begins after transcription; it cannot recover an entity or amount that recognition transcribed incorrectly.

---

## Slide 10 (deck eyebrow 09) — Why tool calling: tool calling keeps business facts outside the model

Native realtime APIs can call tools; this design keeps multilingual speech and governed reasoning as separate, replaceable components.

WHY NOT A SINGLE REALTIME MODEL

- Native audio APIs (OpenAI Realtime, Gemini Live) support function calling
- But recognition, reasoning, and voice stay coupled to one vendor model
- Multilingual STT and TTS span ~30 languages per model (24 shared), and each must stay replaceable
- Governed ontology access must remain text-native and auditable

TOOL-CALLING APPROACHES

- Native realtime — the audio model emits the call; fewest boundaries, vendor-coupled
- Text-mediated (this system) — the reasoning model selects governed tools; transcript and arguments stay inspectable
- MCP / managed-agent — tools, authentication, and data are decoupled; long tasks need progress and async handling

Read tools execute immediately; account-changing actions require explicit confirmation enforced by the tool, not the model.

---

## Slide 11 (deck eyebrow 10) — Multilingual evaluation: language-appropriate metrics and scarce matched data constrain evaluation

Error metrics must follow the script, and public corpora matching spoken business use are limited for these languages.

METRIC FOLLOWS THE SCRIPT

- Spaced scripts — Word Error Rate (WER)
- Thai, Mandarin, Japanese — Character Error Rate (CER)
- Code-switched turns — Mixed Error Rate (MER)
- Synthesis — independent-ASR WER/CER, MOS, speaker similarity, TTFA

MATCHED DATA IS SCARCE

- Public corpora are read speech, not business dialogue
- No shared word unit; scoring stays per-language
- Code-switch sets exist mainly for Mandarin–English
- Conversational and telephone speech is licence-restricted

PUBLIC EVALUATION DATASETS (hyperlinked on the slide)

- [FLEURS](https://huggingface.co/datasets/google/fleurs) — used here (100 clips × 24 languages)
- [Common Voice](https://commonvoice.mozilla.org)
- [CS-FLEURS](https://huggingface.co/datasets/byan/cs-fleurs)
- [ASCEND](https://huggingface.co/datasets/CAiRE/ASCEND)
- [IndicVoices](https://huggingface.co/datasets/ai4bharat/IndicVoices)

Reported here: FLEURS recognition, plus synthesis latency and completion. Business-entity accuracy uses a private in-domain holdout (`multilingual_gold_holdout` in `api/app/routers/asr_benchmark.py`, Deepgram Nova-3 vs a Databricks fine-tuned Whisper) — no public Asian equivalent exists.

---

## Slide 12 (deck eyebrow 11) — Asian-language recognition: recognition error across six Asian languages and English

_Visualization: forest plot (`charts/focal_forest.png`) — all seven focal languages, with point estimate as a colored dot, 95% bootstrap interval as the bar, and separate WER and CER panels._

**Speaker notes:** Two panels. Left WER (languages with spaces). Right CER (Thai, Mandarin, Japanese). Dot = measured error; bar = 95% interval. Compare only within a panel. WER: English 3.1%, Indonesian 5.4%, Hindi 12.9%, Filipino 24.9%. CER: Japanese 5.6%, Mandarin 6.8%, Thai 8.8%. Filipino is the hardest. Transition: next is where they sit in the full 24.

WER ↓ · WORD-TOKEN LANGUAGES

- English: 3.1% [2.3–4.0]
- Indonesian: 5.4% [4.0–7.2]
- Hindi: 12.9% [11.1–15.1]
- Filipino: 24.9% [22.2–28.0]

CER ↓ · NON-SPACED SCRIPTS

- Japanese: 5.6% [4.5–6.9]
- Mandarin: 6.8% [4.0–9.9]
- Thai: 8.8% [6.1–12.4]

Dot = measured error; line = 95% uncertainty interval. Compare values only within a panel because WER counts words and CER counts characters.

---

## Slide 13 (deck eyebrow 12) — Asian languages in the 24-language view: where the six Asian languages sit among 24

_Visualization: sorted horizontal bar chart (`charts/all24_bars.png`) — focal WER languages in coral, the three focal CER scripts in teal, and the other 17 benchmark languages in gray._

**Speaker notes:** Coral = WER; teal = CER. Shorter is better. Point is the distribution: a cluster under 10%, a tail past 20%. Coverage is not uniform (3% to 30%). Only three CER bars because CER is correct only for Japanese, Mandarin, Thai. Transition: the agent also has to speak back.

WER ≤ 10%

- Italian: 2.6%
- Spanish: 2.9%
- English: 3.1%
- German: 4.0%
- Portuguese: 4.6%
- French: 5.2%
- Indonesian: 5.4%
- Vietnamese: 5.6%
- Russian: 5.6%
- Dutch: 7.9%
- Malay: 9.8%
- Turkish: 9.9%

WER > 10%

- Korean: 12.3%
- Hindi: 12.9%
- Arabic: 13.0%
- Polish: 13.7%
- Danish: 20.0%
- Swedish: 20.7%
- Filipino: 24.9%
- Finnish: 25.9%
- Greek: 30.1%

CER

- Japanese: 5.6%
- Mandarin: 6.8%
- Thai: 8.8%

Why only three CER bars? Japanese, Mandarin, and Thai are the three scripts scored with CER; the other 21 languages use WER. Confidence intervals remain on the focal view.

---

## Slide 14 (deck eyebrow 13) — Latency benchmarks: STT scales with utterance length; TTS startup is stable

Same decomposition in every language, from 700 FLEURS samples (`20260830T080032Z`).

**Speaker notes:** Left is STT p50–p95 plus r with utterance length. Right is TTS generation plus a nearly fixed 158 ms delivery. TTS is language-stable. STT varies with length; duration was not stored, so language rank is not a causal claim.

STT · p50 / p95 · r with length · short→long quartile

- Mandarin: 0.93 / 1.71 s · r=0.84 · 0.72→1.30 s
- English: 1.07 / 1.67 s · r=0.87 · 0.74→1.45 s
- Indonesian: 1.38 / 2.42 s · r=0.97 · 1.00→2.16 s
- Japanese: 1.41 / 2.10 s · r=0.93 · 1.03→1.90 s
- Filipino: 2.03 / 3.46 s · r=0.95 · 1.37→2.72 s
- Thai: 2.47 / 4.18 s · r=0.84 · 1.61→3.23 s
- Hindi: 3.99 / 7.07 s · r=0.99 · 2.90→6.46 s

TTS · generation + delivery = client TTFA (p50)

- English 482 + 158 = 639 ms
- Indonesian 485 + 158 = 642 ms
- Mandarin 486 + 158 = 643 ms
- Japanese 485 + 158 = 643 ms
- Filipino 488 + 158 = 647 ms
- Thai 503 + 157 = 660 ms
- Hindi 503 + 157 = 662 ms

Audio duration was not stored, so language rank is not a causal claim. Next measurement: STT real-time factor.

---

## Slide 15 (deck eyebrow 14) — Deployment learnings: the serving environment, not the model weights, set first-turn latency

_Visualization: four-row table — effect · what we measured and what it means · control (`charts/deployment_learnings_research.png`)._

DEPENDENCIES — CUDA/PyTorch compatibility

- The GPU image's CUDA build did not match the model wheels; unpinned, resolution drifts to an incompatible wheel and the missing-operator error appears on the first inference, not at deploy. Plain: the endpoint reports healthy, then the GPU kernels fail on the first forward pass.
- Control: pin the whole stack — `torch==2.7.1` (cu118) with matching `torchvision`.

STARTUP GRAPH — `torch.compile` shape specialization

- `torch.compile` optimizes for one input tensor shape; the voice-clone path prepends reference-audio tokens (a new shape), forcing a ~15 s recompile on the first cloned turn. Plain: the model re-optimizes when input shape changes, so we trigger every shape before traffic.
- Control: warm both shapes at container startup; scale-to-zero off.

REFERENCE AUDIO — session-scoped caching

- A ~500 KB reference clip re-sent every turn cost ~1.7 s of time-to-first-audio, while materializing it server-side is ~25 ms — the cost was upload, not cloning. Plain: the delay was re-sending the voice sample, not the cloning computation.
- Control: send the clip once, reuse by `voice_id` (in-process LRU cache; a miss returns `voice_cache_miss` and the client retries with the clip).

SAMPLER STEPS — diffusion timesteps vs intelligibility

- Steps 4/6/8/10 are latency-flat (~2.4–3.3 s/sentence, RTF < 1); at 4 steps Thai round-trip intelligibility (transcribe the synthesized audio back, compare to input) falls to 0.76, recovering to 1.00 at 6 across en/th/id/zh. Plain: fewer denoising passes saved no time here, so there is no reason to trade away clarity.
- Control: fix 6 steps, CFG 2.0 (lowering CFG garbles audio; tone is set by the pinned reference clip).

Interpretation: reproducing these numbers means reporting the serving image, compiled graph shapes, reference caching, and sampler settings — not only weights and error rates.

---

## Slide 16 (deck eyebrow 15) — Guardrails: guards fire at the input boundary, and every check is logged

_Visualization: one voice turn as a left-to-right pipeline (`charts/runtime_guardrails.png`). Source: `GET /traces/guardrails?limit=300`, queried 06 Sep 2026 — 985 checks over 290 roster-bearing turns, 17 languages + auto-detect. Fires partition by pipeline stage, so the figure shows where guards act rather than a per-rail bar dashboard._

THE TURN, STAGE BY STAGE (fired / checks)

- Perception · STT (Qwen3-ASR): **20 fired** of 580 — language ID + silence suppression; a fire is an empty transcript that stops the turn.
- Routing (runtime policy): **32 fired** of 397 — reply-language gate, semantic route, selection cues; fires drop wrong-language turns or defer low-confidence routes to the LLM.
- Reasoning & tools (internal): **no guardrail checks** — the tool loop is bounded to ≤3 iterations but is not a guardrail surface.
- Grounding (runtime): **1 fired** of 8 — cite-or-silence blocked an answer citing an unsupported `$40` / `INV-90114`.

THREE TAKEAWAYS

- **3.4 checks per turn** — defense in depth across three stages of the cascade.
- **52 of 53 fires happen before the answer** — bad input is rejected or redirected up front, not caught at output.
- **220 checks logged "not evaluated"** (e.g. language-ID on a pinned-language call) — a skipped check is an auditable abstention, never counted as a pass.

Semantics: `fired` = changed the turn (not an alert); `delegated` = the speech model / runtime owns the signal (233 to Qwen3-ASR); `not_evaluated` = did not run. Rows with `surface=internal` are excluded so mechanics do not inflate the counts.
