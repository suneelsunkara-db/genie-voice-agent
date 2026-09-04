# Building Voice Agents for Asian Languages — Applied Learnings

Qwen3-ASR-1.7B and VoxCPM2 on one serving path

**Deck:** [Google Slides](https://docs.google.com/presentation/d/1k2krQZrYTw74O4rezTKHhEbfMTEbH4R4KTdYKhLdsnU/edit)

Cohere Labs · Open Science  
Suneel Sunkara · Databricks

Main deck: 13 content slides plus the title.

**Story:** Multilingual requirement → language challenges → Qwen3-ASR → VoxCPM2 → why the pair → model-serving architecture → ontology → tool calling → multilingual evaluation → findings → deployment lessons.

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

## Slide 4 (deck eyebrow 03) — Qwen3-ASR-1.7B: multilingual speech recognition

Built from Qwen3-Omni: a 300M-parameter AuT audio encoder feeds speech representations through a projector to a Qwen3-1.7B language decoder. [Model card](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) · [Technical report](https://arxiv.org/abs/2601.21337)

_Architecture visual:_ 16 kHz audio → AuT encoder (300M, 12.5 Hz representation rate) → projector → Qwen3 decoder → transcript + language ID.

COVERAGE

- 30 languages plus 22 Chinese dialects
- The six focal Asian languages and English are in the published language list
- The dialect count refers specifically to Chinese dialects; it does not imply uniform low-resource performance

INFERENCE

- The upstream model supports offline and vLLM streaming inference
- This deployment calls `transcribe()` after the utterance is complete
- The benchmark therefore measures completed-utterance recognition, not streaming latency

FORCED ALIGNMENT

- The optional Qwen3-ForcedAligner-0.6B timestamps a supplied text–speech pair in 11 languages
- It does not validate or repair the recognized transcript
- It is not used in this deployed endpoint

Sources: [Qwen3-ASR model card](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) · [Qwen3-ASR technical report](https://arxiv.org/abs/2601.21337) · [deployed wrapper](https://github.com/suneelsunkara-db/genie-voice-agent/blob/main/scripts/ml_asr/realtime_stt_agent.py)

---

## Slide 5 (deck eyebrow 04) — VoxCPM2: tokenizer-free speech generation

2B parameters · 30 languages · 48 kHz output · text-designed voices or cloning from a short recording. [Model card](https://huggingface.co/openbmb/VoxCPM2) · [Repository](https://github.com/OpenBMB/VoxCPM)

_Architecture visual:_ text + optional voice reference → TSLM semantic/prosodic plan → RALM acoustic sequence → LocDiT diffusion + AudioVAE → 48 kHz waveform.

CONTINUOUS AUDIO

- Unlike codec-token TTS, VoxCPM2 models continuous AudioVAE representations
- Its diffusion-autoregressive stack is designed to preserve fine acoustic and prosodic detail

TWO VOICE CONTROLS

- Voice design creates a new voice from a natural-language description
- Voice cloning preserves a speaker from a short reference recording
- These are separate operating modes; text-designed voice is not reference-free cloning

HOW THIS DEPLOYMENT USES IT

- Reference cloning maintains one agent voice across turns
- `generate_streaming()` emits PCM audio chunks
- Six diffusion steps at CFG 2.0 were the lowest tested setting that preserved the Thai reference voice

Sources: [VoxCPM2 model card](https://huggingface.co/openbmb/VoxCPM2) · [official repository](https://github.com/OpenBMB/VoxCPM) · [VoxCPM architecture paper](https://arxiv.org/abs/2509.24650)

---

## Slide 6 (deck eyebrow 05) — The models: why these two models fit the experiment

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

## Slide 7 (deck eyebrow 06) — Model serving architecture: Two voice models around a tool-calling LLM

_Visualization: three-plane architecture diagram (`charts/serving_architecture.png`)._

VOICE SERVING PLANE — the two models we deploy ourselves

- Qwen3-ASR-1.7B (STT) and VoxCPM2 (TTS) are each packaged as an MLflow ResponsesAgent, registered in Unity Catalog under the `candidate` alias, and deployed as separate GPU Model Serving endpoints (`realtime_voice_stt_qwen3_asr_1_7b`, `realtime_voice_tts_voxcpm2`).

REASONING PLANE — the LLM in the middle

- Speech becomes a transcript, which goes to a Databricks foundation-model endpoint (`qwen3-next-80b`). A tool-calling loop (≤3 iterations) decides what information to fetch, then produces the response text sent back to VoxCPM2.

DATA & ONTOLOGY PLANE — where tool calls gather governed business data

- Tools reach the **Genie semantic layer** (Unity Catalog tables + instructions + entity matching that turn natural language into governed SQL — this is what the UI calls the "business ontology"), **Lakebase** for sub-millisecond account and billing facts, and **Genie One** for governed workspace answers.

Governance: Unity Catalog governs the self-registered STT/TTS endpoints and the Genie-served data; Lakebase changes flow back into UC history.

---

## Slide 8 (deck eyebrow 07) — Why ontology: grounding spoken requests in a governed semantic layer

A correct transcript still lacks the entities, definitions, and relationships an answer requires.

_Visualization: ontology knowledge graph (`charts/ontology_graph.png`) — business entities (Customer, Account, Invoice, Billing Cycle, Adjustment, Plan, Payment, Usage) linked by named relationships. The spoken request "Why did my bill increase?" enters on the left; a coral resolution path highlights the customer, invoice, prior billing cycle, and the adjustment that changed the total._

WHAT THE SEMANTIC LAYER ADDS (bottom strip of the diagram)

- Definitions — certified metrics and term meanings
- Governance — permission-aware, account-scoped access
- Sources — governed Unity Catalog tables
- Continuity — carried across follow-up questions

Grounding begins after transcription; it cannot recover an entity or amount that recognition transcribed incorrectly.

---

## Slide 9 (deck eyebrow 08) — Why tool calling: tool calling keeps business facts outside the model

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

## Slide 10 (deck eyebrow 09) — Multilingual evaluation: language-appropriate metrics and scarce matched data constrain evaluation

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

## Slide 11 (deck eyebrow 10) — Asian-language recognition: recognition error across six Asian languages and English

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

## Slide 12 (deck eyebrow 11) — Asian languages in the 24-language view: where the six Asian languages sit among 24

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

## Slide 13 (deck eyebrow 12) — Latency benchmarks: STT scales with utterance length; TTS startup is stable

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

## Slide 14 (deck eyebrow 13) — Four lessons from running these models in production

01  PIN THE SOFTWARE STACK  
The serving GPU shipped a CUDA build our model libraries could not load, so speech recognition failed on the first real request — not at deploy time. Fixing the library versions made it reliable.

02  KEEP THE MODELS WARM  
A cold endpoint answered the first request in about 24 seconds; once warm, about 1.5. We keep the models loaded and warm both the normal and voice-cloned paths before taking real traffic.

03  SEND THE VOICE ONCE  
Resending the reference voice sample on every turn added about 1.7 seconds. We upload it once and refer to it by an ID on later turns, so the lookup is near-instant.

04  DON'T RUSH VOICE GENERATION  
Using fewer generation steps was faster but made the Thai voice noticeably less like the reference. Six steps was the point where speed and voice quality were both acceptable.

_Footer:_ These are practical lessons from operating the system — not results from the recognition benchmark.

**Speaker notes (provenance, not on slide):** pinned stack `torch==2.7.1` / `torchaudio==2.7.1` / `torchvision==0.22.1`; `GPU_MEDIUM` / Small, scale-to-zero off; cold-start ~24 s → warm ~1.5 s; voice reference ~500 KB → `voice_id` lookup ~25 ms; VoxCPM2 six steps at CFG 2.0, four steps drop Thai similarity 1.00 → 0.76. Implementation: `register_realtime_voice_agent.py:36–40`; `realtime_tts_agent.py:82–220`; `realtime_api/services.py:229`; `config/config.yaml:358–360`. References: MLflow ResponsesAgent; Databricks GPU Model Serving; VoxCPM2 model card.
