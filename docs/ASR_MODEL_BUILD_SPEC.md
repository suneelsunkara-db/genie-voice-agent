# ASR Model Build Spec

## Purpose

Build and evaluate a Databricks-hosted speech-to-text model that can replace or complement Deepgram for the Genie Voice Agent app.

The first target is utterance-level transcription for contact-center billing conversations. True low-latency streaming can come later after the model quality and serving contract are proven.

## Use Case Fit

This app does not need generic transcription only. It needs accurate final customer utterances that drive:

- intent and sentiment enrichment
- waiver and payment-plan detection
- invoice ID, amount, date, and account reference extraction
- customer confirmation detection, for example "yes", "go ahead", "that works"
- resolution state transitions and billing adjustment decisions
- agent-facing transcript display

The model should be judged on business-critical transcription quality, not only on word error rate.

## Scope

### In Scope

- Build a reproducible ASR benchmark dataset.
- Baseline Deepgram Nova-3 against that dataset.
- Benchmark one or more Whisper-family base models.
- Fine-tune only if baseline models miss domain-critical speech.
- Track experiments in MLflow.
- Register the best model in Unity Catalog.
- Serve an utterance-level transcription endpoint from Databricks.

### Out of Scope For First Version

- Full Deepgram-compatible live streaming.
- Interim partial transcripts.
- Speaker diarization for mixed-channel audio.
- Real-time endpointing and voice activity detection.
- PCI/PII redaction parity with Deepgram.
- Production autoscaling and high-concurrency load testing.

These can be added after utterance-level quality is proven.

## Success Criteria

The Databricks ASR model is viable if it meets these gates on the locked evaluation set:

- transcript quality is close to or better than Deepgram on contact-center billing utterances
- invoice IDs, amounts, dates, and confirmation phrases are preserved reliably
- transcription latency is acceptable for push-to-talk agent assist
- output contract is stable enough for the existing `/assist` flow
- model, dataset, and metrics are fully versioned in MLflow and Unity Catalog

Recommended initial target:

- Word Error Rate within 10-15 percent relative of Deepgram, or better.
- Entity accuracy equal to or better than Deepgram for invoice IDs, dollar amounts, dates, and billing-action phrases.
- P95 utterance transcription latency under 5 seconds for 5-30 second clips.

Business entity accuracy is the promotion gate. WER alone is not sufficient.

## Databricks Workspace GPU Plan

Workspace inspected: `fe-vm-vdm-classic-rcn6ip`.

Available GPU ML runtimes include:

- `16.4.x-gpu-ml-scala2.13`
- `17.3.x-gpu-ml-scala2.13`
- `18.x-gpu-ml-scala2.13`
- `18.2.x-gpu-ml-scala2.13`

Available GPU node families include:

- `g4dn.*` with NVIDIA T4
- `g5.*` with NVIDIA A10G
- `p3.*` with NVIDIA V100
- `p4d.24xlarge` with 8x NVIDIA A100

Recommended usage:

- Baseline and smoke testing: `g4dn.xlarge`
- Primary LoRA fine-tuning: `g5.2xlarge` or `g5.4xlarge`
- Larger/faster experiments: `g5.12xlarge`
- Full fine-tuning or large-scale training only: `p4d.24xlarge`

Use `g5` A10G as the default training target. T4 is acceptable for smoke tests but not ideal for serious fine-tuning.

## Dataset Design

### Gold Audio Set

Create an evaluation-first dataset before training.

Initial size:

- 50-100 utterance-level clips for the first benchmark
- 300-500 utterance-level clips before claiming fine-tuned model quality
- 1,000+ utterance-level clips for a stronger production-quality model

Clip target:

- 5-30 seconds each
- one dominant speaker per clip for version 1
- contact-center billing language
- realistic microphone and call audio conditions

Include:

- customer complaints about billing
- late-fee waiver requests
- payment-plan requests
- declined payment discussion
- overdue invoice discussion
- confirmation and refusal phrases
- short utterances, for example "yes", "no", "that is fine"
- invoice IDs, dates, dollar amounts, account references
- accents, noise, pauses, crosstalk-like artifacts where available

### Recommended Table Schema

Store metadata in a Delta table, for example:

`<catalog>.<schema>.asr_model_training_clips`

Suggested columns:

- `clip_id`: stable unique ID
- `call_id`: source call ID
- `speaker`: `agent` or `customer`
- `audio_path`: UC Volume path
- `audio_format`: `wav`, `webm`, `mp3`, etc.
- `sample_rate_hz`: normalized sample rate
- `duration_seconds`: clip duration
- `reference_transcript`: human-approved transcript
- `domain`: for example `billing_support`
- `scenario`: for example `late_fee_waiver`
- `accent_label`: optional
- `noise_label`: optional
- `contains_invoice_id`: boolean
- `contains_amount`: boolean
- `contains_date`: boolean
- `contains_confirmation`: boolean
- `expected_entities`: JSON string or struct
- `split`: `train`, `validation`, or `test`
- `created_at`: timestamp
- `dataset_version`: string

### Split Rules

Do not randomly split individual chunks from the same call across train and test.

Use call-level splits:

- train: 70 percent
- validation: 15 percent
- test: 15 percent

Keep a separate locked benchmark set that is never used for training or prompt/model selection.

## Baseline Plan

### Deepgram Baseline

Use Deepgram REST batch transcription against stored audio clips.

Do not use the live browser mic websocket for benchmarking. The live path adds browser, microphone, websocket, and endpointing variability that makes model comparison noisy.

Baseline provider:

- Deepgram Nova-3
- `smart_format=true`
- `punctuate=true`
- preserve raw JSON output

For each clip, store:

- transcript
- raw Deepgram response
- latency
- confidence if available
- word timings if available
- model and request settings
- evaluation run ID

Name the first locked result:

`deepgram_nova3_baseline_v1`

### Whisper Baselines

Benchmark at least:

- Whisper large-v3 or current strongest stable Whisper checkpoint
- Whisper large-v3-turbo or similar faster variant
- Distil-Whisper variant if latency matters more than maximum accuracy

For each model, use identical audio clips and identical scoring logic.

## Evaluation Metrics

### Generic ASR Metrics

- Word Error Rate
- Character Error Rate
- insertion, deletion, and substitution rates
- latency per clip
- throughput per GPU

### Business-Critical Metrics

Track exact or normalized accuracy for:

- invoice IDs
- dollar amounts
- dates
- payment-plan phrases
- waiver phrases
- confirmation phrases
- refusal phrases
- account-status phrases

Recommended entity groups:

```json
{
  "invoice_ids": ["INV-10482"],
  "amounts": ["$248.17"],
  "dates": ["June 30"],
  "billing_actions": ["late fee waiver", "payment plan"],
  "confirmations": ["yes, go ahead"]
}
```

Evaluation should normalize common formatting differences before scoring:

- `$248.17` vs `248 dollars and 17 cents`
- `INV-10482` vs `invoice 10482`
- `June 30` vs `June thirtieth`

### App-Specific Acceptance Tests

For each transcript candidate, optionally run the existing enrichment flow and compare:

- primary intent
- customer signal
- payment plan requested
- waiver requested
- resolution transition
- agent reply availability

This catches cases where WER looks acceptable but the app behavior regresses.

## Training Strategy

### Phase 1: No Fine-Tuning

Start with base model benchmarking. If a base Whisper model is close enough to Deepgram on business-critical metrics, use it first and avoid training complexity.

### Phase 2: LoRA Fine-Tuning

If baseline Whisper misses domain vocabulary or entities, fine-tune with LoRA/PEFT.

Recommended starting point:

- base model: Whisper large-v3 or best benchmarked Whisper-family model
- training style: LoRA adapter
- precision: mixed precision if supported
- data: train split only
- validation: validation split after each run
- promotion check: locked test set only after candidate selection

LoRA is preferred first because it is cheaper, easier to iterate, and lower risk than full fine-tuning.

### Phase 3: Full Fine-Tuning

Only consider full fine-tuning if:

- LoRA does not improve enough
- enough labeled audio exists
- GPU budget supports larger experiments
- regression risk against general speech is acceptable

## MLflow And Unity Catalog

Every run should log:

- dataset version
- split IDs
- model checkpoint
- LoRA config if used
- preprocessing settings
- decoding settings
- training parameters
- WER/CER
- business entity metrics
- latency metrics
- sample predictions
- failure examples
- artifact paths

Register only promotion candidates in Unity Catalog.

Suggested model naming:

`<catalog>.<schema>.genie_voice_asr_whisper`

Suggested aliases:

- `Baseline`
- `Candidate`
- `Champion`
- `Archived`

## Serving Contract

The first Databricks serving endpoint should support utterance-level transcription.

Request:

```json
{
  "audio_b64": "<base64-audio>",
  "mime_type": "audio/wav",
  "sample_rate_hz": 16000,
  "language": "en",
  "call_id": "CALL-2028",
  "speaker": 1
}
```

Response:

```json
{
  "transcript": "Yes, go ahead and set up the payment plan.",
  "confidence": null,
  "language": "en",
  "duration_seconds": 8.4,
  "words": [],
  "provider": "databricks_whisper",
  "model_version": "1",
  "latency_ms": 1830
}
```

The app can then route the transcript into the existing `/assist` flow.

## Integration Plan After Model Approval

After the model beats or matches the baseline on business metrics:

1. Add a `DatabricksWhisperSTT` provider.
2. Add provider config under `providers.stt.adapters`.
3. Refactor `/calls/{call_id}/mic-transcribe` to use the active STT provider instead of hardcoding Deepgram.
4. Keep Deepgram as a fallback provider during rollout.
5. Later refactor `/calls/{call_id}/mic-stream` after a streaming strategy is defined.

## Risks And Mitigations

### Risk: WER Improves But App Behavior Gets Worse

Mitigation:

- evaluate business entities separately
- run app-level enrichment regression checks
- promote based on business metrics, not WER alone

### Risk: Not Enough Labeled Audio

Mitigation:

- start with base Whisper benchmarking
- fine-tune only after a meaningful gold set exists
- prioritize labeling business-critical utterances

### Risk: Latency Is Too High

Mitigation:

- start with utterance-level push-to-talk
- benchmark faster model variants
- use `g5` A10G for serving tests
- defer websocket streaming until endpoint latency is understood

### Risk: Deepgram Features Are Missing

Mitigation:

- treat diarization, interim transcripts, endpointing, and redaction as separate product features
- do not require full parity for model approval

## Immediate Next Steps

1. Create the gold clip manifest table and UC Volume folder layout.
2. Collect or synthesize the first 50-100 representative utterance clips.
3. Manually approve reference transcripts and expected business entities.
4. Run `deepgram_nova3_baseline_v1`.
5. Run Whisper baseline inference on the same clips.
6. Compare Deepgram vs Whisper on WER, entity metrics, and app-level behavior.
7. Decide whether LoRA fine-tuning is justified.

