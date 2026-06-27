# ASR Model-Training Harness

This package prepares and benchmarks the utterance-level ASR dataset used for Whisper fine-tuning and model selection.

The first benchmark is Deepgram Nova-3 on the locked training/evaluation manifest. Whisper and Databricks model-serving baselines use the same manifest and scoring functions so model comparisons are fair.

## Manifest

Use JSONL with one utterance-level clip per line. Required fields:

- `clip_id`
- `audio_path`
- `reference_transcript`

Recommended fields:

- `call_id`
- `speaker`
- `audio_format`
- `sample_rate_hz`
- `duration_seconds`
- `scenario`
- `split`
- `dataset_version`
- `expected_entities`

See `docs/asr_model_training_manifest.example.jsonl`.

## Run The Workflow

From the repo root:

```bash
scripts/asr/01_asr_model_training.sh
```

That one command:

- creates the ASR model-training Volume folders
- creates the local training manifest if it does not exist
- ingests up to 20 Mozilla Common Voice validated English clips if a Common Voice archive is present in the ASR model-training Volume
- otherwise downloads LibriSpeech `dev-clean` from OpenSLR into the ASR model-training Volume and ingests up to 20 clips
- generates billing/contact-center WAV audio from the app's own datagen scenarios
- adds noisy and phone-band augmented billing clips for training robustness
- uploads processed WAV audio to the Databricks Volume
- syncs the manifest to the Databricks Volume
- validates the manifest
- checks whether the listed audio files exist
- runs the Deepgram baseline only after external and billing supplement audio are present
- syncs the Deepgram baseline JSONL back to the Databricks Volume

When the Deepgram baseline runs, the output is JSONL. Each row includes:

- Deepgram transcript
- raw Deepgram response
- latency
- WER/CER
- billing entity scores

The example manifest references placeholder audio paths. For real runs, put audio and manifests in the ASR model-training UC Volume folders created by `prepare`.

## Numbered Scripts

The ASR workflow is intentionally split so baseline runs and actual model training do not get mixed:

- `scripts/asr/01_asr_model_training.sh`: shared data prep, manifest validation, Volume layout, and GPU cluster lifecycle.
- `scripts/asr/02_asr_baseline_runs.sh`: candidate evaluation only, including Deepgram and Whisper baselines.
- `scripts/asr/03_asr_model_finetuning.sh`: actual model fine-tuning only, gated by fair comparison and a dry-run job.

The next command is:

```bash
scripts/asr/02_asr_baseline_runs.sh whisper-full
```

That runs the full-manifest Whisper baseline on the persistent ASR GPU cluster. It does not train a model.

For fair Deepgram-vs-Whisper comparison after metrics or manifest fixes:

```bash
scripts/asr/02_asr_baseline_runs.sh fair-compare
```

That command:

- rescores the full Whisper output with the current manifest and metrics
- runs Deepgram Nova-3 on the same 403-clip manifest using local cached audio
- writes `.run/asr_model_training/evaluations/asr_baseline_fair_comparison.json`

After fair comparison is complete, validate the training lane:

```bash
scripts/asr/03_asr_model_finetuning.sh preflight
scripts/asr/03_asr_model_finetuning.sh dry-run
```

Only run actual LoRA training after dry-run succeeds:

```bash
scripts/asr/03_asr_model_finetuning.sh train-lora
```

For individual baseline runs, use explicit input/output behavior:

```bash
scripts/asr/02_asr_baseline_runs.sh deepgram-full
scripts/asr/01_asr_model_training.sh deepgram --audio-source local-cache --sync-results false
```

## Shared Engine Script

Use `scripts/asr/01_asr_model_training.sh` for the full ASR model-training preparation and baseline flow. It handles environment setup automatically, so there is no separate setup script to remember.

If you forget what to do, run it with no arguments:

```bash
scripts/asr/01_asr_model_training.sh
```

It runs all safe repeatable steps and stops only when it needs real audio or corrected transcript labels.

Commands:

- `next`: show the one command to run next
- `volume`: show the ASR model-training UC Volume paths
- `prepare`: create the ASR model-training UC Volume folders
- `validate`: validate manifest shape and local scoring
- `augment`: generate and upload augmented billing-domain audio
- `deepgram`: run the locked Deepgram baseline
- `whisper`: run a Whisper baseline with optional ML dependencies
- `whisper-db`: run a Databricks GPU Whisper smoke baseline on the dedicated ASR GPU cluster
- `gpu-status`: show the dedicated ASR GPU cluster status
- `gpu-start`: create or start the dedicated ASR GPU cluster
- `gpu-stop`: stop the dedicated ASR GPU cluster after training
- `summarize`: summarize a JSONL result file
- `all`: run validate, Deepgram baseline, and summarize

The normal path is the no-argument command above. The named commands are for debugging only.

Run the Whisper baseline on a Databricks GPU cluster or ML workstation with optional ASR dependencies installed:

```bash
pip install transformers torch accelerate librosa soundfile
scripts/asr/01_asr_model_training.sh whisper --limit 20
```

Remove `--limit 20` once the small smoke run succeeds.

For the Databricks GPU path, start the dedicated training cluster once:

```bash
scripts/asr/01_asr_model_training.sh gpu-start
scripts/asr/01_asr_model_training.sh whisper-db
```

`whisper-db` uploads a self-contained runner to the ASR model-training Volume and submits the smoke job to that persistent cluster using `existing_cluster_id`. It does not create a new job cluster every time.

When model training/evaluation is finished, stop the cluster explicitly:

```bash
scripts/asr/01_asr_model_training.sh gpu-stop
```

To override the dedicated cluster, pass any existing running Databricks cluster:

```bash
scripts/asr/01_asr_model_training.sh whisper-db --cluster-id <cluster-id>
```

or set:

```bash
export ASR_WHISPER_EXISTING_CLUSTER_ID=<cluster-id>
scripts/asr/01_asr_model_training.sh whisper-db
```

Common Voice is preferred for acoustic diversity, not billing-domain coverage. If you have a Common Voice corpus archive from Mozilla Data Collective, place it under:

```text
/Volumes/<catalog>/<schema>/<streaming_volume>/asr_model_training/external_raw/common_voice
```

It does not mirror or redistribute Mozilla's dataset.

If Common Voice is not present, the script automatically uses LibriSpeech `dev-clean` from OpenSLR as the available external acoustic-diversity source. LibriSpeech is CC BY 4.0 and is less conversational than Common Voice, but it is directly available and lets the model-training workflow proceed.

## Scoring

`score_transcript()` computes:

- Word Error Rate
- Character Error Rate
- invoice ID accuracy
- amount accuracy
- date accuracy
- billing-action phrase accuracy
- confirmation/refusal phrase accuracy

Business entity accuracy is the main promotion signal for this app. Generic WER is supporting evidence only.
