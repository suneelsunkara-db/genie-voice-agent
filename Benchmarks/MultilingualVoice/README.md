# Multilingual Voice Benchmarks

End-to-end benchmarks for the Genie realtime voice API. Scores are measured
**on the deployed API** (over the WebSocket), never on the individual models in
isolation. Native FLEURS audio is streamed in and the API's transcript is scored
against the reference — this sweep focuses on **speech-to-text accuracy**.

## What it measures

| Dataset | Source | What it tests | Languages | Metric |
|---|---|---|---|---|
| **FLEURS** | `google/fleurs` | STT accuracy on read speech | 24 (all supported) | WER / CER (↓) |

FLEURS is a **read-speech ASR corpus**, so it is used to score **speech-to-text
only**. TTS *quality* is intentionally **not** scored here: round-tripping ASR
audio through TTS and re-transcribing it conflates STT and TTS error (the
round-trip number is bounded by the STT judge, not the TTS quality). TTS quality
needs its own benchmark and is out of scope for this sweep.

**Latency (TTFT).** FLEURS accuracy uses the STT-only route (the LLM is never in
the loop), so the accuracy turn has no spoken reply to time. To get a meaningful
**TTFT** we run a **TTS round-trip** (`--tts-roundtrip`, on by default): the
reference text is synthesized through the text-to-speech route and we record the
TTS engine's **time-to-first-audio** (`client_ttfa_ms`) — a model-level "how fast
the voice starts speaking" number. (Routing FLEURS through the full agent
capability instead would time the tool-assisted LLM — tens of seconds on
read-speech — which measures the app, not the voice stack.) TTFT is the headline
latency in the UI, with the STT stage time shown alongside it. The UI only
promotes a new run once it has swept the full language set with accuracy **and**
TTFT on every unit — a partial/in-flight run keeps showing the last full run.

> **Deprecated:** the earlier LLM-QA datasets (**2M-Belebele**, **CCFQA**) are no
> longer part of the comparison. They exercised the contact-center billing-agent
> LLM persona rather than STT/TTS, and covered far fewer of our languages. The
> dataset adapters still exist in `run_benchmark.py` for reference, but the
> references, UI, and default `eval.sh` flow are FLEURS-STT only.

## STT model comparison

`eval.sh` runs two Databricks jobs **in sequence** so we get a like-for-like
comparison on the same audio:

1. **Main job** — measures the Genie realtime STT API on FLEURS (WER/CER per
   language) and stages the FLEURS audio on the UC Volume.
2. **Vendor comparison job** — replays the **same staged FLEURS audio** through a
   vendor STT (Deepgram Nova) and writes it under a distinct dataset id
   (`fleurs_deepgram_stt`) so it never collides with the Genie run. It runs
   **after** the main job (which stages the audio) and only for the `fleurs`/`all`
   sweep; `eval.sh` waits on the main job when vendors are enabled.

**Published references** (Whisper large-v2/v3, MMS-1B / 1B-LSAH, SeamlessM4T
Medium/Large/Large-v2) live in `realtime_api/benchmark_references.py` and render
in the UI as `published` rows next to Genie's `measured` bar. Vendor datasets
(`fleurs_deepgram_stt`) are measured and stored in Delta but **hidden from the
UI** (`HIDDEN_DATASETS` in `frontend/src/components/VoiceBenchmarksPage.tsx`).

| Vendor flag | Default | Meaning |
|---|---|---|
| `--vendors` | `deepgram` | comma-separated vendor STT tracks to run |
| `--no-vendors` | — | skip the vendor comparison job entirely |

## Quick start

```bash
cd Benchmarks/MultilingualVoice

# Offline smoke test (writes to ./results/ only):
./eval.sh --fixture

# Submit serverless Databricks job(s) (default): Genie STT + Deepgram vendor STT
./eval.sh

# Subsets:
./eval.sh --dataset fleurs --languages en,ja,zh,ar --limit 40
./eval.sh --wait            # also block on the vendor job

# Genie STT only (skip the Deepgram vendor comparison):
./eval.sh --no-vendors

# Local dev only (requires workspace IP allowlist for serving):
./eval.sh --local --languages en --limit 5
```

Scores are the source of truth in the Delta tables
``{catalog}.{schema}.benchmark_runs`` (one row per run_id×dataset×language) and
``benchmark_samples`` — this is what ``GET /v1/benchmarks`` reads. Serverless job
tasks write straight to Delta; ``summary.json`` is written only for offline
inspection in ``--fixture`` / ``--local`` modes and is **not** read by the API.

Logs still land on the UC Volume at ``volume.multilingual_voice_benchmark_path``
(see `config/config.yaml`):

- ``logs/run_<timestamp>.log`` — full run log mirrored from stdout
- ``logs/issues.jsonl`` — structured per-turn issues

## Authentication

By default ``eval.sh`` submits a serverless job. The job uses the Databricks SDK
(OAuth) to call the deployed app's WebSocket API and write results to the UC
Volume. Your CLI profile (``config/config.local.yaml`` → ``databricks.profile``)
is only used to submit the job.

```bash
databricks auth login --profile fe-vm-vdm-classic-rcn6ip
```

## Configuration

Connection + target settings come from config (no env overrides):

| Setting | Source (config) |
|---|---|
| App base URL | `realtime_voice.benchmark.api_host` |
| Path prefix on the app | `realtime_voice.benchmark.api_prefix` (default `realtime`) |
| UC Volume output dir | `volume.multilingual_voice_benchmark_path` |
| Databricks host / profile | `databricks.host` / `databricks.profile` |
| SP auth (app M2M) | `realtime_voice.benchmark.auth.{client_id,client_secret}` |

Per-run parameters are CLI flags on `run_benchmark.py`:

| Flag | Default | Meaning |
|---|---|---|
| `--dataset` | `all` | `all` \| `fleurs` \| `belebele` \| `ccfqa` |
| `--languages` | *(all)* | comma-separated 2-letter codes |
| `--limit` | `20` | samples per (dataset, language) |
| `--run-id` / `--run-label` | *(auto)* | sweep id / log suffix on the UC Volume |

## Running the Python directly

```bash
source .venv/bin/activate
python run_benchmark.py \
  --transport ws \
  --api-host https://…databricksapps.com --api-prefix realtime \
  --dataset fleurs --languages en,ja --limit 20
```

Use `--transport inprocess` to drive the API in-process via FastAPI TestClient
(verify/echo mode) for CI without a network.

## Files

- `run_benchmark.py` — dataset adapters, turn loop, scoring, summary writer.
- `realtime_client.py` — WebSocket + in-process clients, `TurnResult`.
- `evaluators.py` — pure-Python WER/CER, MCQ extraction, QA matching (no torch/jiwer).
- `languages.py` — 24-language maps to FLEURS / FLORES-200 / CCFQA codes.
- `vendor_fleurs_benchmark.py` — vendor STT runner (Deepgram) over the staged FLEURS audio.
- `fixtures/` — offline sample rows for `--fixture`.
- `eval.sh` — submits the Genie STT job then the Deepgram vendor STT job in sequence (or `--local` / `--fixture`).
- `scripts/ml_asr/submit_multilingual_voice_benchmark_job.py` — main Genie STT job launcher used by `eval.sh`.
- `scripts/ml_asr/submit_vendor_fleurs_benchmark_job.py` — Deepgram vendor STT job launcher used by `eval.sh`.
- `paths.py` — resolves UC Volume output path + app API host from config.
- `results/` — fixture-only scratch dir (not used for live runs).
- Published baselines: `realtime_api/benchmark_references.py` (FLEURS WER references shown in the UI).
