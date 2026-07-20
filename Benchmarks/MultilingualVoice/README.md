# Multilingual Voice Benchmarks

End-to-end benchmarks for the Genie realtime voice API. Every score is measured
**on the API as a whole** (STT → LLM → TTS over the WebSocket), never on the
individual models inside it. Native audio is streamed in; the API's transcript,
reply text, and synthesized speech are scored.

## What it measures

| Dataset | Source | What it tests | Languages | Metric |
|---|---|---|---|---|
| **FLEURS** | `google/fleurs` | STT accuracy on read speech **+** TTS round-trip intelligibility | 24 (all supported) | WER / CER (↓) |
| **2M-Belebele** | `facebook/2M-Belebele` | Spoken reading-comprehension MCQ | 74+ (our 24 subset) | Accuracy (↑) |
| **CCFQA** | `yxdu/ccfqa` | Spoken factual QA | 7 (zh/en/fr/ja/ko/ru/es) | Match / GPT-judge acc (↑) |

**TTS quality** is captured two ways: latency (client time-to-first-audio and
the API's own `tts_first_ms`) and **round-trip intelligibility** — the API's
synthesized speech is fed back through its own STT and the re-heard text is
compared to what it meant to say (`tts_roundtrip_wer`/`_cer`).

> **Note on Belebele:** passages + question + 4 options are long. This API
> finalizes a turn on VAD silence / `max_turn_seconds`, so audio is capped
> (`--max-audio-seconds`, default 18s). For faithful Belebele scores, raise the
> server's `max_turn_seconds`. FLEURS and CCFQA fit a normal turn.

## Quick start

```bash
cd Benchmarks/MultilingualVoice

# Offline smoke test (writes to ./results/ only):
./eval.sh --fixture

# Submit serverless Databricks job (default):
./eval.sh

# Subsets:
./eval.sh --dataset fleurs --languages en,ja,zh,ar --limit 40
./eval.sh --wait

# Local dev only (requires workspace IP allowlist for serving):
./eval.sh --local --languages en --limit 5
```

Live results and logs are written to the UC Volume at
``volume.multilingual_voice_benchmark_path`` (see `config/config.yaml`):

- ``summary.json`` — latest scores (read by ``GET /v1/benchmarks``)
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

## Configuration (env overrides)

| Variable | Default | Meaning |
|---|---|---|
| `MLV_API_HOST` | from `realtime_voice.benchmark.api_host` | Databricks Apps base URL |
| `MLV_API_PREFIX` | `realtime` | path prefix on the app |
| `MLV_RESULTS_DIR` | from `volume.multilingual_voice_benchmark_path` | UC Volume output dir |
| `MLV_LANGUAGES` | *(all)* | comma-separated 2-letter codes |
| `MLV_DATASET` | `all` | `all` \| `fleurs` \| `belebele` \| `ccfqa` |
| `MLV_LIMIT` | `20` | samples per (dataset, language) |
| `MLV_RUN_LABEL` | *(auto)* | suffix for ``logs/run_<label>.log`` on the UC Volume |
| `DATABRICKS_PROFILE` | from config | CLI profile for auth |

## Running the Python directly

```bash
source .venv/bin/activate
python run_benchmark.py \
  --transport ws \
  --api-host https://…databricksapps.com --api-prefix realtime \
  --dataset fleurs --languages en,ja --limit 20 --tts-roundtrip
```

Use `--transport inprocess` to drive the API in-process via FastAPI TestClient
(verify/echo mode) for CI without a network.

## Files

- `run_benchmark.py` — dataset adapters, turn loop, scoring, summary writer.
- `realtime_client.py` — WebSocket + in-process clients, `TurnResult`, TTS round-trip.
- `evaluators.py` — pure-Python WER/CER, MCQ extraction, QA matching (no torch/jiwer).
- `languages.py` — 24-language maps to FLEURS / FLORES-200 / CCFQA codes.
- `fixtures/` — offline sample rows for `--fixture`.
- `eval.sh` — submits the serverless Databricks job (or `--local` / `--fixture`).
- `scripts/ml_asr/submit_multilingual_voice_benchmark_job.py` — job launcher used by `eval.sh`.
- `paths.py` — resolves UC Volume output path + app API host from config.
- `results/` — fixture-only scratch dir (not used for live runs).
