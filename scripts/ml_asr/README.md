# ML ASR scripts

Five steps for dataset prep, deployment, and eval. Settings live in `config/ml_asr_eval.yaml`.

| Step | Script | What it does |
|------|--------|----------------|
| 1 | `01_datasets.sh` | Build FLEURS holdouts on UC Volume (serverless) |
| 2 | `02_quality.sh` | Semantic dataset quality (serverless) |
| 3 | `03_register.sh` | Register UC model candidates (Databricks only) |
| 4 | `04_serve.sh` | Deploy Model Serving endpoints (SDK from laptop) |
| 5 | `05_eval.sh` | Score all eval routes (serverless) |

**One entry point:**

```bash
./scripts/ml_asr.sh datasets
./scripts/ml_asr.sh quality
./scripts/ml_asr.sh register all
./scripts/ml_asr.sh serve deploy-all
./scripts/ml_asr.sh eval
```

## Model routes

| Route | Kind | Steps 3–4? |
|-------|------|------------|
| `deepgram_nova3` | Commercial API (Deepgram Nova-3) | No — API key only |
| `databricks_*` | UC-registered models on Model Serving | Yes |

`eval_matrix` lists every route scored in step 5. `model_serving` lists only `databricks_*` models for register/serve.

## Full workflow

```bash
# 1–2: data + quality gates
./scripts/ml_asr.sh datasets
./scripts/ml_asr.sh quality

# 3–4: deploy Databricks models (skip if endpoints already live)
./scripts/ml_asr/03_register.sh all
./scripts/ml_asr/04_serve.sh deploy-all
./scripts/ml_asr/04_serve.sh smoke databricks_en_finetuned_whisper_lora

# 5: score Deepgram + Databricks endpoints
./scripts/ml_asr.sh eval
./scripts/ml_asr.sh status
```

`03_register.sh` uses `genie_voice.ml_asr.serving` — OSS models register via `scripts/ml_asr/` on Databricks serverless; EN finetuned Whisper still bridges to `scripts/asr/05_register*` until migrated.

`04_serve.sh` uses `genie_voice.ml_asr.serving` with the Databricks Python SDK from your laptop.

## Config

`model_serving` in `config/ml_asr_eval.yaml` maps each `databricks_*` eval model to:

- `registered_model_leaf` — UC model name
- `register.type` — `oss` (OSS baseline on Databricks) or `finetuned_whisper`
- `serve.workload_type` / `workload_size` — endpoint compute

Endpoint names come from `models.*.endpoint` (same names `05_eval.sh` uses).

## Benchmark UI

After `./scripts/ml_asr.sh eval` completes summarize, sync the Volume index for the cockpit ASR benchmark page:

```bash
./scripts/ml_asr/05_eval.sh sync-index
# or: ./scripts/ml_asr/sync_benchmark_index.sh
```

Open **http://localhost:5173/#/asr-benchmark** — prefers `ml_asr` FLEURS results when `.run/ml_asr_eval/index.json` exists; falls back to legacy `voice_model_deep_eval` holdout.

Use the **Eval tier** dropdown for business (entity readiness) vs acoustic (WER/CER). Optional API: `?source=ml_asr|legacy|auto&tier=business|acoustic`.

## Dev-only local runs

```bash
./scripts/ml_asr/01_datasets.sh local
./scripts/ml_asr/02_quality.sh local
```

## Related

- Overview: [scripts/README.md](../README.md)
- Config: `config/ml_asr_eval.yaml`
- Python: `backend/genie_voice/ml_asr/`
- OSS register/serve assets: `scripts/ml_asr/`
- Legacy ASR bake-off: `scripts/asr/` (EN finetuned register only; scheduled for decommission)
