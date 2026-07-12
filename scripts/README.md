# Scripts

Operational entry points for ASR dataset prep, model deployment, and evaluation on Databricks.

## Start here: ML ASR pipeline

Use **`scripts/ml_asr.sh`** for the config-driven multilingual bake-off. All artifacts live on a UC Volume; dataset and eval steps run on serverless by default.

| Step | Command | What it does |
|------|---------|--------------|
| 1 | `./scripts/ml_asr.sh datasets` | Build FLEURS holdouts (business + acoustic) |
| 2 | `./scripts/ml_asr.sh quality` | Semantic dataset quality gates |
| 3 | `./scripts/ml_asr.sh register` | UC registration (**Databricks models only**) |
| 4 | `./scripts/ml_asr.sh serve` | Deploy Model Serving endpoints |
| 5 | `./scripts/ml_asr.sh eval` | Score all eval routes (serverless) |

```bash
./scripts/ml_asr.sh datasets
./scripts/ml_asr.sh quality
./scripts/ml_asr.sh register all
./scripts/ml_asr.sh serve deploy-all
./scripts/ml_asr.sh eval
./scripts/ml_asr.sh status   # eval pipeline state
```

**Config:** `config/ml_asr_eval.yaml`  
**Python:** `backend/genie_voice/ml_asr/`  
**Details:** [scripts/ml_asr/README.md](ml_asr/README.md)

### Model routes

| Route prefix | Kind | Register / serve? |
|--------------|------|-------------------|
| `deepgram_nova3` | Commercial API (Deepgram Nova-3) | No — API key only, eval in step 5 |
| `databricks_*` | UC-registered models on Model Serving | Yes — steps 3–4, then eval in step 5 |

`eval_matrix` in config lists every route scored at eval time. `model_serving` lists only `databricks_*` models for register and deploy.

---

## Legacy ASR scripts (`scripts/asr/`)

Lower-level training, registration, and bake-off scripts from the original EN-centric workflow. The ML ASR pipeline **reuses** some of these for UC registration; you normally do not run the full legacy sequence for eval.

| Script | Purpose |
|--------|---------|
| `01_asr_model_training.sh` | Data prep, manifests, GPU cluster lifecycle, baseline runs |
| `02_asr_baseline_runs.sh` | Candidate evaluation (Deepgram, Whisper baselines) |
| `03_asr_model_finetuning.sh` | Whisper LoRA fine-tuning |
| `04_asr_real_audio_holdout.sh` | Realistic held-out evaluation gate |
| `05_register_asr_model_candidate.sh` | Register EN finetuned Whisper LoRA in UC |
| `06_deploy_asr_model_serving_endpoint.sh` | Deploy a single ASR serving endpoint |
| `07_deep_voice_model_eval.sh` | Deepgram vs serving endpoint eval |
| `10_register_multilingual_asr_candidates.sh` | Register multilingual OSS baselines in UC |
| `11_multilingual_asr_finetuning.sh` | Multilingual LoRA fine-tuning |
| `12_multilingual_business_holdout_loop.sh` | Business holdout build + base vs LoRA loop |
| `13_multilingual_asr_promotion_gate.sh` | Read-only promotion decision report |

**Python modules:** `scripts/asr/databricks_*.py`, `mlflow_*_pyfunc.py` — Databricks job payloads invoked by the shell scripts.

**When to use legacy vs ML ASR:**

- **Multilingual FLEURS bake-off** → `scripts/ml_asr.sh` (steps 1–5 above).
- **EN LoRA training from a custom manifest** → `scripts/asr/01` → `03` → `05` → `06`.
- **Promotion gate after multilingual fine-tune** → `scripts/asr/13`.

More context: `backend/genie_voice/asr_eval/README.md`.

---

## Layout

```
scripts/
  ml_asr.sh              # ML ASR entry point (steps 1–5)
  ml_asr/
    01_datasets.sh
    02_quality.sh
    03_register.sh       # delegates to scripts/asr/05 and 10
    04_serve.sh
    05_eval.sh
    README.md
  asr/                   # legacy training + registration jobs
    01_ … 13_*.sh
    databricks_*.py
  README.md              # this file
```
