#!/usr/bin/env bash
# =============================================================================
# 11_multilingual_asr_finetuning.sh
#
# Fine-tune multilingual ASR winners after OSS-baseline validation.
#
# This is intentionally separate from 10_register_multilingual_asr_candidates.sh:
#   - 10_ registers/evaluates OSS baselines.
#   - 11_ trains real fine-tuned LoRA candidates and refuses to run without
#     real train/validation/holdout data.
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENGINE="$ROOT/scripts/asr/01_asr_model_training.sh"

ASR_ML_FINETUNE_CANDIDATE="${ASR_ML_FINETUNE_CANDIDATE:-}"
ASR_ML_FINETUNE_MANIFEST="${ASR_ML_FINETUNE_MANIFEST:-}"
ASR_ML_FINETUNE_OUTPUT_DIR="${ASR_ML_FINETUNE_OUTPUT_DIR:-$ROOT/.run/asr_model_training/multilingual_finetuning}"
ASR_ML_FINETUNE_PROFILE="${ASR_ML_FINETUNE_PROFILE:-${ASR_DATABRICKS_PROFILE:-${DATABRICKS_CONFIG_PROFILE:-fe-vm-vdm-classic-rcn6ip}}}"
ASR_ML_FINETUNE_GPU_CLUSTER_NAME="${ASR_ML_FINETUNE_GPU_CLUSTER_NAME:-genie-asr-gpu-training}"
ASR_ML_FINETUNE_MIN_TRAIN="${ASR_ML_FINETUNE_MIN_TRAIN:-50}"
ASR_ML_FINETUNE_MIN_VALIDATION="${ASR_ML_FINETUNE_MIN_VALIDATION:-10}"
ASR_ML_FINETUNE_MIN_HOLDOUT="${ASR_ML_FINETUNE_MIN_HOLDOUT:-10}"
ASR_ML_FINETUNE_MIN_ENTITY_ROWS="${ASR_ML_FINETUNE_MIN_ENTITY_ROWS:-10}"
ASR_ML_FINETUNE_EPOCHS="${ASR_ML_FINETUNE_EPOCHS:-1}"
ASR_ML_FINETUNE_LIMIT_TRAIN="${ASR_ML_FINETUNE_LIMIT_TRAIN:-}"
ASR_ML_FINETUNE_LIMIT_EVAL="${ASR_ML_FINETUNE_LIMIT_EVAL:-}"
ASR_ML_FINETUNE_REMOTE_ROOT="${ASR_ML_FINETUNE_REMOTE_ROOT:-}"
ASR_ML_FINETUNE_TEMPLATE="${ASR_ML_FINETUNE_TEMPLATE:-$ASR_ML_FINETUNE_OUTPUT_DIR/multilingual_business_manifest.template.jsonl}"
ASR_ML_FINETUNE_CSV="${ASR_ML_FINETUNE_CSV:-$ASR_ML_FINETUNE_OUTPUT_DIR/multilingual_business_manifest.template.csv}"
ASR_ML_FINETUNE_BUILT_MANIFEST="${ASR_ML_FINETUNE_BUILT_MANIFEST:-$ASR_ML_FINETUNE_OUTPUT_DIR/multilingual_business_manifest.built.jsonl}"
ASR_ML_FINETUNE_RUN_NAME="${ASR_ML_FINETUNE_RUN_NAME:-}"
ASR_ML_FINETUNE_RUN_TRAIN="${ASR_ML_FINETUNE_RUN_TRAIN:-false}"
ASR_ML_FINETUNE_RECORDING_PLAN="${ASR_ML_FINETUNE_RECORDING_PLAN:-$ASR_ML_FINETUNE_OUTPUT_DIR/multilingual_recording_plan.csv}"
ASR_ML_FINETUNE_RECORDING_PLAN_SPLIT="${ASR_ML_FINETUNE_RECORDING_PLAN_SPLIT:-}"
ASR_ML_FINETUNE_SYNTHETIC_BOOTSTRAP="${ASR_ML_FINETUNE_SYNTHETIC_BOOTSTRAP:-false}"
ASR_ML_FINETUNE_SYNTHETIC_SPLITS="${ASR_ML_FINETUNE_SYNTHETIC_SPLITS:-train,validation}"
ASR_ML_FINETUNE_SYNTHETIC_SOURCE="${ASR_ML_FINETUNE_SYNTHETIC_SOURCE:-synthetic_macos_say_bootstrap}"
ASR_ML_FINETUNE_SYNTHETIC_PREFLIGHT="${ASR_ML_FINETUNE_SYNTHETIC_PREFLIGHT:-true}"
ASR_ML_FINETUNE_SAY_VOICE="${ASR_ML_FINETUNE_SAY_VOICE:-}"
ASR_ML_FINETUNE_ALL_CANDIDATES="${ASR_ML_FINETUNE_ALL_CANDIDATES:-th_finetuned_pathumma_whisper_lora id_finetuned_whisper_large_v3_lora zh_finetuned_whisper_large_v3_lora}"
ASR_ML_FINETUNE_ALL_MIN_HOLDOUT="${ASR_ML_FINETUNE_ALL_MIN_HOLDOUT:-0}"
ASR_ML_FINETUNE_EVAL_SPLIT="${ASR_ML_FINETUNE_EVAL_SPLIT:-holdout}"
ASR_ML_FINETUNE_EVAL_LABEL="${ASR_ML_FINETUNE_EVAL_LABEL:-}"
ASR_ML_FINETUNE_FLEURS_LIMIT="${ASR_ML_FINETUNE_FLEURS_LIMIT:-10}"

COMMAND="${1:-help}"
if [[ $# -gt 0 ]]; then
  shift
fi

log() { printf "\033[36m[asr-ml-ft]\033[0m %s\n" "$*"; }
err() { printf "\033[31m[asr-ml-ft]\033[0m %s\n" "$*" >&2; }

usage() {
  cat <<'EOF'
Fine-tune multilingual ASR candidates.

Commands:
  run                One safe command: prepare, scaffold, build/validate, then stop or continue by gates.
  run-all-languages  Generate data and submit separate fine-tuning jobs for all trainable languages.
  evaluate-all-languages
                     Evaluate latest trainable language runs on ASR_ML_FINETUNE_EVAL_SPLIT.
  evaluate-base-all-languages
                     Evaluate base models on the same split/manifests for comparison.
  list               Print supported fine-tuning targets.
  plan               Explain current supported and blocked targets.
  volume             Show resolved Databricks Volume paths.
  prepare            Create multilingual ASR fine-tuning Volume folders.
  scaffold-recording-plan
                     Write a target-language recording checklist CSV.
  scaffold-recording-plans
                     Write recording checklist CSVs for th, id, and zh.
  scaffold-holdout-packs
                     Write holdout-only CSV packs for real recording/transcription.
  bootstrap-synthetic-recording-plan
                     Generate macOS say train/validation WAVs from a recording plan.
  bootstrap-synthetic-holdout-all
                     Generate synthetic holdout smoke manifests for all trainable languages.
  bootstrap-fleurs-holdout-all
                     Download FLEURS real speech and build external holdout manifests.
  materialize-recording-plan
                     Upload completed local recordings and build strict JSONL manifest.
  materialize-holdout-all
                     Upload completed holdout rows and build one manifest per language.
  scaffold-manifest  Write a JSONL template for real multilingual business data.
  scaffold-csv       Write a CSV template for easier data collection.
  build-manifest-from-csv
                     Convert ASR_ML_FINETUNE_CSV into strict JSONL manifest.
  validate-manifest  Validate candidate and real multilingual manifest; does not start training.
  preflight          Alias for validate-manifest.
  dry-run            Submit a tiny LoRA dry-run for supported Whisper targets.
  train-one          Submit real LoRA fine-tuning for supported Whisper targets.
  evaluate-lora      Evaluate a trained multilingual LoRA adapter.
  register-candidate Validate registration gates for a trained multilingual candidate.
  help               Show this help.

Environment:
  ASR_ML_FINETUNE_CANDIDATE       Required for preflight/dry-run/train-one.
  ASR_ML_FINETUNE_MANIFEST        Required JSONL manifest with train/validation/holdout rows.
  ASR_ML_FINETUNE_TEMPLATE        Output path for scaffold-manifest.
  ASR_ML_FINETUNE_CSV             Input/output CSV path for scaffold/build.
  ASR_ML_FINETUNE_BUILT_MANIFEST  Output JSONL path for build-manifest-from-csv.
  ASR_ML_FINETUNE_RECORDING_PLAN  Output CSV path for scaffold-recording-plan.
  ASR_ML_FINETUNE_RECORDING_PLAN_SPLIT
                                    Optional split filter for materialize-recording-plan.
  ASR_ML_FINETUNE_MIN_TRAIN       Min train rows/language. Default: 50.
  ASR_ML_FINETUNE_MIN_VALIDATION  Min validation rows/language. Default: 10.
  ASR_ML_FINETUNE_MIN_HOLDOUT     Min holdout rows/language. Default: 10.
  ASR_ML_FINETUNE_MIN_ENTITY_ROWS Min rows with expected_entities. Default: 10.
  ASR_ML_FINETUNE_RUN_NAME        Optional multilingual LoRA run name for evaluation/registration.
  ASR_ML_FINETUNE_RUN_TRAIN       true lets run submit train-one after dry-run gates. Default: false.
  ASR_ML_FINETUNE_SYNTHETIC_BOOTSTRAP
                                    true lets run bootstrap train/validation audio like English.
  ASR_ML_FINETUNE_SYNTHETIC_SPLITS  Comma-separated splits for synthetic generation.
  ASR_ML_FINETUNE_SYNTHETIC_PREFLIGHT
                                    true runs preflight after synthetic generation.
  ASR_ML_FINETUNE_SAY_VOICE       Optional macOS say voice override for synthetic bootstrap.
  ASR_ML_FINETUNE_ALL_CANDIDATES  Space-separated targets for run-all-languages.
  ASR_ML_FINETUNE_ALL_MIN_HOLDOUT Holdout minimum for run-all-languages. Default: 0.
  ASR_ML_FINETUNE_EVAL_SPLIT      Split for evaluate-lora. Default: holdout.
  ASR_ML_FINETUNE_FLEURS_LIMIT    External FLEURS clips/language. Default: 10.

Fine-tuning targets:
  th_finetuned_pathumma_whisper_lora       supported: Whisper LoRA
  id_finetuned_whisper_large_v3_lora       supported: Whisper LoRA
  zh_finetuned_whisper_large_v3_lora       supported: Whisper LoRA
  id_finetuned_qwen3_asr_0_6b_lora         blocked until Qwen ASR LoRA recipe is verified
  zh_finetuned_qwen3_asr_0_6b_lora         blocked until Qwen ASR LoRA recipe is verified

Example:
  ASR_ML_FINETUNE_CANDIDATE=th_finetuned_pathumma_whisper_lora \
  ASR_ML_FINETUNE_MANIFEST=/path/to/multilingual_business_manifest.jsonl \
    scripts/asr/11_multilingual_asr_finetuning.sh preflight

One-command workflow:
  ASR_ML_FINETUNE_CANDIDATE=th_finetuned_pathumma_whisper_lora \
  ASR_ML_FINETUNE_RECORDING_PLAN=/path/to/filled_th_recording_plan.csv \
    scripts/asr/11_multilingual_asr_finetuning.sh run

English-style synthetic bootstrap:
  ASR_ML_FINETUNE_CANDIDATE=th_finetuned_pathumma_whisper_lora \
  ASR_ML_FINETUNE_SYNTHETIC_BOOTSTRAP=true \
  ASR_ML_FINETUNE_MIN_HOLDOUT=0 \
    scripts/asr/11_multilingual_asr_finetuning.sh run

All-language bootstrap and submission:
  ASR_ML_FINETUNE_MIN_HOLDOUT=0 \
    scripts/asr/11_multilingual_asr_finetuning.sh run-all-languages

All-language bootstrap validation evaluation:
  ASR_ML_FINETUNE_EVAL_SPLIT=validation \
    scripts/asr/11_multilingual_asr_finetuning.sh evaluate-all-languages

All-language base model comparison:
  ASR_ML_FINETUNE_EVAL_SPLIT=holdout \
    scripts/asr/11_multilingual_asr_finetuning.sh evaluate-base-all-languages

All-language real holdout materialization:
  scripts/asr/11_multilingual_asr_finetuning.sh materialize-holdout-all
EOF
}

setup_env() {
  cd "$ROOT"
  mkdir -p "$ASR_ML_FINETUNE_OUTPUT_DIR"
  if [[ -x "$ROOT/.venv/bin/python" ]]; then
    PYTHON_BIN="$ROOT/.venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
  DBX=(databricks --profile "$ASR_ML_FINETUNE_PROFILE")
  export DATABRICKS_CONFIG_PROFILE="$ASR_ML_FINETUNE_PROFILE"
  export ASR_DATABRICKS_PROFILE="$ASR_ML_FINETUNE_PROFILE"
}

resolve_paths() {
  if [[ -n "$ASR_ML_FINETUNE_REMOTE_ROOT" ]]; then
    ASR_ML_FINETUNE_DATASETS="$ASR_ML_FINETUNE_REMOTE_ROOT/datasets/multilingual_gold"
    ASR_ML_FINETUNE_AUDIO="$ASR_ML_FINETUNE_DATASETS/audio"
    ASR_ML_FINETUNE_MANIFESTS="$ASR_ML_FINETUNE_DATASETS/manifests"
    ASR_ML_FINETUNE_EVALUATIONS="$ASR_ML_FINETUNE_REMOTE_ROOT/evaluations/multilingual"
    ASR_ML_FINETUNE_MODEL_ARTIFACTS="$ASR_ML_FINETUNE_REMOTE_ROOT/model_artifacts"
    ASR_ML_FINETUNE_LORA_RUNS="$ASR_ML_FINETUNE_MODEL_ARTIFACTS/multilingual_lora_runs"
    ASR_ML_FINETUNE_JOBS="$ASR_ML_FINETUNE_MODEL_ARTIFACTS/jobs"
    return
  fi
  ASR_ML_FINETUNE_REMOTE_ROOT="$("$PYTHON_BIN" <<'PY'
from __future__ import annotations

from genie_voice.config import get_settings

s = get_settings()
catalog = s.databricks.catalog
schema = s.databricks.schema_name
volume = s.volume.streaming_name
if any("<" in str(v) or not str(v).strip() for v in (catalog, schema, volume)):
    raise SystemExit("Databricks catalog/schema/streaming volume are not configured.")
print(f"/Volumes/{catalog}/{schema}/{volume}/asr_model_training")
PY
)"
  ASR_ML_FINETUNE_DATASETS="$ASR_ML_FINETUNE_REMOTE_ROOT/datasets/multilingual_gold"
  ASR_ML_FINETUNE_AUDIO="$ASR_ML_FINETUNE_DATASETS/audio"
  ASR_ML_FINETUNE_MANIFESTS="$ASR_ML_FINETUNE_DATASETS/manifests"
  ASR_ML_FINETUNE_EVALUATIONS="$ASR_ML_FINETUNE_REMOTE_ROOT/evaluations/multilingual"
  ASR_ML_FINETUNE_MODEL_ARTIFACTS="$ASR_ML_FINETUNE_REMOTE_ROOT/model_artifacts"
  ASR_ML_FINETUNE_LORA_RUNS="$ASR_ML_FINETUNE_MODEL_ARTIFACTS/multilingual_lora_runs"
  ASR_ML_FINETUNE_JOBS="$ASR_ML_FINETUNE_MODEL_ARTIFACTS/jobs"
}

show_volume() {
  setup_env
  resolve_paths
  cat <<EOF
Multilingual ASR fine-tuning Volume layout:

Root:
  $ASR_ML_FINETUNE_REMOTE_ROOT

Gold data:
  audio:     $ASR_ML_FINETUNE_AUDIO
  manifests: $ASR_ML_FINETUNE_MANIFESTS

Training artifacts:
  jobs:      $ASR_ML_FINETUNE_JOBS
  lora runs: $ASR_ML_FINETUNE_LORA_RUNS

Evaluations:
  $ASR_ML_FINETUNE_EVALUATIONS

EOF
}

prepare_volume() {
  setup_env
  resolve_paths
  for path in \
    "$ASR_ML_FINETUNE_AUDIO/th/train" \
    "$ASR_ML_FINETUNE_AUDIO/th/validation" \
    "$ASR_ML_FINETUNE_AUDIO/th/holdout" \
    "$ASR_ML_FINETUNE_AUDIO/id/train" \
    "$ASR_ML_FINETUNE_AUDIO/id/validation" \
    "$ASR_ML_FINETUNE_AUDIO/id/holdout" \
    "$ASR_ML_FINETUNE_AUDIO/zh/train" \
    "$ASR_ML_FINETUNE_AUDIO/zh/validation" \
    "$ASR_ML_FINETUNE_AUDIO/zh/holdout" \
    "$ASR_ML_FINETUNE_MANIFESTS" \
    "$ASR_ML_FINETUNE_EVALUATIONS" \
    "$ASR_ML_FINETUNE_LORA_RUNS" \
    "$ASR_ML_FINETUNE_JOBS"
  do
    "${DBX[@]}" fs mkdirs "dbfs:$path" >/dev/null
  done
  show_volume
}

target_ids() {
  printf '%s\n' \
    th_finetuned_pathumma_whisper_lora \
    id_finetuned_whisper_large_v3_lora \
    zh_finetuned_whisper_large_v3_lora \
    id_finetuned_qwen3_asr_0_6b_lora \
    zh_finetuned_qwen3_asr_0_6b_lora
}

scaffold_manifest() {
  setup_env
  mkdir -p "$(dirname "$ASR_ML_FINETUNE_TEMPLATE")"
  "$PYTHON_BIN" - "$ASR_ML_FINETUNE_TEMPLATE" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
rows = [
    {
        "clip_id": "th_train_0001",
        "audio_path": "/Volumes/<catalog>/<schema>/<volume>/asr_model_training/datasets/multilingual_gold/audio/th/train/th_train_0001.wav",
        "reference_transcript": "สวัสดีค่ะ ฉันโทรมาเรื่องใบแจ้งหนี้ INV-91000 ยอด 49 ดอลลาร์ไม่ถูกต้อง",
        "language": "th",
        "split": "train",
        "scenario": "billing_dispute",
        "speaker": 1,
        "duration_seconds": 7.2,
        "expected_entities": {
            "invoice_ids": ["INV-91000"],
            "amounts": ["49 dollars"],
            "billing_actions": ["dispute"],
            "confirmations": [],
            "refusals": [],
            "account_terms": ["invoice"],
        },
        "metadata": {
            "source": "real_recorded_business_holdout",
            "recording_channel": "browser_or_phone",
            "accent": "required",
            "background_noise": "required",
            "human_transcript_approved": True,
        },
    },
    {
        "clip_id": "id_validation_0001",
        "audio_path": "/Volumes/<catalog>/<schema>/<volume>/asr_model_training/datasets/multilingual_gold/audio/id/validation/id_validation_0001.wav",
        "reference_transcript": "Saya ingin mengonfirmasi pembayaran untuk invoice INV-81234 sebesar 125 dolar",
        "language": "id",
        "split": "validation",
        "scenario": "payment_confirmation",
        "speaker": 1,
        "duration_seconds": 6.5,
        "expected_entities": {
            "invoice_ids": ["INV-81234"],
            "amounts": ["125 dollars"],
            "billing_actions": ["payment"],
            "confirmations": ["mengonfirmasi"],
            "refusals": [],
            "account_terms": ["invoice"],
        },
        "metadata": {
            "source": "real_recorded_business_holdout",
            "recording_channel": "browser_or_phone",
            "accent": "required",
            "background_noise": "required",
            "human_transcript_approved": True,
        },
    },
    {
        "clip_id": "zh_holdout_0001",
        "audio_path": "/Volumes/<catalog>/<schema>/<volume>/asr_model_training/datasets/multilingual_gold/audio/zh/holdout/zh_holdout_0001.wav",
        "reference_transcript": "我想拒绝这张 INV-70021 发票上的 88 美元收费",
        "language": "zh",
        "split": "holdout",
        "scenario": "charge_refusal",
        "speaker": 1,
        "duration_seconds": 6.8,
        "expected_entities": {
            "invoice_ids": ["INV-70021"],
            "amounts": ["88 dollars"],
            "billing_actions": ["charge"],
            "confirmations": [],
            "refusals": ["拒绝"],
            "account_terms": ["发票"],
        },
        "metadata": {
            "source": "real_recorded_business_holdout",
            "recording_channel": "browser_or_phone",
            "accent": "required",
            "background_noise": "required",
            "human_transcript_approved": True,
        },
    },
]
with output.open("w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
print(json.dumps({"template": str(output), "rows": len(rows)}, indent=2, ensure_ascii=False))
PY
  cat >"$(dirname "$ASR_ML_FINETUNE_TEMPLATE")/README.md" <<'EOF'
# Multilingual ASR Fine-Tuning Manifest

This directory is for real recorded multilingual business ASR data.

The template JSONL is schema-only. It must not be used for training until every
placeholder path and metadata value is replaced with real data.

Required per row:
- `clip_id`
- `audio_path`
- `reference_transcript`
- `language`: `th`, `id`, or `zh`
- `split`: `train`, `validation`, or `holdout`
- `scenario`
- `speaker`
- `duration_seconds`
- `expected_entities.invoice_ids`
- `expected_entities.amounts`
- `expected_entities.billing_actions`
- `expected_entities.confirmations`
- `expected_entities.refusals`
- `expected_entities.account_terms`
- `metadata.source`
- `metadata.recording_channel`
- `metadata.accent`
- `metadata.background_noise`
- `metadata.human_transcript_approved`: must be `true`

Minimum gate before training:
- 50 train rows for the target language
- 10 validation rows for the target language
- 10 holdout rows for the target language
- 10 target-language rows with expected entities

Validate before training:

```bash
ASR_ML_FINETUNE_CANDIDATE=th_finetuned_pathumma_whisper_lora \
ASR_ML_FINETUNE_MANIFEST=/path/to/real_multilingual_business_manifest.jsonl \
scripts/asr/11_multilingual_asr_finetuning.sh validate-manifest
```

CSV workflow:

```bash
scripts/asr/11_multilingual_asr_finetuning.sh scaffold-csv

ASR_ML_FINETUNE_CSV=/path/to/filled_recordings.csv \
ASR_ML_FINETUNE_BUILT_MANIFEST=/path/to/real_multilingual_business_manifest.jsonl \
scripts/asr/11_multilingual_asr_finetuning.sh build-manifest-from-csv
```
EOF
}

scaffold_csv() {
  setup_env
  mkdir -p "$(dirname "$ASR_ML_FINETUNE_CSV")"
  "$PYTHON_BIN" - "$ASR_ML_FINETUNE_CSV" <<'PY'
from __future__ import annotations

import csv
import sys
from pathlib import Path

output = Path(sys.argv[1])
columns = [
    "clip_id",
    "audio_path",
    "reference_transcript",
    "language",
    "split",
    "scenario",
    "speaker",
    "duration_seconds",
    "invoice_ids",
    "amounts",
    "billing_actions",
    "confirmations",
    "refusals",
    "account_terms",
    "metadata_source",
    "recording_channel",
    "accent",
    "background_noise",
    "human_transcript_approved",
]
rows = [
    {
        "clip_id": "th_train_0001",
        "audio_path": "/Volumes/<catalog>/<schema>/<volume>/asr_model_training/datasets/multilingual_gold/audio/th/train/th_train_0001.wav",
        "reference_transcript": "สวัสดีค่ะ ฉันโทรมาเรื่องใบแจ้งหนี้ INV-91000 ยอด 49 ดอลลาร์ไม่ถูกต้อง",
        "language": "th",
        "split": "train",
        "scenario": "billing_dispute",
        "speaker": "1",
        "duration_seconds": "7.2",
        "invoice_ids": "INV-91000",
        "amounts": "49 dollars",
        "billing_actions": "dispute",
        "confirmations": "",
        "refusals": "",
        "account_terms": "invoice",
        "metadata_source": "real_recorded_business_holdout",
        "recording_channel": "browser_or_phone",
        "accent": "thai_central",
        "background_noise": "office_low",
        "human_transcript_approved": "true",
    }
]
with output.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
print(f"Wrote CSV template: {output}")
PY
}

scaffold_recording_plan() {
  setup_env
  resolve_paths
  if [[ -z "$ASR_ML_FINETUNE_CANDIDATE" ]]; then
    err "ASR_ML_FINETUNE_CANDIDATE is required for scaffold-recording-plan"
    target_ids >&2
    exit 2
  fi
  load_target "$ASR_ML_FINETUNE_CANDIDATE"
  mkdir -p "$(dirname "$ASR_ML_FINETUNE_RECORDING_PLAN")"
  "$PYTHON_BIN" - "$ASR_ML_FINETUNE_RECORDING_PLAN" "$LANGUAGE_CODE" "$ASR_ML_FINETUNE_MIN_TRAIN" "$ASR_ML_FINETUNE_MIN_VALIDATION" "$ASR_ML_FINETUNE_MIN_HOLDOUT" "$ASR_ML_FINETUNE_AUDIO" <<'PY'
from __future__ import annotations

import csv
import sys
from pathlib import Path

output = Path(sys.argv[1])
language = sys.argv[2]
mins = {
    "train": int(sys.argv[3]),
    "validation": int(sys.argv[4]),
    "holdout": int(sys.argv[5]),
}

audio_root = sys.argv[6]

localized = {
    "th": {
        "accent": "thai_central",
        "confirm": "ยืนยัน",
        "refuse": "ปฏิเสธ",
        "invoice": "ใบแจ้งหนี้",
        "currency": "ดอลลาร์",
        "scenario_specs": [
            ("billing_dispute", "สวัสดีค่ะ ฉันโทรมาเรื่องใบแจ้งหนี้ {invoice_id} ยอด {amount} ดอลลาร์ไม่ถูกต้อง", "dispute", "", ""),
            ("payment_lookup", "ช่วยตรวจสอบการชำระเงินของใบแจ้งหนี้ {invoice_id} จำนวน {amount} ดอลลาร์ให้หน่อย", "payment", "", ""),
            ("payment_confirmation", "ฉันต้องการยืนยันว่าใบแจ้งหนี้ {invoice_id} ถูกชำระแล้ว", "payment", "ยืนยัน", ""),
            ("charge_refusal", "ฉันขอปฏิเสธค่าบริการ {amount} ดอลลาร์ในใบแจ้งหนี้ {invoice_id}", "charge", "", "ปฏิเสธ"),
            ("account_balance", "บัญชีของฉันยังแสดงยอดค้างชำระสำหรับใบแจ้งหนี้ {invoice_id}", "balance", "", ""),
        ],
    },
    "id": {
        "accent": "indonesian_jakarta",
        "confirm": "konfirmasi",
        "refuse": "menolak",
        "invoice": "invoice",
        "currency": "dolar",
        "scenario_specs": [
            ("billing_dispute", "Saya menelepon tentang invoice {invoice_id} dengan tagihan {amount} dolar yang salah", "dispute", "", ""),
            ("payment_lookup", "Tolong periksa pembayaran untuk invoice {invoice_id} sebesar {amount} dolar", "payment", "", ""),
            ("payment_confirmation", "Saya ingin mengonfirmasi bahwa invoice {invoice_id} sudah dibayar", "payment", "konfirmasi", ""),
            ("charge_refusal", "Saya menolak biaya {amount} dolar pada invoice {invoice_id}", "charge", "", "menolak"),
            ("account_balance", "Akun saya masih menunjukkan saldo tertunggak untuk invoice {invoice_id}", "balance", "", ""),
        ],
    },
    "zh": {
        "accent": "mandarin_mainland",
        "confirm": "确认",
        "refuse": "拒绝",
        "invoice": "发票",
        "currency": "美元",
        "scenario_specs": [
            ("billing_dispute", "我打电话是因为发票 {invoice_id} 上的 {amount} 美元收费不正确", "dispute", "", ""),
            ("payment_lookup", "请帮我检查发票 {invoice_id} 的 {amount} 美元付款", "payment", "", ""),
            ("payment_confirmation", "我想确认发票 {invoice_id} 已经付款", "payment", "确认", ""),
            ("charge_refusal", "我想拒绝发票 {invoice_id} 上的 {amount} 美元收费", "charge", "", "拒绝"),
            ("account_balance", "我的账户仍然显示发票 {invoice_id} 有未付余额", "balance", "", ""),
        ],
    },
}
cfg = localized[language]
columns = [
    "clip_id",
    "language",
    "split",
    "scenario",
    "recording_prompt",
    "local_audio_path",
    "approved_reference_transcript",
    "suggested_audio_path",
    "duration_seconds",
    "invoice_ids",
    "amounts",
    "billing_actions",
    "confirmations",
    "refusals",
    "account_terms",
    "recording_channel",
    "accent",
    "background_noise",
    "human_transcript_approved",
    "notes",
]
rows = []
for split, count in mins.items():
    for idx in range(1, count + 1):
        invoice_id = f"INV-{language.upper()}{idx:05d}"
        amount = str(25 + (idx * 7) % 300)
        scenario, template, billing_action, confirmation, refusal = cfg["scenario_specs"][
            (idx - 1) % len(cfg["scenario_specs"])
        ]
        prompt = template.format(
            invoice_id=invoice_id,
            amount=amount,
        )
        clip_id = f"{language}_{split}_{idx:04d}"
        rows.append(
            {
                "clip_id": clip_id,
                "language": language,
                "split": split,
                "scenario": scenario,
                "recording_prompt": prompt,
                "local_audio_path": "",
                "approved_reference_transcript": "",
                "suggested_audio_path": f"{audio_root}/{language}/{split}/{clip_id}.wav",
                "duration_seconds": "",
                "invoice_ids": invoice_id,
                "amounts": f"{amount} dollars",
                "billing_actions": billing_action,
                "confirmations": confirmation,
                "refusals": refusal,
                "account_terms": cfg["invoice"],
                "recording_channel": "browser_or_phone",
                "accent": cfg["accent"],
                "background_noise": "office_low",
                "human_transcript_approved": "false",
                "notes": "Record real audio, then replace false with true only after human transcript approval.",
            }
        )
with output.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
print(f"Wrote recording plan: {output} ({len(rows)} rows)")
PY
}

scaffold_recording_plans() {
  setup_env
  mkdir -p "$ASR_ML_FINETUNE_OUTPUT_DIR"
  local candidate
  for candidate in \
    th_finetuned_pathumma_whisper_lora \
    id_finetuned_whisper_large_v3_lora \
    zh_finetuned_whisper_large_v3_lora
  do
    local language
    case "$candidate" in
      th_*) language=th ;;
      id_*) language=id ;;
      zh_*) language=zh ;;
      *) err "Unable to infer language for $candidate"; exit 2 ;;
    esac
    ASR_ML_FINETUNE_CANDIDATE="$candidate" \
    ASR_ML_FINETUNE_RECORDING_PLAN="$ASR_ML_FINETUNE_OUTPUT_DIR/${language}_recording_plan.csv" \
      "$0" scaffold-recording-plan
  done
}

scaffold_holdout_packs() {
  setup_env
  mkdir -p "$ASR_ML_FINETUNE_OUTPUT_DIR/holdout_packs"
  scaffold_recording_plans
  "$PYTHON_BIN" - "$ASR_ML_FINETUNE_OUTPUT_DIR" <<'PY'
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
pack_dir = output_dir / "holdout_packs"
pack_dir.mkdir(parents=True, exist_ok=True)
languages = {
    "th": "Thai (th-TH, Thailand)",
    "id": "Indonesian (id-ID, Indonesia)",
    "zh": "Mandarin Chinese (zh-CN, Mainland China)",
}
summary = []
for language, label in languages.items():
    source = output_dir / f"{language}_recording_plan.csv"
    target = pack_dir / f"{language}_holdout_recording_pack.csv"
    if not source.exists():
        raise SystemExit(f"Missing recording plan: {source}")
    with source.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader if row.get("split") == "holdout"]
        fieldnames = reader.fieldnames or []
    with target.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary.append(
        {
            "language": language,
            "label": label,
            "rows": len(rows),
            "pack": str(target),
            "required_fill_fields": [
                "local_audio_path",
                "approved_reference_transcript",
                "duration_seconds",
                "human_transcript_approved",
            ],
        }
    )
(pack_dir / "holdout_recording_pack_summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False),
    encoding="utf-8",
)
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY
}

build_manifest_from_csv() {
  setup_env
  if [[ ! -s "$ASR_ML_FINETUNE_CSV" ]]; then
    err "ASR_ML_FINETUNE_CSV does not exist or is empty: $ASR_ML_FINETUNE_CSV"
    exit 2
  fi
  mkdir -p "$(dirname "$ASR_ML_FINETUNE_BUILT_MANIFEST")"
  "$PYTHON_BIN" - "$ASR_ML_FINETUNE_CSV" "$ASR_ML_FINETUNE_BUILT_MANIFEST" <<'PY'
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

csv_path = Path(sys.argv[1])
output = Path(sys.argv[2])
required = {
    "clip_id",
    "audio_path",
    "reference_transcript",
    "language",
    "split",
    "scenario",
    "speaker",
    "duration_seconds",
    "metadata_source",
    "recording_channel",
    "accent",
    "background_noise",
    "human_transcript_approved",
}
entity_fields = [
    "invoice_ids",
    "amounts",
    "billing_actions",
    "confirmations",
    "refusals",
    "account_terms",
]

def values(text: str) -> list[str]:
    return [item.strip() for item in str(text or "").split("|") if item.strip()]

with csv_path.open("r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    missing = required - set(reader.fieldnames or [])
    if missing:
        raise SystemExit(f"CSV missing required columns: {sorted(missing)}")
    rows = []
    for row in reader:
        rows.append(
            {
                "clip_id": row["clip_id"].strip(),
                "audio_path": row["audio_path"].strip(),
                "reference_transcript": row["reference_transcript"].strip(),
                "language": row["language"].strip(),
                "split": row["split"].strip(),
                "scenario": row["scenario"].strip(),
                "speaker": int(row["speaker"] or 1),
                "duration_seconds": float(row["duration_seconds"] or 0),
                "expected_entities": {field: values(row.get(field, "")) for field in entity_fields},
                "metadata": {
                    "source": row["metadata_source"].strip(),
                    "recording_channel": row["recording_channel"].strip(),
                    "accent": row["accent"].strip(),
                    "background_noise": row["background_noise"].strip(),
                    "human_transcript_approved": row["human_transcript_approved"].strip().lower() == "true",
                },
            }
        )
with output.open("w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
print(json.dumps({"manifest": str(output), "rows": len(rows)}, indent=2, ensure_ascii=False))
PY
  if [[ -n "$ASR_ML_FINETUNE_CANDIDATE" ]]; then
    ASR_ML_FINETUNE_MANIFEST="$ASR_ML_FINETUNE_BUILT_MANIFEST" preflight
  fi
}

materialize_recording_plan() {
  setup_env
  if [[ ! -s "$ASR_ML_FINETUNE_RECORDING_PLAN" ]]; then
    err "ASR_ML_FINETUNE_RECORDING_PLAN does not exist or is empty: $ASR_ML_FINETUNE_RECORDING_PLAN"
    exit 2
  fi
  mkdir -p "$(dirname "$ASR_ML_FINETUNE_BUILT_MANIFEST")"
  "$PYTHON_BIN" - "$ASR_ML_FINETUNE_RECORDING_PLAN" "$ASR_ML_FINETUNE_BUILT_MANIFEST" "$ASR_ML_FINETUNE_RECORDING_PLAN_SPLIT" <<'PY'
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

plan = Path(sys.argv[1])
manifest = Path(sys.argv[2])
split_filter = sys.argv[3].strip()
required = {
    "clip_id",
    "language",
    "split",
    "scenario",
    "recording_prompt",
    "local_audio_path",
    "approved_reference_transcript",
    "suggested_audio_path",
    "duration_seconds",
    "invoice_ids",
    "amounts",
    "billing_actions",
    "confirmations",
    "refusals",
    "account_terms",
    "recording_channel",
    "accent",
    "background_noise",
    "human_transcript_approved",
}

def values(text: str) -> list[str]:
    return [item.strip() for item in str(text or "").split("|") if item.strip()]

with plan.open("r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    missing = required - set(reader.fieldnames or [])
    if missing:
        raise SystemExit(f"recording plan missing required columns: {sorted(missing)}")
    upload_pairs: list[tuple[str, str]] = []
    manifest_rows = []
    errors: list[str] = []
    for line_no, row in enumerate(reader, start=2):
        if split_filter and row["split"].strip() != split_filter:
            continue
        local_audio_text = row["local_audio_path"].strip()
        local_audio = Path(local_audio_text)
        suggested_audio = row["suggested_audio_path"].strip()
        approved_transcript = row["approved_reference_transcript"].strip()
        approved = row["human_transcript_approved"].strip().lower() == "true"
        try:
            duration_seconds = float(row["duration_seconds"] or 0)
        except ValueError:
            duration_seconds = 0
        if not local_audio_text:
            errors.append(f"row {line_no} local_audio_path is required")
        elif not local_audio.exists():
            errors.append(f"row {line_no} local_audio_path does not exist: {local_audio}")
        if not suggested_audio.startswith("/Volumes/"):
            errors.append(f"row {line_no} suggested_audio_path must be a /Volumes path: {suggested_audio}")
        if not approved_transcript:
            errors.append(f"row {line_no} approved_reference_transcript is required")
        if not approved:
            errors.append(f"row {line_no} human_transcript_approved must be true")
        if duration_seconds <= 0:
            errors.append(f"row {line_no} duration_seconds must be > 0")
        upload_pairs.append((str(local_audio), suggested_audio))
        manifest_rows.append(
            {
                "clip_id": row["clip_id"].strip(),
                "audio_path": suggested_audio,
                "reference_transcript": approved_transcript,
                "language": row["language"].strip(),
                "split": row["split"].strip(),
                "scenario": row["scenario"].strip(),
                "speaker": 1,
                "duration_seconds": duration_seconds,
                "expected_entities": {
                    "invoice_ids": values(row["invoice_ids"]),
                    "amounts": values(row["amounts"]),
                    "billing_actions": values(row["billing_actions"]),
                    "confirmations": values(row["confirmations"]),
                    "refusals": values(row["refusals"]),
                    "account_terms": values(row["account_terms"]),
                },
                "metadata": {
                    "source": "real_recorded_business_holdout",
                    "recording_channel": row["recording_channel"].strip(),
                    "accent": row["accent"].strip(),
                    "background_noise": row["background_noise"].strip(),
                    "human_transcript_approved": approved,
                    "recording_prompt": row["recording_prompt"].strip(),
                },
            }
        )
if errors:
    print(
        json.dumps(
            {
                "status": "failed",
                "total_errors": len(errors),
                "shown_errors": min(len(errors), 50),
                "errors": errors[:50],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    raise SystemExit(2)
if not manifest_rows:
    raise SystemExit(f"No recording plan rows selected for split={split_filter!r}")
manifest.with_suffix(".uploads.json").write_text(
    json.dumps([{"local": local, "remote": remote} for local, remote in upload_pairs], indent=2),
    encoding="utf-8",
)
with manifest.open("w", encoding="utf-8") as f:
    for row in manifest_rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
print(json.dumps({"manifest": str(manifest), "rows": len(manifest_rows), "uploads": str(manifest.with_suffix('.uploads.json'))}, indent=2))
PY

  local uploads_json="${ASR_ML_FINETUNE_BUILT_MANIFEST%.jsonl}.uploads.json"
  "$PYTHON_BIN" - "$uploads_json" <<'PY' | while IFS=$'\t' read -r local_path remote_path; do
from __future__ import annotations

import json
import sys
from pathlib import Path

for item in json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")):
    print(f"{item['local']}\t{item['remote']}")
PY
    "${DBX[@]}" fs mkdirs "dbfs:$(dirname "$remote_path")" >/dev/null
    "${DBX[@]}" fs cp "$local_path" "dbfs:$remote_path" --overwrite >/dev/null
  done

  if [[ -n "$ASR_ML_FINETUNE_CANDIDATE" ]]; then
    ASR_ML_FINETUNE_MANIFEST="$ASR_ML_FINETUNE_BUILT_MANIFEST" preflight
  fi
}

bootstrap_synthetic_recording_plan() {
  setup_env
  resolve_paths
  if [[ -z "$ASR_ML_FINETUNE_CANDIDATE" ]]; then
    err "ASR_ML_FINETUNE_CANDIDATE is required for bootstrap-synthetic-recording-plan"
    target_ids >&2
    exit 2
  fi
  load_target "$ASR_ML_FINETUNE_CANDIDATE"
  if [[ ! -s "$ASR_ML_FINETUNE_RECORDING_PLAN" ]]; then
    scaffold_recording_plan
  fi
  if ! command -v say >/dev/null 2>&1; then
    err "macOS 'say' is unavailable; cannot generate synthetic bootstrap audio."
    exit 2
  fi
  if ! command -v afconvert >/dev/null 2>&1; then
    err "macOS 'afconvert' is unavailable; cannot convert synthetic audio to WAV."
    exit 2
  fi

  local local_audio_dir="$ASR_ML_FINETUNE_OUTPUT_DIR/synthetic_bootstrap/audio/$LANGUAGE_CODE"
  mkdir -p "$local_audio_dir" "$(dirname "$ASR_ML_FINETUNE_BUILT_MANIFEST")"
  "$PYTHON_BIN" - "$ASR_ML_FINETUNE_RECORDING_PLAN" "$ASR_ML_FINETUNE_BUILT_MANIFEST" "$local_audio_dir" "$ASR_ML_FINETUNE_AUDIO" "$LANGUAGE_CODE" "$ASR_ML_FINETUNE_SAY_VOICE" "$ASR_ML_FINETUNE_SYNTHETIC_SPLITS" "$ASR_ML_FINETUNE_SYNTHETIC_SOURCE" <<'PY'
from __future__ import annotations

import csv
import json
import subprocess
import sys
import wave
from pathlib import Path

plan = Path(sys.argv[1])
manifest = Path(sys.argv[2])
local_audio_dir = Path(sys.argv[3])
volume_audio_dir = sys.argv[4].rstrip("/")
language = sys.argv[5]
voice_override = sys.argv[6].strip()
wanted_splits = {split.strip() for split in sys.argv[7].split(",") if split.strip()}
synthetic_source = sys.argv[8].strip() or "synthetic_macos_say_bootstrap"

voice_preferences = {
    "th": ["Kanya"],
    "id": ["Damayanti"],
    "zh": ["Tingting", "Sin-ji", "Mei-Jia", "Li-mu"],
}
locale_preferences = {
    "th": ["th_TH"],
    "id": ["id_ID"],
    "zh": ["zh_CN", "zh_TW", "zh_HK"],
}

def available_voices() -> list[tuple[str, str]]:
    result = subprocess.run(["say", "-v", "?"], text=True, capture_output=True, check=True)
    voices = []
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            voices.append((parts[0], parts[1]))
    return voices

def choose_voice() -> str:
    if voice_override:
        return voice_override
    voices = available_voices()
    for preferred in voice_preferences.get(language, []):
        if any(name == preferred for name, _locale in voices):
            return preferred
    for locale_prefix in locale_preferences.get(language, []):
        for name, locale in voices:
            if locale.startswith(locale_prefix):
                return name
    raise SystemExit(
        "No suitable macOS say voice found for language="
        f"{language}. Set ASR_ML_FINETUNE_SAY_VOICE to an installed voice."
    )

def values(text: str) -> list[str]:
    return [item.strip() for item in str(text or "").split("|") if item.strip()]

def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        return round(wav.getnframes() / float(wav.getframerate()), 3)

voice = choose_voice()
rows = []
uploads = []
with plan.open("r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    required = {
        "clip_id",
        "language",
        "split",
        "scenario",
        "recording_prompt",
        "invoice_ids",
        "amounts",
        "billing_actions",
        "confirmations",
        "refusals",
        "account_terms",
        "recording_channel",
        "accent",
        "background_noise",
    }
    missing = required - set(reader.fieldnames or [])
    if missing:
        raise SystemExit(f"recording plan missing required columns: {sorted(missing)}")
    for row in reader:
        if row["language"].strip() != language:
            continue
        split = row["split"].strip()
        if split not in wanted_splits:
            continue
        prompt = row["recording_prompt"].strip()
        if not prompt:
            continue
        clip_id = row["clip_id"].strip()
        wav_path = local_audio_dir / split / f"{clip_id}.wav"
        aiff_path = wav_path.with_suffix(".aiff")
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        if not wav_path.exists():
            subprocess.run(["say", "-v", voice, "-o", str(aiff_path), prompt], check=True)
            subprocess.run(["afconvert", str(aiff_path), str(wav_path), "-f", "WAVE", "-d", "LEI16@16000"], check=True)
            aiff_path.unlink(missing_ok=True)
        remote_audio = f"{volume_audio_dir}/{language}/{split}/{clip_id}.wav"
        uploads.append({"local": str(wav_path), "remote": remote_audio})
        rows.append(
            {
                "clip_id": clip_id,
                "audio_path": remote_audio,
                "audio_format": "audio/wav",
                "sample_rate_hz": 16000,
                "duration_seconds": wav_duration(wav_path),
                "reference_transcript": prompt,
                "language": language,
                "split": split,
                "scenario": row["scenario"].strip(),
                "speaker": "synthetic",
                "domain": "billing_support",
                "dataset_version": "multilingual_synthetic_macos_say_bootstrap_v1",
                "expected_entities": {
                    "invoice_ids": values(row["invoice_ids"]),
                    "amounts": values(row["amounts"]),
                    "billing_actions": values(row["billing_actions"]),
                    "confirmations": values(row["confirmations"]),
                    "refusals": values(row["refusals"]),
                    "account_terms": values(row["account_terms"]),
                },
                "metadata": {
                    "source": synthetic_source,
                    "synthetic_audio_source": "macos_say",
                    "say_voice": voice,
                    "recording_channel": row["recording_channel"].strip(),
                    "accent": row["accent"].strip(),
                    "background_noise": row["background_noise"].strip(),
                    "human_transcript_approved": False,
                    "replace_with_real_call_audio_before_production": True,
                },
            }
        )
if not rows:
    raise SystemExit(f"No rows found in recording plan for language={language} splits={sorted(wanted_splits)}")
manifest.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
manifest.with_suffix(".uploads.json").write_text(json.dumps(uploads, indent=2), encoding="utf-8")
print(json.dumps({"manifest": str(manifest), "rows": len(rows), "voice": voice}, indent=2, ensure_ascii=False))
PY

  local uploads_json="${ASR_ML_FINETUNE_BUILT_MANIFEST%.jsonl}.uploads.json"
  "$PYTHON_BIN" - "$uploads_json" <<'PY' | while IFS=$'\t' read -r local_path remote_path; do
from __future__ import annotations

import json
import sys
from pathlib import Path

for item in json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")):
    print(f"{item['local']}\t{item['remote']}")
PY
    "${DBX[@]}" fs mkdirs "dbfs:$(dirname "$remote_path")" >/dev/null
    "${DBX[@]}" fs cp "$local_path" "dbfs:$remote_path" --overwrite >/dev/null
  done

  if [[ "$ASR_ML_FINETUNE_SYNTHETIC_PREFLIGHT" == "true" ]]; then
    ASR_ML_FINETUNE_MANIFEST="$ASR_ML_FINETUNE_BUILT_MANIFEST" preflight
  fi
}

load_target() {
  local candidate="$1"
  CANDIDATE_ID="$candidate"
  case "$candidate" in
    th_finetuned_pathumma_whisper_lora)
      RECIPE=whisper_lora
      FAMILY=whisper
      BASE_MODEL="nectec/Pathumma-whisper-th-large-v3"
      LANGUAGE_CODE=th
      LANGUAGE_NAME=Thai
      LANGUAGE_LOCALE_DISPLAY="Thai (th-TH, Thailand)"
      ;;
    id_finetuned_whisper_large_v3_lora)
      RECIPE=whisper_lora
      FAMILY=whisper
      BASE_MODEL="openai/whisper-large-v3"
      LANGUAGE_CODE=id
      LANGUAGE_NAME=Indonesian
      LANGUAGE_LOCALE_DISPLAY="Indonesian (id-ID, Indonesia)"
      ;;
    zh_finetuned_whisper_large_v3_lora)
      RECIPE=whisper_lora
      FAMILY=whisper
      BASE_MODEL="openai/whisper-large-v3"
      LANGUAGE_CODE=zh
      LANGUAGE_NAME=Chinese
      LANGUAGE_LOCALE_DISPLAY="Mandarin Chinese (zh-CN, Mainland China)"
      ;;
    id_finetuned_qwen3_asr_0_6b_lora)
      RECIPE=blocked_qwen_lora
      FAMILY=qwen3
      BASE_MODEL="Qwen/Qwen3-ASR-0.6B"
      LANGUAGE_CODE=id
      LANGUAGE_NAME=Indonesian
      LANGUAGE_LOCALE_DISPLAY="Indonesian (id-ID, Indonesia)"
      ;;
    zh_finetuned_qwen3_asr_0_6b_lora)
      RECIPE=blocked_qwen_lora
      FAMILY=qwen3
      BASE_MODEL="Qwen/Qwen3-ASR-0.6B"
      LANGUAGE_CODE=zh
      LANGUAGE_NAME=Chinese
      LANGUAGE_LOCALE_DISPLAY="Mandarin Chinese (zh-CN, Mainland China)"
      ;;
    *)
      err "Unknown fine-tuning target: $candidate"
      target_ids >&2
      exit 2
      ;;
  esac
}

sync_manifest_to_local() {
  if [[ -z "$ASR_ML_FINETUNE_MANIFEST" ]]; then
    err "ASR_ML_FINETUNE_MANIFEST is required"
    exit 2
  fi
  local local_manifest="$ASR_ML_FINETUNE_OUTPUT_DIR/manifest_for_preflight.jsonl"
  if [[ "$ASR_ML_FINETUNE_MANIFEST" == /Volumes/* ]]; then
    "${DBX[@]}" fs cp "dbfs:$ASR_ML_FINETUNE_MANIFEST" "$local_manifest" --overwrite >/dev/null
    printf '%s\n' "$local_manifest"
    return
  fi
  if [[ ! -s "$ASR_ML_FINETUNE_MANIFEST" ]]; then
    err "Manifest does not exist or is empty: $ASR_ML_FINETUNE_MANIFEST"
    exit 2
  fi
  printf '%s\n' "$ASR_ML_FINETUNE_MANIFEST"
}

validate_manifest() {
  local manifest_local="$1"
  "$PYTHON_BIN" - "$manifest_local" "$LANGUAGE_CODE" "$ASR_ML_FINETUNE_MIN_TRAIN" "$ASR_ML_FINETUNE_MIN_VALIDATION" "$ASR_ML_FINETUNE_MIN_HOLDOUT" "$ASR_ML_FINETUNE_MIN_ENTITY_ROWS" <<'PY'
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

manifest = Path(sys.argv[1])
language = sys.argv[2]
mins = {
    "train": int(sys.argv[3]),
    "validation": int(sys.argv[4]),
    "holdout": int(sys.argv[5]),
}
min_entity_rows = int(sys.argv[6])

counts: Counter[str] = Counter()
entity_rows = 0
errors: list[str] = []
allowed_languages = {"th", "id", "zh"}
allowed_splits = {"train", "validation", "holdout"}
required_entity_groups = {
    "invoice_ids",
    "amounts",
    "billing_actions",
    "confirmations",
    "refusals",
    "account_terms",
}
required_metadata = {
    "source",
    "recording_channel",
    "accent",
    "background_noise",
    "human_transcript_approved",
}
for line_no, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), start=1):
    if not line.strip() or line.lstrip().startswith("#"):
        continue
    row = json.loads(line)
    for key in ("clip_id", "audio_path", "reference_transcript", "language", "split"):
        if not row.get(key):
            errors.append(f"row {line_no} missing {key}")
    audio_path = str(row.get("audio_path") or "")
    if "<" in audio_path or ">" in audio_path:
        errors.append(f"row {line_no} audio_path still contains a placeholder: {audio_path}")
    elif not audio_path.startswith("/Volumes/") and not Path(audio_path).exists():
        errors.append(f"row {line_no} local audio_path does not exist: {audio_path}")
    if row.get("language") not in allowed_languages:
        errors.append(f"row {line_no} language must be one of {sorted(allowed_languages)}")
    if row.get("split") not in allowed_splits:
        errors.append(f"row {line_no} split must be one of {sorted(allowed_splits)}")
    if not str(row.get("reference_transcript") or "").strip():
        errors.append(f"row {line_no} reference_transcript is empty")
    if float(row.get("duration_seconds") or 0) <= 0:
        errors.append(f"row {line_no} duration_seconds must be > 0")
    expected_entities = row.get("expected_entities")
    if not isinstance(expected_entities, dict):
        errors.append(f"row {line_no} expected_entities must be an object")
        expected_entities = {}
    else:
        missing_groups = required_entity_groups - set(expected_entities)
        if missing_groups:
            errors.append(f"row {line_no} expected_entities missing groups: {sorted(missing_groups)}")
        for group, values in expected_entities.items():
            if not isinstance(values, list):
                errors.append(f"row {line_no} expected_entities.{group} must be a list")
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        errors.append(f"row {line_no} metadata must be an object")
        metadata = {}
    else:
        missing_metadata = required_metadata - set(metadata)
        if missing_metadata:
            errors.append(f"row {line_no} metadata missing fields: {sorted(missing_metadata)}")
        source = str(metadata.get("source") or "").strip().lower()
        split = str(row.get("split") or "")
        is_synthetic = source.startswith("synthetic_") or "synthetic" in source
        if split == "holdout":
            if metadata.get("human_transcript_approved") is not True:
                errors.append(f"row {line_no} holdout metadata.human_transcript_approved must be true")
            if is_synthetic:
                errors.append(f"row {line_no} holdout rows cannot use synthetic source: {source!r}")
        elif not is_synthetic and metadata.get("human_transcript_approved") is not True:
            errors.append(
                f"row {line_no} metadata.human_transcript_approved must be true unless source is synthetic"
            )
        for key in ("source", "recording_channel", "accent", "background_noise"):
            value = str(metadata.get(key) or "").strip().lower()
            if not value or value in {"required", "todo", "unknown", "n/a"}:
                errors.append(f"row {line_no} metadata.{key} must be a real value, not {value!r}")
    if row.get("language") != language:
        continue
    split = str(row.get("split") or "")
    counts[split] += 1
    if any(values for values in expected_entities.values() if isinstance(values, list)):
        entity_rows += 1

for split, minimum in mins.items():
    if counts[split] < minimum:
        errors.append(f"language={language} split={split} has {counts[split]} rows, needs >= {minimum}")
if entity_rows < min_entity_rows:
    errors.append(
        f"language={language} has {entity_rows} rows with expected_entities, needs >= {min_entity_rows}"
    )

if errors:
    print(json.dumps({"status": "failed", "language": language, "counts": counts, "errors": errors}, indent=2))
    raise SystemExit(2)
print(json.dumps({"status": "ok", "language": language, "counts": counts, "entity_rows": entity_rows}, indent=2))
PY
}

gpu_cluster_id() {
  local clusters_json
  if ! clusters_json="$("${DBX[@]}" clusters list --output json 2>&1)"; then
    err "Unable to list Databricks clusters."
    err "$clusters_json"
    err "Run: databricks auth login --profile $ASR_ML_FINETUNE_PROFILE"
    return 2
  fi
  local cluster_id
  cluster_id="$("$PYTHON_BIN" - "$ASR_ML_FINETUNE_GPU_CLUSTER_NAME" "$clusters_json" <<'PY'
import json
import sys

name = sys.argv[1]
try:
    clusters = json.loads(sys.argv[2])
except json.JSONDecodeError as exc:
    raise SystemExit(f"Databricks clusters output was not valid JSON: {exc}") from exc
for cluster in clusters:
    if cluster.get("cluster_name") == name:
        print(cluster.get("cluster_id", ""))
        break
PY
)"
  if [[ -z "$cluster_id" ]]; then
    err "Databricks GPU cluster not found: $ASR_ML_FINETUNE_GPU_CLUSTER_NAME"
    err "Create it with scripts/asr/01_asr_model_training.sh gpu-start, then rerun."
    return 2
  fi

  local state
  if ! state="$("${DBX[@]}" clusters get "$cluster_id" --output json | "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin).get("state", ""))')"; then
    err "Unable to read Databricks GPU cluster state: $cluster_id"
    return 2
  fi
  if [[ "$state" == "TERMINATED" ]]; then
    log "starting dedicated ASR GPU cluster: $cluster_id" >&2
    "${DBX[@]}" clusters start "$cluster_id" --output json >/dev/null
  elif [[ "$state" == "TERMINATING" ]]; then
    err "Cluster $cluster_id is terminating. Wait for TERMINATED, then rerun."
    return 2
  else
    log "dedicated ASR GPU cluster is $state: $cluster_id" >&2
  fi
  printf '%s\n' "$cluster_id"
}

preflight() {
  setup_env
  resolve_paths
  load_target "$ASR_ML_FINETUNE_CANDIDATE"
  local manifest_local
  manifest_local="$(sync_manifest_to_local)"
  validate_manifest "$manifest_local"
  if [[ "$RECIPE" == "blocked_qwen_lora" ]]; then
    cat <<EOF
Fine-tuning preflight stopped.

Target:
  $CANDIDATE_ID

Reason:
  Qwen ASR LoRA fine-tuning is not implemented in this repo yet. Do not register
  final Qwen fine-tuned models until we add and verify a Qwen-specific training
  recipe against a small dry-run.

EOF
    return 2
  fi
  cat <<EOF
Fine-tuning preflight passed.

Target:
  $CANDIDATE_ID

Recipe:
  $RECIPE

Base model:
  $BASE_MODEL

Language:
  ${LANGUAGE_LOCALE_DISPLAY:-$LANGUAGE_NAME ($LANGUAGE_CODE)}

EOF
}

submit_whisper_lora_job() {
  local mode="$1"
  preflight
  local cluster_id
  if ! cluster_id="$(gpu_cluster_id)"; then
    exit 2
  fi
  if [[ -z "$cluster_id" ]]; then
    err "Unable to resolve GPU cluster: $ASR_ML_FINETUNE_GPU_CLUSTER_NAME"
    exit 2
  fi
  local remote_jobs_dir="$ASR_ML_FINETUNE_JOBS"
  local runner_remote="$remote_jobs_dir/databricks_multilingual_whisper_lora_finetune.py"
  local manifest_remote="$ASR_ML_FINETUNE_MANIFESTS/$(basename "$ASR_ML_FINETUNE_MANIFEST")"
  local run_name="${CANDIDATE_ID}_$(date +%Y%m%d_%H%M%S)"
  local output_dir="$ASR_ML_FINETUNE_LORA_RUNS/$run_name"

  "${DBX[@]}" fs mkdirs "dbfs:$remote_jobs_dir"
  "${DBX[@]}" fs mkdirs "dbfs:$(dirname "$manifest_remote")"
  "${DBX[@]}" fs cp "$ROOT/scripts/asr/databricks_multilingual_whisper_lora_finetune.py" "dbfs:$runner_remote" --overwrite
  if [[ "$ASR_ML_FINETUNE_MANIFEST" == /Volumes/* ]]; then
    manifest_remote="$ASR_ML_FINETUNE_MANIFEST"
  else
    "${DBX[@]}" fs cp "$ASR_ML_FINETUNE_MANIFEST" "dbfs:$manifest_remote" --overwrite
  fi

  local job_json="$ASR_ML_FINETUNE_OUTPUT_DIR/${mode}_${CANDIDATE_ID}.job.json"
  "$PYTHON_BIN" - "$job_json" <<PY
import json
from pathlib import Path

params = [
    "--manifest", "${manifest_remote}",
    "--output-dir", "${output_dir}",
    "--base-model", "${BASE_MODEL}",
    "--language-code", "${LANGUAGE_CODE}",
    "--language-name", "${LANGUAGE_NAME}",
    "--candidate-id", "${CANDIDATE_ID}",
    "--epochs", "${ASR_ML_FINETUNE_EPOCHS}",
]
if "${ASR_ML_FINETUNE_LIMIT_TRAIN}":
    params.extend(["--max-train-samples", "${ASR_ML_FINETUNE_LIMIT_TRAIN}"])
if "${ASR_ML_FINETUNE_LIMIT_EVAL}":
    params.extend(["--max-eval-samples", "${ASR_ML_FINETUNE_LIMIT_EVAL}"])
if "${mode}" == "dry-run":
    params.extend(["--dry-run", "--max-train-samples", "8", "--max-eval-samples", "4"])

payload = {
    "run_name": "genie-asr-${CANDIDATE_ID}-${mode}",
    "tasks": [
        {
            "task_key": "multilingual_whisper_lora",
            "existing_cluster_id": "${cluster_id}",
            "spark_python_task": {
                "python_file": "dbfs:${runner_remote}",
                "parameters": params,
            },
            "libraries": [
                {"pypi": {"package": "transformers"}},
                {"pypi": {"package": "accelerate"}},
                {"pypi": {"package": "datasets"}},
                {"pypi": {"package": "peft"}},
                {"pypi": {"package": "evaluate"}},
                {"pypi": {"package": "jiwer"}},
                {"pypi": {"package": "librosa"}},
                {"pypi": {"package": "soundfile"}},
            ],
        }
    ],
}
Path("${job_json}").write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY

  "${DBX[@]}" jobs submit --json @"$job_json" --output json
  cat <<EOF

Submitted multilingual Whisper LoRA $mode job.

Candidate:
  $CANDIDATE_ID

Output directory:
  $output_dir

Cluster:
  $cluster_id

EOF
}

latest_lora_run_name() {
  if [[ -n "$ASR_ML_FINETUNE_RUN_NAME" ]]; then
    printf '%s\n' "$ASR_ML_FINETUNE_RUN_NAME"
    return
  fi
  local listing
  listing="$("${DBX[@]}" fs ls "dbfs:$ASR_ML_FINETUNE_LORA_RUNS" 2>/dev/null || true)"
  "$PYTHON_BIN" - "$CANDIDATE_ID" "$listing" <<'PY'
from __future__ import annotations

import sys

candidate = sys.argv[1]
listing = sys.argv[2]
names = sorted(
    line.strip().rstrip("/")
    for line in listing.splitlines()
    if line.strip().rstrip("/").startswith(f"{candidate}_")
)
if names:
    print(names[-1])
PY
}

volume_path_exists() {
  local path="$1"
  local parent="${path%/*}"
  local name="${path##*/}"
  local listing
  listing="$("${DBX[@]}" fs ls "dbfs:$parent" 2>/dev/null || true)"
  [[ "$listing" == *"$name"* ]]
}

eval_label_suffix() {
  if [[ -n "$ASR_ML_FINETUNE_EVAL_LABEL" ]]; then
    printf '_%s' "$ASR_ML_FINETUNE_EVAL_LABEL"
  fi
}

require_trained_adapter_artifacts() {
  local run_name="$1"
  local run_dir="$ASR_ML_FINETUNE_LORA_RUNS/$run_name"
  local missing=0
  for path in \
    "$run_dir/adapter" \
    "$run_dir/processor" \
    "$run_dir/run_config.json" \
    "$run_dir/train_metrics.json"
  do
    if ! volume_path_exists "$path"; then
      err "Missing trained adapter artifact: $path"
      missing=1
    fi
  done
  return "$missing"
}

require_evaluation_artifacts() {
  local run_name="$1"
  local eval_dir="$ASR_ML_FINETUNE_EVALUATIONS/$run_name"
  local missing=0
  for path in \
    "$eval_dir/lora_evaluation_results.jsonl" \
    "$eval_dir/lora_evaluation_summary.json"
  do
    if ! volume_path_exists "$path"; then
      err "Missing evaluation artifact: $path"
      missing=1
    fi
  done
  return "$missing"
}

evaluate_lora_job() {
  setup_env
  resolve_paths
  load_target "$ASR_ML_FINETUNE_CANDIDATE"
  if [[ "$RECIPE" != "whisper_lora" ]]; then
    err "Evaluation is currently implemented only for Whisper LoRA targets."
    exit 2
  fi
  local run_name
  run_name="$(latest_lora_run_name)"
  if [[ -z "$run_name" ]]; then
    err "No multilingual LoRA run found under $ASR_ML_FINETUNE_LORA_RUNS for $CANDIDATE_ID"
    exit 2
  fi
  require_trained_adapter_artifacts "$run_name"
  if [[ -z "$ASR_ML_FINETUNE_MANIFEST" ]]; then
    err "ASR_ML_FINETUNE_MANIFEST is required for evaluation."
    exit 2
  fi
  local cluster_id
  cluster_id="$(gpu_cluster_id)"
  local runner_remote="$ASR_ML_FINETUNE_JOBS/databricks_whisper_lora_evaluate.py"
  local manifest_remote="$ASR_ML_FINETUNE_MANIFESTS/$(basename "$ASR_ML_FINETUNE_MANIFEST")"
  local run_dir="$ASR_ML_FINETUNE_LORA_RUNS/$run_name"
  local label_suffix
  label_suffix="$(eval_label_suffix)"
  local eval_dir="$ASR_ML_FINETUNE_EVALUATIONS/${run_name}${label_suffix}"
  local job_json="$ASR_ML_FINETUNE_OUTPUT_DIR/evaluate_${run_name}${label_suffix}.job.json"
  local language_arg
  language_arg="$(printf '%s' "$LANGUAGE_NAME" | tr '[:upper:]' '[:lower:]')"

  "${DBX[@]}" fs mkdirs "dbfs:$ASR_ML_FINETUNE_JOBS"
  "${DBX[@]}" fs mkdirs "dbfs:$ASR_ML_FINETUNE_MANIFESTS"
  "${DBX[@]}" fs cp "$ROOT/scripts/asr/databricks_whisper_lora_evaluate.py" "dbfs:$runner_remote" --overwrite
  if [[ "$ASR_ML_FINETUNE_MANIFEST" == /Volumes/* ]]; then
    manifest_remote="$ASR_ML_FINETUNE_MANIFEST"
  else
    "${DBX[@]}" fs cp "$ASR_ML_FINETUNE_MANIFEST" "dbfs:$manifest_remote" --overwrite
  fi

  "$PYTHON_BIN" - "$job_json" <<PY
import json
from pathlib import Path

payload = {
    "run_name": "genie-asr-${run_name}-evaluate",
    "tasks": [
        {
            "task_key": "multilingual_whisper_lora_evaluate",
            "existing_cluster_id": "${cluster_id}",
            "spark_python_task": {
                "python_file": "dbfs:${runner_remote}",
                "parameters": [
                    "--manifest", "${manifest_remote}",
                    "--adapter-dir", "${run_dir}/adapter",
                    "--output", "${eval_dir}/lora_evaluation_results.jsonl",
                    "--summary-output", "${eval_dir}/lora_evaluation_summary.json",
                    "--base-model", "${BASE_MODEL}",
                    "--language", "${language_arg}",
                    "--split", "${ASR_ML_FINETUNE_EVAL_SPLIT}",
                ],
            },
            "libraries": [
                {"pypi": {"package": "transformers"}},
                {"pypi": {"package": "accelerate"}},
                {"pypi": {"package": "peft"}},
                {"pypi": {"package": "librosa"}},
                {"pypi": {"package": "soundfile"}},
            ],
        }
    ],
}
Path("${job_json}").write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY

  "${DBX[@]}" jobs submit --json @"$job_json" --output json
  cat <<EOF

Submitted multilingual LoRA evaluation job.

Run:
  $run_name

Evaluation output:
  $eval_dir

EOF
}

register_candidate_gate() {
  setup_env
  resolve_paths
  load_target "$ASR_ML_FINETUNE_CANDIDATE"
  if [[ "$RECIPE" != "whisper_lora" ]]; then
    err "Registration is blocked until a verified fine-tuning and registration recipe exists for $FAMILY."
    exit 2
  fi
  local run_name
  run_name="$(latest_lora_run_name)"
  if [[ -z "$run_name" ]]; then
    err "No multilingual LoRA run found under $ASR_ML_FINETUNE_LORA_RUNS for $CANDIDATE_ID"
    exit 2
  fi
  require_trained_adapter_artifacts "$run_name"
  require_evaluation_artifacts "$run_name"
  cat <<EOF
Registration gate passed for multilingual fine-tuned candidate.

Candidate:
  $CANDIDATE_ID

LoRA run:
  $run_name

Required next implementation:
  Register this adapter as a UC pyfunc model with adaptation_type=finetuned_lora
  and fine_tuned_by_us=true. Do not deploy until registration and smoke test pass.

EOF
}

run_all() {
  if [[ -z "$ASR_ML_FINETUNE_CANDIDATE" ]]; then
    err "ASR_ML_FINETUNE_CANDIDATE is required for run"
    target_ids >&2
    exit 2
  fi

  setup_env
  resolve_paths
  load_target "$ASR_ML_FINETUNE_CANDIDATE"

  log "step 1/7: create aligned Databricks Volume folders"
  prepare_volume

  log "step 2/7: scaffold JSONL and CSV templates"
  if [[ ! -e "$ASR_ML_FINETUNE_TEMPLATE" ]]; then
    scaffold_manifest
  else
    log "JSONL template already exists: $ASR_ML_FINETUNE_TEMPLATE"
  fi
  local default_csv="$ASR_ML_FINETUNE_OUTPUT_DIR/multilingual_business_manifest.template.csv"
  if [[ "$ASR_ML_FINETUNE_CSV" == "$default_csv" && ! -e "$ASR_ML_FINETUNE_CSV" ]]; then
    scaffold_csv
  elif [[ "$ASR_ML_FINETUNE_CSV" == "$default_csv" ]]; then
    log "CSV template already exists: $ASR_ML_FINETUNE_CSV"
  else
    log "using provided CSV without modifying it: $ASR_ML_FINETUNE_CSV"
  fi
  if [[ ! -e "$ASR_ML_FINETUNE_RECORDING_PLAN" ]]; then
    scaffold_recording_plan
  else
    log "recording plan already exists: $ASR_ML_FINETUNE_RECORDING_PLAN"
  fi

  if [[ "$ASR_ML_FINETUNE_CSV" != "$default_csv" && ! -s "$ASR_ML_FINETUNE_CSV" ]]; then
    err "Provided ASR_ML_FINETUNE_CSV does not exist or is empty: $ASR_ML_FINETUNE_CSV"
    exit 2
  fi

  local default_recording_plan="$ASR_ML_FINETUNE_OUTPUT_DIR/multilingual_recording_plan.csv"
  if [[ "$ASR_ML_FINETUNE_SYNTHETIC_BOOTSTRAP" == "true" ]]; then
    log "step 3/7: bootstrap synthetic train/validation audio from recording plan"
    bootstrap_synthetic_recording_plan
    ASR_ML_FINETUNE_MANIFEST="$ASR_ML_FINETUNE_BUILT_MANIFEST"
  elif [[ -s "$ASR_ML_FINETUNE_RECORDING_PLAN" && "$ASR_ML_FINETUNE_RECORDING_PLAN" != "$default_recording_plan" ]]; then
    log "step 3/7: materialize completed recording plan"
    materialize_recording_plan
    ASR_ML_FINETUNE_MANIFEST="$ASR_ML_FINETUNE_BUILT_MANIFEST"
  elif [[ -n "${ASR_ML_FINETUNE_CSV:-}" && -s "$ASR_ML_FINETUNE_CSV" && "$ASR_ML_FINETUNE_CSV" != "$default_csv" ]]; then
    log "step 3/7: build manifest from CSV"
    build_manifest_from_csv
    ASR_ML_FINETUNE_MANIFEST="$ASR_ML_FINETUNE_BUILT_MANIFEST"
  else
    log "step 3/7: no filled recording plan or CSV provided; using ASR_ML_FINETUNE_MANIFEST if set"
  fi

  log "step 4/7: validate manifest and candidate recipe"
  preflight

  if [[ "$RECIPE" != "whisper_lora" ]]; then
    err "run stopped: only Whisper LoRA targets are trainable right now"
    exit 2
  fi

  if [[ -n "$ASR_ML_FINETUNE_RUN_NAME" ]]; then
    log "step 5/7: completed run provided; skipping training submission"
    log "step 6/7: evaluate adapter"
    evaluate_lora_job
    log "step 7/7: run registration gate"
    register_candidate_gate
    return
  fi

  log "step 5/7: submit dry-run training"
  submit_whisper_lora_job dry-run

  if [[ "$ASR_ML_FINETUNE_RUN_TRAIN" != "true" ]]; then
    cat <<EOF
Run stopped after dry-run submission.

Reason:
  ASR_ML_FINETUNE_RUN_TRAIN is not true.

After the dry-run succeeds, rerun with:

  ASR_ML_FINETUNE_RUN_TRAIN=true \\
  ASR_ML_FINETUNE_CANDIDATE=$CANDIDATE_ID \\
  ASR_ML_FINETUNE_MANIFEST=$ASR_ML_FINETUNE_MANIFEST \\
    scripts/asr/11_multilingual_asr_finetuning.sh run

EOF
    return
  fi

  log "step 6/7: submit real training"
  submit_whisper_lora_job train

  cat <<EOF
Training submitted.

After the training job completes, run the same command again with
ASR_ML_FINETUNE_RUN_NAME set to the completed run name so the script can execute
evaluation and the registration gate:

  ASR_ML_FINETUNE_RUN_NAME=<completed_run_name> \\
  ASR_ML_FINETUNE_CANDIDATE=$CANDIDATE_ID \\
  ASR_ML_FINETUNE_MANIFEST=$ASR_ML_FINETUNE_MANIFEST \\
    scripts/asr/11_multilingual_asr_finetuning.sh run

EOF

  log "step 7/7: waiting for completed run name before evaluation and registration gate"
}

run_all_languages() {
  setup_env
  resolve_paths
  prepare_volume

  local submitted=0
  local skipped=0
  local failed=0
  local candidate language recording_plan built_manifest
  for candidate in $ASR_ML_FINETUNE_ALL_CANDIDATES; do
    load_target "$candidate"
    if [[ "$RECIPE" != "whisper_lora" ]]; then
      skipped=$((skipped + 1))
      cat <<EOF
Skipping blocked fine-tuning target.

Target:
  $candidate

Reason:
  $RECIPE is not a verified trainable recipe in this repo yet.

EOF
      continue
    fi

    language="$LANGUAGE_CODE"
    recording_plan="$ASR_ML_FINETUNE_OUTPUT_DIR/${language}_recording_plan.csv"
    built_manifest="$ASR_ML_FINETUNE_OUTPUT_DIR/${language}_business_manifest.built.jsonl"

    cat <<EOF
Submitting multilingual ASR job.

Target:
  $candidate

Language:
  ${LANGUAGE_LOCALE_DISPLAY:-$LANGUAGE_NAME ($LANGUAGE_CODE)}

Recording plan:
  $recording_plan

Manifest:
  $built_manifest

EOF

    if ASR_ML_FINETUNE_CANDIDATE="$candidate" \
      ASR_ML_FINETUNE_SYNTHETIC_BOOTSTRAP=true \
      ASR_ML_FINETUNE_MIN_HOLDOUT="$ASR_ML_FINETUNE_ALL_MIN_HOLDOUT" \
      ASR_ML_FINETUNE_RECORDING_PLAN="$recording_plan" \
      ASR_ML_FINETUNE_BUILT_MANIFEST="$built_manifest" \
        "$0" run
    then
      submitted=$((submitted + 1))
    else
      failed=$((failed + 1))
      err "all-language submission failed for target: $candidate"
    fi
  done

  cat <<EOF
All-language multilingual ASR submission complete.

Submitted trainable targets:
  $submitted

Skipped blocked targets:
  $skipped

Failed targets:
  $failed

EOF
  if [[ "$failed" -gt 0 ]]; then
    return 2
  fi
}

evaluate_all_languages() {
  setup_env
  resolve_paths

  local evaluated=0
  local skipped=0
  local failed=0
  local candidate language manifest
  for candidate in $ASR_ML_FINETUNE_ALL_CANDIDATES; do
    load_target "$candidate"
    if [[ "$RECIPE" != "whisper_lora" ]]; then
      skipped=$((skipped + 1))
      err "Skipping non-trainable evaluation target: $candidate ($RECIPE)"
      continue
    fi

    language="$LANGUAGE_CODE"
    manifest="$(eval_manifest_for_language "$language")"
    if [[ ! -s "$manifest" ]]; then
      failed=$((failed + 1))
      err "Missing local manifest for $candidate: $manifest"
      continue
    fi

    cat <<EOF
Submitting multilingual ASR evaluation.

Target:
  $candidate

Language:
  ${LANGUAGE_LOCALE_DISPLAY:-$LANGUAGE_NAME ($LANGUAGE_CODE)}

Split:
  $ASR_ML_FINETUNE_EVAL_SPLIT

Manifest:
  $manifest

EOF

    if ASR_ML_FINETUNE_CANDIDATE="$candidate" \
      ASR_ML_FINETUNE_MANIFEST="$manifest" \
      ASR_ML_FINETUNE_EVAL_SPLIT="$ASR_ML_FINETUNE_EVAL_SPLIT" \
        "$0" evaluate-lora
    then
      evaluated=$((evaluated + 1))
    else
      failed=$((failed + 1))
      err "evaluation failed for target: $candidate"
    fi
  done

  cat <<EOF
All-language multilingual ASR evaluation submission complete.

Evaluated trainable targets:
  $evaluated

Skipped blocked targets:
  $skipped

Failed targets:
  $failed

EOF
  if [[ "$failed" -gt 0 ]]; then
    return 2
  fi
}

eval_manifest_for_language() {
  local language="$1"
  if [[ "$ASR_ML_FINETUNE_EVAL_SPLIT" == "holdout" && -s "$ASR_ML_FINETUNE_OUTPUT_DIR/${language}_fleurs_holdout_manifest.built.jsonl" ]]; then
    printf '%s\n' "$ASR_ML_FINETUNE_OUTPUT_DIR/${language}_fleurs_holdout_manifest.built.jsonl"
  elif [[ "$ASR_ML_FINETUNE_EVAL_SPLIT" == "holdout" && -s "$ASR_ML_FINETUNE_OUTPUT_DIR/${language}_holdout_manifest.built.jsonl" ]]; then
    printf '%s\n' "$ASR_ML_FINETUNE_OUTPUT_DIR/${language}_holdout_manifest.built.jsonl"
  else
    printf '%s\n' "$ASR_ML_FINETUNE_OUTPUT_DIR/${language}_business_manifest.built.jsonl"
  fi
}

evaluate_base_job() {
  setup_env
  resolve_paths
  load_target "$ASR_ML_FINETUNE_CANDIDATE"
  if [[ -z "$ASR_ML_FINETUNE_MANIFEST" ]]; then
    err "ASR_ML_FINETUNE_MANIFEST is required for base evaluation."
    exit 2
  fi
  local cluster_id
  cluster_id="$(gpu_cluster_id)"
  local runner_remote="$ASR_ML_FINETUNE_JOBS/databricks_whisper_lora_evaluate.py"
  local manifest_remote="$ASR_ML_FINETUNE_MANIFESTS/$(basename "$ASR_ML_FINETUNE_MANIFEST")"
  local label_suffix
  label_suffix="$(eval_label_suffix)"
  local eval_dir="$ASR_ML_FINETUNE_EVALUATIONS/base_${CANDIDATE_ID}_${ASR_ML_FINETUNE_EVAL_SPLIT}${label_suffix}"
  local job_json="$ASR_ML_FINETUNE_OUTPUT_DIR/evaluate_base_${CANDIDATE_ID}_${ASR_ML_FINETUNE_EVAL_SPLIT}${label_suffix}.job.json"
  local language_arg
  language_arg="$(printf '%s' "$LANGUAGE_NAME" | tr '[:upper:]' '[:lower:]')"

  "${DBX[@]}" fs mkdirs "dbfs:$ASR_ML_FINETUNE_JOBS"
  "${DBX[@]}" fs mkdirs "dbfs:$ASR_ML_FINETUNE_MANIFESTS"
  "${DBX[@]}" fs cp "$ROOT/scripts/asr/databricks_whisper_lora_evaluate.py" "dbfs:$runner_remote" --overwrite
  if [[ "$ASR_ML_FINETUNE_MANIFEST" == /Volumes/* ]]; then
    manifest_remote="$ASR_ML_FINETUNE_MANIFEST"
  else
    "${DBX[@]}" fs cp "$ASR_ML_FINETUNE_MANIFEST" "dbfs:$manifest_remote" --overwrite
  fi

  "$PYTHON_BIN" - "$job_json" <<PY
import json
from pathlib import Path

payload = {
    "run_name": "genie-asr-base-${CANDIDATE_ID}-${ASR_ML_FINETUNE_EVAL_SPLIT}-evaluate",
    "tasks": [
        {
            "task_key": "multilingual_whisper_base_evaluate",
            "existing_cluster_id": "${cluster_id}",
            "spark_python_task": {
                "python_file": "dbfs:${runner_remote}",
                "parameters": [
                    "--manifest", "${manifest_remote}",
                    "--adapter-dir", "unused",
                    "--base-only",
                    "--output", "${eval_dir}/base_evaluation_results.jsonl",
                    "--summary-output", "${eval_dir}/base_evaluation_summary.json",
                    "--base-model", "${BASE_MODEL}",
                    "--language", "${language_arg}",
                    "--split", "${ASR_ML_FINETUNE_EVAL_SPLIT}",
                ],
            },
            "libraries": [
                {"pypi": {"package": "transformers"}},
                {"pypi": {"package": "accelerate"}},
                {"pypi": {"package": "peft"}},
                {"pypi": {"package": "librosa"}},
                {"pypi": {"package": "soundfile"}},
            ],
        }
    ],
}
Path("${job_json}").write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY

  "${DBX[@]}" jobs submit --json @"$job_json" --output json
  cat <<EOF

Submitted multilingual base Whisper evaluation job.

Target:
  $CANDIDATE_ID

Evaluation output:
  $eval_dir

EOF
}

evaluate_base_all_languages() {
  setup_env
  resolve_paths

  local evaluated=0
  local skipped=0
  local failed=0
  local candidate language manifest
  for candidate in $ASR_ML_FINETUNE_ALL_CANDIDATES; do
    load_target "$candidate"
    if [[ "$RECIPE" != "whisper_lora" ]]; then
      skipped=$((skipped + 1))
      err "Skipping non-trainable base evaluation target: $candidate ($RECIPE)"
      continue
    fi
    language="$LANGUAGE_CODE"
    manifest="$(eval_manifest_for_language "$language")"
    if [[ ! -s "$manifest" ]]; then
      failed=$((failed + 1))
      err "Missing manifest for $candidate: $manifest"
      continue
    fi
    cat <<EOF
Submitting base ASR evaluation.

Target:
  $candidate

Language:
  ${LANGUAGE_LOCALE_DISPLAY:-$LANGUAGE_NAME ($LANGUAGE_CODE)}

Split:
  $ASR_ML_FINETUNE_EVAL_SPLIT

Manifest:
  $manifest

EOF
    if ASR_ML_FINETUNE_CANDIDATE="$candidate" \
      ASR_ML_FINETUNE_MANIFEST="$manifest" \
      ASR_ML_FINETUNE_EVAL_SPLIT="$ASR_ML_FINETUNE_EVAL_SPLIT" \
        "$0" evaluate-base
    then
      evaluated=$((evaluated + 1))
    else
      failed=$((failed + 1))
      err "base evaluation failed for target: $candidate"
    fi
  done

  cat <<EOF
All-language base ASR evaluation submission complete.

Evaluated trainable targets:
  $evaluated

Skipped blocked targets:
  $skipped

Failed targets:
  $failed

EOF
  if [[ "$failed" -gt 0 ]]; then
    return 2
  fi
}

materialize_holdout_all() {
  setup_env
  resolve_paths

  local materialized=0
  local skipped=0
  local failed=0
  local candidate language recording_plan built_manifest
  for candidate in $ASR_ML_FINETUNE_ALL_CANDIDATES; do
    load_target "$candidate"
    language="$LANGUAGE_CODE"
    recording_plan="$ASR_ML_FINETUNE_OUTPUT_DIR/${language}_recording_plan.csv"
    built_manifest="$ASR_ML_FINETUNE_OUTPUT_DIR/${language}_holdout_manifest.built.jsonl"
    if [[ ! -s "$recording_plan" ]]; then
      skipped=$((skipped + 1))
      err "Missing recording plan for $candidate: $recording_plan"
      continue
    fi

    cat <<EOF
Materializing multilingual ASR holdout.

Target:
  $candidate

Language:
  ${LANGUAGE_LOCALE_DISPLAY:-$LANGUAGE_NAME ($LANGUAGE_CODE)}

Recording plan:
  $recording_plan

Holdout manifest:
  $built_manifest

EOF

    if ASR_ML_FINETUNE_CANDIDATE="$candidate" \
      ASR_ML_FINETUNE_RECORDING_PLAN_SPLIT=holdout \
      ASR_ML_FINETUNE_RECORDING_PLAN="$recording_plan" \
      ASR_ML_FINETUNE_BUILT_MANIFEST="$built_manifest" \
      ASR_ML_FINETUNE_MIN_TRAIN=0 \
      ASR_ML_FINETUNE_MIN_VALIDATION=0 \
      ASR_ML_FINETUNE_MIN_HOLDOUT="${ASR_ML_FINETUNE_MIN_HOLDOUT:-10}" \
        "$0" materialize-recording-plan
    then
      materialized=$((materialized + 1))
    else
      failed=$((failed + 1))
      err "holdout materialization failed for target: $candidate"
    fi
  done

  cat <<EOF
All-language holdout materialization complete.

Materialized languages:
  $materialized

Skipped languages:
  $skipped

Failed languages:
  $failed

EOF
  if [[ "$failed" -gt 0 ]]; then
    return 2
  fi
}

bootstrap_synthetic_holdout_all() {
  setup_env
  resolve_paths

  local generated=0
  local skipped=0
  local failed=0
  local candidate language recording_plan built_manifest
  for candidate in $ASR_ML_FINETUNE_ALL_CANDIDATES; do
    load_target "$candidate"
    if [[ "$RECIPE" != "whisper_lora" ]]; then
      skipped=$((skipped + 1))
      err "Skipping non-trainable synthetic holdout target: $candidate ($RECIPE)"
      continue
    fi
    language="$LANGUAGE_CODE"
    recording_plan="$ASR_ML_FINETUNE_OUTPUT_DIR/${language}_recording_plan.csv"
    built_manifest="$ASR_ML_FINETUNE_OUTPUT_DIR/${language}_holdout_manifest.built.jsonl"

    cat <<EOF
Generating synthetic holdout smoke data.

Target:
  $candidate

Language:
  ${LANGUAGE_LOCALE_DISPLAY:-$LANGUAGE_NAME ($LANGUAGE_CODE)}

Recording plan:
  $recording_plan

Holdout smoke manifest:
  $built_manifest

EOF

    if ASR_ML_FINETUNE_CANDIDATE="$candidate" \
      ASR_ML_FINETUNE_RECORDING_PLAN="$recording_plan" \
      ASR_ML_FINETUNE_BUILT_MANIFEST="$built_manifest" \
      ASR_ML_FINETUNE_SYNTHETIC_SPLITS=holdout \
      ASR_ML_FINETUNE_SYNTHETIC_SOURCE=synthetic_macos_say_holdout_smoke \
      ASR_ML_FINETUNE_SYNTHETIC_PREFLIGHT=false \
        "$0" bootstrap-synthetic-recording-plan
    then
      generated=$((generated + 1))
    else
      failed=$((failed + 1))
      err "synthetic holdout generation failed for target: $candidate"
    fi
  done

  cat <<EOF
All-language synthetic holdout smoke generation complete.

Generated languages:
  $generated

Skipped languages:
  $skipped

Failed languages:
  $failed

EOF
  if [[ "$failed" -gt 0 ]]; then
    return 2
  fi
}

bootstrap_fleurs_holdout_all() {
  setup_env
  resolve_paths
  mkdir -p "$ASR_ML_FINETUNE_OUTPUT_DIR/external_fleurs"
  "$PYTHON_BIN" - "$ASR_ML_FINETUNE_OUTPUT_DIR" "$ASR_ML_FINETUNE_AUDIO" "$ASR_ML_FINETUNE_FLEURS_LIMIT" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

try:
    import pandas as pd
    import soundfile as sf
    from huggingface_hub import hf_hub_download, list_repo_files
except Exception as exc:  # noqa: BLE001
    raise SystemExit(
        "bootstrap-fleurs-holdout-all requires huggingface_hub, pandas, pyarrow, and soundfile. "
        "Install project dependencies, then rerun."
    ) from exc

output_dir = Path(sys.argv[1])
volume_audio_dir = sys.argv[2].rstrip("/")
limit = int(sys.argv[3])
configs = {"th": "th_th", "id": "id_id", "zh": "cmn_hans_cn"}
local_audio_root = output_dir / "external_fleurs" / "audio"
cache_root = output_dir / "external_fleurs" / "hf_cache"
local_audio_root.mkdir(parents=True, exist_ok=True)
cache_root.mkdir(parents=True, exist_ok=True)
repo_files = list_repo_files("google/fleurs", repo_type="dataset")
created = []

def safe_json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return str(value)

def write_audio(audio: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(audio, dict) and audio.get("array") is not None:
        sf.write(path, audio["array"], int(audio.get("sampling_rate") or 16000))
        return
    if isinstance(audio, (bytes, bytearray, memoryview)):
        path.write_bytes(bytes(audio))
        return
    raise RuntimeError(f"Unsupported FLEURS audio payload: {type(audio).__name__}")

for language, config in configs.items():
    parquet_candidates = [
        name
        for name in repo_files
        if name.endswith(".parquet") and f"/{config}/" in name and "validation" in name
    ]
    if not parquet_candidates:
        raise RuntimeError(f"No FLEURS validation parquet found for {config}")
    parquet_path = hf_hub_download(
        repo_id="google/fleurs",
        repo_type="dataset",
        filename=sorted(parquet_candidates)[0],
        cache_dir=str(cache_root / "hub"),
    )
    dataset = pd.read_parquet(parquet_path)
    rows = []
    written = 0
    for _, item in dataset.iterrows():
        if written >= limit:
            break
        transcript = str(item.get("transcription") or item.get("raw_transcription") or "").strip()
        if not transcript:
            continue
        clip_id = f"{language}_fleurs_holdout_{written + 1:04d}"
        local_audio = local_audio_root / language / f"{clip_id}.wav"
        write_audio(item.get("audio"), local_audio)
        remote_audio = f"{volume_audio_dir}/{language}/external_fleurs_holdout/{clip_id}.wav"
        rows.append(
            {
                "clip_id": clip_id,
                "audio_path": remote_audio,
                "audio_format": "audio/wav",
                "sample_rate_hz": 16000,
                "duration_seconds": None,
                "reference_transcript": transcript,
                "language": language,
                "split": "holdout",
                "scenario": "external_fleurs_acoustic_holdout",
                "speaker": "external_fleurs",
                "domain": "public_read_speech",
                "dataset_version": "google_fleurs_validation_cc_by_4_0",
                "expected_entities": {
                    "invoice_ids": [],
                    "amounts": [],
                    "billing_actions": [],
                    "confirmations": [],
                    "refusals": [],
                    "account_terms": [],
                },
                "metadata": {
                    "source": "google/fleurs",
                    "license": "CC-BY-4.0",
                    "config": config,
                    "id": safe_json_value(item.get("id")),
                    "external_holdout_type": "acoustic_only",
                },
            }
        )
        created.append({"local": str(local_audio), "remote": remote_audio})
        written += 1
    manifest = output_dir / f"{language}_fleurs_holdout_manifest.built.jsonl"
    manifest.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"language": language, "config": config, "manifest": str(manifest), "rows": len(rows)}, ensure_ascii=False))
(output_dir / "external_fleurs" / "uploads.json").write_text(json.dumps(created, indent=2), encoding="utf-8")
PY

  local uploads_json="$ASR_ML_FINETUNE_OUTPUT_DIR/external_fleurs/uploads.json"
  "$PYTHON_BIN" - "$uploads_json" <<'PY' | while IFS=$'\t' read -r local_path remote_path; do
from __future__ import annotations

import json
import sys
from pathlib import Path

for item in json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")):
    print(f"{item['local']}\t{item['remote']}")
PY
    "${DBX[@]}" fs mkdirs "dbfs:$(dirname "$remote_path")" >/dev/null
    "${DBX[@]}" fs cp "$local_path" "dbfs:$remote_path" --overwrite >/dev/null
  done
}

plan() {
  cat <<'EOF'
Multilingual fine-tuning plan

Supported now:
  th_finetuned_pathumma_whisper_lora
    - recipe: Whisper LoRA
    - base: nectec/Pathumma-whisper-th-large-v3
    - requires real Thai train/validation/holdout rows with expected_entities
    - lifecycle: prepare -> validate-manifest -> dry-run -> train-one -> evaluate-lora -> register-candidate gate
  id_finetuned_whisper_large_v3_lora
    - recipe: Whisper LoRA
    - base: openai/whisper-large-v3
    - language: Indonesian (id-ID, Indonesia)
    - lifecycle: prepare -> bootstrap synthetic train/validation -> dry-run -> train-one -> evaluate-lora
  zh_finetuned_whisper_large_v3_lora
    - recipe: Whisper LoRA
    - base: openai/whisper-large-v3
    - language: Mandarin Chinese (zh-CN, Mainland China)
    - lifecycle: prepare -> bootstrap synthetic train/validation -> dry-run -> train-one -> evaluate-lora

Blocked until implemented:
  id_finetuned_qwen3_asr_0_6b_lora
  zh_finetuned_qwen3_asr_0_6b_lora
    - reason: Qwen ASR LoRA training recipe is not verified in this repo
    - do not register/deploy final fine-tuned Qwen models until dry-run training works

OSS-baseline endpoints may remain only as temporary baseline evidence.
EOF
}

case "$COMMAND" in
  run)
    run_all
    ;;
  run-all-languages)
    run_all_languages
    ;;
  evaluate-all-languages)
    evaluate_all_languages
    ;;
  evaluate-base-all-languages)
    evaluate_base_all_languages
    ;;
  list)
    target_ids
    ;;
  plan)
    plan
    ;;
  volume)
    show_volume
    ;;
  prepare)
    prepare_volume
    ;;
  scaffold-recording-plan)
    scaffold_recording_plan
    ;;
  scaffold-recording-plans)
    scaffold_recording_plans
    ;;
  scaffold-holdout-packs)
    scaffold_holdout_packs
    ;;
  bootstrap-synthetic-recording-plan)
    bootstrap_synthetic_recording_plan
    ;;
  bootstrap-synthetic-holdout-all)
    bootstrap_synthetic_holdout_all
    ;;
  bootstrap-fleurs-holdout-all)
    bootstrap_fleurs_holdout_all
    ;;
  scaffold-manifest)
    scaffold_manifest
    ;;
  scaffold-csv)
    scaffold_csv
    ;;
  build-manifest-from-csv)
    build_manifest_from_csv
    ;;
  materialize-recording-plan)
    materialize_recording_plan
    ;;
  materialize-holdout-all)
    materialize_holdout_all
    ;;
  validate-manifest|preflight)
    if [[ -z "$ASR_ML_FINETUNE_CANDIDATE" ]]; then
      err "ASR_ML_FINETUNE_CANDIDATE is required"
      exit 2
    fi
    preflight
    ;;
  dry-run)
    if [[ -z "$ASR_ML_FINETUNE_CANDIDATE" ]]; then
      err "ASR_ML_FINETUNE_CANDIDATE is required"
      exit 2
    fi
    submit_whisper_lora_job dry-run
    ;;
  train-one)
    if [[ -z "$ASR_ML_FINETUNE_CANDIDATE" ]]; then
      err "ASR_ML_FINETUNE_CANDIDATE is required"
      exit 2
    fi
    submit_whisper_lora_job train
    ;;
  evaluate-lora)
    if [[ -z "$ASR_ML_FINETUNE_CANDIDATE" ]]; then
      err "ASR_ML_FINETUNE_CANDIDATE is required"
      exit 2
    fi
    evaluate_lora_job
    ;;
  evaluate-base)
    if [[ -z "$ASR_ML_FINETUNE_CANDIDATE" ]]; then
      err "ASR_ML_FINETUNE_CANDIDATE is required"
      exit 2
    fi
    evaluate_base_job
    ;;
  register-candidate)
    if [[ -z "$ASR_ML_FINETUNE_CANDIDATE" ]]; then
      err "ASR_ML_FINETUNE_CANDIDATE is required"
      exit 2
    fi
    register_candidate_gate
    ;;
  help|--help|-h)
    usage
    ;;
  *)
    err "Unknown command: $COMMAND"
    usage
    exit 2
    ;;
esac
