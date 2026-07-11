#!/usr/bin/env bash
# =============================================================================
# 10_register_multilingual_asr_candidates.sh
#
# Proper multilingual OSS-baseline ASR candidate path:
#   1. Download/cache model snapshot once during registration.
#   2. Register candidate as an MLflow pyfunc model in Unity Catalog.
#   3. Evaluate via models:/<registered_model>@candidate.
#
# This is the production-shaped quality path. Use 09_ only for raw/model API
# plumbing and quick bake-off scaffolding.
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEFAULT_OUTPUT_DIR="$ROOT/.run/asr_model_training/evaluations/multilingual_registered_candidates"
DEFAULT_REMOTE_SUBDIR="registered_candidates/multilingual_asr"

ASR_ML_REGISTER_OUTPUT_DIR="${ASR_ML_REGISTER_OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}"
ASR_ML_REGISTER_PROFILE="${ASR_ML_REGISTER_PROFILE:-${ASR_DATABRICKS_PROFILE:-${DATABRICKS_CONFIG_PROFILE:-fe-vm-vdm-classic-rcn6ip}}}"
ASR_ML_REGISTER_REMOTE_ROOT="${ASR_ML_REGISTER_REMOTE_ROOT:-}"
ASR_ML_REGISTER_ENVIRONMENT_VERSION="${ASR_ML_REGISTER_ENVIRONMENT_VERSION:-2}"
ASR_ML_REGISTER_CANDIDATE="${ASR_ML_REGISTER_CANDIDATE:-}"
ASR_ML_REGISTER_LANGUAGE="${ASR_ML_REGISTER_LANGUAGE:-}"
ASR_ML_REGISTER_MANIFEST="${ASR_ML_REGISTER_MANIFEST:-}"
ASR_ML_REGISTER_LIMIT="${ASR_ML_REGISTER_LIMIT:-1}"
ASR_ML_REGISTER_FORCE_DOWNLOAD="${ASR_ML_REGISTER_FORCE_DOWNLOAD:-false}"
ASR_ML_REGISTER_PUBLIC_LIMIT="${ASR_ML_REGISTER_PUBLIC_LIMIT:-1}"
ASR_ML_REGISTERED_MODEL="${ASR_ML_REGISTERED_MODEL:-}"

COMMAND="${1:-help}"
if [[ $# -gt 0 ]]; then
  shift
fi

log() { printf "\033[36m[asr-uc-candidate]\033[0m %s\n" "$*"; }
err() { printf "\033[31m[asr-uc-candidate]\033[0m %s\n" "$*" >&2; }

usage() {
  cat <<EOF
Register/evaluate multilingual OSS-baseline ASR candidates through UC/MLflow.

Commands:
  list             Print supported candidate ids.
  register-one     Register ASR_ML_REGISTER_CANDIDATE as models:/...@candidate.
  evaluate-one     Evaluate ASR_ML_REGISTER_CANDIDATE through registered model alias.
  register-all     Register all configured candidates.
  evaluate-all     Evaluate all configured candidates against ASR_ML_REGISTER_MANIFEST.
  bootstrap-public-manifest
                   Create public FLEURS manifest/audio under UC Volume.
  help             Show this help.

Environment:
  ASR_ML_REGISTER_CANDIDATE       Candidate id for one-candidate commands.
  ASR_ML_REGISTER_MANIFEST        Manifest JSONL for evaluate commands.
  ASR_ML_REGISTER_LIMIT           Max holdout clips per candidate. Default: 1.
  ASR_ML_REGISTER_PUBLIC_LIMIT    Public FLEURS clips/language. Default: 1.
  ASR_ML_REGISTERED_MODEL         Optional UC FQDN override for one-candidate commands.
  ASR_ML_REGISTER_REMOTE_ROOT     UC Volume root for job/package artifacts.
  ASR_ML_REGISTER_FORCE_DOWNLOAD  true = refresh HF snapshot during registration.

Examples:
  ASR_ML_REGISTER_CANDIDATE=id_oss_qwen3_asr_0_6b \\
    scripts/asr/10_register_multilingual_asr_candidates.sh register-one

  ASR_ML_REGISTER_CANDIDATE=id_oss_qwen3_asr_0_6b \\
  ASR_ML_REGISTER_MANIFEST=/Volumes/.../multilingual_pilot_manifest.jsonl \\
    scripts/asr/10_register_multilingual_asr_candidates.sh evaluate-one
EOF
}

setup_env() {
  cd "$ROOT"
  mkdir -p "$ASR_ML_REGISTER_OUTPUT_DIR"
  if [[ -x "$ROOT/.venv/bin/python" ]]; then
    PYTHON_BIN="$ROOT/.venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
  DBX=(databricks --profile "$ASR_ML_REGISTER_PROFILE")
  export DATABRICKS_CONFIG_PROFILE="$ASR_ML_REGISTER_PROFILE"
}

resolve_remote_root() {
  if [[ -n "$ASR_ML_REGISTER_REMOTE_ROOT" ]]; then
    return
  fi
  ASR_ML_REGISTER_REMOTE_ROOT="$("$PYTHON_BIN" - "$DEFAULT_REMOTE_SUBDIR" <<'PY'
from __future__ import annotations

import sys
from genie_voice.config import get_settings

s = get_settings()
catalog = s.databricks.catalog
schema = s.databricks.schema_name
volume = s.volume.streaming_name
if any("<" in str(v) or not str(v).strip() for v in (catalog, schema, volume)):
    raise SystemExit("Databricks catalog/schema/streaming volume are not configured.")
print(f"/Volumes/{catalog}/{schema}/{volume}/asr_model_training/{sys.argv[1]}")
PY
)"
}

candidate_ids() {
  printf '%s\n' \
    th_oss_qwen3_asr_1_7b \
    th_oss_typhoon_whisper_large_v3 \
    th_oss_pathumma_whisper_large_v3 \
    id_oss_qwen3_asr_1_7b \
    id_oss_qwen3_asr_0_6b \
    zh_oss_qwen3_asr_1_7b \
    zh_oss_qwen3_asr_0_6b
}

load_candidate() {
  local candidate="$1"
  CANDIDATE_ID="$candidate"
  case "$candidate" in
    th_oss_qwen3_asr_1_7b)
      FAMILY=qwen3; BASE_MODEL="Qwen/Qwen3-ASR-1.7B"; LANGUAGE_CODE=th; LANGUAGE_NAME=Thai ;;
    th_oss_typhoon_whisper_large_v3)
      FAMILY=whisper; BASE_MODEL="typhoon-ai/typhoon-whisper-large-v3"; LANGUAGE_CODE=th; LANGUAGE_NAME=Thai ;;
    th_oss_pathumma_whisper_large_v3)
      FAMILY=whisper; BASE_MODEL="nectec/Pathumma-whisper-th-large-v3"; LANGUAGE_CODE=th; LANGUAGE_NAME=Thai ;;
    id_oss_qwen3_asr_1_7b)
      FAMILY=qwen3; BASE_MODEL="Qwen/Qwen3-ASR-1.7B"; LANGUAGE_CODE=id; LANGUAGE_NAME=Indonesian ;;
    id_oss_qwen3_asr_0_6b)
      FAMILY=qwen3; BASE_MODEL="Qwen/Qwen3-ASR-0.6B"; LANGUAGE_CODE=id; LANGUAGE_NAME=Indonesian ;;
    zh_oss_qwen3_asr_1_7b)
      FAMILY=qwen3; BASE_MODEL="Qwen/Qwen3-ASR-1.7B"; LANGUAGE_CODE=zh; LANGUAGE_NAME=Chinese ;;
    zh_oss_qwen3_asr_0_6b)
      FAMILY=qwen3; BASE_MODEL="Qwen/Qwen3-ASR-0.6B"; LANGUAGE_CODE=zh; LANGUAGE_NAME=Chinese ;;
    *)
      err "Unknown candidate: $candidate"
      candidate_ids >&2
      exit 2
      ;;
  esac
  if [[ -n "$ASR_ML_REGISTER_LANGUAGE" && "$ASR_ML_REGISTER_LANGUAGE" != "$LANGUAGE_CODE" ]]; then
    err "Candidate $candidate is language=$LANGUAGE_CODE, not ASR_ML_REGISTER_LANGUAGE=$ASR_ML_REGISTER_LANGUAGE"
    exit 2
  fi
}

registered_model_name() {
  if [[ -n "$ASR_ML_REGISTERED_MODEL" ]]; then
    printf '%s\n' "$ASR_ML_REGISTERED_MODEL"
    return
  fi
  "$PYTHON_BIN" - "$CANDIDATE_ID" <<'PY'
from __future__ import annotations

import re
import sys
from genie_voice.config import get_settings

s = get_settings()
leaf = "genie_asr_" + re.sub(r"[^a-zA-Z0-9_]", "_", sys.argv[1]).lower()
print(f"{s.databricks.catalog}.{s.databricks.schema_name}.{leaf}")
PY
}

copy_runners() {
  local remote_jobs_dir="$ASR_ML_REGISTER_REMOTE_ROOT/jobs"
  "${DBX[@]}" fs mkdirs "dbfs:$remote_jobs_dir"
  "${DBX[@]}" fs cp "$ROOT/scripts/asr/databricks_register_multilingual_asr_candidate.py" "dbfs:$remote_jobs_dir/databricks_register_multilingual_asr_candidate.py" --overwrite
  "${DBX[@]}" fs cp "$ROOT/scripts/asr/databricks_eval_registered_multilingual_asr_candidate.py" "dbfs:$remote_jobs_dir/databricks_eval_registered_multilingual_asr_candidate.py" --overwrite
  "${DBX[@]}" fs cp "$ROOT/scripts/asr/mlflow_multilingual_asr_pyfunc.py" "dbfs:$remote_jobs_dir/mlflow_multilingual_asr_pyfunc.py" --overwrite
}

submit_job() {
  local job_json="$1"
  local run_json="$2"
  local label="$3"
  "${DBX[@]}" api post /api/2.1/jobs/runs/submit --json @"$job_json" --output json >"$run_json"
  local run_id
  run_id="$("$PYTHON_BIN" - "$run_json" <<'PY'
import json, sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["run_id"])
PY
)"
  wait_for_job_run "$run_id" "$label"
}

wait_for_job_run() {
  local run_id="$1"
  local label="$2"
  local lifecycle=""
  local result=""
  local url=""
  log "waiting for Databricks $label run $run_id"
  while true; do
    local payload
    payload="$("${DBX[@]}" jobs get-run "$run_id" --output json)"
    lifecycle="$("$PYTHON_BIN" - "$payload" <<'PY'
import json, sys
print((json.loads(sys.argv[1]).get("state") or {}).get("life_cycle_state") or "")
PY
)"
    result="$("$PYTHON_BIN" - "$payload" <<'PY'
import json, sys
print((json.loads(sys.argv[1]).get("state") or {}).get("result_state") or "")
PY
)"
    url="$("$PYTHON_BIN" - "$payload" <<'PY'
import json, sys
print(json.loads(sys.argv[1]).get("run_page_url") or "")
PY
)"
    printf "  lifecycle=%s result=%s url=%s\n" "$lifecycle" "${result:-}" "$url"
    case "$lifecycle" in
      TERMINATED|SKIPPED|INTERNAL_ERROR) break ;;
    esac
    sleep 20
  done
  if [[ "$result" != "SUCCESS" ]]; then
    err "Databricks $label failed: lifecycle=$lifecycle result=$result url=$url"
    return 2
  fi
}

registration_job_json() {
  local registered_model="$1"
  local output_json_remote="$2"
  local package_dir="$ASR_ML_REGISTER_REMOTE_ROOT/packages"
  local job_json="$ASR_ML_REGISTER_OUTPUT_DIR/register_${CANDIDATE_ID}.job.json"
  "$PYTHON_BIN" - "$job_json" <<PY
import json
import sys
from pathlib import Path

parameters = [
    "--candidate-id", "${CANDIDATE_ID}",
    "--family", "${FAMILY}",
    "--base-model", "${BASE_MODEL}",
    "--language-code", "${LANGUAGE_CODE}",
    "--language-name", "${LANGUAGE_NAME}",
    "--adaptation-type", "oss_baseline",
    "--fine-tuned-by-us", "false",
    "--registered-model", "${registered_model}",
    "--package-dir", "${package_dir}",
    "--wrapper-path", "${ASR_ML_REGISTER_REMOTE_ROOT}/jobs/mlflow_multilingual_asr_pyfunc.py",
    "--output-json", "${output_json_remote}",
]
if "${ASR_ML_REGISTER_FORCE_DOWNLOAD}" == "true":
    parameters.append("--force-download")

job = {
    "run_name": "register-multilingual-asr-${CANDIDATE_ID}",
    "tasks": [
        {
            "task_key": "register_candidate",
            "environment_key": "asr_register_env",
            "spark_python_task": {
                "python_file": "dbfs:${ASR_ML_REGISTER_REMOTE_ROOT}/jobs/databricks_register_multilingual_asr_candidate.py",
                "parameters": parameters,
            },
        }
    ],
    "environments": [
        {
            "environment_key": "asr_register_env",
            "spec": {
                "environment_version": "${ASR_ML_REGISTER_ENVIRONMENT_VERSION}",
                "dependencies": [
                    "mlflow",
                    "huggingface_hub",
                    "torch",
                    "transformers",
                    "accelerate",
                    "qwen-asr",
                    "librosa",
                    "soundfile",
                    "pandas"
                ],
            },
        }
    ],
}
Path(sys.argv[1]).write_text(json.dumps(job, indent=2), encoding="utf-8")
PY
  printf '%s\n' "$job_json"
}

evaluate_job_json() {
  local registered_model="$1"
  local manifest_remote="$2"
  local output_remote="$3"
  local summary_remote="$4"
  local job_json="$ASR_ML_REGISTER_OUTPUT_DIR/evaluate_${CANDIDATE_ID}.job.json"
  "$PYTHON_BIN" - "$job_json" <<PY
import json
import sys
from pathlib import Path

job = {
    "run_name": "evaluate-multilingual-asr-${CANDIDATE_ID}",
    "tasks": [
        {
            "task_key": "evaluate_candidate",
            "environment_key": "asr_eval_env",
            "spark_python_task": {
                "python_file": "dbfs:${ASR_ML_REGISTER_REMOTE_ROOT}/jobs/databricks_eval_registered_multilingual_asr_candidate.py",
                "parameters": [
                    "evaluate",
                    "--registered-model", "${registered_model}",
                    "--candidate-id", "${CANDIDATE_ID}",
                    "--manifest", "${manifest_remote}",
                    "--language", "${LANGUAGE_CODE}",
                    "--output", "${output_remote}",
                    "--summary-output", "${summary_remote}",
                    "--limit", "${ASR_ML_REGISTER_LIMIT}",
                ],
            },
        }
    ],
    "environments": [
        {
            "environment_key": "asr_eval_env",
            "spec": {
                "environment_version": "${ASR_ML_REGISTER_ENVIRONMENT_VERSION}",
                "dependencies": [
                    "mlflow",
                    "torch",
                    "transformers",
                    "accelerate",
                    "qwen-asr",
                    "huggingface_hub",
                    "librosa",
                    "soundfile",
                    "pyarrow",
                    "pandas"
                ],
            },
        }
    ],
}
Path(sys.argv[1]).write_text(json.dumps(job, indent=2), encoding="utf-8")
PY
  printf '%s\n' "$job_json"
}

bootstrap_public_manifest_job_json() {
  local manifest_remote="$1"
  local scaffold_remote="$ASR_ML_REGISTER_REMOTE_ROOT/public_manifest"
  local job_json="$ASR_ML_REGISTER_OUTPUT_DIR/bootstrap_public_manifest.job.json"
  "$PYTHON_BIN" - "$job_json" <<PY
import json
import sys
from pathlib import Path

job = {
    "run_name": "bootstrap-multilingual-public-asr-manifest",
    "tasks": [
        {
            "task_key": "bootstrap_public_manifest",
            "environment_key": "asr_public_manifest_env",
            "spark_python_task": {
                "python_file": "dbfs:${ASR_ML_REGISTER_REMOTE_ROOT}/jobs/databricks_eval_registered_multilingual_asr_candidate.py",
                "parameters": [
                    "bootstrap-public-manifest",
                    "--manifest", "${manifest_remote}",
                    "--public-limit", "${ASR_ML_REGISTER_PUBLIC_LIMIT}",
                    "--scaffold-dir", "${scaffold_remote}",
                ],
            },
        }
    ],
    "environments": [
        {
            "environment_key": "asr_public_manifest_env",
            "spec": {
                "environment_version": "${ASR_ML_REGISTER_ENVIRONMENT_VERSION}",
                "dependencies": [
                    "mlflow",
                    "huggingface_hub",
                    "pandas",
                    "pyarrow",
                    "soundfile"
                ],
            },
        }
    ],
}
Path(sys.argv[1]).write_text(json.dumps(job, indent=2), encoding="utf-8")
PY
  printf '%s\n' "$job_json"
}

sync_manifest() {
  if [[ -z "$ASR_ML_REGISTER_MANIFEST" ]]; then
    err "ASR_ML_REGISTER_MANIFEST is required for evaluation"
    exit 2
  fi
  if [[ "$ASR_ML_REGISTER_MANIFEST" == /Volumes/* ]]; then
    printf '%s\n' "$ASR_ML_REGISTER_MANIFEST"
    return
  fi
  local remote_inputs_dir="$ASR_ML_REGISTER_REMOTE_ROOT/inputs"
  local manifest_remote="$remote_inputs_dir/$(basename "$ASR_ML_REGISTER_MANIFEST")"
  "${DBX[@]}" fs mkdirs "dbfs:$remote_inputs_dir" >/dev/null
  "${DBX[@]}" fs cp "$ASR_ML_REGISTER_MANIFEST" "dbfs:$manifest_remote" --overwrite >/dev/null
  printf '%s\n' "$manifest_remote"
}

bootstrap_public_manifest() {
  setup_env
  resolve_remote_root
  copy_runners
  local remote_inputs_dir="$ASR_ML_REGISTER_REMOTE_ROOT/inputs"
  local manifest_remote="${ASR_ML_REGISTER_MANIFEST:-$remote_inputs_dir/multilingual_public_fleurs_manifest.jsonl}"
  local local_manifest="$ASR_ML_REGISTER_OUTPUT_DIR/multilingual_public_fleurs_manifest.jsonl"
  "${DBX[@]}" fs mkdirs "dbfs:$remote_inputs_dir"
  local job_json
  job_json="$(bootstrap_public_manifest_job_json "$manifest_remote")"
  local run_json="$ASR_ML_REGISTER_OUTPUT_DIR/bootstrap_public_manifest_run.json"
  submit_job "$job_json" "$run_json" "public manifest bootstrap"
  "${DBX[@]}" fs cp "dbfs:$manifest_remote" "$local_manifest" --overwrite
  log "public manifest written to $manifest_remote"
  log "local copy: $local_manifest"
}

register_one() {
  setup_env
  resolve_remote_root
  copy_runners
  load_candidate "$ASR_ML_REGISTER_CANDIDATE"
  local registered_model
  registered_model="$(registered_model_name)"
  local remote_outputs_dir="$ASR_ML_REGISTER_REMOTE_ROOT/outputs/$CANDIDATE_ID"
  local local_dir="$ASR_ML_REGISTER_OUTPUT_DIR/$CANDIDATE_ID"
  mkdir -p "$local_dir"
  "${DBX[@]}" fs mkdirs "dbfs:$remote_outputs_dir"
  local output_json_remote="$remote_outputs_dir/registration.json"
  local job_json
  job_json="$(registration_job_json "$registered_model" "$output_json_remote")"
  local run_json="$local_dir/register_run.json"
  submit_job "$job_json" "$run_json" "registration"
  "${DBX[@]}" fs cp "dbfs:$output_json_remote" "$local_dir/registration.json" --overwrite
  log "registered $CANDIDATE_ID as $registered_model@candidate"
}

evaluate_one() {
  setup_env
  resolve_remote_root
  copy_runners
  load_candidate "$ASR_ML_REGISTER_CANDIDATE"
  local registered_model
  registered_model="$(registered_model_name)"
  local manifest_remote
  manifest_remote="$(sync_manifest)"
  local remote_outputs_dir="$ASR_ML_REGISTER_REMOTE_ROOT/outputs/$CANDIDATE_ID"
  local local_dir="$ASR_ML_REGISTER_OUTPUT_DIR/$CANDIDATE_ID"
  mkdir -p "$local_dir"
  "${DBX[@]}" fs mkdirs "dbfs:$remote_outputs_dir"
  local output_remote="$remote_outputs_dir/eval_results.jsonl"
  local summary_remote="$remote_outputs_dir/eval_summary.json"
  local job_json
  if ! job_json="$(evaluate_job_json "$registered_model" "$manifest_remote" "$output_remote" "$summary_remote")"; then
    err "Failed to write evaluation job JSON"
    exit 2
  fi
  local run_json="$local_dir/evaluate_run.json"
  submit_job "$job_json" "$run_json" "evaluation"
  "${DBX[@]}" fs cp "dbfs:$output_remote" "$local_dir/eval_results.jsonl" --overwrite
  "${DBX[@]}" fs cp "dbfs:$summary_remote" "$local_dir/eval_summary.json" --overwrite
  log "evaluated $CANDIDATE_ID through $registered_model@candidate"
}

run_for_all() {
  local subcommand="$1"
  local failed=0
  while IFS= read -r candidate; do
    ASR_ML_REGISTER_CANDIDATE="$candidate" "$0" "$subcommand" || failed=1
  done < <(candidate_ids)
  return "$failed"
}

case "$COMMAND" in
  list)
    candidate_ids
    ;;
  register-one)
    if [[ -z "$ASR_ML_REGISTER_CANDIDATE" ]]; then
      err "ASR_ML_REGISTER_CANDIDATE is required"
      exit 2
    fi
    register_one
    ;;
  evaluate-one)
    if [[ -z "$ASR_ML_REGISTER_CANDIDATE" ]]; then
      err "ASR_ML_REGISTER_CANDIDATE is required"
      exit 2
    fi
    evaluate_one
    ;;
  register-all)
    run_for_all register-one
    ;;
  evaluate-all)
    run_for_all evaluate-one
    ;;
  bootstrap-public-manifest)
    bootstrap_public_manifest
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
