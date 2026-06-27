#!/usr/bin/env bash
# =============================================================================
# 05_register_asr_model_candidate.sh
#
# Register the trained ASR LoRA adapter as a Unity Catalog *candidate* model.
# This packages lineage and artifacts; it does not start or require a GPU cluster.
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOCAL_REGISTRY_DIR="$ROOT/.run/asr_model_training/registry_package"
ASR_BASE_MODEL="${ASR_LORA_BASE_MODEL:-openai/whisper-small.en}"
ASR_REGISTERED_MODEL_NAME="${ASR_REGISTERED_MODEL_NAME:-genie_asr_whisper_lora}"
ASR_LORA_RUN_NAME="${ASR_LORA_RUN_NAME:-}"
ASR_SKIP_MLFLOW_REGISTER="${ASR_SKIP_MLFLOW_REGISTER:-false}"
ASR_REGISTRATION_MODE="${ASR_REGISTRATION_MODE:-databricks-serverless}"
ASR_SERVERLESS_ENVIRONMENT_VERSION="${ASR_SERVERLESS_ENVIRONMENT_VERSION:-2}"
ASR_SERVERLESS_PERFORMANCE_TARGET="${ASR_SERVERLESS_PERFORMANCE_TARGET:-PERFORMANCE_OPTIMIZED}"

COMMAND="${1:-register-candidate}"
if [[ $# -gt 0 ]]; then
  shift
fi

log() { printf "\033[35m[asr-register]\033[0m %s\n" "$*"; }
err() { printf "\033[31m[asr-register]\033[0m %s\n" "$*" >&2; }

usage() {
  cat <<'EOF'
ASR UC model registration

Purpose:
  Package and register the trained Whisper LoRA ASR adapter as a Unity Catalog
  candidate model. This does not promote to production.

Commands:
  register-candidate  Package artifacts and register candidate model version.
  smoke-test-candidate
                      Load models:/<registered_model>@candidate and run one
                      app-contract prediction against a manifest clip.
  preflight           Validate auth, config, artifacts, and reports.
  package-only        Copy Volume artifacts into the local registry package cache.
  help                Show this help.

Environment:
  ASR_DATABRICKS_PROFILE       Databricks CLI profile. Default: fe-vm-vdm-classic-rcn6ip
  ASR_LORA_RUN_NAME            LoRA run name. Default: latest lora_* under Volume.
  ASR_REGISTERED_MODEL_NAME    UC model leaf name. Default: genie_asr_whisper_lora
  ASR_REGISTRATION_MODE        databricks-serverless or local. Default: databricks-serverless.
  ASR_SERVERLESS_PERFORMANCE_TARGET
                                Serverless job target. Default: PERFORMANCE_OPTIMIZED.
  ASR_SERVERLESS_ENVIRONMENT_VERSION
                                Serverless environment version. Default: 2.
  ASR_SKIP_MLFLOW_REGISTER     true = package only, skip MLflow registration.

Important:
  - No GPU or classic cluster is started. Default registration uses a one-off
    Databricks serverless job with PERFORMANCE_OPTIMIZED target.
  - Registered pyfunc contract matches the app upload path:
    input:  audio_b64, mime_type, speaker
    output: raw_transcript, transcript, confidence, model, base_model,
            lora_run_name, requires_invoice_postprocessing
  - Batch/eval audio_path input is also supported by the wrapper because the
    model was trained/evaluated on utterance-level audio files.
  - Model is tagged candidate-only.
  - Production promotion requires real recorded holdout approval.

EOF
}

setup_env() {
  cd "$ROOT"
  if [[ -f "$ROOT/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$ROOT/.env"
    set +a
  fi
  if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
    python3 -m venv "$ROOT/.venv"
  fi
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
  export PYTHONPATH="$ROOT/backend${PYTHONPATH:+:$PYTHONPATH}"
  if ! python -c "import genie_voice" >/dev/null 2>&1; then
    python -m pip install -q --upgrade pip
    python -m pip install -q -e "$ROOT/backend"
  fi
}

setup_databricks_cli() {
  if ! command -v databricks >/dev/null 2>&1; then
    err "Databricks CLI is not installed or not on PATH."
    exit 1
  fi
  local profile="${ASR_DATABRICKS_PROFILE:-${DATABRICKS_CONFIG_PROFILE:-fe-vm-vdm-classic-rcn6ip}}"
  DBX=(databricks --profile "$profile")
  export DATABRICKS_CONFIG_PROFILE="$profile"
  export ASR_DATABRICKS_PROFILE="$profile"
}

resolve_paths() {
  eval "$(python - <<'PY'
import shlex
from genie_voice.config import get_settings

s = get_settings()
catalog = s.databricks.catalog
schema = s.databricks.schema_name
volume = s.volume.streaming_name
if any("<" in str(v) or not str(v).strip() for v in (catalog, schema, volume)):
    raise SystemExit("Databricks catalog/schema/streaming volume are not configured.")

root = f"/Volumes/{catalog}/{schema}/{volume}/asr_model_training"
registered_model = f"{catalog}.{schema}."
paths = {
    "ASR_CATALOG": catalog,
    "ASR_SCHEMA": schema,
    "ASR_TRAINING_ROOT": root,
    "ASR_TRAINING_MANIFEST": f"{root}/datasets/gold/manifests/asr_training_gold_v1.jsonl",
    "ASR_HOLDOUT_MANIFEST": f"{root}/datasets/holdout/manifests/asr_real_audio_holdout_v1.jsonl",
    "ASR_MODEL_ARTIFACTS": f"{root}/model_artifacts",
    "ASR_EVALUATIONS": f"{root}/evaluations",
    "ASR_REGISTERED_MODEL_FQDN_PREFIX": registered_model,
    "DATABRICKS_HOST_RESOLVED": s.databricks_host,
}
for key, value in paths.items():
    print(f"{key}={shlex.quote(str(value))}")
PY
)"
  ASR_REGISTERED_MODEL_FQDN="${ASR_REGISTERED_MODEL_FQDN_PREFIX}${ASR_REGISTERED_MODEL_NAME}"
}

latest_lora_run_name() {
  if [[ -n "$ASR_LORA_RUN_NAME" ]]; then
    printf "%s\n" "$ASR_LORA_RUN_NAME"
    return
  fi
  local runs
  runs="$("${DBX[@]}" fs ls "dbfs:$ASR_MODEL_ARTIFACTS/lora_runs")"
  python - "$runs" <<'PY'
import sys
names = sorted(line.strip().rstrip("/") for line in sys.argv[1].splitlines() if line.strip().startswith("lora_"))
if names:
    print(names[-1])
PY
}

volume_path_exists() {
  local path="$1"
  if "${DBX[@]}" fs ls "dbfs:$path" >/dev/null 2>&1; then
    return 0
  fi
  local parent
  local name
  local listing
  parent="$(dirname "$path")"
  name="$(basename "$path")"
  listing="$("${DBX[@]}" fs ls "dbfs:$parent" 2>/dev/null || true)"
  python - "$name" "$listing" <<'PY'
import sys

name = sys.argv[1]
listing = sys.argv[2]
for line in listing.splitlines():
    if line.strip().rstrip("/") == name:
        raise SystemExit(0)
raise SystemExit(1)
PY
}

preflight() {
  setup_env
  setup_databricks_cli
  resolve_paths
  ASR_LORA_RUN_NAME="$(latest_lora_run_name)"
  if [[ -z "$ASR_LORA_RUN_NAME" ]]; then
    err "No LoRA run found under $ASR_MODEL_ARTIFACTS/lora_runs"
    exit 2
  fi

  local lora_dir="$ASR_MODEL_ARTIFACTS/lora_runs/$ASR_LORA_RUN_NAME"
  for path in \
    "$lora_dir/adapter" \
    "$lora_dir/processor" \
    "$lora_dir/run_config.json" \
    "$lora_dir/train_metrics.json" \
    "$ASR_EVALUATIONS/asr_lora_invoice_postprocessed_comparison_${ASR_LORA_RUN_NAME}.json" \
    "$ASR_EVALUATIONS/asr_entity_error_analysis_invoice_postprocessed_${ASR_LORA_RUN_NAME}.md"
  do
    if ! volume_path_exists "$path"; then
      err "Missing required Volume artifact: $path"
      exit 2
    fi
  done

  if [[ "$ASR_SKIP_MLFLOW_REGISTER" != "true" && "$ASR_REGISTRATION_MODE" == "local" ]]; then
    if ! python - <<'PY' >/dev/null 2>&1
import mlflow
PY
    then
      err "Python package 'mlflow' is required for registration. Install it in .venv or set ASR_SKIP_MLFLOW_REGISTER=true for package-only."
      exit 2
    fi
  fi

  cat <<EOF
ASR registration preflight passed.

LoRA run:
  $ASR_LORA_RUN_NAME

Registered model:
  $ASR_REGISTERED_MODEL_FQDN

Volume artifacts:
  $lora_dir

Registered model app contract:
  input:  audio_b64, mime_type, speaker
  output: raw_transcript, transcript, confidence, model, base_model, lora_run_name,
          requires_invoice_postprocessing

Training/eval compatibility:
  wrapper also accepts audio_path for utterance-level batch evaluation.

EOF
}

upload_registration_scripts() {
  local scripts_dir="$ASR_TRAINING_ROOT/registration_scripts"
  "${DBX[@]}" fs mkdirs "dbfs:$scripts_dir"
  "${DBX[@]}" fs cp \
    "$ROOT/scripts/asr/databricks_register_asr_model_candidate.py" \
    "dbfs:$scripts_dir/databricks_register_asr_model_candidate.py" \
    --overwrite
  "${DBX[@]}" fs cp \
    "$ROOT/scripts/asr/mlflow_whisper_lora_pyfunc.py" \
    "dbfs:$scripts_dir/mlflow_whisper_lora_pyfunc.py" \
    --overwrite
  "${DBX[@]}" fs cp \
    "$ROOT/scripts/asr/databricks_smoke_test_asr_model_candidate.py" \
    "dbfs:$scripts_dir/databricks_smoke_test_asr_model_candidate.py" \
    --overwrite
  ASR_REGISTRATION_RUNNER="$scripts_dir/databricks_register_asr_model_candidate.py"
  ASR_REGISTRATION_WRAPPER="$scripts_dir/mlflow_whisper_lora_pyfunc.py"
  ASR_SMOKE_TEST_RUNNER="$scripts_dir/databricks_smoke_test_asr_model_candidate.py"
}

resolve_smoke_manifest() {
  if volume_path_exists "$ASR_HOLDOUT_MANIFEST"; then
    ASR_SMOKE_MANIFEST="$ASR_HOLDOUT_MANIFEST"
  elif volume_path_exists "$ASR_TRAINING_MANIFEST"; then
    ASR_SMOKE_MANIFEST="$ASR_TRAINING_MANIFEST"
  else
    err "No manifest found for smoke test: $ASR_HOLDOUT_MANIFEST or $ASR_TRAINING_MANIFEST"
    exit 2
  fi
}

wait_for_job_run() {
  local run_id="$1"
  local label="$2"
  local task_run_id=""
  local lifecycle=""
  local result=""
  local url=""

  log "waiting for Databricks $label run $run_id"
  while true; do
    local state_json
    state_json="$("${DBX[@]}" jobs get-run "$run_id" --output json)"
    lifecycle="$(python - "$state_json" <<'PY'
import json, sys
payload = json.loads(sys.argv[1])
print(payload.get("state", {}).get("life_cycle_state", ""))
PY
)"
    result="$(python - "$state_json" <<'PY'
import json, sys
payload = json.loads(sys.argv[1])
print(payload.get("state", {}).get("result_state", ""))
PY
)"
    url="$(python - "$state_json" <<'PY'
import json, sys
payload = json.loads(sys.argv[1])
print(payload.get("run_page_url", ""))
PY
)"
    task_run_id="$(python - "$state_json" <<'PY'
import json, sys
payload = json.loads(sys.argv[1])
tasks = payload.get("tasks") or []
print(tasks[0].get("run_id", "") if tasks else "")
PY
)"
    case "$lifecycle" in
      TERMINATED|SKIPPED|INTERNAL_ERROR)
        break
        ;;
    esac
    sleep 20
  done

  if [[ -n "$task_run_id" ]]; then
    "${DBX[@]}" jobs get-run-output "$task_run_id" --output json
  else
    "${DBX[@]}" jobs get-run-output "$run_id" --output json
  fi
  if [[ "$result" != "SUCCESS" ]]; then
    err "Databricks $label failed: lifecycle=$lifecycle result=$result url=$url"
    exit 2
  fi
}

submit_databricks_serverless_registration() {
  preflight
  upload_registration_scripts

  local job_json="$ROOT/.run/asr_model_training/register_candidate_job_${ASR_LORA_RUN_NAME}.json"
  local run_json="$ROOT/.run/asr_model_training/register_candidate_run_${ASR_LORA_RUN_NAME}.json"
  mkdir -p "$(dirname "$job_json")"

  python - "$job_json" <<PY
import json
import sys

job = {
    "run_name": "register-asr-candidate-${ASR_LORA_RUN_NAME}",
    "performance_target": "${ASR_SERVERLESS_PERFORMANCE_TARGET}",
    "tasks": [
        {
            "task_key": "register_candidate",
            "environment_key": "asr_register_env",
            "spark_python_task": {
                "python_file": "dbfs:${ASR_REGISTRATION_RUNNER}",
                "parameters": [
                    "--model-artifacts", "${ASR_MODEL_ARTIFACTS}",
                    "--evaluations", "${ASR_EVALUATIONS}",
                    "--lora-run-name", "${ASR_LORA_RUN_NAME}",
                    "--registered-model", "${ASR_REGISTERED_MODEL_FQDN}",
                    "--base-model", "${ASR_BASE_MODEL}",
                    "--wrapper-path", "${ASR_REGISTRATION_WRAPPER}",
                ],
            },
        }
    ],
    "environments": [
        {
            "environment_key": "asr_register_env",
            "spec": {
                "environment_version": "${ASR_SERVERLESS_ENVIRONMENT_VERSION}",
                "dependencies": [
                    "mlflow",
                    "pandas"
                ],
            },
        }
    ],
}
Path = __import__("pathlib").Path
Path(sys.argv[1]).write_text(json.dumps(job, indent=2), encoding="utf-8")
PY

  log "submitting performance-optimized serverless Databricks registration job"
  "${DBX[@]}" jobs submit --json @"$job_json" --output json >"$run_json"

  local run_id
  run_id="$(python - "$run_json" <<'PY'
import json
import sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(payload["run_id"])
PY
)"

  wait_for_job_run "$run_id" "registration"
}

smoke_test_candidate() {
  preflight
  upload_registration_scripts
  resolve_smoke_manifest

  local job_json="$ROOT/.run/asr_model_training/smoke_test_candidate_job_${ASR_LORA_RUN_NAME}.json"
  local run_json="$ROOT/.run/asr_model_training/smoke_test_candidate_run_${ASR_LORA_RUN_NAME}.json"
  mkdir -p "$(dirname "$job_json")"

  python - "$job_json" <<PY
import json
import sys
from pathlib import Path

job = {
    "run_name": "smoke-test-asr-candidate-${ASR_LORA_RUN_NAME}",
    "performance_target": "${ASR_SERVERLESS_PERFORMANCE_TARGET}",
    "tasks": [
        {
            "task_key": "smoke_test_candidate",
            "environment_key": "asr_smoke_env",
            "spark_python_task": {
                "python_file": "dbfs:${ASR_SMOKE_TEST_RUNNER}",
                "parameters": [
                    "--registered-model", "${ASR_REGISTERED_MODEL_FQDN}",
                    "--manifest", "${ASR_SMOKE_MANIFEST}",
                ],
            },
        }
    ],
    "environments": [
        {
            "environment_key": "asr_smoke_env",
            "spec": {
                "environment_version": "${ASR_SERVERLESS_ENVIRONMENT_VERSION}",
                "dependencies": [
                    "mlflow",
                    "pandas",
                    "torch",
                    "transformers",
                    "peft",
                    "librosa",
                    "soundfile"
                ],
            },
        }
    ],
}
Path(sys.argv[1]).write_text(json.dumps(job, indent=2), encoding="utf-8")
PY

  log "submitting performance-optimized serverless ASR candidate smoke test"
  "${DBX[@]}" jobs submit --json @"$job_json" --output json >"$run_json"

  local run_id
  run_id="$(python - "$run_json" <<'PY'
import json
import sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(payload["run_id"])
PY
)"
  wait_for_job_run "$run_id" "smoke test"
}

copy_path_from_volume() {
  local remote_path="$1"
  local local_path="$2"
  rm -rf "$local_path"
  "${DBX[@]}" fs cp "dbfs:$remote_path" "$local_path" --recursive --overwrite
}

package_artifacts() {
  preflight

  local package_dir="$LOCAL_REGISTRY_DIR/$ASR_LORA_RUN_NAME"
  local lora_dir="$ASR_MODEL_ARTIFACTS/lora_runs/$ASR_LORA_RUN_NAME"
  mkdir -p "$package_dir"

  log "copying LoRA artifacts from Volume into local package cache"
  copy_path_from_volume "$lora_dir/adapter" "$package_dir/adapter"
  copy_path_from_volume "$lora_dir/processor" "$package_dir/processor"
  "${DBX[@]}" fs cp "dbfs:$lora_dir/run_config.json" "$package_dir/run_config.json" --overwrite
  "${DBX[@]}" fs cp "dbfs:$lora_dir/train_metrics.json" "$package_dir/train_metrics.json" --overwrite
  "${DBX[@]}" fs cp \
    "dbfs:$ASR_EVALUATIONS/asr_lora_invoice_postprocessed_comparison_${ASR_LORA_RUN_NAME}.json" \
    "$package_dir/asr_lora_invoice_postprocessed_comparison.json" \
    --overwrite
  "${DBX[@]}" fs cp \
    "dbfs:$ASR_EVALUATIONS/asr_entity_error_analysis_invoice_postprocessed_${ASR_LORA_RUN_NAME}.md" \
    "$package_dir/asr_entity_error_analysis_invoice_postprocessed.md" \
    --overwrite

  cat >"$package_dir/ASR_MODEL_CARD.md" <<EOF
# Genie ASR Whisper LoRA Candidate

- Status: candidate
- Base model: $ASR_BASE_MODEL
- LoRA run: $ASR_LORA_RUN_NAME
- Adapter Volume path: $lora_dir/adapter
- Processor Volume path: $lora_dir/processor
- Registration target: $ASR_REGISTERED_MODEL_FQDN
- Production gate: real recorded holdout required before promotion

This package includes adapter/processor artifacts and evaluation reports. The
runtime ASR path also depends on candidate-aware invoice-ID postprocessing.
EOF

  cat >"$package_dir/model_metadata.json" <<EOF
{
  "status": "candidate",
  "base_model": "$ASR_BASE_MODEL",
  "lora_run_name": "$ASR_LORA_RUN_NAME",
  "registered_model": "$ASR_REGISTERED_MODEL_FQDN",
  "adapter_volume_path": "$lora_dir/adapter",
  "processor_volume_path": "$lora_dir/processor",
  "requires_invoice_postprocessing": true,
  "requires_real_recorded_holdout_before_production": true
}
EOF

  cat <<EOF
Packaged ASR candidate artifacts.

Local package:
  $package_dir

EOF
}

register_candidate() {
  if [[ "$ASR_REGISTRATION_MODE" == "databricks-serverless" ]]; then
    submit_databricks_serverless_registration
    return
  fi

  package_artifacts
  if [[ "$ASR_SKIP_MLFLOW_REGISTER" == "true" ]]; then
    log "Skipping MLflow registration because ASR_SKIP_MLFLOW_REGISTER=true"
    return
  fi

  export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-databricks}"
  export MLFLOW_REGISTRY_URI="${MLFLOW_REGISTRY_URI:-databricks-uc}"
  export DATABRICKS_HOST="${DATABRICKS_HOST:-$DATABRICKS_HOST_RESOLVED}"

  local package_dir="$LOCAL_REGISTRY_DIR/$ASR_LORA_RUN_NAME"
  python - "$package_dir" "$ASR_REGISTERED_MODEL_FQDN" "$ASR_LORA_RUN_NAME" "$ASR_BASE_MODEL" <<'PY'
import json
import sys
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

package_dir = Path(sys.argv[1])
registered_model = sys.argv[2]
lora_run_name = sys.argv[3]
base_model = sys.argv[4]
metadata = json.loads((package_dir / "model_metadata.json").read_text(encoding="utf-8"))

sys.path.insert(0, str(Path.cwd() / "scripts" / "asr"))
from mlflow_whisper_lora_pyfunc import WhisperLoraASRModel  # noqa: E402

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

experiment_name = "/Users/suneel.sunkara@databricks.com/genie_asr_model_registration"
try:
    experiment = client.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id if experiment else client.create_experiment(experiment_name)
except Exception:
    experiment_id = None

with mlflow.start_run(experiment_id=experiment_id, run_name=f"register-{lora_run_name}") as run:
    mlflow.log_param("status", "candidate")
    mlflow.log_param("base_model", base_model)
    mlflow.log_param("lora_run_name", lora_run_name)
    mlflow.log_param("requires_invoice_postprocessing", True)
    mlflow.log_param("requires_real_recorded_holdout_before_production", True)
    mlflow.log_artifacts(str(package_dir), artifact_path="asr_candidate_package_raw")

    input_example = pd.DataFrame(
        [
            {
                "audio_b64": "UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA=",
                "mime_type": "audio/wav",
                "speaker": 1,
            }
        ]
    )
    output_example = pd.DataFrame(
        [
            {
                "raw_transcript": "example transcript",
                "transcript": "example transcript",
                "confidence": 0.0,
                "model": "whisper_lora",
                "base_model": base_model,
                "lora_run_name": lora_run_name,
                "requires_invoice_postprocessing": True,
            }
        ]
    )
    signature = infer_signature(input_example, output_example)

    model_info = mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=WhisperLoraASRModel(metadata),
        artifacts={
            "adapter": str(package_dir / "adapter"),
            "processor": str(package_dir / "processor"),
        },
        registered_model_name=registered_model,
        signature=signature,
        input_example=input_example,
        pip_requirements=[
            "mlflow",
            "torch",
            "transformers",
            "peft",
            "librosa",
            "soundfile",
            "pandas",
        ],
        code_paths=[str(Path.cwd() / "scripts" / "asr" / "mlflow_whisper_lora_pyfunc.py")],
        metadata=metadata,
    )

    source = model_info.model_uri

latest_versions = client.search_model_versions(f"name = '{registered_model}'")
version = max(latest_versions, key=lambda item: int(item.version))
client.set_model_version_tag(registered_model, version.version, "status", "candidate")
client.set_model_version_tag(registered_model, version.version, "base_model", base_model)
client.set_model_version_tag(registered_model, version.version, "lora_run_name", lora_run_name)
client.set_model_version_tag(registered_model, version.version, "requires_invoice_postprocessing", "true")
client.set_model_version_tag(registered_model, version.version, "requires_real_recorded_holdout_before_production", "true")
client.set_registered_model_alias(registered_model, "candidate", version.version)

print(
    json.dumps(
        {
            "registered_model": registered_model,
            "version": version.version,
            "alias": "candidate",
            "run_id": version.run_id,
            "source": source,
            "metadata": metadata,
        },
        indent=2,
    )
)
PY
}

case "$COMMAND" in
  preflight)
    preflight
    ;;
  smoke-test-candidate)
    smoke_test_candidate
    ;;
  package-only)
    ASR_SKIP_MLFLOW_REGISTER=true package_artifacts
    ;;
  register-candidate)
    register_candidate
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    err "Unknown command: $COMMAND"
    usage >&2
    exit 2
    ;;
esac
