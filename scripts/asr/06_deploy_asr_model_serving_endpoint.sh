#!/usr/bin/env bash
# =============================================================================
# 06_deploy_asr_model_serving_endpoint.sh
#
# Deploy the registered ASR candidate model to a responsive Databricks Model
# Serving endpoint for app smoke tests as a Deepgram alternative.
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

ASR_REGISTERED_MODEL_NAME="${ASR_REGISTERED_MODEL_NAME:-genie_asr_whisper_lora}"
ASR_MODEL_ALIAS="${ASR_MODEL_ALIAS:-candidate}"
ASR_MODEL_VERSION="${ASR_MODEL_VERSION:-}"
ASR_SERVING_ENDPOINT_NAME="${ASR_SERVING_ENDPOINT_NAME:-voice_finetuned_whisper_model}"
ASR_SERVING_SERVED_ENTITY_NAME="${ASR_SERVING_SERVED_ENTITY_NAME:-asr_candidate}"
ASR_SERVING_WORKLOAD_TYPE="${ASR_SERVING_WORKLOAD_TYPE:-CPU}"
ASR_SERVING_WORKLOAD_SIZE="${ASR_SERVING_WORKLOAD_SIZE:-Medium}"
ASR_SERVING_SCALE_TO_ZERO="${ASR_SERVING_SCALE_TO_ZERO:-false}"
ASR_SERVING_ROUTE_OPTIMIZED="${ASR_SERVING_ROUTE_OPTIMIZED:-false}"
ASR_SERVING_TIMEOUT="${ASR_SERVING_TIMEOUT:-60m}"

COMMAND="${1:-deploy}"
if [[ $# -gt 0 ]]; then
  shift
fi

log() { printf "\033[36m[asr-serving]\033[0m %s\n" "$*"; }
err() { printf "\033[31m[asr-serving]\033[0m %s\n" "$*" >&2; }

usage() {
  cat <<'EOF'
ASR Databricks Model Serving deployment

Commands:
  deploy      Create or update the ASR serving endpoint.
  status      Show endpoint status/config.
  smoke-test  Query the serving endpoint with one manifest audio clip.
  preflight   Validate model alias/version and serving config.
  help        Show this help.

Environment:
  ASR_DATABRICKS_PROFILE        Databricks CLI profile. Default: fe-vm-vdm-classic-rcn6ip
  ASR_REGISTERED_MODEL_NAME     UC model leaf name. Default: genie_asr_whisper_lora
  ASR_MODEL_ALIAS               Model alias to deploy. Default: candidate
  ASR_MODEL_VERSION             Explicit version override. Default: resolve alias.
  ASR_SERVING_ENDPOINT_NAME     Endpoint name. Default: voice_finetuned_whisper_model
  ASR_SERVING_WORKLOAD_TYPE     GPU_SMALL, GPU_MEDIUM, CPU, etc. Default: CPU.
  ASR_SERVING_WORKLOAD_SIZE     Small, Medium, or Large. Default: Medium.
  ASR_SERVING_SCALE_TO_ZERO     false keeps endpoint warm. Default: false.
  ASR_SERVING_ROUTE_OPTIMIZED   true optimizes routing on create. Default: false.

Compute choice:
  - Uses Databricks Model Serving serverless compute.
  - Default CPU Medium warm serving keeps cost lower after GPU did not improve latency enough.
  - scale_to_zero=false avoids cold starts for speaking-input app tests.
  - route_optimized=false matches the app's current SDK/unified-auth path.
    Use true only after adding scoped OAuth authorization_details handling.
  - Override ASR_SERVING_WORKLOAD_TYPE/ASR_SERVING_WORKLOAD_SIZE to compare GPU or larger CPU settings.

Contract:
  input:  dataframe_records with audio_b64, mime_type, speaker
  output: raw_transcript, transcript, confidence, model, base_model,
          lora_run_name, requires_invoice_postprocessing

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
paths = {
    "ASR_CATALOG": catalog,
    "ASR_SCHEMA": schema,
    "ASR_REGISTERED_MODEL_FQDN_PREFIX": f"{catalog}.{schema}.",
    "ASR_HOLDOUT_MANIFEST": f"{root}/datasets/holdout/manifests/asr_real_audio_holdout_v1.jsonl",
    "ASR_TRAINING_MANIFEST": f"{root}/datasets/gold/manifests/asr_training_gold_v1.jsonl",
    "DATABRICKS_HOST_RESOLVED": s.databricks_host,
}
for key, value in paths.items():
    print(f"{key}={shlex.quote(str(value))}")
PY
)"
  ASR_REGISTERED_MODEL_FQDN="${ASR_REGISTERED_MODEL_FQDN_PREFIX}${ASR_REGISTERED_MODEL_NAME}"
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

resolve_model_version() {
  if [[ -n "$ASR_MODEL_VERSION" ]]; then
    return
  fi
  local alias_json
  alias_json="$("${DBX[@]}" api get "/api/2.1/unity-catalog/models/${ASR_REGISTERED_MODEL_FQDN}/aliases/${ASR_MODEL_ALIAS}")"
  ASR_MODEL_VERSION="$(python - "$alias_json" <<'PY'
import json
import sys
payload = json.loads(sys.argv[1])
version = payload.get("version")
if not version:
    raise SystemExit("Alias response did not include a version.")
print(version)
PY
)"
}

resolve_smoke_manifest() {
  if volume_path_exists "$ASR_HOLDOUT_MANIFEST"; then
    ASR_SMOKE_MANIFEST="$ASR_HOLDOUT_MANIFEST"
  elif volume_path_exists "$ASR_TRAINING_MANIFEST"; then
    ASR_SMOKE_MANIFEST="$ASR_TRAINING_MANIFEST"
  else
    err "No manifest found for smoke test."
    exit 2
  fi
}

preflight() {
  setup_env
  setup_databricks_cli
  resolve_paths
  resolve_model_version
  case "$ASR_SERVING_WORKLOAD_SIZE" in
    Small|Medium|Large) ;;
    *)
      err "ASR_SERVING_WORKLOAD_SIZE must be Small, Medium, or Large."
      exit 2
      ;;
  esac
  case "$ASR_SERVING_WORKLOAD_TYPE" in
    CPU|CPU_MEDIUM|CPU_LARGE|GPU_SMALL|GPU_MEDIUM|GPU_LARGE|MULTIGPU_MEDIUM|GPU_MEDIUM_8) ;;
    *)
      err "ASR_SERVING_WORKLOAD_TYPE must be CPU, CPU_MEDIUM, CPU_LARGE, GPU_SMALL, GPU_MEDIUM, GPU_LARGE, MULTIGPU_MEDIUM, or GPU_MEDIUM_8."
      exit 2
      ;;
  esac
  case "$ASR_SERVING_SCALE_TO_ZERO" in
    true|false) ;;
    *)
      err "ASR_SERVING_SCALE_TO_ZERO must be true or false."
      exit 2
      ;;
  esac

  cat <<EOF
ASR serving preflight passed.

Endpoint:
  $ASR_SERVING_ENDPOINT_NAME

Model:
  $ASR_REGISTERED_MODEL_FQDN version $ASR_MODEL_VERSION (alias: $ASR_MODEL_ALIAS)

Compute:
  Databricks Model Serving serverless
  workload_type: $ASR_SERVING_WORKLOAD_TYPE
  workload_size: $ASR_SERVING_WORKLOAD_SIZE
  scale_to_zero_enabled: $ASR_SERVING_SCALE_TO_ZERO
  route_optimized_on_create: $ASR_SERVING_ROUTE_OPTIMIZED

Latency posture:
  Warm endpoint for speaking-input app tests; no cold-start scale-to-zero.

EOF
}

endpoint_exists() {
  "${DBX[@]}" serving-endpoints get "$ASR_SERVING_ENDPOINT_NAME" --output json >/dev/null 2>&1
}

endpoint_route_optimized() {
  "${DBX[@]}" serving-endpoints get "$ASR_SERVING_ENDPOINT_NAME" --output json \
    | python -c 'import json, sys; payload = json.load(sys.stdin); print("true" if payload.get("route_optimized") else "false")'
}

write_serving_config() {
  local output="$1"
  python - "$output" <<PY
import json
import sys
from pathlib import Path

scale_to_zero = "${ASR_SERVING_SCALE_TO_ZERO}".lower() == "true"
served_entity = {
    "name": "${ASR_SERVING_SERVED_ENTITY_NAME}",
    "entity_name": "${ASR_REGISTERED_MODEL_FQDN}",
    "entity_version": "${ASR_MODEL_VERSION}",
    "workload_type": "${ASR_SERVING_WORKLOAD_TYPE}",
    "workload_size": "${ASR_SERVING_WORKLOAD_SIZE}",
    "scale_to_zero_enabled": scale_to_zero,
}
payload = {
    "served_entities": [served_entity],
    "traffic_config": {
        "routes": [
            {
                "served_model_name": "${ASR_SERVING_SERVED_ENTITY_NAME}",
                "traffic_percentage": 100,
            }
        ]
    },
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY
}

deploy() {
  preflight

  local config_json="$ROOT/.run/asr_model_training/serving_endpoint_${ASR_SERVING_ENDPOINT_NAME}.json"
  local create_json="$ROOT/.run/asr_model_training/serving_endpoint_create_${ASR_SERVING_ENDPOINT_NAME}.json"
  mkdir -p "$(dirname "$config_json")"
  write_serving_config "$config_json"
python - "$config_json" "$create_json" "$ASR_SERVING_ENDPOINT_NAME" <<'PY'
import json
import sys
from pathlib import Path
config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
Path(sys.argv[2]).write_text(
    json.dumps({"name": sys.argv[3], "config": config}, indent=2),
    encoding="utf-8",
)
PY

  if endpoint_exists; then
    local current_route_optimized
    current_route_optimized="$(endpoint_route_optimized)"
    if [[ "$current_route_optimized" != "$ASR_SERVING_ROUTE_OPTIMIZED" ]]; then
      log "recreating endpoint to change route_optimized from $current_route_optimized to $ASR_SERVING_ROUTE_OPTIMIZED"
      "${DBX[@]}" serving-endpoints delete "$ASR_SERVING_ENDPOINT_NAME"
    fi
  fi

  if endpoint_exists; then
    log "updating existing serving endpoint: $ASR_SERVING_ENDPOINT_NAME"
    "${DBX[@]}" serving-endpoints update-config "$ASR_SERVING_ENDPOINT_NAME" \
      --json @"$config_json" \
      --timeout "$ASR_SERVING_TIMEOUT" \
      --output json
  else
    log "creating serving endpoint: $ASR_SERVING_ENDPOINT_NAME"
    if [[ "$ASR_SERVING_ROUTE_OPTIMIZED" == "true" ]]; then
      "${DBX[@]}" serving-endpoints create \
        --json @"$create_json" \
        --route-optimized \
        --timeout "$ASR_SERVING_TIMEOUT" \
        --output json
    else
      "${DBX[@]}" serving-endpoints create \
        --json @"$create_json" \
        --timeout "$ASR_SERVING_TIMEOUT" \
        --output json
    fi
  fi
}

status() {
  setup_env
  setup_databricks_cli
  "${DBX[@]}" serving-endpoints get "$ASR_SERVING_ENDPOINT_NAME" --output json
}

smoke_test() {
  preflight
  resolve_smoke_manifest

  local local_dir="$ROOT/.run/asr_model_training/serving_smoke"
  local manifest_local="$local_dir/manifest.jsonl"
  local audio_local="$local_dir/audio"
  local request_json="$local_dir/request.json"
  mkdir -p "$audio_local"

  "${DBX[@]}" fs cp "dbfs:$ASR_SMOKE_MANIFEST" "$manifest_local" --overwrite
  local audio_path
  audio_path="$(python - "$manifest_local" <<'PY'
import json
import sys
from pathlib import Path
for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    if line.strip() and not line.lstrip().startswith("#"):
        print(json.loads(line)["audio_path"])
        break
PY
)"
  "${DBX[@]}" fs cp "dbfs:$audio_path" "$audio_local/clip$(python - "$audio_path" <<'PY'
import sys
from pathlib import Path
print(Path(sys.argv[1]).suffix or ".wav")
PY
)" --overwrite

  python - "$manifest_local" "$audio_local" "$request_json" <<'PY'
import base64
import json
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
audio_dir = Path(sys.argv[2])
request = Path(sys.argv[3])
row = None
for line in manifest.read_text(encoding="utf-8").splitlines():
    if line.strip() and not line.lstrip().startswith("#"):
        row = json.loads(line)
        break
if row is None:
    raise SystemExit("No smoke-test row found.")
audio = next(audio_dir.iterdir())
mime = "audio/wav" if audio.suffix.lower() == ".wav" else "application/octet-stream"
payload = {
    "dataframe_records": [
        {
            "audio_b64": base64.b64encode(audio.read_bytes()).decode("ascii"),
            "mime_type": mime,
            "speaker": 1,
        }
    ]
}
request.write_text(json.dumps(payload), encoding="utf-8")
print(json.dumps({"clip_id": row.get("clip_id"), "audio_path": row.get("audio_path")}, indent=2))
PY

  "${DBX[@]}" serving-endpoints query "$ASR_SERVING_ENDPOINT_NAME" \
    --json @"$request_json" \
    --output json
}

case "$COMMAND" in
  preflight)
    preflight
    ;;
  deploy)
    deploy
    ;;
  status)
    status
    ;;
  smoke-test)
    smoke_test
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
