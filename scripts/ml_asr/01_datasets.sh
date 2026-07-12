#!/usr/bin/env bash
# Download FLEURS holdouts to UC Volume (serverless by default).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_common.sh"
cd "$ML_ASR_ROOT"

MODE="${1:-run}"
if [[ $# -gt 0 ]]; then shift; fi

usage() {
  cat <<EOF
Build smoke datasets on Databricks serverless (defaults from config/ml_asr_eval.yaml).

  ./scripts/ml_asr/01_datasets.sh           # serverless (recommended)
  ./scripts/ml_asr/01_datasets.sh local       # dev only: download on laptop

Optional: export HF_TOKEN=hf_... for faster Hub downloads on serverless.
EOF
}

ml_asr_require_cli

LOCAL_ARGS=()
[[ "$MODE" == "local" ]] && LOCAL_ARGS=(--local)

run_prepare() {
  local dataset="$1"
  local limit="$2"
  ml_asr_log "prepare $dataset (limit=$limit, mode=${MODE:-serverless})"
  "$ML_ASR_CLI" "${LOCAL_ARGS[@]}" step prepare --dataset "$dataset" --limit "$limit"
}

case "$MODE" in
  run|"")
    biz_limit="$(ml_asr_smoke_limit business)"
    aco_limit="$(ml_asr_smoke_limit acoustic)"
    run_prepare fleurs_business_v1 "$biz_limit"
    run_prepare fleurs_acoustic_v1 "$aco_limit"
    ml_asr_log "datasets ready on Volume"
    ;;
  local)
    biz_limit="$(ml_asr_smoke_limit business)"
    aco_limit="$(ml_asr_smoke_limit acoustic)"
    run_prepare fleurs_business_v1 "$biz_limit"
    run_prepare fleurs_acoustic_v1 "$aco_limit"
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    ml_asr_err "Unknown mode: $MODE"
    usage >&2
    exit 2
    ;;
esac
