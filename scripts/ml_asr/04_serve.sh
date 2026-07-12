#!/usr/bin/env bash
# Deploy Model Serving endpoints for ml_asr eval (step 4 — after register).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_common.sh"
cd "$ML_ASR_ROOT"

MODE="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi

usage() {
  cat <<EOF
Step 4 — deploy Databricks Model Serving endpoints (not Deepgram).

  ./scripts/ml_asr/04_serve.sh list
  ./scripts/ml_asr/04_serve.sh preflight databricks_en_finetuned_whisper_lora
  ./scripts/ml_asr/04_serve.sh deploy databricks_en_finetuned_whisper_lora
  ./scripts/ml_asr/04_serve.sh deploy-all
  ./scripts/ml_asr/04_serve.sh status databricks_en_finetuned_whisper_lora
  ./scripts/ml_asr/04_serve.sh smoke databricks_en_finetuned_whisper_lora

Config: config/ml_asr_eval.yaml -> model_serving.models
Runs from your laptop via Databricks Python SDK.
EOF
}

case "$MODE" in
  list)
    ml_asr_serving serve list "$@"
    ;;
  preflight)
    [[ $# -ge 1 ]] || { ml_asr_err "Usage: 04_serve.sh preflight <model_id>"; exit 2; }
    ml_asr_serving serve preflight "$@"
    ;;
  deploy)
    [[ $# -ge 1 ]] || { ml_asr_err "Usage: 04_serve.sh deploy <model_id>"; exit 2; }
    ml_asr_log "deploy $1"
    ml_asr_serving serve deploy "$@"
    ;;
  deploy-all)
    ml_asr_log "deploy all databricks endpoints"
    ml_asr_serving serve deploy-all "$@"
    ;;
  status)
    [[ $# -ge 1 ]] || { ml_asr_err "Usage: 04_serve.sh status <model_id>"; exit 2; }
    ml_asr_serving serve status "$@"
    ;;
  smoke)
    [[ $# -ge 1 ]] || { ml_asr_err "Usage: 04_serve.sh smoke <model_id>"; exit 2; }
    ml_asr_serving serve smoke "$@"
    ;;
  help|-h|--help|"")
    usage
    ;;
  *)
    ml_asr_err "Unknown mode: $MODE"
    usage >&2
    exit 2
    ;;
esac
