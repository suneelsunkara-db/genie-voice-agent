#!/usr/bin/env bash
# Register Databricks UC models for ml_asr eval (step 3 — before serve/eval).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_common.sh"
cd "$ML_ASR_ROOT"

MODE="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi

usage() {
  cat <<EOF
Step 3 — register Databricks UC models (not Deepgram).

  ./scripts/ml_asr/03_register.sh list
  ./scripts/ml_asr/03_register.sh one databricks_th_pathumma_whisper_large_v3
  ./scripts/ml_asr/03_register.sh all

Delegates to `genie_voice.ml_asr.serving` (OSS via scripts/ml_asr; EN finetuned still uses scripts/asr until migrated).
Config: config/ml_asr_eval.yaml -> model_serving.models (databricks_* only)
EOF
}

case "$MODE" in
  list)
    ml_asr_serving register list "$@"
    ;;
  one)
    [[ $# -ge 1 ]] || { ml_asr_err "Usage: 03_register.sh one <model_id>"; exit 2; }
    ml_asr_log "register $1"
    ml_asr_serving register one "$@"
    ;;
  all)
    ml_asr_log "register all databricks models in model_serving"
    ml_asr_serving register all "$@"
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
