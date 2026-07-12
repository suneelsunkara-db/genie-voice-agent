#!/usr/bin/env bash
# Holistic ASR eval on serverless (step 5 — after datasets, quality, register, serve).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_common.sh"
cd "$ML_ASR_ROOT"

MODE="${1:-next}"
if [[ $# -gt 0 ]]; then shift; fi

usage() {
  cat <<EOF
Step 5 — score models on holdout clips (serverless).

  ./scripts/ml_asr/05_eval.sh              # next smoke step
  ./scripts/ml_asr/05_eval.sh status       # pipeline state + gates
  ./scripts/ml_asr/05_eval.sh plan         # pending steps
  ./scripts/ml_asr/05_eval.sh sync-index   # pull index for benchmark UI

Routes scored (from eval_matrix):
  - deepgram_nova3          commercial API (no UC register/serve)
  - databricks_*            Model Serving endpoints (steps 3–4)
EOF
}

ml_asr_require_cli

case "$MODE" in
  next|run|"")
    ml_asr_log "eval iterate next (serverless, smoke limits from config)"
    "$ML_ASR_CLI" --smoke iterate next
    ;;
  status)
    "$ML_ASR_CLI" --smoke status
    ;;
  plan)
    "$ML_ASR_CLI" --smoke iterate plan
    ;;
  sync-index)
    exec "$SCRIPT_DIR/sync_benchmark_index.sh"
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
