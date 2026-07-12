#!/usr/bin/env bash
# Semantic dataset quality check (serverless by default).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_common.sh"
cd "$ML_ASR_ROOT"

MODE="${1:-run}"
if [[ $# -gt 0 ]]; then shift; fi

usage() {
  cat <<EOF
Check label/audio quality on Volume manifests (not just clip counts).

  ./scripts/ml_asr/02_quality.sh            # serverless (recommended)
  ./scripts/ml_asr/02_quality.sh show       # re-print last report
  ./scripts/ml_asr/02_quality.sh local      # dev only
EOF
}

ml_asr_require_cli

case "$MODE" in
  run|""|serverless)
    ml_asr_log "dataset quality eval (serverless)"
    "$ML_ASR_CLI" dataset-eval "$@"
    ml_asr_print_quality_summary "$ML_ASR_REPORT"
    ;;
  local)
    ml_asr_log "dataset quality eval (local)"
    "$ML_ASR_CLI" --local dataset-eval "$@"
    ml_asr_print_quality_summary "$ML_ASR_REPORT"
    ;;
  show)
    ml_asr_print_quality_summary "${1:-$ML_ASR_REPORT}"
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
