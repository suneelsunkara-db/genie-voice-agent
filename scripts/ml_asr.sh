#!/usr/bin/env bash
# ML ASR pipeline — serverless by default.
#
#   1 datasets   build holdouts
#   2 quality    dataset quality gates
#   3 register   UC models (Databricks only)
#   4 serve      Model Serving endpoints (Databricks only)
#   5 eval       score all routes (Deepgram API + Databricks endpoints)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cmd="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi

case "$cmd" in
  datasets|1) exec "$ROOT/scripts/ml_asr/01_datasets.sh" run "$@" ;;
  quality|2) exec "$ROOT/scripts/ml_asr/02_quality.sh" run "$@" ;;
  register|3) exec "$ROOT/scripts/ml_asr/03_register.sh" "${1:-list}" "${@:2}" ;;
  serve|4) exec "$ROOT/scripts/ml_asr/04_serve.sh" "${1:-help}" "${@:2}" ;;
  eval|5) exec "$ROOT/scripts/ml_asr/05_eval.sh" "${1:-next}" "${@:2}" ;;
  status) exec "$ROOT/scripts/ml_asr/05_eval.sh" status "$@" ;;
  models)
    printf "\033[33m[ml-asr]\033[0m 'models' renamed to 'eval' (step 5)\n" >&2
    exec "$ROOT/scripts/ml_asr/05_eval.sh" "${1:-next}" "${@:2}"
    ;;
  help|-h|--help|"")
    cat <<EOF
ML ASR pipeline (config/ml_asr_eval.yaml). Serverless unless noted.

  1  ./scripts/ml_asr.sh datasets    build FLEURS holdouts on UC Volume
  2  ./scripts/ml_asr.sh quality     semantic dataset quality check
  3  ./scripts/ml_asr.sh register    UC registration (Databricks models only)
  4  ./scripts/ml_asr.sh serve       deploy Model Serving endpoints
  5  ./scripts/ml_asr.sh eval        score Deepgram + Databricks routes
     ./scripts/ml_asr.sh status      eval pipeline state

Deepgram Nova-3 is a commercial API baseline (API key) — not UC/OSS.
Steps 3–4 apply only to databricks_* models in model_serving.

See README.md (repo root) -> "ML ASR pipeline & model serving"
EOF
    ;;
  *)
    printf "Unknown command: %s\n" "$cmd" >&2
    exec "$ROOT/scripts/ml_asr.sh" help
    ;;
esac
