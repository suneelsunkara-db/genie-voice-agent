#!/usr/bin/env bash
# Pull ml_asr results index from UC Volume for the local ASR benchmark UI.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_common.sh"
cd "$ML_ASR_ROOT"

REMOTE_INDEX="${ML_ASR_REMOTE_INDEX:-/Volumes/partner_demo_catalog/genie_voice_contact_center/raw_streaming_data/asr_model_training/evaluations/ml_asr_eval/results/index.json}"
LOCAL_INDEX="$ML_ASR_ROOT/.run/ml_asr_eval/index.json"

mkdir -p "$(dirname "$LOCAL_INDEX")"
ml_asr_log "sync $REMOTE_INDEX -> $LOCAL_INDEX"
databricks fs cp "dbfs:$REMOTE_INDEX" "$LOCAL_INDEX"
ml_asr_log "done — open http://localhost:5173/#/asr-benchmark"
