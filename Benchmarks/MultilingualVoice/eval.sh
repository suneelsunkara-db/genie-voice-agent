#!/usr/bin/env bash
#
# Submit (default) or run locally the multilingual voice benchmark.
#
# FLEURS scores speech-to-text accuracy (WER/CER) on the STT transcript (STT-only
# route — the LLM is never in the loop). It ALSO runs a TTS round-trip: the
# reference text is synthesized through the text-to-speech route, which records
# the TTS engine's time-to-first-audio as TTFT (a model-level "how fast the voice
# starts speaking" number, not the agent's tool-assisted latency), then re-STTs the
# audio for intelligibility. Pass --no-roundtrip to skip it (WER/CER only).
#
# Default: serverless Databricks job -> realtime app WebSocket APIs -> Delta,
# then a vendor STT comparison job (Deepgram) that reuses the SAME staged FLEURS
# audio. The vendor job runs AFTER the main job so the staged data exists, so
# `eval.sh` waits on the main job when vendors are enabled.
# Results: Delta tables {catalog}.{schema}.benchmark_runs + benchmark_samples
#          (read by GET /realtime/v1/benchmarks in the Databricks app; summary.json
#           is written only in --fixture/--local mode and is NOT read by the API)
# Logs:    volume.multilingual_voice_benchmark_path/logs/run_<timestamp>.log
#
# Usage:
#   ./eval.sh                                    # main + vendor STT jobs in sequence
#   ./eval.sh --dataset fleurs --languages en,ja --limit 5
#   ./eval.sh --wait                             # also block on the vendor job
#   ./eval.sh --no-vendors                       # skip the Deepgram STT comparison
#   ./eval.sh --vendors deepgram                 # choose which vendor tracks to run
#   ./eval.sh --local --languages en --limit 2   # run on this machine (dev only)
#   ./eval.sh --fixture                          # offline fixture scoring
#
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
cd "$HERE"

DATABRICKS_PROFILE="${DATABRICKS_PROFILE:-fe-vm-vdm-classic-rcn6ip}"
LANGUAGES="${MLV_LANGUAGES:-}"
# FLEURS-STT is the documented default sweep (the UI + references are FLEURS-only;
# belebele/ccfqa are deprecated). Pass --dataset all to also run the deprecated sets.
DATASET="${MLV_DATASET:-fleurs}"
LIMIT="${MLV_LIMIT:-20}"
VENDORS="${MLV_VENDORS:-deepgram}"
VENDORS_ENABLED=1

FIXTURE=0
LOCAL=0
WAIT=0
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --fixture) FIXTURE=1; shift ;;
    --local) LOCAL=1; shift ;;
    --wait) WAIT=1; shift ;;
    --resume) EXTRA_ARGS+=("--resume"); shift ;;
    --languages) LANGUAGES="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --vendors) VENDORS="$2"; shift 2 ;;
    --no-vendors) VENDORS_ENABLED=0; shift ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

VENV="$REPO_ROOT/.venv"
if [[ ! -d "$VENV" ]]; then
  echo "ERROR: repo virtualenv not found at $VENV" >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

CONFIG_YAML="$REPO_ROOT/config/config.local.yaml"
if [[ -f "$CONFIG_YAML" ]]; then
  read -r CFG_PROFILE CFG_HF_TOKEN < <(python - <<'PY' "$CONFIG_YAML"
import sys
from pathlib import Path
import yaml
cfg = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8")) or {}
profile = str((cfg.get("databricks") or {}).get("profile") or "").strip()
token = str((cfg.get("secrets") or {}).get("hf_token") or "").strip()
print(profile, token)
PY
)
  [[ -n "${CFG_PROFILE:-}" ]] && DATABRICKS_PROFILE="$CFG_PROFILE"
  if [[ -n "${CFG_HF_TOKEN:-}" && "${CFG_HF_TOKEN}" == hf_* ]]; then
    export HF_TOKEN="$CFG_HF_TOKEN"
    export HUGGING_FACE_HUB_TOKEN="$CFG_HF_TOKEN"
  fi
fi

if [[ "$FIXTURE" -eq 1 ]]; then
  echo "[run] fixture mode (offline scoring -> Benchmarks/MultilingualVoice/results/)"
  python run_benchmark.py --fixture --out-dir "$HERE/results"
  exit 0
fi

if [[ "$LOCAL" -eq 1 ]]; then
  echo "[run] local mode (dev only — subject to workspace IP ACL on serving endpoints)"
  export DATABRICKS_PROFILE="$DATABRICKS_PROFILE"
  python run_benchmark.py \
    --transport ws \
    --dataset "$DATASET" \
    --languages "$LANGUAGES" \
    --limit "$LIMIT" \
    --databricks-profile "$DATABRICKS_PROFILE" \
    ${EXTRA_ARGS+"${EXTRA_ARGS[@]}"}
  exit 0
fi

# The vendor STT comparison (Deepgram) reuses the FLEURS audio the main job
# stages on the Volume, so it only applies to the fleurs/all sweep.
if [[ "$VENDORS_ENABLED" -eq 1 && "$DATASET" != "fleurs" && "$DATASET" != "all" ]]; then
  echo "[submit] vendors skipped (only run on the fleurs/all sweep; dataset=$DATASET)"
  VENDORS_ENABLED=0
fi

echo "[submit] multilingual voice benchmark Databricks job"
SUBMIT_ARGS=(--dataset "$DATASET" --languages "$LANGUAGES" --limit "$LIMIT")
# Wait on the main job when vendors follow it, so its staged FLEURS data exists
# before the vendor job starts (vendors reuse that staged audio).
if [[ "$VENDORS_ENABLED" -eq 1 || "$WAIT" -eq 1 ]]; then
  SUBMIT_ARGS+=(--wait)
fi
python "$REPO_ROOT/scripts/ml_asr/submit_multilingual_voice_benchmark_job.py" "${SUBMIT_ARGS[@]}" ${EXTRA_ARGS+"${EXTRA_ARGS[@]}"}

if [[ "$VENDORS_ENABLED" -eq 1 ]]; then
  echo "[submit] vendor FLEURS STT comparison job (Deepgram): $VENDORS"
  VENDOR_ARGS=(--vendors "$VENDORS" --languages "$LANGUAGES" --limit "$LIMIT")
  [[ "$WAIT" -eq 1 ]] && VENDOR_ARGS+=(--wait)
  python "$REPO_ROOT/scripts/ml_asr/submit_vendor_fleurs_benchmark_job.py" "${VENDOR_ARGS[@]}"
fi
