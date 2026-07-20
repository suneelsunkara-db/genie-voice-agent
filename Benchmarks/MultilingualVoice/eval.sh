#!/usr/bin/env bash
#
# Submit (default) or run locally the multilingual voice benchmark.
#
# Default: serverless Databricks job -> realtime app WebSocket APIs -> UC Volume.
# Results: volume.multilingual_voice_benchmark_path/summary.json
# Logs:    volume.multilingual_voice_benchmark_path/logs/run_<timestamp>.log
#          (read by GET /realtime/v1/benchmarks in the Databricks app)
#
# Usage:
#   ./eval.sh                                    # submit full job
#   ./eval.sh --dataset fleurs --languages en,ja --limit 5
#   ./eval.sh --wait                             # submit and block until done
#   ./eval.sh --local --languages en --limit 2   # run on this machine (dev only)
#   ./eval.sh --fixture                          # offline fixture scoring
#
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
cd "$HERE"

DATABRICKS_PROFILE="${DATABRICKS_PROFILE:-fe-vm-vdm-classic-rcn6ip}"
LANGUAGES="${MLV_LANGUAGES:-}"
DATASET="${MLV_DATASET:-all}"
LIMIT="${MLV_LIMIT:-20}"

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
    --tts-roundtrip \
    ${EXTRA_ARGS+"${EXTRA_ARGS[@]}"}
  exit 0
fi

echo "[submit] multilingual voice benchmark Databricks job"
SUBMIT_ARGS=(--dataset "$DATASET" --languages "$LANGUAGES" --limit "$LIMIT")
[[ "$WAIT" -eq 1 ]] && SUBMIT_ARGS+=(--wait)
python "$REPO_ROOT/scripts/ml_asr/submit_multilingual_voice_benchmark_job.py" "${SUBMIT_ARGS[@]}" ${EXTRA_ARGS+"${EXTRA_ARGS[@]}"}
