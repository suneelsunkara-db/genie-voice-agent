#!/usr/bin/env bash
# =============================================================================
# 02_asr_baseline_runs.sh
#
# ASR candidate evaluation only. This script runs baselines against the locked
# ASR model-training manifest. It does not fine-tune or register a model.
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENGINE="$ROOT/scripts/asr/01_asr_model_training.sh"

COMMAND="${1:-next}"
if [[ $# -gt 0 ]]; then
  shift
fi

usage() {
  cat <<'EOF'
ASR baseline/evaluation runs

Purpose:
  Compare STT candidates against the locked ASR model-training manifest.
  This does not train a model.

Commands:
  next             Show the next baseline command to run.
  gpu-status       Show the dedicated ASR GPU cluster status.
  gpu-start        Create/start the dedicated ASR GPU cluster.
  whisper-smoke    Run 20-clip Databricks Whisper smoke baseline.
  whisper-full     Run full-manifest Databricks Whisper baseline.
  deepgram-full    Run full-manifest Deepgram baseline using local cached audio.
  rescore          Rescore an existing baseline JSONL. Pass --input and --output.
  fair-compare     Rescore Whisper, run Deepgram 403, and write comparison JSON.
  compare-existing Write comparison JSON from existing rescored Whisper and Deepgram outputs.
  summarize        Summarize a baseline JSONL file. Pass --output <path>.
  gpu-stop         Stop GPU cluster after all baseline/training work is done.
  help             Show this help.

Recommended order:
  scripts/asr/02_asr_baseline_runs.sh gpu-start
  scripts/asr/02_asr_baseline_runs.sh whisper-smoke
  scripts/asr/02_asr_baseline_runs.sh whisper-full

EOF
}

next_step() {
  cat <<'EOF'
Next ASR baseline step

Run the full Whisper baseline on the persistent GPU cluster:

  scripts/asr/02_asr_baseline_runs.sh whisper-full

This evaluates Whisper against the full locked manifest. It does not train a model.
EOF
}

write_comparison() {
  local whisper_input="$ROOT/.run/asr_model_training/baselines/whisper_small_databricks_full_baseline.jsonl"
  local whisper_rescored="$ROOT/.run/asr_model_training/baselines/whisper_small_databricks_full_rescored.jsonl"
  local deepgram_output="$ROOT/.run/asr_model_training/baselines/deepgram_nova3_403_baseline.jsonl"
  local comparison_output="$ROOT/.run/asr_model_training/evaluations/asr_baseline_fair_comparison.json"

  if [[ "${1:-}" == "--rescore-whisper" ]]; then
    "$ENGINE" rescore \
      --input "$whisper_input" \
      --output "$whisper_rescored"
  fi

  mkdir -p "$(dirname "$comparison_output")"
  python3 - "$whisper_rescored" "$deepgram_output" "$comparison_output" <<'PY'
import json
import sys
from collections import defaultdict
from pathlib import Path

whisper_path = Path(sys.argv[1])
deepgram_path = Path(sys.argv[2])
output_path = Path(sys.argv[3])

def summarize(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    def avg(values):
        values = [value for value in values if value is not None]
        return None if not values else sum(values) / len(values)
    entity_totals = defaultdict(lambda: {"expected": 0, "matched": 0})
    for row in rows:
        for group, score in (row.get("score", {}).get("entity_scores") or {}).items():
            entity_totals[group]["expected"] += score.get("expected") or 0
            entity_totals[group]["matched"] += score.get("matched") or 0
    return {
        "results": str(path),
        "clips": len(rows),
        "provider": sorted({row.get("provider") for row in rows if row.get("provider")}),
        "models": sorted({row.get("model") for row in rows if row.get("model")}),
        "avg_wer": avg(row.get("score", {}).get("wer") for row in rows),
        "avg_cer": avg(row.get("score", {}).get("cer") for row in rows),
        "avg_entity_accuracy": avg(row.get("score", {}).get("entity_accuracy") for row in rows),
        "avg_latency_ms": avg(row.get("latency_ms") for row in rows),
        "entity_groups": {
            group: {
                **counts,
                "accuracy": None if counts["expected"] == 0 else counts["matched"] / counts["expected"],
            }
            for group, counts in sorted(entity_totals.items())
        },
    }

comparison = {
    "whisper": summarize(whisper_path),
    "deepgram": summarize(deepgram_path),
}
output_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
print(json.dumps(comparison, indent=2))
PY
}

fair_compare() {
  local whisper_input="$ROOT/.run/asr_model_training/baselines/whisper_small_databricks_full_baseline.jsonl"
  local whisper_rescored="$ROOT/.run/asr_model_training/baselines/whisper_small_databricks_full_rescored.jsonl"
  local deepgram_output="$ROOT/.run/asr_model_training/baselines/deepgram_nova3_403_baseline.jsonl"

  "$ENGINE" rescore \
    --input "$whisper_input" \
    --output "$whisper_rescored"

  "$ENGINE" deepgram \
    --audio-source local-cache \
    --sync-results false \
    --output "$deepgram_output"

  write_comparison
}

case "$COMMAND" in
  next)
    next_step
    ;;
  gpu-status)
    "$ENGINE" gpu-status "$@"
    ;;
  gpu-start)
    "$ENGINE" gpu-start "$@"
    ;;
  whisper-smoke)
    "$ENGINE" whisper-db --limit 20 "$@"
    ;;
  whisper-full)
    "$ENGINE" whisper-db --limit all "$@"
    ;;
  deepgram-full)
    "$ENGINE" deepgram \
      --audio-source local-cache \
      --sync-results false \
      --output "$ROOT/.run/asr_model_training/baselines/deepgram_nova3_403_baseline.jsonl" \
      "$@"
    ;;
  rescore)
    "$ENGINE" rescore "$@"
    ;;
  fair-compare)
    fair_compare "$@"
    ;;
  compare-existing)
    write_comparison "$@"
    ;;
  summarize)
    "$ENGINE" summarize "$@"
    ;;
  gpu-stop)
    "$ENGINE" gpu-stop "$@"
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    printf "Unknown command: %s\n\n" "$COMMAND" >&2
    usage >&2
    exit 2
    ;;
esac
