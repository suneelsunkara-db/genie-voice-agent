#!/usr/bin/env bash
# Shared helpers for scripts/ml_asr/*
set -euo pipefail

ML_ASR_ROOT="$(cd "$(dirname "${BASH_SOURCE[1]:-${BASH_SOURCE[0]:-$0}}")/../.." && pwd)"
ML_ASR_CLI="$ML_ASR_ROOT/scripts/ml_asr/_cli.sh"
ML_ASR_REPORT="$ML_ASR_ROOT/.run/ml_asr_eval/dataset_quality_eval.json"

ml_asr_log() { printf "\033[36m[ml-asr]\033[0m %s\n" "$*"; }
ml_asr_warn() { printf "\033[33m[ml-asr]\033[0m %s\n" "$*"; }
ml_asr_err() { printf "\033[31m[ml-asr]\033[0m %s\n" "$*" >&2; }

ml_asr_require_cli() {
  if [[ ! -x "$ML_ASR_CLI" ]]; then
    ml_asr_err "Missing CLI: $ML_ASR_CLI"
    exit 1
  fi
}

ml_asr_ensure_venv() {
  ml_asr_require_cli
  if [[ ! -x "$ML_ASR_ROOT/.venv/bin/python" ]]; then
    "$ML_ASR_CLI" status >/dev/null 2>&1 || true
  fi
}

ml_asr_serving() {
  ml_asr_ensure_venv
  exec "$ML_ASR_ROOT/.venv/bin/python" -m genie_voice.ml_asr.serving.cli "$@"
}

ml_asr_smoke_limit() {
  local tier="$1"
  local key="business_clip_limit"
  [[ "$tier" == "acoustic" ]] && key="acoustic_clip_limit"
  local value
  value="$(grep -E "[[:space:]]${key}:[[:space:]]*[0-9]+" "$ML_ASR_ROOT/config/ml_asr_eval.yaml" | head -1 | sed -E 's/.*: *//')"
  echo "${value:-25}"
}

ml_asr_print_quality_summary() {
  local report_path="${1:-$ML_ASR_REPORT}"
  if [[ ! -f "$report_path" ]]; then
    ml_asr_err "Report not found: $report_path"
    exit 1
  fi
  python3 - "$report_path" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
gates = report.get("gates", {})

print()
print("=" * 72)
print("DATASET QUALITY")
print("=" * 72)
print(f"dataset_quality_ready:   {gates.get('dataset_quality_ready')}")
print(f"recommended_next_step:   {gates.get('recommended_next_step')}")
print()

for tier_name in ("business", "acoustic"):
    rows = report.get("tiers", {}).get(tier_name, [])
    if not rows:
        continue
    print("-" * 72)
    print(f"{tier_name.upper()}")
    print("-" * 72)
    if tier_name == "business":
        print(f"{'lang':<8} {'clips':>5} {'scen%':>6} {'susp%':>6} {'issues':>6}")
        for row in rows:
            print(
                f"{row['language']:<8} {row['clip_count']:>5} "
                f"{row.get('scenario_consistency_rate', 0):>6.3f} "
                f"{row.get('suspicious_label_rate', 0):>6.3f} "
                f"{row.get('clips_with_issues', 0):>6}"
            )
    else:
        print(f"{'lang':<8} {'clips':>5} {'dup%':>6} {'audio_fail':>10}")
        for row in rows:
            audio = row.get("audio_sample") or {}
            print(
                f"{row['language']:<8} {row['clip_count']:>5} "
                f"{row.get('duplicate_transcript_rate', 0):>6.3f} "
                f"{audio.get('failures', 0):>10}"
            )
    print()

print("=" * 72)
print(f"Full report: {sys.argv[1]}")
print("=" * 72)
PY
}
