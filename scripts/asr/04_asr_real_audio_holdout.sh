#!/usr/bin/env bash
# =============================================================================
# 04_asr_real_audio_holdout.sh
#
# Real/realistic held-out ASR evaluation gate. This script is deliberately
# separate from training so holdout clips cannot be mixed back into LoRA data.
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENGINE="$ROOT/scripts/asr/01_asr_model_training.sh"
FINETUNE="$ROOT/scripts/asr/03_asr_model_finetuning.sh"
LOCAL_HOLDOUT_DIR="$ROOT/.run/asr_model_training/holdout"
HOLDOUT_MANIFEST="$LOCAL_HOLDOUT_DIR/manifests/asr_real_audio_holdout_v1.jsonl"
HOLDOUT_RESULTS_DIR="$LOCAL_HOLDOUT_DIR/results"
HOLDOUT_EVAL_DIR="$LOCAL_HOLDOUT_DIR/evaluations"
HOLDOUT_MIN_CLIPS="${ASR_HOLDOUT_MIN_CLIPS:-50}"
ASR_TRAINING_ROOT="${ASR_TRAINING_ROOT:-}"
ASR_HOLDOUT_ROOT="${ASR_HOLDOUT_ROOT:-}"
ASR_HOLDOUT_MANIFEST="${ASR_HOLDOUT_MANIFEST:-}"
ASR_MODEL_ARTIFACTS="${ASR_MODEL_ARTIFACTS:-}"
ASR_TRAINING_EVALUATIONS="${ASR_TRAINING_EVALUATIONS:-}"
ASR_BASE_MODEL="${ASR_LORA_BASE_MODEL:-openai/whisper-small.en}"
ASR_GPU_CLUSTER_NAME="${ASR_GPU_CLUSTER_NAME:-genie-asr-gpu-training}"

COMMAND="${1:-next}"
if [[ $# -gt 0 ]]; then
  shift
fi

log() { printf "\033[35m[asr-holdout]\033[0m %s\n" "$*"; }
err() { printf "\033[31m[asr-holdout]\033[0m %s\n" "$*" >&2; }

usage() {
  cat <<'EOF'
ASR real-audio holdout gate

Purpose:
  Evaluate Deepgram, Base Whisper, and LoRA Whisper on held-out real/realistic
  audio that was not used for training.

Commands:
  next       Show the next required holdout step.
  prepare    Create local/Volume holdout folders and a manifest template.
  validate   Validate the holdout manifest and enforce the minimum clip count.
  evaluate   Run the full holdout gate and write a decision report.
  help       Show this help.

Important:
  - Holdout clips must not be used for training.
  - Transcripts must be human-approved.
  - This script does not auto-stop the GPU cluster.

EOF
}

setup_databricks_cli() {
  resolve_training_volume_paths
  if ! command -v databricks >/dev/null 2>&1; then
    err "Databricks CLI is not installed or not on PATH."
    exit 1
  fi
  local profile="${ASR_DATABRICKS_PROFILE:-${DATABRICKS_CONFIG_PROFILE:-fe-vm-vdm-classic-rcn6ip}}"
  DBX=(databricks --profile "$profile")
  export DATABRICKS_CONFIG_PROFILE="$profile"
  export ASR_DATABRICKS_PROFILE="$profile"
  export DATABRICKS_AUTH_STORAGE="${DATABRICKS_AUTH_STORAGE:-plaintext}"
}

resolve_training_volume_paths() {
  local py="$ROOT/.venv/bin/python"
  if [[ ! -x "$py" ]]; then
    py="python3"
  fi
  eval "$("$py" <<'PY'
import shlex
from genie_voice.config import get_settings

s = get_settings()
catalog = s.databricks.catalog
schema = s.databricks.schema_name
volume = s.volume.streaming_name
if any("<" in str(v) or not str(v).strip() for v in (catalog, schema, volume)):
    raise SystemExit(
        "Databricks catalog/schema/streaming volume are not configured. "
        "Set config/config.local.yaml or GENIE_DATABRICKS__CATALOG / "
        "GENIE_DATABRICKS__SCHEMA / GENIE_VOLUME__STREAMING_NAME."
    )

root = f"/Volumes/{catalog}/{schema}/{volume}/asr_model_training"
paths = {
    "ASR_TRAINING_ROOT": root,
    "ASR_HOLDOUT_ROOT": f"{root}/datasets/holdout",
    "ASR_HOLDOUT_MANIFEST": f"{root}/datasets/holdout/manifests/asr_real_audio_holdout_v1.jsonl",
    "ASR_MODEL_ARTIFACTS": f"{root}/model_artifacts",
    "ASR_TRAINING_EVALUATIONS": f"{root}/evaluations",
}
for key, value in paths.items():
    print(f"{key}={shlex.quote(value)}")
PY
)"
}

sync_local_file_to_volume() {
  local local_path="$1"
  local remote_dir="$2"
  if [[ -s "$local_path" ]]; then
    setup_databricks_cli
    "${DBX[@]}" fs mkdirs "dbfs:$remote_dir"
    "${DBX[@]}" fs cp "$local_path" "dbfs:$remote_dir/$(basename "$local_path")" --overwrite
  fi
}

next_step() {
  resolve_training_volume_paths
  if [[ ! -s "$HOLDOUT_MANIFEST" ]]; then
    cat <<EOF
Next ASR holdout step

Run:
  scripts/asr/04_asr_real_audio_holdout.sh prepare

Then add at least $HOLDOUT_MIN_CLIPS human-approved holdout rows here:
  $HOLDOUT_MANIFEST

Holdout audio should live under:
  $ASR_HOLDOUT_ROOT/audio

EOF
    return
  fi

  if ! validate_manifest >/dev/null; then
    cat <<EOF
Next ASR holdout step

Fix the holdout manifest, then run:
  scripts/asr/04_asr_real_audio_holdout.sh validate

Manifest:
  $HOLDOUT_MANIFEST

EOF
    return
  fi

  cat <<'EOF'
Next ASR holdout step

Run the full held-out evaluation gate:

  scripts/asr/04_asr_real_audio_holdout.sh evaluate

This may start/reuse the dedicated GPU cluster for Whisper evaluation.
EOF
}

prepare() {
  resolve_training_volume_paths
  mkdir -p "$LOCAL_HOLDOUT_DIR/manifests" "$HOLDOUT_RESULTS_DIR" "$HOLDOUT_EVAL_DIR"
  if [[ ! -e "$HOLDOUT_MANIFEST" ]]; then
    cat >"$HOLDOUT_MANIFEST" <<EOF
{"clip_id":"HOLDOUT-001","audio_path":"$ASR_HOLDOUT_ROOT/audio/HOLDOUT-001.wav","reference_transcript":"Customer says the human-approved transcript here.","call_id":"HOLDOUT-CALL-001","speaker":"customer","split":"holdout","dataset_version":"real_audio_holdout_v1","expected_entities":{"invoice_ids":["INV-90022"],"amounts":[],"dates":[],"billing_actions":[],"confirmations":[],"refusals":[],"account_terms":["invoice"]}}
EOF
  fi
  python3 - "$HOLDOUT_MANIFEST" "$ASR_HOLDOUT_ROOT/audio/HOLDOUT-001.wav" <<'PY'
import json
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
template_audio = sys.argv[2]
lines = [line for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
if len(lines) != 1:
    raise SystemExit(0)
row = json.loads(lines[0])
if (
    row.get("clip_id") == "HOLDOUT-001"
    and row.get("reference_transcript") == "Customer says the human-approved transcript here."
):
    row["audio_path"] = template_audio
    manifest.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
PY

  setup_databricks_cli
  "${DBX[@]}" fs mkdirs "dbfs:$ASR_HOLDOUT_ROOT/audio"
  "${DBX[@]}" fs mkdirs "dbfs:$ASR_HOLDOUT_ROOT/manifests"
  "${DBX[@]}" fs mkdirs "dbfs:$ASR_HOLDOUT_ROOT/results"

  cat <<EOF
Prepared holdout workspace.

Local manifest:
  $HOLDOUT_MANIFEST

Volume folders:
  $ASR_HOLDOUT_ROOT/audio
  $ASR_HOLDOUT_ROOT/manifests
  $ASR_HOLDOUT_ROOT/results

Replace the template row with at least $HOLDOUT_MIN_CLIPS real/realistic clips
and human-approved transcripts before running evaluate.

EOF
}

validate_manifest() {
  resolve_training_volume_paths
  python3 - "$HOLDOUT_MANIFEST" "$HOLDOUT_MIN_CLIPS" <<'PY'
import json
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
min_clips = int(sys.argv[2])
if not manifest.exists() or manifest.stat().st_size == 0:
    raise SystemExit(f"Missing holdout manifest: {manifest}")

rows = []
for line_no, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), start=1):
    if not line.strip() or line.lstrip().startswith("#"):
        continue
    row = json.loads(line)
    missing = [key for key in ("clip_id", "audio_path", "reference_transcript", "expected_entities") if not row.get(key)]
    if missing:
        raise SystemExit(f"{manifest}:{line_no} missing {missing}")
    if row.get("split") != "holdout":
        raise SystemExit(f"{manifest}:{line_no} split must be holdout")
    if row.get("replace_with_real_call_audio_before_training"):
        raise SystemExit(f"{manifest}:{line_no} row is marked as training placeholder, not holdout")
    rows.append(row)

if len(rows) < min_clips:
    raise SystemExit(f"Holdout manifest has {len(rows)} clips; require at least {min_clips}")

print(json.dumps({"clips": len(rows), "manifest": str(manifest)}, indent=2))
PY
}

sync_manifest() {
  setup_databricks_cli
  "${DBX[@]}" fs cp "$HOLDOUT_MANIFEST" "dbfs:$ASR_HOLDOUT_MANIFEST" --overwrite
}

latest_lora_run_name() {
  setup_databricks_cli
  if [[ -n "${ASR_LORA_RUN_NAME:-}" ]]; then
    printf "%s\n" "$ASR_LORA_RUN_NAME"
    return
  fi
  local runs
  runs="$("${DBX[@]}" fs ls "dbfs:$ASR_MODEL_ARTIFACTS/lora_runs")"
  python3 - "$runs" <<'PY'
import sys
names = sorted(line.strip().rstrip("/") for line in sys.argv[1].splitlines() if line.strip().startswith("lora_"))
if names:
    print(names[-1])
PY
}

find_gpu_cluster() {
  setup_databricks_cli
  local clusters_json
  clusters_json="$("${DBX[@]}" clusters list --output json)"
  python3 - "$ASR_GPU_CLUSTER_NAME" "$clusters_json" <<'PY'
import json
import sys
name = sys.argv[1]
for cluster in json.loads(sys.argv[2]):
    if cluster.get("cluster_name") == name:
        print(cluster.get("cluster_id", ""))
        break
PY
}

submit_base_whisper() {
  local cluster_id="$1"
  local runner_local="$ROOT/scripts/asr/databricks_whisper_baseline.py"
  local runner_remote="$ASR_MODEL_ARTIFACTS/jobs/databricks_whisper_baseline.py"
  local output_remote="$ASR_HOLDOUT_ROOT/results/base_whisper_holdout.jsonl"
  local output_local="$HOLDOUT_RESULTS_DIR/base_whisper_holdout.jsonl"

  "${DBX[@]}" fs mkdirs "dbfs:$ASR_MODEL_ARTIFACTS/jobs"
  "${DBX[@]}" fs cp "$runner_local" "dbfs:$runner_remote" --overwrite

  local job_json
  job_json="$(mktemp)"
  python3 - "$job_json" "$runner_remote" "$ASR_HOLDOUT_MANIFEST" "$output_remote" "$ASR_BASE_MODEL" "$cluster_id" <<'PY'
import json
import sys
from pathlib import Path

job_json, runner_remote, manifest, output, model, cluster_id = sys.argv[1:]
payload = {
    "run_name": "genie-asr-holdout-base-whisper",
    "tasks": [{
        "task_key": "base_whisper_holdout",
        "existing_cluster_id": cluster_id,
        "spark_python_task": {
            "python_file": f"dbfs:{runner_remote}",
            "parameters": ["--manifest", manifest, "--output", output, "--model", model, "--language", "english"],
        },
        "libraries": [
            {"pypi": {"package": "transformers"}},
            {"pypi": {"package": "accelerate"}},
            {"pypi": {"package": "librosa"}},
            {"pypi": {"package": "soundfile"}},
        ],
    }],
}
Path(job_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY
  "${DBX[@]}" jobs submit --json @"$job_json" --output json
  rm -f "$job_json"
  "${DBX[@]}" fs cp "dbfs:$output_remote" "$output_local" --overwrite
}

submit_lora_whisper() {
  local cluster_id="$1"
  local lora_run="$2"
  local runner_local="$ROOT/scripts/asr/databricks_whisper_lora_evaluate.py"
  local runner_remote="$ASR_MODEL_ARTIFACTS/jobs/databricks_whisper_lora_evaluate.py"
  local adapter_dir="$ASR_MODEL_ARTIFACTS/lora_runs/$lora_run/adapter"
  local output_remote="$ASR_HOLDOUT_ROOT/results/lora_whisper_holdout.jsonl"
  local summary_remote="$ASR_HOLDOUT_ROOT/results/lora_whisper_holdout_summary.json"
  local output_local="$HOLDOUT_RESULTS_DIR/lora_whisper_holdout.jsonl"

  "${DBX[@]}" fs mkdirs "dbfs:$ASR_MODEL_ARTIFACTS/jobs"
  "${DBX[@]}" fs cp "$runner_local" "dbfs:$runner_remote" --overwrite

  local job_json
  job_json="$(mktemp)"
  python3 - "$job_json" "$runner_remote" "$ASR_HOLDOUT_MANIFEST" "$adapter_dir" "$output_remote" "$summary_remote" "$ASR_BASE_MODEL" "$cluster_id" <<'PY'
import json
import sys
from pathlib import Path

job_json, runner_remote, manifest, adapter, output, summary, model, cluster_id = sys.argv[1:]
payload = {
    "run_name": "genie-asr-holdout-lora-whisper",
    "tasks": [{
        "task_key": "lora_whisper_holdout",
        "existing_cluster_id": cluster_id,
        "spark_python_task": {
            "python_file": f"dbfs:{runner_remote}",
            "parameters": [
                "--manifest", manifest,
                "--adapter-dir", adapter,
                "--output", output,
                "--summary-output", summary,
                "--base-model", model,
            ],
        },
        "libraries": [
            {"pypi": {"package": "transformers"}},
            {"pypi": {"package": "accelerate"}},
            {"pypi": {"package": "peft"}},
            {"pypi": {"package": "librosa"}},
            {"pypi": {"package": "soundfile"}},
        ],
    }],
}
Path(job_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY
  "${DBX[@]}" jobs submit --json @"$job_json" --output json
  rm -f "$job_json"
  "${DBX[@]}" fs cp "dbfs:$output_remote" "$output_local" --overwrite
}

write_decision_report() {
  local deepgram="$HOLDOUT_RESULTS_DIR/deepgram_holdout_invoice_postprocessed.jsonl"
  local base="$HOLDOUT_RESULTS_DIR/base_whisper_holdout_invoice_postprocessed.jsonl"
  local lora="$HOLDOUT_RESULTS_DIR/lora_whisper_holdout_invoice_postprocessed.jsonl"
  local report="$HOLDOUT_EVAL_DIR/asr_real_audio_holdout_decision_report.json"

  python3 - "$deepgram" "$base" "$lora" "$report" <<'PY'
import json
import sys
from collections import defaultdict
from pathlib import Path

deepgram, base, lora, report = map(Path, sys.argv[1:])

def avg(values):
    values = [v for v in values if v is not None]
    return None if not values else sum(values) / len(values)

def summarize(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    entity_totals = defaultdict(lambda: {"expected": 0, "matched": 0})
    for row in rows:
        for group, score in (row.get("score", {}).get("entity_scores") or {}).items():
            entity_totals[group]["expected"] += score.get("expected") or 0
            entity_totals[group]["matched"] += score.get("matched") or 0
    return {
        "results": str(path),
        "clips": len(rows),
        "avg_wer": avg(row.get("score", {}).get("wer") for row in rows),
        "avg_cer": avg(row.get("score", {}).get("cer") for row in rows),
        "avg_entity_accuracy": avg(row.get("score", {}).get("entity_accuracy") for row in rows),
        "entity_groups": {
            group: {
                **counts,
                "accuracy": None if counts["expected"] == 0 else counts["matched"] / counts["expected"],
            }
            for group, counts in sorted(entity_totals.items())
        },
    }

comparison = {
    "deepgram": summarize(deepgram),
    "base_whisper_postprocessed": summarize(base),
    "lora_whisper_postprocessed": summarize(lora),
}
winner = min(comparison, key=lambda name: (comparison[name]["avg_wer"], -comparison[name]["avg_entity_accuracy"]))
decision = {
    "comparison": comparison,
    "winner_by_wer_then_entity_accuracy": winner,
    "production_gate": {
        "minimum_clips": 50,
        "requires_real_or_realistic_holdout": True,
        "passed": comparison[winner]["clips"] >= 50 and (comparison[winner]["avg_entity_accuracy"] or 0) >= 0.95,
    },
}
report.parent.mkdir(parents=True, exist_ok=True)
report.write_text(json.dumps(decision, indent=2), encoding="utf-8")
print(json.dumps(decision, indent=2))
PY
}

evaluate() {
  validate_manifest
  prepare >/dev/null
  sync_manifest

  log "running Deepgram holdout baseline"
  "$ENGINE" deepgram \
    --manifest "$HOLDOUT_MANIFEST" \
    --output "$HOLDOUT_RESULTS_DIR/deepgram_holdout.jsonl" \
    --audio-source volume \
    --sync-results false
  sync_local_file_to_volume "$HOLDOUT_RESULTS_DIR/deepgram_holdout.jsonl" "$ASR_HOLDOUT_ROOT/results"

  log "starting/reusing dedicated GPU cluster for holdout Whisper evaluation"
  "$FINETUNE" gpu-start
  local cluster_id
  cluster_id="$(find_gpu_cluster)"
  if [[ -z "$cluster_id" ]]; then
    err "Could not resolve GPU cluster after gpu-start."
    exit 1
  fi

  local lora_run
  lora_run="$(latest_lora_run_name)"
  if [[ -z "$lora_run" ]]; then
    err "No LoRA run found. Train/evaluate LoRA before holdout gate."
    exit 1
  fi

  log "running Base Whisper holdout evaluation"
  submit_base_whisper "$cluster_id"

  log "running LoRA Whisper holdout evaluation: $lora_run"
  submit_lora_whisper "$cluster_id" "$lora_run"

  log "rescoring all holdout outputs with invoice postprocessing"
  "$ENGINE" rescore --postprocess-invoice-ids \
    --manifest "$HOLDOUT_MANIFEST" \
    --input "$HOLDOUT_RESULTS_DIR/deepgram_holdout.jsonl" \
    --output "$HOLDOUT_RESULTS_DIR/deepgram_holdout_invoice_postprocessed.jsonl"
  "$ENGINE" rescore --postprocess-invoice-ids \
    --manifest "$HOLDOUT_MANIFEST" \
    --input "$HOLDOUT_RESULTS_DIR/base_whisper_holdout.jsonl" \
    --output "$HOLDOUT_RESULTS_DIR/base_whisper_holdout_invoice_postprocessed.jsonl"
  "$ENGINE" rescore --postprocess-invoice-ids \
    --manifest "$HOLDOUT_MANIFEST" \
    --input "$HOLDOUT_RESULTS_DIR/lora_whisper_holdout.jsonl" \
    --output "$HOLDOUT_RESULTS_DIR/lora_whisper_holdout_invoice_postprocessed.jsonl"
  sync_local_file_to_volume "$HOLDOUT_RESULTS_DIR/deepgram_holdout_invoice_postprocessed.jsonl" "$ASR_HOLDOUT_ROOT/results"
  sync_local_file_to_volume "$HOLDOUT_RESULTS_DIR/base_whisper_holdout_invoice_postprocessed.jsonl" "$ASR_HOLDOUT_ROOT/results"
  sync_local_file_to_volume "$HOLDOUT_RESULTS_DIR/lora_whisper_holdout_invoice_postprocessed.jsonl" "$ASR_HOLDOUT_ROOT/results"

  write_decision_report
  sync_local_file_to_volume "$HOLDOUT_EVAL_DIR/asr_real_audio_holdout_decision_report.json" "$ASR_TRAINING_EVALUATIONS"
  sync_local_file_to_volume "$HOLDOUT_EVAL_DIR/asr_real_audio_holdout_decision_report.json" "$ASR_HOLDOUT_ROOT/results"

  cat <<EOF

Holdout evaluation complete.

Decision report:
  $HOLDOUT_EVAL_DIR/asr_real_audio_holdout_decision_report.json

GPU cluster was not stopped automatically. Stop later only when no more GPU
work is needed:
  scripts/asr/03_asr_model_finetuning.sh gpu-stop

EOF
}

case "$COMMAND" in
  next)
    next_step
    ;;
  prepare)
    prepare
    ;;
  validate)
    validate_manifest
    ;;
  evaluate)
    evaluate
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    err "Unknown command: $COMMAND"
    usage >&2
    exit 2
    ;;
esac
