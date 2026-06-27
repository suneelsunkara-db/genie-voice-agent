#!/usr/bin/env bash
# =============================================================================
# 03_asr_model_finetuning.sh
#
# Actual ASR model fine-tuning and trained-model evaluation only.
# This script is intentionally separated from 02_asr_baseline_runs.sh so model
# training is explicit and cannot be confused with vendor/base-model baselines.
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENGINE="$ROOT/scripts/asr/01_asr_model_training.sh"
ASR_GPU_CLUSTER_NAME="${ASR_GPU_CLUSTER_NAME:-genie-asr-gpu-training}"
ASR_GPU_RUNTIME="${ASR_GPU_RUNTIME:-16.4.x-gpu-ml-scala2.13}"
ASR_TRAINING_ROOT="${ASR_TRAINING_ROOT:-}"
ASR_TRAINING_MANIFEST="${ASR_TRAINING_MANIFEST:-}"
ASR_TRAINING_BASELINES="${ASR_TRAINING_BASELINES:-}"
ASR_TRAINING_EVALUATIONS="${ASR_TRAINING_EVALUATIONS:-}"
ASR_TRAINING_MODEL_ARTIFACTS="${ASR_TRAINING_MODEL_ARTIFACTS:-}"
BASE_MODEL="${ASR_LORA_BASE_MODEL:-openai/whisper-small.en}"

COMMAND="${1:-next}"
if [[ $# -gt 0 ]]; then
  shift
fi

usage() {
  cat <<'EOF'
ASR model fine-tuning

Purpose:
  Fine-tune and evaluate the Databricks-hosted Whisper alternative to Deepgram.
  This script is for actual trained-model work, not vendor/base-model baselines.

Commands:
  next        Show the next training gate.
  gpu-status  Show the dedicated ASR GPU cluster status.
  gpu-start   Create/start the dedicated ASR GPU cluster.
  preflight   Verify the training gates before fine-tuning.
  dry-run     Submit a Databricks LoRA dry-run job without training.
  train-lora  Submit Databricks LoRA fine-tuning job.
  evaluate-lora
              Evaluate the latest LoRA adapter against the locked manifest.
  analyze-entity-errors
              Compare Base Whisper, LoRA Whisper, and Deepgram entity misses.
  rescore-entity-metrics
              Rescore outputs with current metrics and write corrected comparison.
  rescore-invoice-postprocess
              Apply candidate-aware invoice-ID post-processing and rescore.
  gpu-stop    Stop GPU cluster after training/evaluation work is done.
  help        Show this help.

EOF
}

next_step() {
  cat <<'EOF'
Actual model training is not the next action yet.

Training gate:
  1. Fair Deepgram-vs-Whisper comparison must exist.
  2. Dedicated ASR GPU cluster must be running.
  3. LoRA dry-run must pass before real training.

Run this first:

  scripts/asr/03_asr_model_finetuning.sh dry-run

EOF
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
    "ASR_TRAINING_MANIFEST": f"{root}/datasets/gold/manifests/asr_training_gold_v1.jsonl",
    "ASR_TRAINING_BASELINES": f"{root}/baselines",
    "ASR_TRAINING_EVALUATIONS": f"{root}/evaluations",
    "ASR_TRAINING_MODEL_ARTIFACTS": f"{root}/model_artifacts",
}
for key, value in paths.items():
    print(f"{key}={shlex.quote(value)}")
PY
)"
}

preflight() {
  local comparison="$ROOT/.run/asr_model_training/evaluations/asr_baseline_fair_comparison.json"
  if [[ ! -s "$comparison" ]]; then
    cat <<EOF
Fine-tuning preflight failed.

Missing fair comparison artifact:
  $comparison

Run:
  scripts/asr/02_asr_baseline_runs.sh fair-compare

EOF
    return 2
  fi

  local cluster_id
  cluster_id="$(find_gpu_cluster)"
  if [[ -z "$cluster_id" ]]; then
    cat <<EOF
Fine-tuning preflight failed.

Missing dedicated ASR GPU cluster:
  $ASR_GPU_CLUSTER_NAME

Run:
  scripts/asr/03_asr_model_finetuning.sh gpu-start

EOF
    return 2
  fi

  local state
  state="$(cluster_state "$cluster_id")"
  if [[ "$state" != "RUNNING" ]]; then
    cat <<EOF
Fine-tuning preflight failed.

Dedicated ASR GPU cluster is $state:
  $cluster_id

Run:
  scripts/asr/03_asr_model_finetuning.sh gpu-start

EOF
    return 2
  fi

  cat <<EOF
Fine-tuning preflight passed.

Fair comparison:
  $comparison

Dedicated GPU cluster:
  $cluster_id

EOF
}

setup_databricks_cli() {
  resolve_training_volume_paths
  if ! command -v databricks >/dev/null 2>&1; then
    printf "Databricks CLI is not installed or not on PATH.\n" >&2
    exit 1
  fi
  local profile="${ASR_DATABRICKS_PROFILE:-${DATABRICKS_CONFIG_PROFILE:-fe-vm-vdm-classic-rcn6ip}}"
  DBX=(databricks --profile "$profile")
  export DATABRICKS_CONFIG_PROFILE="$profile"
  export ASR_DATABRICKS_PROFILE="$profile"
  export DATABRICKS_AUTH_STORAGE="${DATABRICKS_AUTH_STORAGE:-plaintext}"
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

find_gpu_cluster() {
  setup_databricks_cli
  local clusters_json
  clusters_json="$("${DBX[@]}" clusters list --output json)"
  python3 - "$ASR_GPU_CLUSTER_NAME" "$clusters_json" <<'PY'
import json
import sys

name = sys.argv[1]
clusters = json.loads(sys.argv[2])
for cluster in clusters:
    if cluster.get("cluster_name") == name:
        print(cluster.get("cluster_id", ""))
        break
PY
}

cluster_state() {
  setup_databricks_cli
  "${DBX[@]}" clusters get "$1" --output json | python3 -c 'import json,sys; print(json.load(sys.stdin).get("state", ""))'
}

submit_lora_job() {
  local mode="$1"
  preflight

  setup_databricks_cli
  local cluster_id
  cluster_id="$(find_gpu_cluster)"
  local runner_local="$ROOT/scripts/asr/databricks_whisper_lora_finetune.py"
  local runner_remote="$ASR_TRAINING_MODEL_ARTIFACTS/jobs/databricks_whisper_lora_finetune.py"
  local manifest_local="$ROOT/.run/asr_model_training/manifests/asr_training_gold_v1.jsonl"
  local run_name
  run_name="lora_$(date +%Y%m%d_%H%M%S)"
  local output_dir="$ASR_TRAINING_MODEL_ARTIFACTS/lora_runs/$run_name"

  "${DBX[@]}" fs cp "$manifest_local" "dbfs:$ASR_TRAINING_MANIFEST" --overwrite
  "${DBX[@]}" fs mkdirs "dbfs:$ASR_TRAINING_MODEL_ARTIFACTS/jobs"
  "${DBX[@]}" fs cp "$runner_local" "dbfs:$runner_remote" --overwrite

  local job_json
  job_json="$(mktemp)"
  python3 - "$job_json" "$runner_remote" "$ASR_TRAINING_MANIFEST" "$output_dir" "$BASE_MODEL" "$cluster_id" "$mode" <<'PY'
import json
import sys
from pathlib import Path

job_json, runner_remote, manifest, output_dir, base_model, cluster_id, mode = sys.argv[1:]
params = [
    "--manifest", manifest,
    "--output-dir", output_dir,
    "--base-model", base_model,
    "--epochs", "1",
    "--max-eval-samples", "40",
]
if mode == "dry-run":
    params.extend(["--dry-run", "--max-train-samples", "8", "--max-eval-samples", "4"])

payload = {
    "run_name": f"genie-asr-whisper-lora-{mode}",
    "tasks": [
        {
            "task_key": "whisper_lora",
            "existing_cluster_id": cluster_id,
            "spark_python_task": {
                "python_file": f"dbfs:{runner_remote}",
                "parameters": params,
            },
            "libraries": [
                {"pypi": {"package": "transformers"}},
                {"pypi": {"package": "accelerate"}},
                {"pypi": {"package": "datasets"}},
                {"pypi": {"package": "peft"}},
                {"pypi": {"package": "evaluate"}},
                {"pypi": {"package": "jiwer"}},
                {"pypi": {"package": "librosa"}},
                {"pypi": {"package": "soundfile"}},
            ],
        }
    ],
}
Path(job_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY

  "${DBX[@]}" jobs submit --json @"$job_json" --output json
  rm -f "$job_json"
  cat <<EOF

Submitted LoRA $mode job.

Output directory:
  $output_dir

Cluster:
  $cluster_id

EOF
}

latest_lora_run_name() {
  setup_databricks_cli
  if [[ -n "${ASR_LORA_RUN_NAME:-}" ]]; then
    printf "%s\n" "$ASR_LORA_RUN_NAME"
    return
  fi

  local runs
  runs="$("${DBX[@]}" fs ls "dbfs:$ASR_TRAINING_MODEL_ARTIFACTS/lora_runs")"
  python3 - "$runs" <<'PY'
import sys

run_names = sorted(line.strip().rstrip("/") for line in sys.argv[1].splitlines() if line.strip().startswith("lora_"))
if run_names:
    print(run_names[-1])
PY
}

write_lora_comparison() {
  local run_name="$1"
  local lora_results="$2"
  local comparison_input="$ROOT/.run/asr_model_training/evaluations/asr_baseline_fair_comparison.json"
  local comparison_output="$ROOT/.run/asr_model_training/evaluations/asr_lora_fair_comparison_${run_name}.json"

  mkdir -p "$(dirname "$comparison_output")"
  python3 - "$comparison_input" "$lora_results" "$comparison_output" <<'PY'
import json
import sys
from collections import defaultdict
from pathlib import Path

comparison_input = Path(sys.argv[1])
lora_path = Path(sys.argv[2])
comparison_output = Path(sys.argv[3])

def avg(values):
    values = [value for value in values if value is not None]
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
        "provider": sorted({row.get("provider") for row in rows if row.get("provider")}),
        "models": sorted({row.get("model") for row in rows if row.get("model")}),
        "adapters": sorted({row.get("adapter_dir") for row in rows if row.get("adapter_dir")}),
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

comparison = json.loads(comparison_input.read_text(encoding="utf-8"))
comparison["lora_whisper"] = summarize(lora_path)
comparison_output.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
print(json.dumps(comparison, indent=2))
PY
}

evaluate_lora_job() {
  preflight

  setup_databricks_cli
  local cluster_id
  cluster_id="$(find_gpu_cluster)"
  local runner_local="$ROOT/scripts/asr/databricks_whisper_lora_evaluate.py"
  local runner_remote="$ASR_TRAINING_MODEL_ARTIFACTS/jobs/databricks_whisper_lora_evaluate.py"
  local manifest_local="$ROOT/.run/asr_model_training/manifests/asr_training_gold_v1.jsonl"
  local run_name
  run_name="$(latest_lora_run_name)"
  if [[ -z "$run_name" ]]; then
    cat <<EOF
LoRA evaluation failed.

No LoRA run found under:
  $ASR_TRAINING_MODEL_ARTIFACTS/lora_runs

Run:
  scripts/asr/03_asr_model_finetuning.sh train-lora

EOF
    return 2
  fi

  local adapter_dir="$ASR_TRAINING_MODEL_ARTIFACTS/lora_runs/$run_name/adapter"
  local eval_dir="$ASR_TRAINING_MODEL_ARTIFACTS/lora_runs/$run_name/evaluation"
  local results_remote="$eval_dir/lora_evaluation_results.jsonl"
  local summary_remote="$eval_dir/lora_evaluation_summary.json"
  local results_local="$ROOT/.run/asr_model_training/baselines/whisper_lora_${run_name}_evaluation.jsonl"
  local summary_local="$ROOT/.run/asr_model_training/evaluations/whisper_lora_${run_name}_summary.json"

  "${DBX[@]}" fs cp "$manifest_local" "dbfs:$ASR_TRAINING_MANIFEST" --overwrite
  "${DBX[@]}" fs mkdirs "dbfs:$ASR_TRAINING_MODEL_ARTIFACTS/jobs"
  "${DBX[@]}" fs cp "$runner_local" "dbfs:$runner_remote" --overwrite

  local job_json
  job_json="$(mktemp)"
  python3 - "$job_json" "$runner_remote" "$ASR_TRAINING_MANIFEST" "$adapter_dir" "$results_remote" "$summary_remote" "$BASE_MODEL" "$cluster_id" <<'PY'
import json
import sys
from pathlib import Path

job_json, runner_remote, manifest, adapter_dir, results_remote, summary_remote, base_model, cluster_id = sys.argv[1:]
payload = {
    "run_name": "genie-asr-whisper-lora-evaluate",
    "tasks": [
        {
            "task_key": "whisper_lora_evaluate",
            "existing_cluster_id": cluster_id,
            "spark_python_task": {
                "python_file": f"dbfs:{runner_remote}",
                "parameters": [
                    "--manifest", manifest,
                    "--adapter-dir", adapter_dir,
                    "--output", results_remote,
                    "--summary-output", summary_remote,
                    "--base-model", base_model,
                ],
            },
            "libraries": [
                {"pypi": {"package": "transformers"}},
                {"pypi": {"package": "accelerate"}},
                {"pypi": {"package": "peft"}},
                {"pypi": {"package": "librosa"}},
                {"pypi": {"package": "soundfile"}},
            ],
        }
    ],
}
Path(job_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY

  "${DBX[@]}" jobs submit --json @"$job_json" --output json
  rm -f "$job_json"

  mkdir -p "$(dirname "$results_local")" "$(dirname "$summary_local")"
  "${DBX[@]}" fs cp "dbfs:$results_remote" "$results_local" --overwrite
  "${DBX[@]}" fs cp "dbfs:$summary_remote" "$summary_local" --overwrite
  write_lora_comparison "$run_name" "$results_local"
  sync_local_file_to_volume "$results_local" "$ASR_TRAINING_BASELINES/whisper"
  sync_local_file_to_volume "$summary_local" "$ASR_TRAINING_EVALUATIONS"
  sync_local_file_to_volume \
    "$ROOT/.run/asr_model_training/evaluations/asr_lora_fair_comparison_${run_name}.json" \
    "$ASR_TRAINING_EVALUATIONS"

  cat <<EOF

Evaluated LoRA adapter.

LoRA run:
  $run_name

Results:
  $results_local

Summary:
  $summary_local

Comparison:
  $ROOT/.run/asr_model_training/evaluations/asr_lora_fair_comparison_${run_name}.json

Cluster:
  $cluster_id

EOF
}

analyze_entity_errors() {
  local run_name
  run_name="${ASR_LORA_RUN_NAME:-}"
  if [[ -z "$run_name" ]]; then
    setup_databricks_cli
    run_name="$(latest_lora_run_name)"
  fi
  if [[ -z "$run_name" ]]; then
    cat <<EOF
Entity analysis failed.

No LoRA run found. Run:
  scripts/asr/03_asr_model_finetuning.sh evaluate-lora

EOF
    return 2
  fi

  local base_results="$ROOT/.run/asr_model_training/baselines/whisper_small_databricks_full_rescored.jsonl"
  local lora_results="$ROOT/.run/asr_model_training/baselines/whisper_lora_${run_name}_evaluation.jsonl"
  local deepgram_results="$ROOT/.run/asr_model_training/baselines/deepgram_nova3_403_baseline.jsonl"
  local json_output="$ROOT/.run/asr_model_training/evaluations/asr_entity_error_analysis_${run_name}.json"
  local markdown_output="$ROOT/.run/asr_model_training/evaluations/asr_entity_error_analysis_${run_name}.md"

  for path in "$base_results" "$lora_results" "$deepgram_results"; do
    if [[ ! -s "$path" ]]; then
      cat <<EOF
Entity analysis failed.

Missing result file:
  $path

Run:
  scripts/asr/03_asr_model_finetuning.sh evaluate-lora

EOF
      return 2
    fi
  done

  python3 "$ROOT/scripts/asr/analyze_entity_errors.py" \
    --base-whisper "$base_results" \
    --lora-whisper "$lora_results" \
    --deepgram "$deepgram_results" \
    --json-output "$json_output" \
    --markdown-output "$markdown_output"
  sync_local_file_to_volume "$json_output" "$ASR_TRAINING_EVALUATIONS"
  sync_local_file_to_volume "$markdown_output" "$ASR_TRAINING_EVALUATIONS"

  cat <<EOF

Entity error analysis written.

JSON:
  $json_output

Markdown:
  $markdown_output

EOF
}

rescore_entity_metrics() {
  local run_name
  run_name="${ASR_LORA_RUN_NAME:-}"
  if [[ -z "$run_name" ]]; then
    setup_databricks_cli
    run_name="$(latest_lora_run_name)"
  fi
  if [[ -z "$run_name" ]]; then
    cat <<EOF
Entity rescore failed.

No LoRA run found. Run:
  scripts/asr/03_asr_model_finetuning.sh evaluate-lora

EOF
    return 2
  fi

  local base_input="$ROOT/.run/asr_model_training/baselines/whisper_small_databricks_full_rescored.jsonl"
  local lora_input="$ROOT/.run/asr_model_training/baselines/whisper_lora_${run_name}_evaluation.jsonl"
  local deepgram_input="$ROOT/.run/asr_model_training/baselines/deepgram_nova3_403_baseline.jsonl"
  local base_output="$ROOT/.run/asr_model_training/baselines/whisper_small_databricks_full_corrected_rescored.jsonl"
  local lora_output="$ROOT/.run/asr_model_training/baselines/whisper_lora_${run_name}_corrected_rescored.jsonl"
  local deepgram_output="$ROOT/.run/asr_model_training/baselines/deepgram_nova3_403_corrected_rescored.jsonl"
  local comparison_output="$ROOT/.run/asr_model_training/evaluations/asr_lora_corrected_fair_comparison_${run_name}.json"
  local analysis_json="$ROOT/.run/asr_model_training/evaluations/asr_entity_error_analysis_corrected_${run_name}.json"
  local analysis_markdown="$ROOT/.run/asr_model_training/evaluations/asr_entity_error_analysis_corrected_${run_name}.md"

  "$ENGINE" rescore --input "$base_input" --output "$base_output"
  "$ENGINE" rescore --input "$lora_input" --output "$lora_output"
  "$ENGINE" rescore --input "$deepgram_input" --output "$deepgram_output"
  sync_local_file_to_volume "$base_output" "$ASR_TRAINING_BASELINES/whisper"
  sync_local_file_to_volume "$lora_output" "$ASR_TRAINING_BASELINES/whisper"
  sync_local_file_to_volume "$deepgram_output" "$ASR_TRAINING_BASELINES/deepgram"

  python3 - "$base_output" "$lora_output" "$deepgram_output" "$comparison_output" <<'PY'
import json
import sys
from collections import defaultdict
from pathlib import Path

base_path, lora_path, deepgram_path, output_path = map(Path, sys.argv[1:])

def avg(values):
    values = [value for value in values if value is not None]
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
        "provider": sorted({row.get("provider") for row in rows if row.get("provider")}),
        "models": sorted({row.get("model") for row in rows if row.get("model")}),
        "adapters": sorted({row.get("adapter_dir") for row in rows if row.get("adapter_dir")}),
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
    "base_whisper": summarize(base_path),
    "lora_whisper": summarize(lora_path),
    "deepgram": summarize(deepgram_path),
}
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
print(json.dumps(comparison, indent=2))
PY

  python3 "$ROOT/scripts/asr/analyze_entity_errors.py" \
    --base-whisper "$base_output" \
    --lora-whisper "$lora_output" \
    --deepgram "$deepgram_output" \
    --json-output "$analysis_json" \
    --markdown-output "$analysis_markdown"
  sync_local_file_to_volume "$comparison_output" "$ASR_TRAINING_EVALUATIONS"
  sync_local_file_to_volume "$analysis_json" "$ASR_TRAINING_EVALUATIONS"
  sync_local_file_to_volume "$analysis_markdown" "$ASR_TRAINING_EVALUATIONS"

  cat <<EOF

Corrected entity metrics written.

Comparison:
  $comparison_output

Analysis:
  $analysis_markdown

EOF
}

rescore_invoice_postprocess() {
  local run_name
  run_name="${ASR_LORA_RUN_NAME:-}"
  if [[ -z "$run_name" ]]; then
    setup_databricks_cli
    run_name="$(latest_lora_run_name)"
  fi
  if [[ -z "$run_name" ]]; then
    cat <<EOF
Invoice postprocess rescore failed.

No LoRA run found. Run:
  scripts/asr/03_asr_model_finetuning.sh evaluate-lora

EOF
    return 2
  fi

  local base_input="$ROOT/.run/asr_model_training/baselines/whisper_small_databricks_full_rescored.jsonl"
  local lora_input="$ROOT/.run/asr_model_training/baselines/whisper_lora_${run_name}_evaluation.jsonl"
  local deepgram_input="$ROOT/.run/asr_model_training/baselines/deepgram_nova3_403_baseline.jsonl"
  local base_output="$ROOT/.run/asr_model_training/baselines/whisper_small_databricks_full_invoice_postprocessed_rescored.jsonl"
  local lora_output="$ROOT/.run/asr_model_training/baselines/whisper_lora_${run_name}_invoice_postprocessed_rescored.jsonl"
  local deepgram_output="$ROOT/.run/asr_model_training/baselines/deepgram_nova3_403_invoice_postprocessed_rescored.jsonl"
  local comparison_output="$ROOT/.run/asr_model_training/evaluations/asr_lora_invoice_postprocessed_comparison_${run_name}.json"
  local analysis_json="$ROOT/.run/asr_model_training/evaluations/asr_entity_error_analysis_invoice_postprocessed_${run_name}.json"
  local analysis_markdown="$ROOT/.run/asr_model_training/evaluations/asr_entity_error_analysis_invoice_postprocessed_${run_name}.md"

  "$ENGINE" rescore --postprocess-invoice-ids --input "$base_input" --output "$base_output"
  "$ENGINE" rescore --postprocess-invoice-ids --input "$lora_input" --output "$lora_output"
  "$ENGINE" rescore --postprocess-invoice-ids --input "$deepgram_input" --output "$deepgram_output"
  sync_local_file_to_volume "$base_output" "$ASR_TRAINING_BASELINES/whisper"
  sync_local_file_to_volume "$lora_output" "$ASR_TRAINING_BASELINES/whisper"
  sync_local_file_to_volume "$deepgram_output" "$ASR_TRAINING_BASELINES/deepgram"

  python3 - "$base_output" "$lora_output" "$deepgram_output" "$comparison_output" <<'PY'
import json
import sys
from collections import defaultdict
from pathlib import Path

base_path, lora_path, deepgram_path, output_path = map(Path, sys.argv[1:])

def avg(values):
    values = [value for value in values if value is not None]
    return None if not values else sum(values) / len(values)

def summarize(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    entity_totals = defaultdict(lambda: {"expected": 0, "matched": 0})
    corrections = 0
    for row in rows:
        corrections += len((row.get("postprocessing") or {}).get("invoice_id_corrections") or [])
        for group, score in (row.get("score", {}).get("entity_scores") or {}).items():
            entity_totals[group]["expected"] += score.get("expected") or 0
            entity_totals[group]["matched"] += score.get("matched") or 0
    return {
        "results": str(path),
        "clips": len(rows),
        "invoice_id_corrections": corrections,
        "provider": sorted({row.get("provider") for row in rows if row.get("provider")}),
        "models": sorted({row.get("model") for row in rows if row.get("model")}),
        "adapters": sorted({row.get("adapter_dir") for row in rows if row.get("adapter_dir")}),
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
    "base_whisper": summarize(base_path),
    "lora_whisper": summarize(lora_path),
    "deepgram": summarize(deepgram_path),
}
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
print(json.dumps(comparison, indent=2))
PY

  python3 "$ROOT/scripts/asr/analyze_entity_errors.py" \
    --base-whisper "$base_output" \
    --lora-whisper "$lora_output" \
    --deepgram "$deepgram_output" \
    --json-output "$analysis_json" \
    --markdown-output "$analysis_markdown"
  sync_local_file_to_volume "$comparison_output" "$ASR_TRAINING_EVALUATIONS"
  sync_local_file_to_volume "$analysis_json" "$ASR_TRAINING_EVALUATIONS"
  sync_local_file_to_volume "$analysis_markdown" "$ASR_TRAINING_EVALUATIONS"

  cat <<EOF

Invoice postprocessed metrics written.

Comparison:
  $comparison_output

Analysis:
  $analysis_markdown

EOF
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
  preflight)
    preflight
    ;;
  dry-run)
    submit_lora_job "dry-run"
    ;;
  train-lora)
    submit_lora_job "train"
    ;;
  evaluate-lora)
    evaluate_lora_job "$@"
    ;;
  analyze-entity-errors)
    analyze_entity_errors "$@"
    ;;
  rescore-entity-metrics)
    rescore_entity_metrics "$@"
    ;;
  rescore-invoice-postprocess)
    rescore_invoice_postprocess "$@"
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
