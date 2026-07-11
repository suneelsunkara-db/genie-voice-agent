#!/usr/bin/env bash
# =============================================================================
# 09_pilot_multilingual_model_bakeoff.sh
#
# Model-only pilot bake-off for multilingual ASR strategy.
#
# This script is intentionally NOT the app integration path. It does not change
# FastAPI, frontend, Databricks serving endpoints, or runtime config. Its job is
# to validate model strategy before product code changes:
#
#   - Thai: Qwen3-ASR vs Thai-specialized Whisper OSS checkpoints
#   - Indonesian: Qwen3-ASR 1.7B vs 0.6B
#   - Chinese: Qwen3-ASR 1.7B vs 0.6B
#
# Safe default: dry-run mode writes reference transcripts as hypotheses so the
# manifest/results/scoring path can be validated before model deps are installed.
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEFAULT_OUTPUT_DIR="$ROOT/.run/asr_model_training/evaluations/multilingual_model_bakeoff"
DEFAULT_SCAFFOLD_DIR="$DEFAULT_OUTPUT_DIR/scaffold"
DEFAULT_MANIFEST="$DEFAULT_SCAFFOLD_DIR/multilingual_pilot_manifest.example.jsonl"
DEFAULT_RESULTS="$DEFAULT_OUTPUT_DIR/multilingual_pilot_results.jsonl"
DEFAULT_SUMMARY="$DEFAULT_OUTPUT_DIR/multilingual_pilot_summary.json"
DEFAULT_PLAN="$DEFAULT_OUTPUT_DIR/multilingual_pilot_plan.json"
DEFAULT_REMOTE_SUBDIR="evaluations/multilingual_model_bakeoff"

ASR_ML_PILOT_OUTPUT_DIR="${ASR_ML_PILOT_OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}"
ASR_ML_PILOT_SCAFFOLD_DIR="${ASR_ML_PILOT_SCAFFOLD_DIR:-$DEFAULT_SCAFFOLD_DIR}"
ASR_ML_PILOT_MANIFEST="${ASR_ML_PILOT_MANIFEST:-$DEFAULT_MANIFEST}"
ASR_ML_PILOT_RESULTS="${ASR_ML_PILOT_RESULTS:-$DEFAULT_RESULTS}"
ASR_ML_PILOT_SUMMARY="${ASR_ML_PILOT_SUMMARY:-$DEFAULT_SUMMARY}"
ASR_ML_PILOT_PLAN="${ASR_ML_PILOT_PLAN:-$DEFAULT_PLAN}"
ASR_ML_PILOT_DRY_RUN="${ASR_ML_PILOT_DRY_RUN:-true}"
ASR_ML_PILOT_LANGUAGE="${ASR_ML_PILOT_LANGUAGE:-}"
ASR_ML_PILOT_CANDIDATE="${ASR_ML_PILOT_CANDIDATE:-}"
ASR_ML_PILOT_LIMIT="${ASR_ML_PILOT_LIMIT:-}"
ASR_ML_PILOT_MIN_HOLDOUT="${ASR_ML_PILOT_MIN_HOLDOUT:-200}"
ASR_ML_PILOT_MIN_TRAIN_SMOKE="${ASR_ML_PILOT_MIN_TRAIN_SMOKE:-75}"
ASR_ML_PILOT_PUBLIC_LIMIT="${ASR_ML_PILOT_PUBLIC_LIMIT:-2}"
ASR_ML_PILOT_DATABRICKS_PROFILE="${ASR_ML_PILOT_DATABRICKS_PROFILE:-${ASR_DATABRICKS_PROFILE:-${DATABRICKS_CONFIG_PROFILE:-fe-vm-vdm-classic-rcn6ip}}}"
ASR_ML_PILOT_SERVERLESS_ENVIRONMENT_VERSION="${ASR_ML_PILOT_SERVERLESS_ENVIRONMENT_VERSION:-2}"
ASR_ML_PILOT_SERVERLESS_PERFORMANCE_TARGET="${ASR_ML_PILOT_SERVERLESS_PERFORMANCE_TARGET:-PERFORMANCE_OPTIMIZED}"
ASR_ML_PILOT_REMOTE_ROOT="${ASR_ML_PILOT_REMOTE_ROOT:-}"

COMMAND="${1:-pilot}"
if [[ $# -gt 0 ]]; then
  shift
fi

log() { printf "\033[36m[asr-multilingual-pilot]\033[0m %s\n" "$*"; }
err() { printf "\033[31m[asr-multilingual-pilot]\033[0m %s\n" "$*" >&2; }

usage() {
  cat <<EOF
Multilingual ASR model-only pilot bake-off

Commands:
  pilot               One-command safe flow: plan, scaffold, dry-run, summarize, preflight. Default.
  databricks-serverless
                      Submit this pilot as a one-off Databricks serverless job.
  databricks-public-smoke
                      Bootstrap public FLEURS audio on serverless, run real models, summarize.
  plan                Write and print model strategy, candidates, gates, and data requirements.
  scaffold            Write manifest/result examples, candidate matrix, and collection checklist.
  prepare-real-pilot  Validate a real manifest before non-dry-run model execution.
  preflight           Check Python imports, optional model deps, manifest, and output paths.
  run                 Run all configured candidates on the holdout manifest and write results JSONL.
  run-candidate       Run one candidate. Set ASR_ML_PILOT_CANDIDATE and optional ASR_ML_PILOT_LANGUAGE.
  summarize-existing  Score ASR_ML_PILOT_RESULTS and write summary JSON.
  help                Show this help.

Environment:
  ASR_ML_PILOT_MANIFEST           Pilot manifest JSONL. Default: $DEFAULT_MANIFEST
  ASR_ML_PILOT_RESULTS            Result JSONL. Default: $DEFAULT_RESULTS
  ASR_ML_PILOT_SUMMARY            Summary JSON. Default: $DEFAULT_SUMMARY
  ASR_ML_PILOT_SCAFFOLD_DIR       Scaffold/template directory. Default: $DEFAULT_SCAFFOLD_DIR
  ASR_ML_PILOT_DRY_RUN            true/false. Default: true.
  ASR_ML_PILOT_LANGUAGE           Optional language filter: th, id, zh.
  ASR_ML_PILOT_CANDIDATE          Required for run-candidate.
  ASR_ML_PILOT_LIMIT              Optional max holdout clips per candidate.
  ASR_ML_PILOT_MIN_HOLDOUT        Target holdout clips/language. Default: 200.
  ASR_ML_PILOT_MIN_TRAIN_SMOKE    Target train smoke clips/language. Default: 75.
  ASR_ML_PILOT_PUBLIC_LIMIT       Public FLEURS clips/language for smoke. Default: 2.
  ASR_ML_PILOT_DATABRICKS_PROFILE Databricks CLI profile. Default: fe-vm-vdm-classic-rcn6ip.
  ASR_ML_PILOT_REMOTE_ROOT        UC Volume remote root. Default resolves from app config:
                                  /Volumes/<catalog>/<schema>/<volume>/asr_model_training/evaluations/multilingual_model_bakeoff
  ASR_ML_PILOT_SERVERLESS_PERFORMANCE_TARGET
                                  Serverless job target. Default: PERFORMANCE_OPTIMIZED.

Examples:
  scripts/asr/09_pilot_multilingual_model_bakeoff.sh
  scripts/asr/09_pilot_multilingual_model_bakeoff.sh databricks-serverless
  ASR_ML_PILOT_PUBLIC_LIMIT=1 scripts/asr/09_pilot_multilingual_model_bakeoff.sh databricks-public-smoke
  ASR_ML_PILOT_MANIFEST=/path/to/real_manifest.jsonl \\
    scripts/asr/09_pilot_multilingual_model_bakeoff.sh prepare-real-pilot
  scripts/asr/09_pilot_multilingual_model_bakeoff.sh scaffold
  scripts/asr/09_pilot_multilingual_model_bakeoff.sh run
  ASR_ML_PILOT_DRY_RUN=false ASR_ML_PILOT_LANGUAGE=zh \\
    scripts/asr/09_pilot_multilingual_model_bakeoff.sh run
  ASR_ML_PILOT_CANDIDATE=th_oss_qwen3_asr_1_7b \\
    scripts/asr/09_pilot_multilingual_model_bakeoff.sh run-candidate
  scripts/asr/09_pilot_multilingual_model_bakeoff.sh summarize-existing

This is not an app/runtime script. It is only for model strategy validation.
EOF
}

setup_env() {
  cd "$ROOT"
  if [[ -f "$ROOT/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$ROOT/.env"
    set +a
  fi
  mkdir -p "$ASR_ML_PILOT_OUTPUT_DIR" "$ASR_ML_PILOT_SCAFFOLD_DIR"
  export PYTHONPATH="$ROOT/backend${PYTHONPATH:+:$PYTHONPATH}"
  if [[ -x "$ROOT/.venv/bin/python" ]]; then
    PYTHON_BIN="$ROOT/.venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
}

setup_databricks_cli() {
  setup_env
  if ! command -v databricks >/dev/null 2>&1; then
    err "Databricks CLI is not installed or not on PATH."
    exit 1
  fi
  DBX=(databricks --profile "$ASR_ML_PILOT_DATABRICKS_PROFILE")
  export DATABRICKS_CONFIG_PROFILE="$ASR_ML_PILOT_DATABRICKS_PROFILE"
  export ASR_DATABRICKS_PROFILE="$ASR_ML_PILOT_DATABRICKS_PROFILE"
}

resolve_remote_root() {
  if [[ -n "$ASR_ML_PILOT_REMOTE_ROOT" ]]; then
    return
  fi
  ASR_ML_PILOT_REMOTE_ROOT="$("$PYTHON_BIN" - "$DEFAULT_REMOTE_SUBDIR" <<'PY'
from __future__ import annotations

import sys
from genie_voice.config import get_settings

s = get_settings()
catalog = s.databricks.catalog
schema = s.databricks.schema_name
volume = s.volume.streaming_name
if any("<" in str(v) or not str(v).strip() for v in (catalog, schema, volume)):
    raise SystemExit("Databricks catalog/schema/streaming volume are not configured.")
print(f"/Volumes/{catalog}/{schema}/{volume}/asr_model_training/{sys.argv[1]}")
PY
)"
}

python_args() {
  printf '%s\n' \
    --plan-output "$ASR_ML_PILOT_PLAN" \
    --scaffold-dir "$ASR_ML_PILOT_SCAFFOLD_DIR" \
    --manifest "$ASR_ML_PILOT_MANIFEST" \
    --results "$ASR_ML_PILOT_RESULTS" \
    --summary-output "$ASR_ML_PILOT_SUMMARY" \
    --dry-run "$ASR_ML_PILOT_DRY_RUN" \
    --language "$ASR_ML_PILOT_LANGUAGE" \
    --candidate "$ASR_ML_PILOT_CANDIDATE" \
    --limit "$ASR_ML_PILOT_LIMIT" \
    --public-limit "$ASR_ML_PILOT_PUBLIC_LIMIT" \
    --min-holdout "$ASR_ML_PILOT_MIN_HOLDOUT" \
    --min-train-smoke "$ASR_ML_PILOT_MIN_TRAIN_SMOKE"
}

load_python_args() {
  PY_ARGS=()
  while IFS= read -r arg; do
    PY_ARGS+=("$arg")
  done < <(python_args)
}

run_python() {
  local action="$1"
  shift || true
  python3 - "$action" "$@" <<'PY'
from __future__ import annotations

import argparse
import importlib.util
import io
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


LANGUAGES: list[dict[str, Any]] = [
    {
        "language": "Thai",
        "code": "th",
        "decision": "bake_off",
        "metric": "cer",
        "public_eval_sets": ["FLEURS-th", "Common Voice Thai", "GigaSpeech2/Thai where available"],
        "promotion_gate": {
            "critical_entity_accuracy_min": 0.95,
            "cer_max": 0.08,
            "unsafe_for_resolution_rate_max": 0.05,
            "p95_latency_ms_max": 2500,
        },
        "candidates": [
            {
                "id": "th_oss_qwen3_asr_1_7b",
                "family": "qwen3",
                "base_model": "Qwen/Qwen3-ASR-1.7B",
                "license": "Apache-2.0",
                "role": "strategic platform candidate",
            },
            {
                "id": "th_oss_typhoon_whisper_large_v3",
                "family": "whisper",
                "base_model": "typhoon-ai/typhoon-whisper-large-v3",
                "license": "MIT",
                "role": "Thai-specialized Whisper challenger",
            },
            {
                "id": "th_oss_pathumma_whisper_large_v3",
                "family": "whisper",
                "base_model": "nectec/Pathumma-whisper-th-large-v3",
                "license": "model-card review required",
                "role": "Thai-specialized Whisper fallback/challenger",
            },
        ],
    },
    {
        "language": "Indonesian",
        "code": "id",
        "decision": "qwen3_first",
        "metric": "wer",
        "public_eval_sets": ["FLEURS-id", "Common Voice Indonesian"],
        "promotion_gate": {
            "critical_entity_accuracy_min": 0.95,
            "wer_max": 0.12,
            "unsafe_for_resolution_rate_max": 0.05,
            "p95_latency_ms_max": 2200,
        },
        "candidates": [
            {
                "id": "id_oss_qwen3_asr_1_7b",
                "family": "qwen3",
                "base_model": "Qwen/Qwen3-ASR-1.7B",
                "license": "Apache-2.0",
                "role": "quality candidate",
            },
            {
                "id": "id_oss_qwen3_asr_0_6b",
                "family": "qwen3",
                "base_model": "Qwen/Qwen3-ASR-0.6B",
                "license": "Apache-2.0",
                "role": "cost/latency candidate",
            },
        ],
    },
    {
        "language": "Chinese",
        "code": "zh",
        "decision": "qwen3_first",
        "metric": "cer",
        "public_eval_sets": ["FLEURS-zh", "AISHELL", "WenetSpeech"],
        "promotion_gate": {
            "critical_entity_accuracy_min": 0.95,
            "cer_max": 0.06,
            "unsafe_for_resolution_rate_max": 0.05,
            "p95_latency_ms_max": 2200,
        },
        "candidates": [
            {
                "id": "zh_oss_qwen3_asr_1_7b",
                "family": "qwen3",
                "base_model": "Qwen/Qwen3-ASR-1.7B",
                "license": "Apache-2.0",
                "role": "quality candidate",
            },
            {
                "id": "zh_oss_qwen3_asr_0_6b",
                "family": "qwen3",
                "base_model": "Qwen/Qwen3-ASR-0.6B",
                "license": "Apache-2.0",
                "role": "cost/latency candidate",
            },
        ],
    },
]


MODEL_LANGUAGE_NAMES = {
    "th": "Thai",
    "id": "Indonesian",
    "zh": "Chinese",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["plan", "scaffold", "bootstrap-public-smoke", "prepare-real-pilot", "preflight", "run", "run-candidate", "summarize-existing"])
    parser.add_argument("--plan-output", required=True)
    parser.add_argument("--scaffold-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--dry-run", default="true")
    parser.add_argument("--language", default="")
    parser.add_argument("--candidate", default="")
    parser.add_argument("--limit", default="")
    parser.add_argument("--public-limit", type=int, default=2)
    parser.add_argument("--min-holdout", type=int, required=True)
    parser.add_argument("--min-train-smoke", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.action == "plan":
        plan(args)
    elif args.action == "scaffold":
        scaffold(args)
    elif args.action == "bootstrap-public-smoke":
        bootstrap_public_smoke(args)
    elif args.action == "prepare-real-pilot":
        prepare_real_pilot(args)
    elif args.action == "preflight":
        preflight(args)
    elif args.action == "run":
        run_all(args)
    elif args.action == "run-candidate":
        run_one_candidate(args)
    elif args.action == "summarize-existing":
        summarize_existing(args)


def plan(args: argparse.Namespace) -> None:
    data = {
        "goal": "Pilot multilingual OSS ASR model strategy before app/runtime changes.",
        "scope": ["Thai", "Indonesian", "Chinese"],
        "not_in_scope": ["Tagalog", "Filipino", "Philippines", "FastAPI/frontend changes", "serving endpoint changes"],
        "safe_default": "ASR_ML_PILOT_DRY_RUN=true writes reference transcripts to validate plumbing.",
        "pilot_data_requirements": pilot_data_requirements(args.min_train_smoke, args.min_holdout),
        "languages": LANGUAGES,
        "decision_rule": {
            "thai": "Choose Qwen3 only if it matches/beats Thai-specialized Whisper on entity accuracy and CER while improving latency/TCO.",
            "indonesian": "Use Qwen3 unless measured WER/entity/latency gates fail.",
            "chinese": "Use Qwen3 unless measured CER/entity/latency gates fail.",
        },
    }
    write_json(args.plan_output, data)
    print_plan(data, args.plan_output)


def scaffold(args: argparse.Namespace) -> None:
    output_dir = Path(args.scaffold_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "pilot_manifest_example": output_dir / "multilingual_pilot_manifest.example.jsonl",
        "pilot_results_example": output_dir / "multilingual_pilot_results.example.jsonl",
        "candidate_matrix": output_dir / "candidate_matrix.json",
        "collection_checklist": output_dir / "collection_checklist.json",
    }
    write_jsonl(files["pilot_manifest_example"], manifest_example_rows())
    write_jsonl(files["pilot_results_example"], result_example_rows())
    write_json(files["candidate_matrix"], {"languages": LANGUAGES})
    write_json(
        files["collection_checklist"],
        {
            "goal": "Collect only enough data to validate model strategy before product code changes.",
            "minimums": pilot_data_requirements(args.min_train_smoke, args.min_holdout),
            "per_language_required_counts": collection_counts(args.min_train_smoke, args.min_holdout),
            "holdout_rules": [
                "Do not train on holdout clips.",
                "Require human-approved verbatim transcripts.",
                "Keep noise/device/channel metadata on every row.",
                "Include expected_entities for every billing/account utterance.",
                "Evaluate Qwen3 with context biasing enabled for IDs, products, account terms, and amounts.",
            ],
            "result_jsonl_required_fields": [
                "language",
                "candidate",
                "clip_id",
                "reference_transcript",
                "transcript",
                "latency_ms",
                "expected_entities",
            ],
        },
    )
    print_scaffold(files)


def bootstrap_public_smoke(args: argparse.Namespace) -> None:
    cache_root = Path(args.scaffold_dir).parent / "hf_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_root / "home")
    os.environ["HF_DATASETS_CACHE"] = str(cache_root / "datasets")
    try:
        import pandas as pd
        import soundfile as sf
        from huggingface_hub import hf_hub_download, list_repo_files
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("bootstrap-public-smoke requires huggingface_hub, pandas, pyarrow, and soundfile") from exc

    configs = {"th": "th_th", "id": "id_id", "zh": "cmn_hans_cn"}
    base_dir = Path(args.scaffold_dir).parent / "public_smoke_audio"
    base_dir.mkdir(parents=True, exist_ok=True)
    repo_files = list_repo_files("google/fleurs", repo_type="dataset")
    rows: list[dict[str, Any]] = []
    for lang in LANGUAGES:
        code = lang["code"]
        if args.language and args.language != code:
            continue
        config = configs[code]
        parquet_candidates = [
            name
            for name in repo_files
            if name.endswith(".parquet") and f"/{config}/" in name and "validation" in name
        ]
        if not parquet_candidates:
            raise RuntimeError(f"No FLEURS validation parquet found for {config}")
        parquet_file = sorted(parquet_candidates)[0]
        parquet_path = hf_hub_download(
            repo_id="google/fleurs",
            repo_type="dataset",
            filename=parquet_file,
            cache_dir=str(cache_root / "hub"),
        )
        dataset = pd.read_parquet(parquet_path)
        written = 0
        for _, item in dataset.iterrows():
            if written >= args.public_limit:
                break
            audio = item.get("audio")
            clip_id = f"{code}_fleurs_{written + 1:04d}"
            audio_path = write_public_audio(audio, base_dir, clip_id, sf)
            rows.append(
                {
                    "clip_id": clip_id,
                    "audio_path": str(audio_path),
                    "reference_transcript": str(item.get("transcription") or item.get("raw_transcription") or "").strip(),
                    "language": code,
                    "split": "holdout",
                    "scenario": "public_fleurs_smoke",
                    "duration_seconds": None,
                    "expected_entities": {},
                    "metadata": {"source": "google/fleurs", "config": config, "id": safe_json_value(item.get("id"))},
                }
            )
            written += 1
    if not rows:
        raise SystemExit("No public smoke rows were created")
    write_jsonl(args.manifest, rows)
    print("\nPUBLIC FLEURS SMOKE MANIFEST")
    print("=" * 78)
    print(f"Manifest: {args.manifest}")
    print(f"Rows: {len(rows)}")
    for code in sorted({row["language"] for row in rows}):
        print(f"  {code}: {sum(1 for row in rows if row['language'] == code)} clips")
    print("=" * 78)


def write_public_audio(audio: Any, base_dir: Path, clip_id: str, sf: Any) -> Path:
    if isinstance(audio, dict):
        path_hint = str(audio.get("path") or "")
        suffix = Path(path_hint).suffix or ".wav"
        if audio.get("bytes"):
            output = base_dir / f"{clip_id}{suffix}"
            output.write_bytes(bytes(audio["bytes"]))
            return output
        if audio.get("array") is not None and audio.get("sampling_rate"):
            output = base_dir / f"{clip_id}.wav"
            sf.write(output, audio["array"], int(audio["sampling_rate"]))
            return output
    if isinstance(audio, (bytes, bytearray, memoryview)):
        output = base_dir / f"{clip_id}.wav"
        output.write_bytes(bytes(audio))
        return output
    raise RuntimeError(f"Unsupported FLEURS audio payload for {clip_id}: {type(audio).__name__}")


def safe_json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    try:
        if pd_is_na(value):
            return None
    except Exception:  # noqa: BLE001
        pass
    return str(value)


def pd_is_na(value: Any) -> bool:
    import pandas as pd

    return bool(pd.isna(value))


def preflight(args: argparse.Namespace) -> None:
    checks: dict[str, Any] = {
        "python": sys.version.split()[0],
        "dry_run": is_true(args.dry_run),
        "manifest": args.manifest,
        "manifest_exists": Path(args.manifest).exists(),
        "results": args.results,
        "summary_output": args.summary_output,
        "backend_metrics_import": False,
        "qwen_asr_available": has_module("qwen_asr"),
        "transformers_available": has_module("transformers"),
        "torch_available": has_module("torch"),
        "soundfile_available": has_module("soundfile"),
    }
    try:
        from genie_voice.asr_eval.metrics import score_transcript  # noqa: F401

        checks["backend_metrics_import"] = True
    except Exception as exc:  # noqa: BLE001
        checks["backend_metrics_error"] = str(exc)
    if Path(args.manifest).exists():
        rows = read_jsonl(args.manifest, limit=10)
        checks["manifest_rows_sampled"] = len(rows)
        checks["manifest_shape_ok"] = all(
            {"clip_id", "audio_path", "reference_transcript", "language", "split"} <= set(row)
            for row in rows
        )
    print(json.dumps(checks, indent=2))


def prepare_real_pilot(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise SystemExit(f"Manifest does not exist: {manifest_path}")
    rows = read_jsonl(args.manifest)
    report = validate_real_manifest(rows, manifest_path, min_holdout=args.min_holdout, min_train=args.min_train_smoke)
    output = Path(args.summary_output).with_name("multilingual_pilot_manifest_validation.json")
    write_json(output, report)
    print_manifest_validation(report, output)
    if report["errors"]:
        raise SystemExit(2)


def run_all(args: argparse.Namespace) -> None:
    selected = selected_candidates(args.language, "")
    run_candidates(args, selected, append=False)


def run_one_candidate(args: argparse.Namespace) -> None:
    if not args.candidate:
        raise SystemExit("ASR_ML_PILOT_CANDIDATE is required for run-candidate")
    selected = selected_candidates(args.language, args.candidate)
    if not selected:
        raise SystemExit(f"Unknown candidate or language mismatch: {args.candidate}")
    run_candidates(args, selected, append=True)


def run_candidates(args: argparse.Namespace, candidates: list[dict[str, Any]], *, append: bool) -> None:
    manifest_rows = read_jsonl(args.manifest)
    if not is_true(args.dry_run) and not is_public_smoke_manifest(manifest_rows):
        report = validate_real_manifest(
            manifest_rows,
            Path(args.manifest),
            min_holdout=args.min_holdout,
            min_train=args.min_train_smoke,
        )
        if report["errors"]:
            print_manifest_validation(report, Path(args.summary_output).with_name("multilingual_pilot_manifest_validation.json"))
            raise SystemExit("Real model run refused because manifest validation failed.")
    manifest = load_holdout_manifest(args.manifest, language=args.language)
    if not manifest:
        raise SystemExit(f"No holdout clips selected from {args.manifest}")
    limit = int(args.limit) if args.limit else None
    if limit:
        manifest = manifest[:limit]
    output = Path(args.results)
    output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append and output.exists() and not str(output).startswith("/Volumes/") else "w"
    dry_run = is_true(args.dry_run)
    with output.open(mode, encoding="utf-8") as f:
        rows_written = 0
        for candidate in candidates:
            lang_rows = [row for row in manifest if row.get("language") == candidate["language_code"]]
            for row in lang_rows:
                started = time.perf_counter()
                transcript = transcribe(row, candidate, dry_run=dry_run)
                latency_ms = round((time.perf_counter() - started) * 1000)
                if dry_run:
                    latency_ms = int(row.get("dry_run_latency_ms") or 1200)
                f.write(
                    json.dumps(
                        {
                            "language": row.get("language"),
                            "candidate": candidate["id"],
                            "clip_id": row.get("clip_id"),
                            "reference_transcript": row.get("reference_transcript"),
                            "transcript": transcript,
                            "latency_ms": latency_ms,
                            "expected_entities": row.get("expected_entities") or {},
                            "raw": {
                                "dry_run": dry_run,
                                "base_model": candidate["base_model"],
                                "family": candidate["family"],
                            },
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                rows_written += 1
    print(f"Wrote {rows_written} result rows to {output}")
    print("Next: scripts/asr/09_pilot_multilingual_model_bakeoff.sh summarize-existing")


def is_public_smoke_manifest(rows: list[dict[str, Any]]) -> bool:
    return bool(rows) and all(str(row.get("scenario") or "") == "public_fleurs_smoke" for row in rows)


def transcribe(row: dict[str, Any], candidate: dict[str, Any], *, dry_run: bool) -> str:
    if dry_run:
        return str(row.get("reference_transcript") or "")
    if candidate["family"] == "qwen3":
        return transcribe_qwen3(row, candidate)
    if candidate["family"] == "whisper":
        return transcribe_whisper(row, candidate)
    raise RuntimeError(f"Unsupported candidate family: {candidate['family']}")


def transcribe_qwen3(row: dict[str, Any], candidate: dict[str, Any]) -> str:
    try:
        from qwen_asr import Qwen3ASRModel
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Install qwen-asr to run Qwen3 candidates, or keep ASR_ML_PILOT_DRY_RUN=true") from exc
    model = Qwen3ASRModel.from_pretrained(candidate["base_model"])
    # The qwen-asr package API is evolving; this call is intentionally isolated
    # in the pilot script rather than the app path.
    language = MODEL_LANGUAGE_NAMES.get(str(row.get("language") or ""), str(row.get("language") or ""))
    result = model.transcribe(str(row["audio_path"]), language=language)
    text = extract_transcript_text(result)
    if text:
        return text
    return str(result).strip()


def extract_transcript_text(result: Any) -> str:
    if isinstance(result, dict):
        return str(result.get("text") or result.get("transcript") or "").strip()
    if isinstance(result, (list, tuple)):
        return " ".join(text for item in result if (text := extract_transcript_text(item))).strip()
    for attr in ("text", "transcript"):
        if hasattr(result, attr):
            value = getattr(result, attr)
            if value:
                return str(value).strip()
    return ""


def transcribe_whisper(row: dict[str, Any], candidate: dict[str, Any]) -> str:
    try:
        import torch
        from transformers import pipeline
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Install torch and transformers to run Whisper candidates, or keep ASR_ML_PILOT_DRY_RUN=true") from exc
    device = 0 if torch.cuda.is_available() else -1
    asr = pipeline(
        task="automatic-speech-recognition",
        model=candidate["base_model"],
        device=device,
    )
    result = asr(str(row["audio_path"]))
    if isinstance(result, dict):
        return str(result.get("text") or "").strip()
    return str(result).strip()


def summarize_existing(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.results)
    if not rows:
        raise SystemExit(f"No rows found in {args.results}")

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        score = score_row(row)
        enriched = {**row, "score": score, "app_readiness": app_readiness(row)}
        groups[(str(row.get("language") or "unknown"), str(row.get("candidate") or "unknown"))].append(enriched)

    summary = {
        "results": args.results,
        "rows": len(rows),
        "groups": [summarize_group(language, candidate, group_rows) for (language, candidate), group_rows in sorted(groups.items())],
        "decision_points": measured_decision_points(groups),
    }
    write_json(args.summary_output, summary)
    print_measured_summary(summary, args.summary_output)


def score_row(row: dict[str, Any]) -> dict[str, Any]:
    reference = str(row.get("reference_transcript") or "")
    hypothesis = str(row.get("transcript") or "")
    ref_words = normalize_words(reference)
    hyp_words = normalize_words(hypothesis)
    ref_chars = normalize_chars(reference)
    hyp_chars = normalize_chars(hypothesis)
    word_errors = edit_distance(ref_words, hyp_words)
    char_errors = edit_distance(list(ref_chars), list(hyp_chars))
    return {
        "wer": ratio(word_errors, len(ref_words)),
        "cer": ratio(char_errors, len(ref_chars)),
        "word_errors": word_errors,
        "reference_words": len(ref_words),
        "char_errors": char_errors,
        "reference_chars": len(ref_chars),
    }


def app_readiness(row: dict[str, Any]) -> dict[str, Any]:
    critical_groups = ("invoice_ids", "amounts", "dates", "billing_actions", "confirmations", "refusals")
    critical_expected = 0
    critical_matched = 0
    unsafe_reasons: list[str] = []
    transcript = str(row.get("transcript") or "")
    expected_entities = row.get("expected_entities") or {}
    for group in critical_groups:
        expected_values = [str(value) for value in expected_entities.get(group, []) if str(value).strip()]
        missing = [value for value in expected_values if not multilingual_entity_present(value, transcript)]
        critical_expected += len(expected_values)
        critical_matched += len(expected_values) - len(missing)
        if missing:
            unsafe_reasons.append(f"missing_{group}")
    if not transcript.strip():
        unsafe_reasons.append("empty_transcript")
    if numeric_tokens(str(row.get("reference_transcript") or "")) - numeric_tokens(transcript):
        unsafe_reasons.append("missing_numeric_token")
    critical_accuracy = None if critical_expected == 0 else critical_matched / critical_expected
    return {
        "critical_entity_accuracy": critical_accuracy,
        "critical_entities_expected": critical_expected,
        "critical_entities_matched": critical_matched,
        "unsafe_for_resolution": bool(unsafe_reasons),
        "unsafe_reasons": sorted(set(unsafe_reasons)),
    }


def summarize_group(language: str, candidate: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    metric = "wer" if language.lower() in {"id", "indonesian"} else "cer"
    latencies = [row.get("latency_ms") for row in rows if row.get("latency_ms") is not None]
    return {
        "language": language,
        "candidate": candidate,
        "clips": len(rows),
        "primary_metric": metric,
        "avg_wer": avg(row.get("score", {}).get("wer") for row in rows),
        "avg_cer": avg(row.get("score", {}).get("cer") for row in rows),
        "avg_critical_entity_accuracy": avg(row.get("app_readiness", {}).get("critical_entity_accuracy") for row in rows),
        "unsafe_for_resolution_rate": rate(row.get("app_readiness", {}).get("unsafe_for_resolution") for row in rows),
        "p95_latency_ms": percentile(latencies, 95),
        "empty_transcript_rate": rate(not str(row.get("transcript") or "").strip() for row in rows),
        "unsafe_reason_counts": dict(Counter(reason for row in rows for reason in row.get("app_readiness", {}).get("unsafe_reasons", []))),
    }


def measured_decision_points(groups: dict[tuple[str, str], list[dict[str, Any]]]) -> list[dict[str, Any]]:
    by_language: dict[str, list[str]] = defaultdict(list)
    for language, candidate in groups:
        by_language[language].append(candidate)
    points: list[dict[str, Any]] = []
    for language, candidates in sorted(by_language.items()):
        required = "Qwen3 plus one Thai-specialized Whisper candidate" if language == "th" else "Qwen3 1.7B and optional 0.6B challenger"
        points.append(
            {
                "language": language,
                "candidates_measured": sorted(candidates),
                "required_comparison": required,
                "decision": "ready_for_review",
            }
        )
    return points


def selected_candidates(language_filter: str, candidate_filter: str) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for lang in LANGUAGES:
        if language_filter and lang["code"] != language_filter and lang["language"].lower() != language_filter.lower():
            continue
        for candidate in lang["candidates"]:
            if candidate_filter and candidate["id"] != candidate_filter:
                continue
            selected.append({**candidate, "language": lang["language"], "language_code": lang["code"], "metric": lang["metric"]})
    return selected


def load_holdout_manifest(path: str, *, language: str) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    selected = []
    for row in rows:
        if row.get("split") != "holdout":
            continue
        if language and row.get("language") != language:
            continue
        selected.append(row)
    return selected


def manifest_example_rows() -> list[dict[str, Any]]:
    return [
        {
            "clip_id": "th_train_0001",
            "audio_path": "/path/to/th_train_0001.wav",
            "reference_transcript": "ลูกค้าต้องการตรวจสอบใบแจ้งหนี้ INV-10482 จำนวน 1,432.10 บาท",
            "language": "th",
            "split": "train_smoke",
            "scenario": "billing_amount_dispute",
            "duration_seconds": 7.2,
            "expected_entities": {"invoice_ids": ["INV-10482"], "amounts": ["1,432.10"], "account_terms": ["ใบแจ้งหนี้"]},
        },
        {
            "clip_id": "th_holdout_0001",
            "audio_path": "/path/to/th_holdout_0001.wav",
            "reference_transcript": "ใช่ ยืนยันให้ชำระเงินสำหรับใบแจ้งหนี้ INV-29401 วันนี้",
            "language": "th",
            "split": "holdout",
            "scenario": "payment_confirmation",
            "duration_seconds": 5.8,
            "expected_entities": {"invoice_ids": ["INV-29401"], "confirmations": ["ใช่"], "billing_actions": ["ชำระเงิน"]},
        },
        {
            "clip_id": "id_train_0001",
            "audio_path": "/path/to/id_train_0001.wav",
            "reference_transcript": "Saya ingin membantah tagihan invoice INV-55018 sebesar 2.450.000 rupiah",
            "language": "id",
            "split": "train_smoke",
            "scenario": "billing_amount_dispute",
            "duration_seconds": 6.9,
            "expected_entities": {"invoice_ids": ["INV-55018"], "amounts": ["2.450.000"], "billing_actions": ["membantah"], "account_terms": ["invoice"]},
        },
        {
            "clip_id": "id_holdout_0001",
            "audio_path": "/path/to/id_holdout_0001.wav",
            "reference_transcript": "Tidak, jumlah pembayaran untuk INV-77120 itu salah",
            "language": "id",
            "split": "holdout",
            "scenario": "customer_refusal",
            "duration_seconds": 4.7,
            "expected_entities": {"invoice_ids": ["INV-77120"], "refusals": ["Tidak"], "billing_actions": ["pembayaran"]},
        },
        {
            "clip_id": "zh_train_0001",
            "audio_path": "/path/to/zh_train_0001.wav",
            "reference_transcript": "我想查询发票 INV-88017，金额是一千二百三十点五元",
            "language": "zh",
            "split": "train_smoke",
            "scenario": "invoice_lookup",
            "duration_seconds": 5.9,
            "expected_entities": {"invoice_ids": ["INV-88017"], "amounts": ["一千二百三十点五元"], "account_terms": ["发票"]},
        },
        {
            "clip_id": "zh_holdout_0001",
            "audio_path": "/path/to/zh_holdout_0001.wav",
            "reference_transcript": "确认，今天可以处理 INV-10993 的付款",
            "language": "zh",
            "split": "holdout",
            "scenario": "payment_confirmation",
            "duration_seconds": 4.9,
            "expected_entities": {"invoice_ids": ["INV-10993"], "confirmations": ["确认"], "billing_actions": ["付款"]},
        },
    ]


def result_example_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for manifest_row in manifest_example_rows():
        if manifest_row["split"] != "holdout":
            continue
        for candidate in selected_candidates(str(manifest_row["language"]), ""):
            rows.append(
                {
                    "language": manifest_row["language"],
                    "candidate": candidate["id"],
                    "clip_id": manifest_row["clip_id"],
                    "reference_transcript": manifest_row["reference_transcript"],
                    "transcript": manifest_row["reference_transcript"],
                    "latency_ms": 1200,
                    "expected_entities": manifest_row["expected_entities"],
                    "raw": {"note": "Replace transcript/latency/raw with real inference output."},
                }
            )
    return rows


def pilot_data_requirements(min_train_smoke: int, min_holdout: int) -> dict[str, Any]:
    return {
        "train_smoke_clips_per_language": min_train_smoke,
        "holdout_clips_per_language": min_holdout,
        "total_train_smoke_clips": min_train_smoke * len(LANGUAGES),
        "total_holdout_clips": min_holdout * len(LANGUAGES),
        "required_slices_per_language": [
            "quiet",
            "office/cafe noise",
            "phone/browser audio",
            "invoice/account IDs",
            "amounts and dates",
            "confirmations and refusals",
            "code-switching with English product/account terms",
        ],
    }


def collection_counts(min_train_smoke: int, min_holdout: int) -> dict[str, Any]:
    return {
        lang["code"]: {
            "language": lang["language"],
            "train_smoke": min_train_smoke,
            "holdout": min_holdout,
            "quiet_min": round(min_holdout * 0.25),
            "noise_min": round(min_holdout * 0.25),
            "phone_or_browser_min": round(min_holdout * 0.30),
            "entity_heavy_min": round(min_holdout * 0.50),
            "confirm_refusal_min": round(min_holdout * 0.15),
            "code_switch_min": round(min_holdout * 0.20),
        }
        for lang in LANGUAGES
    }


def print_plan(data: dict[str, Any], output: str) -> None:
    print("\nMULTILINGUAL ASR MODEL PILOT")
    print("=" * 78)
    print(f"Scope: {', '.join(data['scope'])}")
    print(f"Plan JSON: {output}")
    req = data["pilot_data_requirements"]
    print(f"Pilot data: {req['train_smoke_clips_per_language']} train_smoke + {req['holdout_clips_per_language']} holdout clips/language")
    print("\nCandidates and gates:")
    for lang in data["languages"]:
        gate = lang["promotion_gate"]
        quality_key = "cer_max" if "cer_max" in gate else "wer_max"
        print(f"  {lang['language']}: {lang['decision']} metric={lang['metric']} {quality_key}<={gate[quality_key]}")
        for candidate in lang["candidates"]:
            print(f"    - {candidate['id']} ({candidate['base_model']})")
    print("=" * 78)


def print_scaffold(files: dict[str, Path]) -> None:
    print("\nMULTILINGUAL ASR PILOT SCAFFOLD")
    print("=" * 78)
    for label, path in files.items():
        print(f"{label}: {path}")
    print("\nNext:")
    print("  1. Replace /path/to/*.wav with real audio paths and expand the manifest.")
    print("  2. Run dry-run plumbing: scripts/asr/09_pilot_multilingual_model_bakeoff.sh run")
    print("  3. Run real models only when ready: ASR_ML_PILOT_DRY_RUN=false ... run")
    print("  4. Score: scripts/asr/09_pilot_multilingual_model_bakeoff.sh summarize-existing")
    print("=" * 78)


def validate_real_manifest(
    rows: list[dict[str, Any]],
    manifest_path: Path,
    *,
    min_holdout: int,
    min_train: int,
) -> dict[str, Any]:
    languages = {lang["code"] for lang in LANGUAGES}
    required = {"clip_id", "audio_path", "reference_transcript", "language", "split", "expected_entities"}
    counts: dict[str, Counter[str]] = {code: Counter() for code in languages}
    entity_counts: dict[str, int] = {code: 0 for code in languages}
    errors: list[str] = []
    warnings: list[str] = []
    seen_clip_ids: set[str] = set()

    if not rows:
        errors.append("Manifest has no rows.")

    for idx, row in enumerate(rows, start=1):
        missing = sorted(key for key in required if not row.get(key))
        if missing:
            errors.append(f"row {idx}: missing required fields {missing}")
            continue
        clip_id = str(row["clip_id"])
        if clip_id in seen_clip_ids:
            errors.append(f"row {idx}: duplicate clip_id {clip_id}")
        seen_clip_ids.add(clip_id)

        language = str(row["language"])
        split = str(row["split"])
        if language not in languages:
            errors.append(f"row {idx}: unsupported language {language!r}; expected one of {sorted(languages)}")
            continue
        if split not in {"train_smoke", "holdout"}:
            errors.append(f"row {idx}: split must be train_smoke or holdout, got {split!r}")
        counts[language][split] += 1

        audio_path = str(row["audio_path"])
        if "/path/to/" in audio_path:
            errors.append(f"row {idx}: placeholder audio_path remains: {audio_path}")
        elif is_local_audio_path(audio_path) and not Path(strip_file_scheme(audio_path)).exists():
            errors.append(f"row {idx}: local audio_path does not exist: {audio_path}")
        elif not is_supported_remote_audio_path(audio_path):
            warnings.append(
                f"row {idx}: audio_path is not local or UC/DBFS-style; serverless may not read it: {audio_path}"
            )

        reference = str(row["reference_transcript"]).strip()
        if len(reference) < 2:
            errors.append(f"row {idx}: reference_transcript is too short")

        expected_entities = row.get("expected_entities") or {}
        if not isinstance(expected_entities, dict):
            errors.append(f"row {idx}: expected_entities must be an object")
            continue
        entity_total = sum(len(values or []) for values in expected_entities.values() if isinstance(values, list))
        entity_counts[language] += entity_total
        if split == "holdout" and entity_total == 0:
            errors.append(f"row {idx}: holdout row has no expected_entities")

    for lang in LANGUAGES:
        code = lang["code"]
        if counts[code]["holdout"] == 0:
            errors.append(f"{code}: missing holdout rows")
        if counts[code]["train_smoke"] == 0:
            warnings.append(f"{code}: no train_smoke rows; fine for zero-shot, insufficient for LoRA smoke")
        if counts[code]["holdout"] < min(10, min_holdout):
            warnings.append(f"{code}: only {counts[code]['holdout']} holdout rows; use >=10 for smoke, target {min_holdout}")
        if counts[code]["train_smoke"] < min(5, min_train):
            warnings.append(f"{code}: only {counts[code]['train_smoke']} train_smoke rows; target {min_train}")
        if entity_counts[code] == 0:
            errors.append(f"{code}: no expected entities across manifest")

    return {
        "manifest": str(manifest_path),
        "rows": len(rows),
        "counts": {code: dict(counter) for code, counter in sorted(counts.items())},
        "entity_counts": entity_counts,
        "required_fields": sorted(required),
        "supported_languages": sorted(languages),
        "errors": errors,
        "warnings": warnings,
        "ready_for_real_inference": not errors,
    }


def print_manifest_validation(report: dict[str, Any], output: Path) -> None:
    print("\nREAL MULTILINGUAL PILOT MANIFEST VALIDATION")
    print("=" * 78)
    print(f"Manifest: {report['manifest']}")
    print(f"Rows: {report['rows']}")
    print(f"Validation JSON: {output}")
    print(f"Ready for real inference: {report['ready_for_real_inference']}")
    print("\nCounts:")
    for code, counts in report["counts"].items():
        print(f"  {code}: train_smoke={counts.get('train_smoke', 0)} holdout={counts.get('holdout', 0)} entities={report['entity_counts'].get(code, 0)}")
    if report["errors"]:
        print("\nErrors:")
        for error in report["errors"][:50]:
            print(f"  - {error}")
    if report["warnings"]:
        print("\nWarnings:")
        for warning in report["warnings"][:50]:
            print(f"  - {warning}")
    print("=" * 78)


def is_local_audio_path(path: str) -> bool:
    return path.startswith("/") or path.startswith("file://")


def strip_file_scheme(path: str) -> str:
    return path.removeprefix("file://")


def is_supported_remote_audio_path(path: str) -> bool:
    return path.startswith("/Volumes/") or path.startswith("dbfs:/") or path.startswith("s3://")


def print_measured_summary(summary: dict[str, Any], output: str) -> None:
    print("\nMEASURED MULTILINGUAL ASR DATA POINTS")
    print("=" * 78)
    print(f"Rows scored: {summary['rows']}")
    print(f"Summary JSON: {output}")
    for group in summary["groups"]:
        primary = "avg_wer" if group["primary_metric"] == "wer" else "avg_cer"
        print(f"\n{group['language']} / {group['candidate']}")
        print(f"  clips: {group['clips']}")
        print(f"  {primary}: {fmt_pct(group[primary])}")
        print(f"  critical_entity_accuracy: {fmt_pct(group['avg_critical_entity_accuracy'])}")
        print(f"  unsafe_for_resolution_rate: {fmt_pct(group['unsafe_for_resolution_rate'])}")
        print(f"  p95_latency_ms: {fmt_num(group['p95_latency_ms'])}")
        print(f"  empty_transcript_rate: {fmt_pct(group['empty_transcript_rate'])}")
    print("=" * 78)


def multilingual_entity_present(expected: str, transcript: str) -> bool:
    expected_norm = normalize_loose_entity(expected)
    transcript_norm = normalize_loose_entity(transcript)
    if not expected_norm:
        return True
    if expected_norm in transcript_norm:
        return True
    invoice_match = re.search(r"inv(\d+)", expected_norm)
    if invoice_match and invoice_match.group(1) in transcript_norm:
        return True
    digits = re.sub(r"\D", "", expected)
    return bool(len(digits) >= 3 and digits in re.sub(r"\D", "", transcript))


def normalize_loose_entity(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", "", text)
    return re.sub(r"[^\w\u0e00-\u0e7f\u3400-\u9fff]", "", text)


def numeric_tokens(text: str) -> set[str]:
    return set(re.findall(r"\b\d+(?:\.\d+)?\b", text.replace(",", "")))


def normalize_words(text: str) -> list[str]:
    text = re.sub(r"[^\w\s\u0e00-\u0e7f\u3400-\u9fff]", " ", text.lower())
    text = re.sub(r"\s+", " ", text).strip()
    return text.split() if text else []


def normalize_chars(text: str) -> str:
    text = re.sub(r"\s+", "", text.lower())
    return re.sub(r"[^\w\u0e00-\u0e7f\u3400-\u9fff]", "", text)


def edit_distance(left: list[str], right: list[str]) -> int:
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for i, left_item in enumerate(left, start=1):
        current = [i]
        for j, right_item in enumerate(right, start=1):
            cost = 0 if left_item == right_item else 1
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost))
        previous = current
    return previous[-1]


def ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0 if numerator == 0 else 1.0
    return numerator / denominator


def read_jsonl(path: str, *, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def is_true(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def avg(values) -> float | None:
    present = [value for value in values if value is not None]
    return None if not present else sum(present) / len(present)


def rate(values) -> float | None:
    present = [bool(value) for value in values]
    return None if not present else sum(1 for value in present if value) / len(present)


def percentile(values: list[float], pct: int) -> float | None:
    present = sorted(value for value in values if value is not None)
    if not present:
        return None
    if len(present) == 1:
        return present[0]
    rank = (len(present) - 1) * (pct / 100)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return present[int(rank)]
    return present[lower] + (present[upper] - present[lower]) * (rank - lower)


def fmt_pct(value: Any) -> str:
    return "n/a" if value is None else f"{float(value) * 100:.2f}%"


def fmt_num(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.0f}"


if __name__ == "__main__":
    main()
PY
}

run_action() {
  local action="$1"
  setup_env
  load_python_args
  run_python "$action" "${PY_ARGS[@]}"
}

write_python_runner() {
  local runner="$ASR_ML_PILOT_OUTPUT_DIR/multilingual_pilot_runner.py"
  python3 - "$0" "$runner" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

source = Path(sys.argv[1]).read_text(encoding="utf-8")
marker = "  python3 - \"$action\" \"$@\" <<'PY'\n"
start = source.index(marker) + len(marker)
end = source.index("\nPY\n}", start)
Path(sys.argv[2]).write_text(source[start:end] + "\n", encoding="utf-8")
PY
  printf "%s\n" "$runner"
}

wait_for_job_run() {
  local run_id="$1"
  local label="$2"
  local lifecycle=""
  local result=""
  local url=""
  local task_run_id=""

  log "waiting for Databricks $label run $run_id"
  while true; do
    local payload
    payload="$("${DBX[@]}" jobs get-run "$run_id" --output json)"
    lifecycle="$(python3 - "$payload" <<'PY'
import json, sys
payload = json.loads(sys.argv[1])
print((payload.get("state") or {}).get("life_cycle_state") or "")
PY
)"
    result="$(python3 - "$payload" <<'PY'
import json, sys
payload = json.loads(sys.argv[1])
print((payload.get("state") or {}).get("result_state") or "")
PY
)"
    url="$(python3 - "$payload" <<'PY'
import json, sys
payload = json.loads(sys.argv[1])
print(payload.get("run_page_url") or "")
PY
)"
    task_run_id="$(python3 - "$payload" <<'PY'
import json, sys
payload = json.loads(sys.argv[1])
tasks = payload.get("tasks") or []
if tasks:
    print(tasks[-1].get("run_id") or "")
PY
)"
    printf "  lifecycle=%s result=%s url=%s\n" "$lifecycle" "${result:-}" "$url"
    case "$lifecycle" in
      TERMINATED|SKIPPED|INTERNAL_ERROR)
        break
        ;;
    esac
    sleep 20
  done

  if [[ -n "$task_run_id" ]]; then
    "${DBX[@]}" jobs get-run-output "$task_run_id" --output json || true
  else
    "${DBX[@]}" jobs get-run-output "$run_id" --output json || true
  fi
  if [[ "$result" != "SUCCESS" ]]; then
    err "Databricks $label failed: lifecycle=$lifecycle result=$result url=$url"
    exit 2
  fi
}

submit_databricks_serverless() {
  local mode="${1:-standard}"
  setup_databricks_cli
  resolve_remote_root

  # Ensure a local scaffold manifest exists for first-time users. If the caller
  # points ASR_ML_PILOT_MANIFEST at a real manifest, that file is uploaded.
  if [[ ! -s "$ASR_ML_PILOT_MANIFEST" ]]; then
    log "manifest not found; generating scaffold manifest first"
    run_action scaffold
  fi
  if [[ "$mode" != "public-smoke" && "$ASR_ML_PILOT_DRY_RUN" != "true" ]]; then
    log "validating real pilot manifest before serverless model run"
    run_action prepare-real-pilot
  fi

  local runner_local
  runner_local="$(write_python_runner)"
  local remote_jobs_dir="$ASR_ML_PILOT_REMOTE_ROOT/jobs"
  local remote_inputs_dir="$ASR_ML_PILOT_REMOTE_ROOT/inputs"
  local remote_outputs_dir="$ASR_ML_PILOT_REMOTE_ROOT/outputs"
  local runner_remote="$remote_jobs_dir/multilingual_pilot_runner.py"
  local manifest_remote="$remote_inputs_dir/$(basename "$ASR_ML_PILOT_MANIFEST")"
  local results_remote="$remote_outputs_dir/multilingual_pilot_results.jsonl"
  local summary_remote="$remote_outputs_dir/multilingual_pilot_summary.json"
  local plan_remote="$remote_outputs_dir/multilingual_pilot_plan.json"
  local job_json="$ASR_ML_PILOT_OUTPUT_DIR/databricks_serverless_multilingual_pilot_job.json"
  local run_json="$ASR_ML_PILOT_OUTPUT_DIR/databricks_serverless_multilingual_pilot_run.json"

  "${DBX[@]}" fs mkdirs "dbfs:$remote_jobs_dir"
  "${DBX[@]}" fs mkdirs "dbfs:$remote_inputs_dir"
  "${DBX[@]}" fs mkdirs "dbfs:$remote_outputs_dir"
  "${DBX[@]}" fs cp "$runner_local" "dbfs:$runner_remote" --overwrite
  "${DBX[@]}" fs cp "$ASR_ML_PILOT_MANIFEST" "dbfs:$manifest_remote" --overwrite

  python3 - "$job_json" "$runner_remote" "$manifest_remote" "$results_remote" "$summary_remote" "$plan_remote" <<PY
import json
import sys
from pathlib import Path

job_json, runner_remote, manifest_remote, results_remote, summary_remote, plan_remote = sys.argv[1:]
common = [
    "--plan-output", plan_remote,
    "--scaffold-dir", "${ASR_ML_PILOT_REMOTE_ROOT}/scaffold",
    "--manifest", manifest_remote,
    "--results", results_remote,
    "--summary-output", summary_remote,
    "--dry-run", "${ASR_ML_PILOT_DRY_RUN}",
    "--language", "${ASR_ML_PILOT_LANGUAGE}",
    "--candidate", "${ASR_ML_PILOT_CANDIDATE}",
    "--limit", "${ASR_ML_PILOT_LIMIT}",
    "--public-limit", "${ASR_ML_PILOT_PUBLIC_LIMIT}",
    "--min-holdout", "${ASR_ML_PILOT_MIN_HOLDOUT}",
    "--min-train-smoke", "${ASR_ML_PILOT_MIN_TRAIN_SMOKE}",
]
public_smoke = "${mode}" == "public-smoke"
tasks = [
    {
        "task_key": "multilingual_pilot_plan",
        "environment_key": "asr_multilingual_pilot_env",
        "spark_python_task": {
            "python_file": f"dbfs:{runner_remote}",
            "parameters": ["plan", *common],
        },
    },
]
run_depends_on = "multilingual_pilot_plan"
run_action = "run-candidate" if "${ASR_ML_PILOT_CANDIDATE}" else "run"
if public_smoke:
    tasks.append(
        {
            "task_key": "bootstrap_public_smoke",
            "depends_on": [{"task_key": "multilingual_pilot_plan"}],
            "environment_key": "asr_multilingual_pilot_env",
            "spark_python_task": {
                "python_file": f"dbfs:{runner_remote}",
                "parameters": ["bootstrap-public-smoke", *common],
            },
        }
    )
    run_depends_on = "bootstrap_public_smoke"
tasks.extend(
    [
        {
            "task_key": "multilingual_pilot_run",
            "depends_on": [{"task_key": run_depends_on}],
            "environment_key": "asr_multilingual_pilot_env",
            "spark_python_task": {
                "python_file": f"dbfs:{runner_remote}",
                "parameters": [run_action, *common],
            },
        },
        {
            "task_key": "multilingual_pilot_summarize",
            "depends_on": [{"task_key": "multilingual_pilot_run"}],
            "environment_key": "asr_multilingual_pilot_env",
            "spark_python_task": {
                "python_file": f"dbfs:{runner_remote}",
                "parameters": ["summarize-existing", *common],
            },
        },
    ]
)
job = {
    "run_name": "genie-asr-multilingual-model-bakeoff",
    "tasks": tasks,
    "environments": [
        {
            "environment_key": "asr_multilingual_pilot_env",
            "spec": {
                "environment_version": "${ASR_ML_PILOT_SERVERLESS_ENVIRONMENT_VERSION}",
                "dependencies": [
                    "qwen-asr",
                    "torch",
                    "transformers",
                    "accelerate",
                    "huggingface_hub",
                    "pandas",
                    "pyarrow",
                    "soundfile"
                ],
            },
        }
    ],
}
Path(job_json).write_text(json.dumps(job, indent=2), encoding="utf-8")
PY

  log "submitting Databricks serverless multilingual pilot job"
  "${DBX[@]}" api post /api/2.1/jobs/runs/submit --json @"$job_json" --output json >"$run_json"
  local run_id
  run_id="$(python3 - "$run_json" <<'PY'
import json, sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["run_id"])
PY
)"
  wait_for_job_run "$run_id" "multilingual pilot"

  log "copying Databricks serverless outputs back locally"
  "${DBX[@]}" fs cp "dbfs:$results_remote" "$ASR_ML_PILOT_RESULTS" --overwrite
  "${DBX[@]}" fs cp "dbfs:$summary_remote" "$ASR_ML_PILOT_SUMMARY" --overwrite
  "${DBX[@]}" fs cp "dbfs:$plan_remote" "$ASR_ML_PILOT_PLAN" --overwrite

  cat <<EOF
Databricks serverless multilingual pilot completed.

Run JSON:
  $run_json
Results:
  $ASR_ML_PILOT_RESULTS
Summary:
  $ASR_ML_PILOT_SUMMARY
EOF
}

case "$COMMAND" in
  pilot)
    log "running complete safe multilingual pilot flow"
    run_action plan
    run_action scaffold
    run_action run
    run_action summarize-existing
    run_action preflight
    ;;
  databricks-serverless)
    log "submitting complete multilingual pilot to Databricks serverless"
    submit_databricks_serverless
    ;;
  databricks-public-smoke)
    log "submitting public FLEURS model-quality smoke test to Databricks serverless"
    ASR_ML_PILOT_DRY_RUN=false submit_databricks_serverless public-smoke
    ;;
  plan)
    log "writing multilingual pilot plan"
    run_action plan
    ;;
  scaffold)
    log "writing multilingual pilot scaffold"
    run_action scaffold
    ;;
  prepare-real-pilot)
    log "validating real multilingual pilot manifest"
    run_action prepare-real-pilot
    ;;
  preflight)
    log "running multilingual pilot preflight"
    run_action preflight
    ;;
  run)
    log "running multilingual pilot bake-off"
    run_action run
    ;;
  run-candidate)
    log "running one multilingual pilot candidate"
    run_action run-candidate
    ;;
  summarize-existing)
    log "summarizing multilingual pilot results"
    run_action summarize-existing
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
