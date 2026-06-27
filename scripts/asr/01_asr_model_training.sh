#!/usr/bin/env bash
# =============================================================================
# 01_asr_model_training.sh
#
# One ASR model-training command to remember:
#   scripts/asr/01_asr_model_training.sh
#
# Run it with no arguments and it tells you the single next command.
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="$ROOT/.venv"
EXAMPLE_MANIFEST="$ROOT/docs/asr_model_training_manifest.example.jsonl"
LOCAL_TRAINING_DIR="$ROOT/.run/asr_model_training"
DEFAULT_MANIFEST="$LOCAL_TRAINING_DIR/manifests/asr_training_gold_v1.jsonl"
DEFAULT_DEEPGRAM_OUTPUT="$ROOT/.run/asr_model_training/baselines/deepgram_nova3_baseline.jsonl"
DEFAULT_WHISPER_OUTPUT="$ROOT/.run/asr_model_training/baselines/whisper_small_baseline.jsonl"
DEFAULT_OUTPUT="$DEFAULT_DEEPGRAM_OUTPUT"

COMMAND="${1:-run}"
if [[ $# -gt 0 ]]; then
  shift
fi

MANIFEST="$DEFAULT_MANIFEST"
OUTPUT="$DEFAULT_OUTPUT"
INPUT=""
MODEL="nova-3"
LANGUAGE="en-US"
LIMIT=""
AUDIO_SOURCE="${ASR_BASELINE_AUDIO_SOURCE:-volume}"
SYNC_RESULTS="${ASR_SYNC_BASELINE_RESULTS:-true}"
WHISPER_CLUSTER_ID="${ASR_WHISPER_EXISTING_CLUSTER_ID:-}"
ASR_GPU_CLUSTER_NAME="${ASR_GPU_CLUSTER_NAME:-genie-asr-gpu-training}"
ASR_GPU_NODE_TYPE="${ASR_GPU_NODE_TYPE:-g4dn.xlarge}"
ASR_GPU_RUNTIME="${ASR_GPU_RUNTIME:-16.4.x-gpu-ml-scala2.13}"
POSTPROCESS_INVOICE_IDS="false"
SPLITS=()

log() { printf "\033[35m[asr-training]\033[0m %s\n" "$*"; }
warn() { printf "\033[33m[asr-training]\033[0m %s\n" "$*"; }
err() { printf "\033[31m[asr-training]\033[0m %s\n" "$*" >&2; }

usage() {
  cat <<EOF
ASR model-training workflow

If you only remember one thing, run:

  scripts/asr/01_asr_model_training.sh

Commands:
  run          Run the safe repeatable workflow. Default when no command is given.
  next         Show the one command to run next.
  volume       Show the Databricks UC Volume paths for ASR model training.
  prepare      Create the ASR model-training UC Volume folder layout.
  validate     Validate the model-training manifest and scoring locally.
  augment      Generate and upload augmented billing-domain audio.
  deepgram     Run Deepgram baseline and write JSONL results.
  whisper      Run Whisper baseline and write JSONL results.
  whisper-db   Run Whisper baseline on the dedicated ASR GPU cluster.
  rescore      Rescore an existing baseline JSONL with current metrics.
  gpu-status   Show the dedicated ASR GPU cluster status.
  gpu-start    Create or start the dedicated ASR GPU cluster.
  gpu-stop     Stop the dedicated ASR GPU cluster after training.
  summarize    Summarize the JSONL results.
  all          Run validate, Deepgram baseline, and summarize.
  help         Show this help.

Defaults:
  manifest: $DEFAULT_MANIFEST
  output:   $DEFAULT_OUTPUT
  model:    nova-3 for deepgram, openai/whisper-small.en for whisper
  language: en-US
  audio:    volume
  sync:     true

Examples:
  scripts/asr/01_asr_model_training.sh
EOF
}

show_next_step() {
  cat <<EOF
ASR model-training next step

Run this now:

  scripts/asr/01_asr_model_training.sh

What it does:
  - creates the Databricks UC Volume folders for ASR model-training data
  - uses Common Voice from the external_raw Volume folder if present
  - otherwise downloads LibriSpeech dev-clean from OpenSLR into the Volume
  - ingests a small external speech sample for acoustic diversity
  - then generates 100+ billing/contact-center utterances from this app's scenarios
  - adds noisy and phone-band augmented billing clips for training robustness
  - creates a local training manifest if missing
  - syncs that manifest to the Databricks Volume
  - validates the manifest
  - checks whether the manifest audio files exist
  - uses a training-oriented layout: datasets/gold, baselines, evaluations, model_artifacts
  - keeps ASR model-training files separate from existing Genie Voice Agent paths
  - only runs the Deepgram baseline when audio files are present
  - does not train a model yet
  - does not change app/runtime behavior
  - does not delete or overwrite existing files

Current defaults:
  manifest: $DEFAULT_MANIFEST
  output:   $DEFAULT_OUTPUT

EOF
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --manifest)
        MANIFEST="$2"
        shift 2
        ;;
      --output)
        OUTPUT="$2"
        shift 2
        ;;
      --input)
        INPUT="$2"
        shift 2
        ;;
      --model)
        MODEL="$2"
        shift 2
        ;;
      --language)
        LANGUAGE="$2"
        shift 2
        ;;
      --limit)
        LIMIT="$2"
        shift 2
        ;;
      --audio-source)
        AUDIO_SOURCE="$2"
        shift 2
        ;;
      --sync-results)
        SYNC_RESULTS="$2"
        shift 2
        ;;
      --postprocess-invoice-ids)
        POSTPROCESS_INVOICE_IDS="true"
        shift
        ;;
      --cluster-id|--existing-cluster-id)
        WHISPER_CLUSTER_ID="$2"
        shift 2
        ;;
      --cluster-name)
        ASR_GPU_CLUSTER_NAME="$2"
        shift 2
        ;;
      --split)
        SPLITS+=("$2")
        shift 2
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        err "Unknown argument: $1"
        usage
        exit 2
        ;;
    esac
  done
}

setup_env() {
  cd "$ROOT"
  if [[ -f "$ROOT/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$ROOT/.env"
    set +a
  fi
  if [[ ! -d "$VENV_DIR" ]]; then
    log "creating virtualenv at $VENV_DIR"
    python3 -m venv "$VENV_DIR"
  fi
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  export PYTHONPATH="$ROOT/backend${PYTHONPATH:+:$PYTHONPATH}"
  if ! python -c "import genie_voice" >/dev/null 2>&1; then
    log "installing backend package"
    python -m pip install -q --upgrade pip
    python -m pip install -q -e "$ROOT/backend"
  fi
}

resolve_training_volume_paths() {
  eval "$(python - <<'PY'
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
    "ASR_TRAINING_EXTERNAL_RAW": f"{root}/external_raw",
    "ASR_TRAINING_COMMON_VOICE_RAW": f"{root}/external_raw/common_voice",
    "ASR_TRAINING_LIBRISPEECH_RAW": f"{root}/external_raw/librispeech",
    "ASR_TRAINING_GOLD_AUDIO": f"{root}/datasets/gold/audio",
    "ASR_TRAINING_GOLD_MANIFESTS": f"{root}/datasets/gold/manifests",
    "ASR_TRAINING_BASELINES": f"{root}/baselines",
    "ASR_TRAINING_EVALUATIONS": f"{root}/evaluations",
    "ASR_TRAINING_MODEL_ARTIFACTS": f"{root}/model_artifacts",
}
for key, value in paths.items():
    print(f"{key}={shlex.quote(value)}")
PY
)"
}

setup_databricks_cli() {
  if ! command -v databricks >/dev/null 2>&1; then
    err "Databricks CLI is not installed or not on PATH."
    exit 1
  fi

  DBX=(databricks)
  local profile="${ASR_DATABRICKS_PROFILE:-${DATABRICKS_CONFIG_PROFILE:-}}"
  if [[ -z "$profile" ]]; then
    profile="$(python - <<'PY'
import json
import subprocess

preferred = "fe-vm-vdm-classic-rcn6ip"
try:
    p = subprocess.run(
        ["databricks", "auth", "profiles", "--output", "json"],
        text=True,
        capture_output=True,
        timeout=30,
    )
    data = json.loads(p.stdout or "{}") if p.returncode == 0 else {}
except Exception:
    data = {}

profiles = [p for p in data.get("profiles", []) if p.get("valid")]
names = [p.get("name") for p in profiles if p.get("name")]
if preferred in names:
    print(preferred)
elif len(names) == 1:
    print(names[0])
PY
)"
  fi

  if [[ -n "$profile" ]]; then
    DBX=(databricks --profile "$profile")
    export DATABRICKS_CONFIG_PROFILE="$profile"
    export ASR_DATABRICKS_PROFILE="$profile"
    export DATABRICKS_AUTH_STORAGE="${DATABRICKS_AUTH_STORAGE:-plaintext}"
    log "using Databricks profile: $profile"
  else
    log "using Databricks default auth"
  fi
}

show_volume_paths() {
  resolve_training_volume_paths
  cat <<EOF
ASR model-training Databricks UC Volume layout

Root:
  $ASR_TRAINING_ROOT

Folders:
  external raw:    $ASR_TRAINING_EXTERNAL_RAW
  common voice:    $ASR_TRAINING_COMMON_VOICE_RAW
  librispeech:      $ASR_TRAINING_LIBRISPEECH_RAW
  gold audio:      $ASR_TRAINING_GOLD_AUDIO
  gold manifests:  $ASR_TRAINING_GOLD_MANIFESTS
  baselines:       $ASR_TRAINING_BASELINES
  evaluations:     $ASR_TRAINING_EVALUATIONS
  model artifacts: $ASR_TRAINING_MODEL_ARTIFACTS

These paths are isolated under asr_model_training and do not change existing Genie Voice Agent tables or app behavior.
EOF
}

prepare_volume_layout() {
  resolve_training_volume_paths
  setup_databricks_cli
  log "creating ASR model-training UC Volume folders if missing"
  "${DBX[@]}" fs mkdirs "dbfs:$ASR_TRAINING_COMMON_VOICE_RAW"
  "${DBX[@]}" fs mkdirs "dbfs:$ASR_TRAINING_LIBRISPEECH_RAW"
  "${DBX[@]}" fs mkdirs "dbfs:$ASR_TRAINING_GOLD_AUDIO"
  "${DBX[@]}" fs mkdirs "dbfs:$ASR_TRAINING_GOLD_MANIFESTS"
  "${DBX[@]}" fs mkdirs "dbfs:$ASR_TRAINING_BASELINES/deepgram"
  "${DBX[@]}" fs mkdirs "dbfs:$ASR_TRAINING_BASELINES/whisper"
  "${DBX[@]}" fs mkdirs "dbfs:$ASR_TRAINING_EVALUATIONS"
  "${DBX[@]}" fs mkdirs "dbfs:$ASR_TRAINING_MODEL_ARTIFACTS"
  show_volume_paths
}

ensure_training_manifest() {
  resolve_training_volume_paths
  mkdir -p "$(dirname "$MANIFEST")"
  if [[ ! -f "$MANIFEST" ]]; then
    log "creating local ASR training manifest: $MANIFEST"
    python - "$EXAMPLE_MANIFEST" "$MANIFEST" "$ASR_TRAINING_ROOT" <<'PY'
import sys
from pathlib import Path

example = Path(sys.argv[1])
target = Path(sys.argv[2])
training_root = sys.argv[3]
text = example.read_text(encoding="utf-8")
text = text.replace(
    "/Volumes/<catalog>/<schema>/raw_streaming_data/asr_model_training",
    training_root,
)
target.write_text(text, encoding="utf-8")
PY
  else
    log "using existing local ASR training manifest: $MANIFEST"
  fi
}

sync_manifest_to_volume() {
  resolve_training_volume_paths
  setup_databricks_cli
  local volume_manifest="$ASR_TRAINING_GOLD_MANIFESTS/$(basename "$MANIFEST")"
  log "syncing manifest to Databricks Volume"
  "${DBX[@]}" fs cp "$MANIFEST" "dbfs:$volume_manifest" --overwrite
  log "volume manifest: $volume_manifest"
}

ingest_common_voice_if_present() {
  resolve_training_volume_paths
  if ! command -v afconvert >/dev/null 2>&1; then
    err "afconvert is unavailable; cannot convert Common Voice MP3 clips to WAV."
    return 1
  fi

  local cv_root="$LOCAL_TRAINING_DIR/external/common_voice"
  local raw_dir="$cv_root/raw"
  local extract_dir="$cv_root/extracted"
  local wav_dir="$cv_root/wav"
  local limit="${ASR_COMMON_VOICE_LIMIT:-20}"
  mkdir -p "$raw_dir" "$extract_dir" "$wav_dir"

  setup_databricks_cli
  local remote_listing
  remote_listing="$("${DBX[@]}" fs ls "dbfs:$ASR_TRAINING_COMMON_VOICE_RAW" 2>/dev/null || true)"
  local archive_name
  archive_name="$(REMOTE_LISTING="$remote_listing" python - <<'PY'
import os

names = [line.strip().split()[-1] for line in os.environ.get("REMOTE_LISTING", "").splitlines() if line.strip()]
candidates = sorted(
    name for name in names
    if (name.startswith("cv-corpus") or name.startswith("common_voice")) and name.endswith(".tar.gz")
)
print(candidates[-1] if candidates else "")
PY
)"
  if [[ -z "$archive_name" ]]; then
    warn "Common Voice archive not found in $ASR_TRAINING_COMMON_VOICE_RAW."
    return 2
  fi

  local archive="$raw_dir/$archive_name"
  if [[ ! -f "$archive" ]]; then
    log "copying Common Voice archive from Databricks Volume"
    "${DBX[@]}" fs cp "dbfs:$ASR_TRAINING_COMMON_VOICE_RAW/$archive_name" "$archive" --overwrite
  fi

  log "using Common Voice archive: $archive"
  if [[ -z "$(ls -A "$extract_dir" 2>/dev/null || true)" ]]; then
    log "extracting Common Voice archive"
    tar -xzf "$archive" -C "$extract_dir"
  fi

  local tsv
  tsv="$(python - "$extract_dir" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
matches = sorted(root.glob("**/en/validated.tsv")) or sorted(root.glob("**/validated.tsv"))
print(matches[0] if matches else "")
PY
)"
  if [[ -z "$tsv" ]]; then
    warn "No validated.tsv found in Common Voice archive; skipping ingest."
    return 0
  fi

  log "ingesting up to $limit Common Voice validated clips"
  python - "$MANIFEST" "$tsv" "$wav_dir" "$ASR_TRAINING_GOLD_AUDIO" "$limit" <<'PY'
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
tsv = Path(sys.argv[2])
wav_dir = Path(sys.argv[3])
volume_audio_dir = sys.argv[4].rstrip("/")
limit = int(sys.argv[5])

existing = []
seen_ids = set()
if manifest.exists():
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        row = json.loads(line)
        existing.append(row)
        seen_ids.add(row["clip_id"])

clips_dir = tsv.parent / "clips"
added = 0
with tsv.open("r", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        if added >= limit:
            break
        rel_path = row.get("path") or ""
        sentence = (row.get("sentence") or "").strip()
        if not rel_path or not sentence:
            continue
        source = clips_dir / rel_path
        if not source.exists():
            continue
        stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(rel_path).stem)
        clip_id = f"CV-EN-{stem}"
        if clip_id in seen_ids:
            continue
        wav_path = wav_dir / f"{clip_id}.wav"
        if not wav_path.exists():
            subprocess.run(
                ["afconvert", str(source), str(wav_path), "-f", "WAVE", "-d", "LEI16@16000"],
                check=True,
            )
        existing.append(
            {
                "clip_id": clip_id,
                "call_id": None,
                "speaker": "common_voice",
                "audio_path": f"{volume_audio_dir}/{clip_id}.wav",
                "audio_format": "audio/wav",
                "sample_rate_hz": 16000,
                "duration_seconds": None,
                "domain": "general_speech",
                "scenario": "mozilla_common_voice_validated",
                "split": "train",
                "dataset_version": "common_voice_bootstrap_v1",
                "reference_transcript": sentence,
                "expected_entities": {
                    "invoice_ids": [],
                    "amounts": [],
                    "dates": [],
                    "billing_actions": [],
                    "confirmations": [],
                    "refusals": [],
                    "account_terms": [],
                },
                "external_dataset": "mozilla_common_voice",
                "replace_with_real_call_audio_before_training": False,
            }
        )
        seen_ids.add(clip_id)
        added += 1

manifest.write_text(
    "\n".join(json.dumps(row, ensure_ascii=False) for row in existing) + "\n",
    encoding="utf-8",
)
print(added)
PY

  setup_databricks_cli
  local uploaded=0
  for audio_file in "$wav_dir"/*.wav; do
    [[ -e "$audio_file" ]] || continue
    "${DBX[@]}" fs cp "$audio_file" "dbfs:$ASR_TRAINING_GOLD_AUDIO/$(basename "$audio_file")" --overwrite
    uploaded=$((uploaded + 1))
  done
  log "Common Voice WAV files uploaded: $uploaded"
  return 0
}

ingest_librispeech_available() {
  resolve_training_volume_paths
  if ! command -v afconvert >/dev/null 2>&1; then
    err "afconvert is unavailable; cannot convert LibriSpeech FLAC clips to WAV."
    return 1
  fi

  local ls_root="$LOCAL_TRAINING_DIR/external/librispeech"
  local raw_dir="$ls_root/raw"
  local extract_dir="$ls_root/extracted"
  local wav_dir="$ls_root/wav"
  local archive_name="dev-clean.tar.gz"
  local archive="$raw_dir/$archive_name"
  local limit="${ASR_LIBRISPEECH_LIMIT:-20}"
  local url="${ASR_LIBRISPEECH_URL:-https://www.openslr.org/resources/12/dev-clean.tar.gz}"
  mkdir -p "$raw_dir" "$extract_dir" "$wav_dir"

  setup_databricks_cli
  if [[ ! -f "$archive" ]]; then
    if "${DBX[@]}" fs ls "dbfs:$ASR_TRAINING_LIBRISPEECH_RAW/$archive_name" >/dev/null 2>&1; then
      log "copying LibriSpeech archive from Databricks Volume"
      "${DBX[@]}" fs cp "dbfs:$ASR_TRAINING_LIBRISPEECH_RAW/$archive_name" "$archive" --overwrite
    else
      log "downloading LibriSpeech dev-clean from OpenSLR"
      python - "$url" "$archive" <<'PY'
import sys
import urllib.request
from pathlib import Path

url = sys.argv[1]
target = Path(sys.argv[2])
target.parent.mkdir(parents=True, exist_ok=True)
urllib.request.urlretrieve(url, target)
PY
      log "syncing raw LibriSpeech archive to Databricks Volume"
      "${DBX[@]}" fs cp "$archive" "dbfs:$ASR_TRAINING_LIBRISPEECH_RAW/$archive_name" --overwrite
    fi
  fi

  if [[ -z "$(ls -A "$extract_dir" 2>/dev/null || true)" ]]; then
    log "extracting LibriSpeech archive"
    tar -xzf "$archive" -C "$extract_dir"
  fi

  log "ingesting up to $limit LibriSpeech clips"
  python - "$MANIFEST" "$extract_dir" "$wav_dir" "$ASR_TRAINING_GOLD_AUDIO" "$limit" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
extract_dir = Path(sys.argv[2])
wav_dir = Path(sys.argv[3])
volume_audio_dir = sys.argv[4].rstrip("/")
limit = int(sys.argv[5])

existing = []
seen_ids = set()
if manifest.exists():
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        row = json.loads(line)
        existing.append(row)
        seen_ids.add(row["clip_id"])

added = 0
for trans_file in sorted(extract_dir.glob("**/*.trans.txt")):
    with trans_file.open("r", encoding="utf-8") as f:
        for line in f:
            if added >= limit:
                break
            parts = line.strip().split(" ", 1)
            if len(parts) != 2:
                continue
            utt_id, transcript = parts
            clip_id = f"LS-{utt_id}"
            if clip_id in seen_ids:
                continue
            flac = trans_file.parent / f"{utt_id}.flac"
            if not flac.exists():
                continue
            wav_path = wav_dir / f"{clip_id}.wav"
            if not wav_path.exists():
                subprocess.run(
                    ["afconvert", str(flac), str(wav_path), "-f", "WAVE", "-d", "LEI16@16000"],
                    check=True,
                )
            existing.append(
                {
                    "clip_id": clip_id,
                    "call_id": None,
                    "speaker": "librispeech",
                    "audio_path": f"{volume_audio_dir}/{clip_id}.wav",
                    "audio_format": "audio/wav",
                    "sample_rate_hz": 16000,
                    "duration_seconds": None,
                    "domain": "general_speech",
                    "scenario": "openslr_librispeech_dev_clean",
                    "split": "train",
                    "dataset_version": "librispeech_dev_clean_bootstrap_v1",
                    "reference_transcript": transcript,
                    "expected_entities": {
                        "invoice_ids": [],
                        "amounts": [],
                        "dates": [],
                        "billing_actions": [],
                        "confirmations": [],
                        "refusals": [],
                        "account_terms": [],
                    },
                    "external_dataset": "openslr_librispeech_dev_clean",
                    "license": "CC-BY-4.0",
                    "replace_with_real_call_audio_before_training": False,
                }
            )
            seen_ids.add(clip_id)
            added += 1
    if added >= limit:
        break

manifest.write_text(
    "\n".join(json.dumps(row, ensure_ascii=False) for row in existing) + "\n",
    encoding="utf-8",
)
print(added)
PY

  setup_databricks_cli
  local uploaded=0
  while IFS= read -r audio_file; do
    [[ -e "$audio_file" ]] || continue
    "${DBX[@]}" fs cp "$audio_file" "dbfs:$ASR_TRAINING_GOLD_AUDIO/$(basename "$audio_file")" --overwrite
    uploaded=$((uploaded + 1))
  done < <(
    python - "$MANIFEST" "$wav_dir" <<'PY'
import json
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
wav_dir = Path(sys.argv[2])
for line in manifest.read_text(encoding="utf-8").splitlines():
    if not line.strip() or line.lstrip().startswith("#"):
        continue
    row = json.loads(line)
    if row.get("dataset_version") != "librispeech_dev_clean_bootstrap_v1":
        continue
    print(wav_dir / f"{row['clip_id']}.wav")
PY
  )
  log "LibriSpeech WAV files uploaded: $uploaded"
}

generate_domain_billing_audio() {
  resolve_training_volume_paths
  if ! command -v say >/dev/null 2>&1; then
    warn "macOS 'say' is unavailable; skipping billing-domain audio generation."
    return 0
  fi

  local local_audio_dir="$LOCAL_TRAINING_DIR/datasets/gold/audio"
  mkdir -p "$local_audio_dir"

  local limit="${ASR_BILLING_UTTERANCE_LIMIT:-120}"
  log "generating up to $limit billing-domain utterance WAV files from app scenarios"
  python - "$MANIFEST" "$local_audio_dir" "$ASR_TRAINING_GOLD_AUDIO" "$limit" <<'PY'
import json
import re
import subprocess
import sys
from pathlib import Path

from genie_voice.datagen import build_dataset

manifest = Path(sys.argv[1])
local_audio_dir = Path(sys.argv[2])
volume_audio_dir = sys.argv[3].rstrip("/")
limit = int(sys.argv[4])

rows = []
seen_ids = set()
if manifest.exists():
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        row = json.loads(line)
        rows.append(row)
        seen_ids.add(row["clip_id"])

def entities_for(text: str) -> dict[str, list[str]]:
    invoice_ids = sorted(set(re.findall(r"\bINV-\d+\b", text, flags=re.I)))
    amounts = sorted(set(re.findall(r"\$\d[\d,]*(?:\.\d{2})?", text)))
    dates = sorted(set(re.findall(
        r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}\b",
        text,
    )))
    lower = text.lower()
    billing_actions = []
    if "payment arrangement" in lower:
        billing_actions.append("payment arrangement")
    if "payment plan" in lower:
        billing_actions.append("payment plan")
    if "waive" in lower or "waived" in lower:
        billing_actions.append("waive")
    if "late fee" in lower:
        billing_actions.append("late fee")
    confirmations = []
    for phrase in ("yes", "that works", "okay", "thank you", "got it", "sure"):
        if re.search(rf"\b{re.escape(phrase)}\b", lower):
            confirmations.append(phrase)
    refusals = []
    for phrase in ("no", "do not", "don't"):
        if re.search(rf"\b{re.escape(phrase)}\b", lower):
            refusals.append(phrase)
    account_terms = []
    for phrase in ("invoice", "account", "autopay", "card", "payment", "refund", "charged", "balance", "overdue"):
        if phrase in lower:
            account_terms.append(phrase)
    return {
        "invoice_ids": invoice_ids,
        "amounts": amounts,
        "dates": dates,
        "billing_actions": sorted(set(billing_actions)),
        "confirmations": sorted(set(confirmations)),
        "refusals": sorted(set(refusals)),
        "account_terms": sorted(set(account_terms)),
    }

added = 0
dataset = build_dataset()
for call in dataset.calls:
    for utterance in call.get("utterances", []):
        if added >= limit:
            break
        role = utterance.get("speaker_role") or "unknown"
        turn_index = int(utterance.get("turn_index", 0))
        text = str(utterance.get("text") or "").strip()
        if not text:
            continue
        clip_id = f"BILLING-{call['call_id']}-U{turn_index:02d}"
        if clip_id in seen_ids:
            continue
        entities = entities_for(text)
        local_audio = local_audio_dir / f"{clip_id}.wav"
        temp_aiff = local_audio_dir / f"{clip_id}.aiff"
        if not local_audio.exists():
            subprocess.run(["say", "-o", str(temp_aiff), text], check=True)
            subprocess.run(
                ["afconvert", str(temp_aiff), str(local_audio), "-f", "WAVE", "-d", "LEI16@16000"],
                check=True,
            )
        rows.append(
            {
                "clip_id": clip_id,
                "call_id": call["call_id"],
                "speaker": role,
                "audio_path": f"{volume_audio_dir}/{clip_id}.wav",
                "audio_format": "audio/wav",
                "sample_rate_hz": 16000,
                "duration_seconds": None,
                "domain": "billing_support",
                "scenario": call.get("primary_intent") or "generated_contact_center",
                "split": "train" if added % 10 else "validation",
                "dataset_version": "billing_scenarios_synthetic_v1",
                "reference_transcript": text,
                "expected_entities": entities,
                "synthetic_audio_source": "macos_say",
                "source_generator": "genie_voice.datagen",
                "replace_with_real_call_audio_before_training": True,
            }
        )
        seen_ids.add(clip_id)
        added += 1
    if added >= limit:
        break

manifest.write_text(
    "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
    encoding="utf-8",
)
print(added)
PY

  setup_databricks_cli
  log "uploading billing-domain audio to Databricks Volume"
  local uploaded=0
  for audio_file in "$local_audio_dir"/BILLING-*.wav; do
    [[ -e "$audio_file" ]] || continue
    "${DBX[@]}" fs cp "$audio_file" "dbfs:$ASR_TRAINING_GOLD_AUDIO/$(basename "$audio_file")" --overwrite
    uploaded=$((uploaded + 1))
  done
  log "billing-domain audio files uploaded: $uploaded"
}

generate_augmented_billing_audio() {
  resolve_training_volume_paths

  local local_audio_dir="$LOCAL_TRAINING_DIR/datasets/gold/audio"
  mkdir -p "$local_audio_dir"

  local limit="${ASR_BILLING_AUGMENT_SOURCE_LIMIT:-60}"
  log "generating noisy and phone-band augmentations for up to $limit billing clips"
  python - "$MANIFEST" "$local_audio_dir" "$ASR_TRAINING_GOLD_AUDIO" "$limit" <<'PY'
import json
import random
import struct
import sys
import wave
from pathlib import Path

manifest = Path(sys.argv[1])
local_audio_dir = Path(sys.argv[2])
volume_audio_dir = sys.argv[3].rstrip("/")
limit = int(sys.argv[4])

rows = []
seen_ids = set()
for line in manifest.read_text(encoding="utf-8").splitlines():
    if not line.strip() or line.lstrip().startswith("#"):
        continue
    row = json.loads(line)
    rows.append(row)
    seen_ids.add(row["clip_id"])

def read_wav(path: Path) -> tuple[list[int], int]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())
    if channels != 1 or sample_width != 2:
        raise ValueError(f"Expected mono 16-bit WAV: {path}")
    return list(struct.unpack(f"<{len(frames) // 2}h", frames)), rate

def write_wav(path: Path, samples: list[int], rate: int) -> None:
    clipped = [max(-32768, min(32767, int(sample))) for sample in samples]
    frames = struct.pack(f"<{len(clipped)}h", *clipped)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(frames)

def add_noise(samples: list[int], seed: str) -> list[int]:
    rng = random.Random(seed)
    return [sample + rng.randint(-450, 450) for sample in samples]

def phone_band(samples: list[int]) -> list[int]:
    # Cheap telephony simulation: downsample to 8 kHz, upsample back to 16 kHz,
    # and reduce amplitude. This keeps the training file contract unchanged.
    downsampled = [(samples[i] + samples[i + 1]) // 2 for i in range(0, len(samples) - 1, 2)]
    upsampled = []
    for sample in downsampled:
        reduced = int(sample * 0.82)
        upsampled.extend([reduced, reduced])
    if len(upsampled) < len(samples):
        upsampled.append(upsampled[-1] if upsampled else 0)
    return upsampled[: len(samples)]

def augmented_row(source: dict, variant: str, clip_id: str) -> dict:
    row = dict(source)
    row["clip_id"] = clip_id
    row["audio_path"] = f"{volume_audio_dir}/{clip_id}.wav"
    row["split"] = "train"
    row["dataset_version"] = "billing_scenarios_augmented_v1"
    row["augmentation"] = variant
    row["source_clip_id"] = source["clip_id"]
    row["synthetic_audio_source"] = "macos_say_augmented"
    row["replace_with_real_call_audio_before_training"] = True
    return row

added = 0
source_rows = [
    row
    for row in rows
    if row.get("dataset_version") == "billing_scenarios_synthetic_v1"
]
for source in source_rows[:limit]:
    source_audio = local_audio_dir / f"{source['clip_id']}.wav"
    if not source_audio.exists():
        continue
    samples, rate = read_wav(source_audio)
    variants = {
        "noise": add_noise(samples, source["clip_id"]),
        "phone_band": phone_band(samples),
    }
    for variant, augmented_samples in variants.items():
        clip_id = f"{source['clip_id']}-AUG-{variant.upper().replace('_', '-')}"
        if clip_id in seen_ids:
            continue
        write_wav(local_audio_dir / f"{clip_id}.wav", augmented_samples, rate)
        rows.append(augmented_row(source, variant, clip_id))
        seen_ids.add(clip_id)
        added += 1

manifest.write_text(
    "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
    encoding="utf-8",
)
print(added)
PY

  setup_databricks_cli
  log "uploading augmented billing audio to Databricks Volume"
  local uploaded=0
  while IFS= read -r audio_file; do
    [[ -e "$audio_file" ]] || continue
    "${DBX[@]}" fs cp "$audio_file" "dbfs:$ASR_TRAINING_GOLD_AUDIO/$(basename "$audio_file")" --overwrite
    uploaded=$((uploaded + 1))
  done < <(
    python - "$MANIFEST" "$local_audio_dir" <<'PY'
import json
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
local_audio_dir = Path(sys.argv[2])
for line in manifest.read_text(encoding="utf-8").splitlines():
    if not line.strip() or line.lstrip().startswith("#"):
        continue
    row = json.loads(line)
    if row.get("dataset_version") != "billing_scenarios_augmented_v1":
        continue
    print(local_audio_dir / f"{row['clip_id']}.wav")
PY
  )
  log "augmented billing audio files uploaded: $uploaded"
}

missing_manifest_audio() {
  local missing=0
  while IFS= read -r audio_path; do
    [[ -z "$audio_path" ]] && continue
    local uri="$audio_path"
    if [[ "$uri" == /Volumes/* ]]; then
      uri="dbfs:$uri"
    fi
    if [[ "$uri" == dbfs:* ]]; then
      local parent="${uri%/*}"
      local name="${uri##*/}"
      local listing
      listing="$("${DBX[@]}" fs ls "$parent" 2>/dev/null || true)"
      if [[ "$listing" != *"$name"* ]]; then
        printf "%s\n" "$audio_path"
        missing=1
      fi
    elif [[ ! -f "$uri" ]]; then
      printf "%s\n" "$audio_path"
      missing=1
    fi
  done < <(
    python - "$MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    if not line.strip() or line.lstrip().startswith("#"):
        continue
    print(json.loads(line)["audio_path"])
PY
  )
  return "$missing"
}

run_repeatable_workflow() {
  setup_env
  prepare_volume_layout
  ensure_training_manifest
  local common_voice_status=0
  ingest_common_voice_if_present || common_voice_status=$?
  if [[ "$common_voice_status" == "2" ]]; then
    ingest_librispeech_available
  elif [[ "$common_voice_status" != "0" ]]; then
    return "$common_voice_status"
  fi
  generate_domain_billing_audio
  generate_augmented_billing_audio
  sync_manifest_to_volume
  validate_manifest

  log "checking manifest audio files"
  local missing
  missing="$(missing_manifest_audio || true)"
  if [[ -n "$missing" ]]; then
    cat <<EOF

ASR model-training is prepared. Baseline is paused because the manifest audio files are not present yet.

Put the listed audio files in the Databricks Volume path below, or edit the manifest to point at the right files:

  $ASR_TRAINING_GOLD_AUDIO

Manifest to edit locally:

  $MANIFEST

Missing audio paths:
$missing

Then rerun the same command:

  scripts/asr/01_asr_model_training.sh

EOF
    return 0
  fi

  run_deepgram
  summarize_results
}

validate_manifest() {
  log "validating model-training manifest: $MANIFEST"
  python -m compileall -q "$ROOT/backend/genie_voice/asr_eval"
  python - "$MANIFEST" <<'PY'
import json
import sys

from genie_voice.asr_eval.manifest import load_manifest
from genie_voice.asr_eval.metrics import score_transcript

manifest = sys.argv[1]
clips = load_manifest(manifest)
if not clips:
    raise SystemExit(f"Manifest has no clips: {manifest}")

scores = [
    score_transcript(c.reference_transcript, c.reference_transcript, c.expected_entities)
    for c in clips
]
summary = {
    "manifest": manifest,
    "clips": len(clips),
    "splits": sorted({c.split for c in clips if c.split}),
    "dataset_versions": sorted({c.dataset_version for c in clips if c.dataset_version}),
    "smoke_avg_wer": sum(s.wer for s in scores) / len(scores),
    "smoke_avg_cer": sum(s.cer for s in scores) / len(scores),
}
print(json.dumps(summary, indent=2))
PY
}

baseline_manifest_for_audio_source() {
  case "$AUDIO_SOURCE" in
    volume)
      printf "%s\n" "$MANIFEST"
      ;;
    local-cache)
      local local_manifest="$LOCAL_TRAINING_DIR/manifests/$(basename "${MANIFEST%.jsonl}").local_audio.jsonl"
      mkdir -p "$(dirname "$local_manifest")"
      python - "$MANIFEST" "$local_manifest" "$LOCAL_TRAINING_DIR" <<'PY'
import json
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
output = Path(sys.argv[2])
training_dir = Path(sys.argv[3])
local_audio_dirs = [
    training_dir / "datasets/gold/audio",
    training_dir / "external/librispeech/wav",
]
rows = []
missing = []
for line in manifest.read_text(encoding="utf-8").splitlines():
    if not line.strip() or line.lstrip().startswith("#"):
        continue
    row = json.loads(line)
    name = Path(row["audio_path"]).name
    local_path = None
    for directory in local_audio_dirs:
        candidate = directory / name
        if candidate.exists():
            local_path = candidate
            break
    if local_path is None:
        missing.append(row["audio_path"])
    else:
        row["audio_path"] = str(local_path)
    rows.append(row)
if missing:
    preview = "\n".join(missing[:20])
    suffix = f"\n... {len(missing)} missing total" if len(missing) > 20 else ""
    raise SystemExit(f"Missing local cached audio files:\n{preview}{suffix}")
output.write_text(
    "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
    encoding="utf-8",
)
print(output)
PY
      ;;
    *)
      err "Unknown --audio-source: $AUDIO_SOURCE (expected volume or local-cache)"
      exit 2
      ;;
  esac
}

should_sync_results() {
  case "$SYNC_RESULTS" in
    true|1|yes)
      return 0
      ;;
    false|0|no)
      return 1
      ;;
    *)
      err "Unknown --sync-results: $SYNC_RESULTS (expected true or false)"
      exit 2
      ;;
  esac
}

run_deepgram() {
  if [[ "$OUTPUT" == "$DEFAULT_WHISPER_OUTPUT" ]]; then
    OUTPUT="$DEFAULT_DEEPGRAM_OUTPUT"
  fi
  setup_databricks_cli
  local baseline_manifest
  baseline_manifest="$(baseline_manifest_for_audio_source)"
  mkdir -p "$(dirname "$OUTPUT")"
  log "running Deepgram baseline"
  log "manifest: $baseline_manifest"
  log "audio source: $AUDIO_SOURCE"
  log "output: $OUTPUT"
  log "model: $MODEL"

  local args=(
    --manifest "$baseline_manifest"
    --output "$OUTPUT"
    --model "$MODEL"
    --language "$LANGUAGE"
  )
  if [[ -n "$LIMIT" ]]; then
    args+=(--limit "$LIMIT")
  fi
  if ((${#SPLITS[@]} > 0)); then
    for split in "${SPLITS[@]}"; do
      args+=(--split "$split")
    done
  fi
  python -m genie_voice.asr_eval.deepgram_baseline "${args[@]}"
  if ! should_sync_results; then
    warn "skipping Deepgram baseline sync because --sync-results=false"
    return 0
  fi
  resolve_training_volume_paths
  setup_databricks_cli
  log "syncing Deepgram baseline results to Databricks Volume"
  "${DBX[@]}" fs cp "$OUTPUT" "dbfs:$ASR_TRAINING_BASELINES/deepgram/$(basename "$OUTPUT")" --overwrite
}

run_whisper() {
  if [[ "$MODEL" == "nova-3" ]]; then
    MODEL="openai/whisper-small.en"
  fi
  if [[ "$LANGUAGE" == "en-US" ]]; then
    LANGUAGE="english"
  fi
  if [[ "$OUTPUT" == "$DEFAULT_DEEPGRAM_OUTPUT" ]]; then
    OUTPUT="$DEFAULT_WHISPER_OUTPUT"
  fi

  mkdir -p "$(dirname "$OUTPUT")"
  log "running Whisper baseline"
  log "manifest: $MANIFEST"
  log "output: $OUTPUT"
  log "model: $MODEL"

  local args=(
    --manifest "$MANIFEST"
    --output "$OUTPUT"
    --model "$MODEL"
    --language "$LANGUAGE"
  )
  if [[ -n "$LIMIT" ]]; then
    args+=(--limit "$LIMIT")
  fi
  if ((${#SPLITS[@]} > 0)); then
    for split in "${SPLITS[@]}"; do
      args+=(--split "$split")
    done
  fi
  python -m genie_voice.asr_eval.whisper_baseline "${args[@]}"
  if ! should_sync_results; then
    warn "skipping Whisper baseline sync because --sync-results=false"
    return 0
  fi
  resolve_training_volume_paths
  setup_databricks_cli
  log "syncing Whisper baseline results to Databricks Volume"
  "${DBX[@]}" fs cp "$OUTPUT" "dbfs:$ASR_TRAINING_BASELINES/whisper/$(basename "$OUTPUT")" --overwrite
}

find_asr_gpu_cluster() {
  local clusters_json
  clusters_json="$("${DBX[@]}" clusters list --output json)"
  python - "$ASR_GPU_CLUSTER_NAME" "$clusters_json" <<'PY'
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

show_asr_gpu_cluster_status() {
  setup_databricks_cli
  local cluster_id="${WHISPER_CLUSTER_ID:-}"
  if [[ -z "$cluster_id" ]]; then
    cluster_id="$(find_asr_gpu_cluster)"
  fi
  if [[ -z "$cluster_id" ]]; then
    cat <<EOF
No dedicated ASR GPU cluster found.

Name:
  $ASR_GPU_CLUSTER_NAME

Create/start it with:
  scripts/asr/01_asr_model_training.sh gpu-start
EOF
    return 0
  fi
  "${DBX[@]}" clusters get "$cluster_id" --output json
}

ensure_asr_gpu_cluster() {
  setup_databricks_cli

  local cluster_id="${WHISPER_CLUSTER_ID:-}"
  if [[ -z "$cluster_id" ]]; then
    cluster_id="$(find_asr_gpu_cluster)"
  fi

  if [[ -z "$cluster_id" ]]; then
    local single_user
    single_user="$("${DBX[@]}" current-user me --output json | python -c 'import json,sys; print(json.load(sys.stdin)["userName"])')"

    local cluster_json
    cluster_json="$(mktemp)"
    python - "$cluster_json" "$ASR_GPU_CLUSTER_NAME" "$ASR_GPU_RUNTIME" "$ASR_GPU_NODE_TYPE" "$single_user" <<'PY'
import json
import sys
from pathlib import Path

cluster_json, name, runtime, node_type, single_user = sys.argv[1:]
payload = {
    "cluster_name": name,
    "spark_version": runtime,
    "node_type_id": node_type,
    "num_workers": 0,
    "autotermination_minutes": 0,
    "spark_conf": {
        "spark.databricks.cluster.profile": "singleNode",
        "spark.master": "local[*]",
    },
    "custom_tags": {
        "ResourceClass": "SingleNode",
        "Purpose": "genie-asr-model-training",
    },
    "data_security_mode": "SINGLE_USER",
    "single_user_name": single_user,
}
Path(cluster_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY
    log "creating dedicated ASR GPU cluster: $ASR_GPU_CLUSTER_NAME ($ASR_GPU_NODE_TYPE)"
    cluster_id="$("${DBX[@]}" clusters create --json @"$cluster_json" --output json | python -c 'import json,sys; print(json.load(sys.stdin)["cluster_id"])')"
    rm -f "$cluster_json"
    WHISPER_CLUSTER_ID="$cluster_id"
    log "dedicated ASR GPU cluster ready: $WHISPER_CLUSTER_ID"
    return 0
  fi

  local state
  state="$("${DBX[@]}" clusters get "$cluster_id" --output json | python -c 'import json,sys; print(json.load(sys.stdin).get("state", ""))')"
  if [[ "$state" == "TERMINATED" ]]; then
    log "starting dedicated ASR GPU cluster: $cluster_id"
    "${DBX[@]}" clusters start "$cluster_id" --output json >/dev/null
  elif [[ "$state" == "TERMINATING" ]]; then
    err "Cluster $cluster_id is terminating. Wait for TERMINATED, then rerun gpu-start."
    return 2
  else
    log "dedicated ASR GPU cluster is $state: $cluster_id"
  fi
  WHISPER_CLUSTER_ID="$cluster_id"
}

stop_asr_gpu_cluster() {
  setup_databricks_cli
  local cluster_id="${WHISPER_CLUSTER_ID:-}"
  if [[ -z "$cluster_id" ]]; then
    cluster_id="$(find_asr_gpu_cluster)"
  fi
  if [[ -z "$cluster_id" ]]; then
    warn "No dedicated ASR GPU cluster found to stop: $ASR_GPU_CLUSTER_NAME"
    return 0
  fi
  log "stopping dedicated ASR GPU cluster: $cluster_id"
  "${DBX[@]}" clusters delete "$cluster_id" --output json >/dev/null
}

run_databricks_whisper() {
  if [[ "$MODEL" == "nova-3" ]]; then
    MODEL="openai/whisper-small.en"
  fi
  if [[ "$LANGUAGE" == "en-US" ]]; then
    LANGUAGE="english"
  fi
  if [[ -z "$LIMIT" ]]; then
    LIMIT="${ASR_WHISPER_DATABRICKS_LIMIT:-20}"
  fi

  resolve_training_volume_paths
  setup_databricks_cli
  ensure_asr_gpu_cluster

  local runner_local="$ROOT/scripts/asr/databricks_whisper_baseline.py"
  local runner_remote="$ASR_TRAINING_MODEL_ARTIFACTS/jobs/databricks_whisper_baseline.py"
  local output_name="whisper_small_databricks_smoke_${LIMIT}_baseline.jsonl"
  if [[ "$LIMIT" == "all" || "$LIMIT" == "0" ]]; then
    output_name="whisper_small_databricks_full_baseline.jsonl"
  fi
  local output_remote="$ASR_TRAINING_BASELINES/whisper/$output_name"
  local manifest_remote="$ASR_TRAINING_GOLD_MANIFESTS/$(basename "$MANIFEST")"

  log "using dedicated ASR GPU cluster for Whisper: $WHISPER_CLUSTER_ID"

  log "uploading Databricks Whisper runner"
  "${DBX[@]}" fs mkdirs "dbfs:$ASR_TRAINING_MODEL_ARTIFACTS/jobs"
  "${DBX[@]}" fs cp "$runner_local" "dbfs:$runner_remote" --overwrite
  sync_manifest_to_volume

  local single_user
  single_user="$("${DBX[@]}" current-user me --output json | python -c 'import json,sys; print(json.load(sys.stdin)["userName"])')"

  local job_json
  job_json="$(mktemp)"
  python - "$job_json" "$runner_remote" "$manifest_remote" "$output_remote" "$MODEL" "$LANGUAGE" "$LIMIT" "$single_user" "$WHISPER_CLUSTER_ID" <<'PY'
import json
import sys
from pathlib import Path

(
    job_json,
    runner_remote,
    manifest_remote,
    output_remote,
    model,
    language,
    limit,
    single_user,
    existing_cluster_id,
) = sys.argv[1:]

cluster_config = {
    "existing_cluster_id": existing_cluster_id,
} if existing_cluster_id else {
    "new_cluster": {
        "spark_version": "16.4.x-gpu-ml-scala2.13",
        "node_type_id": "g4dn.xlarge",
        "num_workers": 0,
        "spark_conf": {
            "spark.databricks.cluster.profile": "singleNode",
            "spark.master": "local[*]",
        },
        "custom_tags": {"ResourceClass": "SingleNode"},
        "data_security_mode": "SINGLE_USER",
        "single_user_name": single_user,
    },
}

parameters = [
    "--manifest",
    manifest_remote,
    "--output",
    output_remote,
    "--model",
    model,
    "--language",
    language,
]
if limit not in {"all", "0"}:
    parameters.extend(["--limit", str(limit)])

payload = {
    "run_name": "genie-asr-whisper-baseline-smoke",
    "tasks": [
        {
            "task_key": "whisper_baseline",
            "spark_python_task": {
                "python_file": f"dbfs:{runner_remote}",
                "parameters": parameters,
            },
            **cluster_config,
            "libraries": [
                {"pypi": {"package": "transformers"}},
                {"pypi": {"package": "accelerate"}},
                {"pypi": {"package": "librosa"}},
                {"pypi": {"package": "soundfile"}},
            ],
        }
    ],
}
Path(job_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY

  log "submitting Databricks Whisper GPU smoke job"
  "${DBX[@]}" jobs submit --json @"$job_json" --output json
  rm -f "$job_json"
  cat <<EOF

Whisper Databricks smoke job submitted.

Expected output:
  $output_remote

Limit:
  $LIMIT

For a full manifest run, use:
  scripts/asr/02_asr_baseline_runs.sh whisper-full

EOF
}

summarize_results() {
  log "summarizing results: $OUTPUT"
  python - "$OUTPUT" <<'PY'
import json
import sys
from collections import defaultdict
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    raise SystemExit(f"Results file does not exist: {path}")

rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
if not rows:
    raise SystemExit(f"Results file has no rows: {path}")

def avg(values):
    values = [v for v in values if v is not None]
    return None if not values else sum(values) / len(values)

entity_totals = defaultdict(lambda: {"expected": 0, "matched": 0})
for row in rows:
    for group, score in (row.get("score", {}).get("entity_scores") or {}).items():
        entity_totals[group]["expected"] += score.get("expected") or 0
        entity_totals[group]["matched"] += score.get("matched") or 0

summary = {
    "results": str(path),
    "clips": len(rows),
    "providers": sorted({row.get("provider") for row in rows if row.get("provider")}),
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
print(json.dumps(summary, indent=2))
PY
}

run_rescore() {
  if [[ -z "$INPUT" ]]; then
    err "rescore requires --input <baseline-jsonl>"
    exit 2
  fi
  log "rescoring baseline results"
  log "manifest: $MANIFEST"
  log "input: $INPUT"
  log "output: $OUTPUT"
  local postprocess_args=()
  if [[ "$POSTPROCESS_INVOICE_IDS" == "true" ]]; then
    postprocess_args+=(--postprocess-invoice-ids)
  fi
  python -m genie_voice.asr_eval.rescore_results \
    --manifest "$MANIFEST" \
    --input "$INPUT" \
    --output "$OUTPUT" \
    "${postprocess_args[@]}"
}

parse_args "$@"

case "$COMMAND" in
  run)
    run_repeatable_workflow
    ;;
  next)
    show_next_step
    ;;
  volume)
    setup_env
    show_volume_paths
    ;;
  prepare)
    setup_env
    prepare_volume_layout
    ;;
  validate)
    setup_env
    validate_manifest
    ;;
  augment)
    setup_env
    prepare_volume_layout
    generate_augmented_billing_audio
    sync_manifest_to_volume
    validate_manifest
    ;;
  deepgram)
    setup_env
    run_deepgram
    ;;
  whisper)
    setup_env
    run_whisper
    ;;
  whisper-db)
    setup_env
    run_databricks_whisper
    ;;
  rescore)
    setup_env
    run_rescore
    ;;
  gpu-status)
    setup_env
    show_asr_gpu_cluster_status
    ;;
  gpu-start)
    setup_env
    ensure_asr_gpu_cluster
    ;;
  gpu-stop)
    setup_env
    stop_asr_gpu_cluster
    ;;
  summarize)
    setup_env
    summarize_results
    ;;
  all)
    setup_env
    validate_manifest
    run_deepgram
    summarize_results
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    err "Unknown command: $COMMAND"
    usage
    exit 2
    ;;
esac
