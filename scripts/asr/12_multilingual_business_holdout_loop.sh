#!/usr/bin/env bash
# =============================================================================
# 12_multilingual_business_holdout_loop.sh
#
# Build the next multilingual ASR improvement loop:
#   - fixed business holdout packs, separate from training
#   - strict manifest validation for entity/action coverage
#   - base-vs-LoRA evaluation using the same evaluator and decision gate
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENGINE="$ROOT/scripts/asr/11_multilingual_asr_finetuning.sh"

ASR_BUSINESS_OUTPUT_DIR="${ASR_BUSINESS_OUTPUT_DIR:-$ROOT/.run/asr_model_training/multilingual_business_holdout}"
ASR_BUSINESS_LANGUAGES="${ASR_BUSINESS_LANGUAGES:-th id zh}"
ASR_BUSINESS_ROWS_PER_LANGUAGE="${ASR_BUSINESS_ROWS_PER_LANGUAGE:-30}"
ASR_BUSINESS_MIN_CLIPS_PER_LANGUAGE="${ASR_BUSINESS_MIN_CLIPS_PER_LANGUAGE:-30}"
ASR_BUSINESS_REQUIRE_APPROVED="${ASR_BUSINESS_REQUIRE_APPROVED:-true}"
ASR_BUSINESS_EVAL_SPLIT="${ASR_BUSINESS_EVAL_SPLIT:-holdout}"
ASR_BUSINESS_PROFILE="${ASR_BUSINESS_PROFILE:-${ASR_ML_FINETUNE_PROFILE:-${ASR_DATABRICKS_PROFILE:-${DATABRICKS_CONFIG_PROFILE:-fe-vm-vdm-classic-rcn6ip}}}}"
ASR_BUSINESS_REMOTE_ROOT="${ASR_BUSINESS_REMOTE_ROOT:-/Volumes/partner_demo_catalog/genie_voice_contact_center/raw_streaming_data/asr_model_training}"
ASR_BUSINESS_PROXY_UPLOAD="${ASR_BUSINESS_PROXY_UPLOAD:-true}"

COMMAND="${1:-help}"
if [[ $# -gt 0 ]]; then
  shift
fi

log() { printf "\033[36m[asr-business]\033[0m %s\n" "$*"; }
err() { printf "\033[31m[asr-business]\033[0m %s\n" "$*" >&2; }

usage() {
  cat <<'EOF'
Multilingual business ASR holdout and improvement loop.

Commands:
  scaffold     Write per-language business holdout CSV + JSONL templates.
  bootstrap-proxy
               Generate clearly labeled TTS/channel proxy holdout audio and manifests.
  validate     Validate built holdout manifests before evaluation.
  validate-proxy
               Validate proxy manifests without requiring human transcript approval.
  evaluate     Run base and latest LoRA evaluations on validated manifests.
  evaluate-proxy
               Run base and latest LoRA evaluations on proxy manifests.
  run          scaffold, then explain the next required data gate.
  help         Show this help.

Environment:
  ASR_BUSINESS_OUTPUT_DIR             Output directory for plans/manifests.
  ASR_BUSINESS_LANGUAGES              Space-separated languages. Default: th id zh.
  ASR_BUSINESS_ROWS_PER_LANGUAGE      Template rows per language. Default: 30.
  ASR_BUSINESS_MIN_CLIPS_PER_LANGUAGE Minimum validated holdout rows/language. Default: 30.
  ASR_BUSINESS_REQUIRE_APPROVED       Require human_transcript_approved=true. Default: true.
  ASR_BUSINESS_EVAL_SPLIT             Eval split passed to 11_. Default: holdout.
  ASR_BUSINESS_PROFILE                Databricks CLI profile.
  ASR_BUSINESS_REMOTE_ROOT            ASR training Volume root.
  ASR_BUSINESS_PROXY_UPLOAD           Upload proxy WAVs during bootstrap. Default: true.

Expected built manifest names:
  $ASR_BUSINESS_OUTPUT_DIR/th_business_holdout.built.jsonl
  $ASR_BUSINESS_OUTPUT_DIR/id_business_holdout.built.jsonl
  $ASR_BUSINESS_OUTPUT_DIR/zh_business_holdout.built.jsonl

Typical flow:
  scripts/asr/12_multilingual_business_holdout_loop.sh scaffold

  # Fill each CSV with externally sourced/local audio + approved transcript,
  # then materialize with 11_ or build a compatible JSONL manifest.

  scripts/asr/12_multilingual_business_holdout_loop.sh validate
  scripts/asr/12_multilingual_business_holdout_loop.sh evaluate

Decision gate:
  - LoRA must beat base on business entity/action accuracy.
  - LoRA must not materially regress FLEURS WER/CER.
  - If quality is tied, prefer the faster/simpler base model.
EOF
}

setup() {
  cd "$ROOT"
  mkdir -p "$ASR_BUSINESS_OUTPUT_DIR"
  DBX=(databricks --profile "$ASR_BUSINESS_PROFILE")
}

candidate_for_language() {
  case "$1" in
    th) printf '%s\n' "th_finetuned_pathumma_whisper_lora" ;;
    id) printf '%s\n' "id_finetuned_whisper_large_v3_lora" ;;
    zh) printf '%s\n' "zh_finetuned_whisper_large_v3_lora" ;;
    *) err "Unsupported language: $1"; return 2 ;;
  esac
}

scaffold() {
  setup
  "$python_bin" - "$ASR_BUSINESS_OUTPUT_DIR" "$ASR_BUSINESS_ROWS_PER_LANGUAGE" "$ASR_BUSINESS_LANGUAGES" "$ASR_BUSINESS_REMOTE_ROOT" <<'PY'
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
rows_per_language = int(sys.argv[2])
languages = sys.argv[3].split()
catalog_root = sys.argv[4]

templates = {
    "th": {
        "display": "Thai",
        "accent": "thai_central",
        "account_term": "ใบแจ้งหนี้",
        "phrases": [
            ("billing_dispute", "สวัสดีค่ะ ฉันโทรมาโต้แย้งใบแจ้งหนี้ {invoice} ยอด {amount} ดอลลาร์ เพราะยอดไม่ถูกต้อง", "dispute", "", ""),
            ("payment_lookup", "ช่วยตรวจสอบการชำระเงินของใบแจ้งหนี้ {invoice} จำนวน {amount} ดอลลาร์ให้หน่อย", "payment", "", ""),
            ("payment_confirmation", "ฉันต้องการยืนยันว่าใบแจ้งหนี้ {invoice} ถูกชำระแล้ว", "payment", "ยืนยัน", ""),
            ("charge_refusal", "ฉันขอปฏิเสธค่าบริการ {amount} ดอลลาร์ในใบแจ้งหนี้ {invoice}", "charge", "", "ปฏิเสธ"),
            ("account_balance", "บัญชีของฉันยังแสดงยอดค้างชำระสำหรับใบแจ้งหนี้ {invoice}", "balance", "", ""),
            ("refund_request", "ช่วยขอคืนเงินสำหรับใบแจ้งหนี้ {invoice} ยอด {amount} ดอลลาร์", "refund", "", ""),
            ("payment_extension", "ฉันต้องการขยายเวลาชำระเงินสำหรับใบแจ้งหนี้ {invoice}", "extension", "", ""),
            ("case_close", "ถ้าใบแจ้งหนี้ {invoice} ได้รับการแก้ไขแล้ว ช่วยปิดเคสนี้", "close", "", ""),
        ],
    },
    "id": {
        "display": "Indonesian",
        "accent": "indonesian_jakarta",
        "account_term": "tagihan",
        "phrases": [
            ("billing_dispute", "Halo saya ingin mengajukan keberatan untuk tagihan {invoice} sebesar {amount} dolar karena jumlahnya salah", "dispute", "", ""),
            ("payment_lookup", "Tolong periksa pembayaran untuk tagihan {invoice} sejumlah {amount} dolar", "payment", "", ""),
            ("payment_confirmation", "Saya ingin konfirmasi bahwa tagihan {invoice} sudah dibayar", "payment", "konfirmasi", ""),
            ("charge_refusal", "Saya menolak biaya {amount} dolar pada tagihan {invoice}", "charge", "", "menolak"),
            ("account_balance", "Akun saya masih menunjukkan saldo tertunggak untuk tagihan {invoice}", "balance", "", ""),
            ("refund_request", "Saya ingin meminta pengembalian dana untuk tagihan {invoice} sebesar {amount} dolar", "refund", "", ""),
            ("payment_extension", "Saya perlu perpanjangan waktu pembayaran untuk tagihan {invoice}", "extension", "", ""),
            ("case_close", "Kalau tagihan {invoice} sudah selesai, tolong tutup kasus ini", "close", "", ""),
        ],
    },
    "zh": {
        "display": "Mandarin Chinese",
        "accent": "mandarin_mainland",
        "account_term": "发票",
        "phrases": [
            ("billing_dispute", "您好 我想对发票 {invoice} 的 {amount} 美元金额提出争议 因为金额不对", "dispute", "", ""),
            ("payment_lookup", "请帮我查询发票 {invoice} 的 {amount} 美元付款状态", "payment", "", ""),
            ("payment_confirmation", "我想确认发票 {invoice} 已经付款", "payment", "确认", ""),
            ("charge_refusal", "我拒绝发票 {invoice} 上的 {amount} 美元费用", "charge", "", "拒绝"),
            ("account_balance", "我的账户仍然显示发票 {invoice} 有未付余额", "balance", "", ""),
            ("refund_request", "我想申请发票 {invoice} 的 {amount} 美元退款", "refund", "", ""),
            ("payment_extension", "我需要延期支付发票 {invoice}", "extension", "", ""),
            ("case_close", "如果发票 {invoice} 已经解决 请关闭这个工单", "close", "", ""),
        ],
    },
}

fieldnames = [
    "clip_id",
    "language",
    "split",
    "scenario",
    "external_audio_url",
    "local_audio_path",
    "approved_reference_transcript",
    "suggested_audio_path",
    "duration_seconds",
    "invoice_ids",
    "amounts",
    "billing_actions",
    "confirmations",
    "refusals",
    "account_terms",
    "recording_channel",
    "accent",
    "background_noise",
    "human_transcript_approved",
    "notes",
]

for language in languages:
    cfg = templates[language]
    csv_path = out_dir / f"{language}_business_holdout_plan.csv"
    jsonl_path = out_dir / f"{language}_business_holdout.template.jsonl"
    csv_rows = []
    json_rows = []
    for idx in range(1, rows_per_language + 1):
        scenario, phrase, action, confirmation, refusal = cfg["phrases"][(idx - 1) % len(cfg["phrases"])]
        invoice = f"INV-{language.upper()}{idx + 90000:05d}"
        amount = str(25 + (idx * 11) % 170)
        amount_entity = f"{amount} dollars" if "{amount}" in phrase else ""
        transcript = phrase.format(invoice=invoice, amount=amount)
        clip_id = f"{language}_business_holdout_{idx:04d}"
        suggested_audio_path = (
            f"{catalog_root}/datasets/multilingual_gold/audio/{language}/holdout/{clip_id}.wav"
        )
        row = {
            "clip_id": clip_id,
            "language": language,
            "split": "holdout",
            "scenario": scenario,
            "external_audio_url": "",
            "local_audio_path": "",
            "approved_reference_transcript": transcript,
            "suggested_audio_path": suggested_audio_path,
            "duration_seconds": "",
            "invoice_ids": invoice,
            "amounts": amount_entity,
            "billing_actions": action,
            "confirmations": confirmation,
            "refusals": refusal,
            "account_terms": cfg["account_term"],
            "recording_channel": "phone_or_browser",
            "accent": cfg["accent"],
            "background_noise": "realistic_low_to_medium",
            "human_transcript_approved": "false",
            "notes": (
                "Use externally sourced or recorded audio with rights to use. "
                "Set local_audio_path, duration_seconds, and human_transcript_approved=true only after transcript review."
            ),
        }
        csv_rows.append(row)
        json_rows.append(
            {
                "clip_id": clip_id,
                "audio_path": "REPLACE_WITH_VOLUME_AUDIO_PATH",
                "audio_format": "audio/wav",
                "sample_rate_hz": 16000,
                "duration_seconds": 0,
                "reference_transcript": transcript,
                "language": language,
                "split": "holdout",
                "scenario": scenario,
                "speaker": "external_or_recorded",
                "domain": "billing_support",
                "dataset_version": "multilingual_business_holdout_v1",
                "expected_entities": {
                    "invoice_ids": [invoice],
                    "amounts": [amount_entity] if amount_entity else [],
                    "billing_actions": [action],
                    "confirmations": [confirmation] if confirmation else [],
                    "refusals": [refusal] if refusal else [],
                    "account_terms": [cfg["account_term"]],
                },
                "metadata": {
                    "source": "external_or_recorded_business_holdout",
                    "recording_channel": "phone_or_browser",
                    "accent": cfg["accent"],
                    "background_noise": "realistic_low_to_medium",
                    "human_transcript_approved": False,
                    "training_exclusion": True,
                },
            }
        )
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in json_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{cfg['display']}: {csv_path}")
    print(f"{cfg['display']}: {jsonl_path}")
PY
}

bootstrap_proxy() {
  setup
  scaffold
  if ! command -v say >/dev/null 2>&1; then
    err "macOS 'say' is unavailable; cannot generate proxy audio."
    exit 2
  fi
  if ! command -v afconvert >/dev/null 2>&1; then
    err "macOS 'afconvert' is unavailable; cannot convert proxy audio to WAV."
    exit 2
  fi

  local upload_tsv="$ASR_BUSINESS_OUTPUT_DIR/proxy_uploads.tsv"
  "$python_bin" - "$ASR_BUSINESS_OUTPUT_DIR" "$ASR_BUSINESS_LANGUAGES" "$ASR_BUSINESS_REMOTE_ROOT" "$upload_tsv" <<'PY'
from __future__ import annotations

import csv
import json
import math
import random
import struct
import subprocess
import sys
import wave
from pathlib import Path

out_dir = Path(sys.argv[1])
languages = sys.argv[2].split()
remote_root = sys.argv[3]
upload_tsv = Path(sys.argv[4])
audio_root = out_dir / "proxy_audio"

locale_prefixes = {
    "th": ("th_", "th-"),
    "id": ("id_", "id-"),
    "zh": ("zh_", "zh-", "cmn_", "cmn-"),
}


def installed_voices() -> list[tuple[str, str]]:
    result = subprocess.run(["say", "-v", "?"], check=True, text=True, capture_output=True)
    voices = []
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        voices.append((parts[0], parts[1].lower()))
    return voices


def voice_for(language: str, voices: list[tuple[str, str]]) -> str:
    prefixes = locale_prefixes[language]
    for name, locale in voices:
        if any(locale.startswith(prefix) for prefix in prefixes):
            return name
    raise SystemExit(f"No macOS say voice found for {language}. Install one or reduce ASR_BUSINESS_LANGUAGES.")


def duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        return round(wav.getnframes() / float(wav.getframerate()), 3)


def add_proxy_channel_noise(path: Path, language: str, idx: int) -> None:
    with wave.open(str(path), "rb") as src:
        params = src.getparams()
        frames = src.readframes(src.getnframes())
    if params.sampwidth != 2:
        return
    rng = random.Random(f"{language}-{idx}")
    samples = list(struct.unpack("<" + "h" * (len(frames) // 2), frames))
    degraded = []
    last = 0
    noise_level = 180 + (idx % 5) * 35
    for sample in samples:
        # Light phone-like smoothing plus deterministic low background noise.
        smoothed = int((sample * 0.82) + (last * 0.18))
        last = sample
        noisy = smoothed + rng.randint(-noise_level, noise_level)
        degraded.append(max(-32768, min(32767, noisy)))
    packed = struct.pack("<" + "h" * len(degraded), *degraded)
    with wave.open(str(path), "wb") as dst:
        dst.setparams(params)
        dst.writeframes(packed)


def list_field(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split("|") if item.strip()]


voices = installed_voices()
uploads = []
for language in languages:
    voice = voice_for(language, voices)
    plan_path = out_dir / f"{language}_business_holdout_plan.csv"
    built_manifest = out_dir / f"{language}_business_holdout.built.jsonl"
    rows = []
    with plan_path.open("r", encoding="utf-8", newline="") as f:
        for idx, row in enumerate(csv.DictReader(f), start=1):
            clip_id = row["clip_id"]
            prompt = row["approved_reference_transcript"]
            local_dir = audio_root / language
            local_dir.mkdir(parents=True, exist_ok=True)
            aiff_path = local_dir / f"{clip_id}.aiff"
            wav_path = local_dir / f"{clip_id}.wav"
            if not wav_path.exists():
                subprocess.run(["say", "-v", voice, "-o", str(aiff_path), prompt], check=True)
                subprocess.run(["afconvert", str(aiff_path), str(wav_path), "-f", "WAVE", "-d", "LEI16@16000"], check=True)
                aiff_path.unlink(missing_ok=True)
                add_proxy_channel_noise(wav_path, language, idx)
            remote_audio = f"{remote_root}/datasets/multilingual_gold/audio/{language}/holdout_proxy/{clip_id}.wav"
            uploads.append((str(wav_path), remote_audio))
            rows.append(
                {
                    "clip_id": clip_id,
                    "audio_path": remote_audio,
                    "audio_format": "audio/wav",
                    "sample_rate_hz": 16000,
                    "duration_seconds": duration_seconds(wav_path),
                    "reference_transcript": prompt,
                    "language": language,
                    "split": "holdout",
                    "scenario": row["scenario"],
                    "speaker": "proxy_tts",
                    "domain": "billing_support",
                    "dataset_version": "multilingual_business_proxy_tts_channel_v1",
                    "expected_entities": {
                        "invoice_ids": list_field(row["invoice_ids"]),
                        "amounts": list_field(row["amounts"]),
                        "billing_actions": list_field(row["billing_actions"]),
                        "confirmations": list_field(row["confirmations"]),
                        "refusals": list_field(row["refusals"]),
                        "account_terms": list_field(row["account_terms"]),
                    },
                    "metadata": {
                        "source": "proxy_tts_channel_holdout",
                        "synthetic_audio_source": "macos_say",
                        "say_voice": voice,
                        "recording_channel": "proxy_phone_or_browser",
                        "accent": row["accent"],
                        "background_noise": "deterministic_low_noise",
                        "human_transcript_approved": False,
                        "proxy_holdout": True,
                        "training_exclusion": True,
                        "replace_with_real_or_licensed_external_audio_before_production": True,
                    },
                }
            )
    with built_manifest.open("w", encoding="utf-8") as f:
        for item in rows:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"{language}: wrote {built_manifest} with voice={voice} rows={len(rows)}")

with upload_tsv.open("w", encoding="utf-8") as f:
    for local_path, remote_path in uploads:
        f.write(f"{local_path}\t{remote_path}\n")
print(f"uploads: {upload_tsv}")
PY

  if [[ "$ASR_BUSINESS_PROXY_UPLOAD" == "true" ]]; then
    while IFS=$'\t' read -r local_path remote_path; do
      [[ -n "$local_path" && -n "$remote_path" ]] || continue
      "${DBX[@]}" fs mkdirs "dbfs:${remote_path%/*}" >/dev/null
      "${DBX[@]}" fs cp "$local_path" "dbfs:$remote_path" --overwrite >/dev/null
    done < "$upload_tsv"
  else
    log "skipping proxy WAV upload because ASR_BUSINESS_PROXY_UPLOAD=false"
  fi

  cat <<EOF

Proxy business holdout bootstrapped.

Important:
  This is a proxy TTS/channel holdout, not real or licensed external speech.
  Use it for model/debug iteration only; production confidence still requires
  approved real or licensed external audio.

Next:
  ASR_BUSINESS_REQUIRE_APPROVED=false scripts/asr/12_multilingual_business_holdout_loop.sh validate
  scripts/asr/12_multilingual_business_holdout_loop.sh evaluate-proxy

EOF
}

validate() {
  setup
  "$python_bin" - "$ASR_BUSINESS_OUTPUT_DIR" "$ASR_BUSINESS_LANGUAGES" "$ASR_BUSINESS_MIN_CLIPS_PER_LANGUAGE" "$ASR_BUSINESS_REQUIRE_APPROVED" <<'PY'
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

out_dir = Path(sys.argv[1])
languages = sys.argv[2].split()
min_rows = int(sys.argv[3])
require_approved = sys.argv[4].lower() == "true"

required_groups = ("invoice_ids", "billing_actions", "account_terms")
failed = False
for language in languages:
    path = out_dir / f"{language}_business_holdout.built.jsonl"
    if not path.exists():
        print(f"ERROR {language}: missing {path}", file=sys.stderr)
        failed = True
        continue
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        rows.append(row)
        prefix = f"{path}:{line_no}"
        if row.get("language") != language:
            print(f"ERROR {prefix}: expected language={language}", file=sys.stderr)
            failed = True
        if row.get("split") != "holdout":
            print(f"ERROR {prefix}: expected split=holdout", file=sys.stderr)
            failed = True
        for key in ("clip_id", "audio_path", "reference_transcript", "duration_seconds"):
            if not row.get(key):
                print(f"ERROR {prefix}: missing {key}", file=sys.stderr)
                failed = True
        if str(row.get("audio_path", "")).startswith("REPLACE_"):
            print(f"ERROR {prefix}: placeholder audio_path", file=sys.stderr)
            failed = True
        entities = row.get("expected_entities") or {}
        for group in required_groups:
            if not entities.get(group):
                print(f"ERROR {prefix}: missing expected_entities.{group}", file=sys.stderr)
                failed = True
        metadata = row.get("metadata") or {}
        if require_approved and metadata.get("human_transcript_approved") is not True:
            print(f"ERROR {prefix}: human_transcript_approved must be true", file=sys.stderr)
            failed = True
        if metadata.get("training_exclusion") is not True:
            print(f"ERROR {prefix}: holdout rows must set metadata.training_exclusion=true", file=sys.stderr)
            failed = True
    if len(rows) < min_rows:
        print(f"ERROR {language}: {len(rows)} rows < required {min_rows}", file=sys.stderr)
        failed = True
    amount_rows = sum(1 for row in rows if (row.get("expected_entities") or {}).get("amounts"))
    if amount_rows == 0:
        print(f"ERROR {language}: no rows with expected amount entities", file=sys.stderr)
        failed = True
    scenarios = Counter(str(row.get("scenario") or "") for row in rows)
    print(
        json.dumps(
            {"language": language, "rows": len(rows), "amount_rows": amount_rows, "scenarios": scenarios},
            ensure_ascii=False,
        )
    )

if failed:
    raise SystemExit(2)
PY
}

evaluate() {
  setup
  validate
  local language candidate manifest
  for language in $ASR_BUSINESS_LANGUAGES; do
    candidate="$(candidate_for_language "$language")"
    manifest="$ASR_BUSINESS_OUTPUT_DIR/${language}_business_holdout.built.jsonl"
    log "evaluating base model for $language using $manifest"
    ASR_ML_FINETUNE_CANDIDATE="$candidate" \
      ASR_ML_FINETUNE_MANIFEST="$manifest" \
      ASR_ML_FINETUNE_EVAL_LABEL="${ASR_BUSINESS_EVAL_LABEL:-}" \
      ASR_ML_FINETUNE_EVAL_SPLIT="$ASR_BUSINESS_EVAL_SPLIT" \
      "$ENGINE" evaluate-base

    log "evaluating latest LoRA model for $language using $manifest"
    ASR_ML_FINETUNE_CANDIDATE="$candidate" \
      ASR_ML_FINETUNE_MANIFEST="$manifest" \
      ASR_ML_FINETUNE_EVAL_LABEL="${ASR_BUSINESS_EVAL_LABEL:-}" \
      ASR_ML_FINETUNE_EVAL_SPLIT="$ASR_BUSINESS_EVAL_SPLIT" \
      "$ENGINE" evaluate-lora
  done
}

python_bin() {
  if [[ -x "$ROOT/.venv/bin/python" ]]; then
    printf '%s\n' "$ROOT/.venv/bin/python"
  else
    printf '%s\n' "python3"
  fi
}

python_bin="$(python_bin)"

case "$COMMAND" in
  scaffold)
    scaffold
    ;;
  bootstrap-proxy)
    bootstrap_proxy
    ;;
  validate)
    validate
    ;;
  validate-proxy)
    ASR_BUSINESS_REQUIRE_APPROVED=false validate
    ;;
  evaluate)
    evaluate
    ;;
  evaluate-proxy)
    ASR_BUSINESS_EVAL_LABEL=business_proxy ASR_BUSINESS_REQUIRE_APPROVED=false evaluate
    ;;
  run)
    scaffold
    cat <<EOF

Business holdout templates are ready.

Next required gate:
  Fill each *_business_holdout_plan.csv with usable external/local audio,
  approved transcripts, durations, and human_transcript_approved=true.

Then build compatible manifests:
  $ASR_BUSINESS_OUTPUT_DIR/th_business_holdout.built.jsonl
  $ASR_BUSINESS_OUTPUT_DIR/id_business_holdout.built.jsonl
  $ASR_BUSINESS_OUTPUT_DIR/zh_business_holdout.built.jsonl

Finally run:
  scripts/asr/12_multilingual_business_holdout_loop.sh validate
  scripts/asr/12_multilingual_business_holdout_loop.sh evaluate

EOF
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
