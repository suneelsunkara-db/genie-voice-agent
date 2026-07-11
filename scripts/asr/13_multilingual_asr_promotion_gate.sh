#!/usr/bin/env bash
# =============================================================================
# 13_multilingual_asr_promotion_gate.sh
#
# Read-only multilingual ASR promotion gate.
#
# Compares existing base/OSS and LoRA evaluation artifacts and writes an explicit
# decision report. It does NOT register models, deploy endpoints, or edit app
# config. Promotion is intentionally conservative: if the evidence is incomplete
# or mixed, the decision is no-promotion.
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

ASR_PROMOTION_OUTPUT_DIR="${ASR_PROMOTION_OUTPUT_DIR:-$ROOT/.run/asr_model_training/promotion_gate}"
ASR_PROMOTION_REMOTE_ROOT="${ASR_PROMOTION_REMOTE_ROOT:-}"
ASR_PROMOTION_ENTITY_DELTA_MIN="${ASR_PROMOTION_ENTITY_DELTA_MIN:-0.02}"
ASR_PROMOTION_WER_REGRESSION_MAX="${ASR_PROMOTION_WER_REGRESSION_MAX:-0.02}"
ASR_PROMOTION_CER_REGRESSION_MAX="${ASR_PROMOTION_CER_REGRESSION_MAX:-0.02}"
ASR_PROMOTION_LATENCY_REGRESSION_MS_MAX="${ASR_PROMOTION_LATENCY_REGRESSION_MS_MAX:-750}"
ASR_PROMOTION_REQUIRE_BUSINESS_HOLDOUT="${ASR_PROMOTION_REQUIRE_BUSINESS_HOLDOUT:-true}"

COMMAND="${1:-report}"
if [[ $# -gt 0 ]]; then
  shift
fi

log() { printf "\033[36m[asr-promotion]\033[0m %s\n" "$*"; }
err() { printf "\033[31m[asr-promotion]\033[0m %s\n" "$*" >&2; }

usage() {
  cat <<'EOF'
Multilingual ASR promotion gate.

Commands:
  report   Read existing Volume artifacts and write JSON/Markdown decision report.
  print    Alias for report.
  help     Show this help.

Environment:
  ASR_PROMOTION_OUTPUT_DIR              Local report directory.
  ASR_PROMOTION_REMOTE_ROOT             ASR training Volume root. Default: app config.
  ASR_PROMOTION_ENTITY_DELTA_MIN        LoRA entity accuracy delta required. Default: 0.02.
  ASR_PROMOTION_WER_REGRESSION_MAX      Max WER regression allowed. Default: 0.02.
  ASR_PROMOTION_CER_REGRESSION_MAX      Max CER regression allowed. Default: 0.02.
  ASR_PROMOTION_LATENCY_REGRESSION_MS_MAX
                                         Max latency regression allowed. Default: 750.
  ASR_PROMOTION_REQUIRE_BUSINESS_HOLDOUT
                                         true blocks promotion without business holdout. Default: true.

Output:
  .run/asr_model_training/promotion_gate/multilingual_asr_promotion_report.json
  .run/asr_model_training/promotion_gate/multilingual_asr_promotion_report.md

Decision rule:
  - Promote LoRA only if it beats/ties base on business entity/action accuracy,
    does not materially regress external WER/CER, does not increase unsafe rate,
    and stays inside latency budget.
  - Missing business holdout or mixed metrics => no-promotion.
  - If tied, prefer deployed OSS/base for simplicity and risk.
EOF
}

setup_env() {
  cd "$ROOT"
  mkdir -p "$ASR_PROMOTION_OUTPUT_DIR"
  if [[ -x "$ROOT/.venv/bin/python" ]]; then
    PYTHON_BIN="$ROOT/.venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
  export PYTHONPATH="$ROOT/backend${PYTHONPATH:+:$PYTHONPATH}"
}

write_report() {
  setup_env
  log "writing promotion report to $ASR_PROMOTION_OUTPUT_DIR"
  "$PYTHON_BIN" - "$ASR_PROMOTION_OUTPUT_DIR" <<'PY'
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from genie_voice.config import get_settings
from genie_voice.databricks.client import get_workspace_client


out_dir = Path(sys.argv[1])
out_dir.mkdir(parents=True, exist_ok=True)

entity_delta_min = float(os.environ.get("ASR_PROMOTION_ENTITY_DELTA_MIN", "0.02"))
wer_regression_max = float(os.environ.get("ASR_PROMOTION_WER_REGRESSION_MAX", "0.02"))
cer_regression_max = float(os.environ.get("ASR_PROMOTION_CER_REGRESSION_MAX", "0.02"))
latency_regression_ms_max = float(os.environ.get("ASR_PROMOTION_LATENCY_REGRESSION_MS_MAX", "750"))
require_business_holdout = os.environ.get("ASR_PROMOTION_REQUIRE_BUSINESS_HOLDOUT", "true").lower() == "true"


@dataclass(frozen=True)
class LanguagePlan:
    code: str
    app_code: str
    label: str
    lora_candidate: str
    latest_lora_run: str
    deployed_endpoint: str
    deployed_candidate: str
    base_eval_prefix: str
    intended_final_model: str
    intended_final_endpoint: str


LANGUAGES = [
    LanguagePlan(
        code="th",
        app_code="th-TH",
        label="Thai",
        lora_candidate="th_finetuned_pathumma_whisper_lora",
        latest_lora_run="th_finetuned_pathumma_whisper_lora_20260709_211456",
        deployed_endpoint="voice_pathumma_whisper_large_v3_th_lora",
        deployed_candidate="pathumma_whisper_large_v3_th_lora",
        base_eval_prefix="base_th_finetuned_pathumma_whisper_lora",
        intended_final_model="genie_asr_th_<winner>",
        intended_final_endpoint="voice_asr_th_<winner>",
    ),
    LanguagePlan(
        code="id",
        app_code="id-ID",
        label="Indonesian",
        lora_candidate="id_finetuned_whisper_large_v3_lora",
        latest_lora_run="id_finetuned_whisper_large_v3_lora_20260709_212613",
        deployed_endpoint="voice_qwen3_asr_0_6b_id_lora",
        deployed_candidate="qwen3_asr_0_6b_id_lora",
        base_eval_prefix="base_id_finetuned_whisper_large_v3_lora",
        intended_final_model="genie_asr_id_<winner>",
        intended_final_endpoint="voice_asr_id_<winner>",
    ),
    LanguagePlan(
        code="zh",
        app_code="zh-CN",
        label="Chinese",
        lora_candidate="zh_finetuned_whisper_large_v3_lora",
        latest_lora_run="zh_finetuned_whisper_large_v3_lora_20260709_213616",
        deployed_endpoint="voice_qwen3_asr_0_6b_zh_lora",
        deployed_candidate="qwen3_asr_0_6b_zh_lora",
        base_eval_prefix="base_zh_finetuned_whisper_large_v3_lora",
        intended_final_model="genie_asr_zh_<winner>",
        intended_final_endpoint="voice_asr_zh_<winner>",
    ),
]


def avg(values: list[float | None]) -> float | None:
    present = [v for v in values if isinstance(v, (int, float))]
    if not present:
        return None
    return sum(present) / len(present)


def safe_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def metric(summary: dict[str, Any] | None, *names: str) -> float | None:
    if not summary:
        return None
    for name in names:
        value = safe_float(summary.get(name))
        if value is not None:
            return value
    for nested_name in ("summary", "aggregate", "metrics", "provider_summary"):
        nested = summary.get(nested_name)
        if isinstance(nested, dict):
            for name in names:
                value = safe_float(nested.get(name))
                if value is not None:
                    return value
    return None


def read_json(client: Any, path: str) -> dict[str, Any] | None:
    try:
        data = client.files.download(path).contents.read().decode("utf-8")
        return json.loads(data)
    except Exception:
        return None


def list_dir(client: Any, path: str) -> list[Any]:
    try:
        return list(client.files.list_directory_contents(path))
    except Exception:
        return []


def endpoint_status(client: Any, endpoint_name: str) -> dict[str, Any]:
    try:
        endpoint = client.serving_endpoints.get(endpoint_name)
    except Exception as exc:
        return {"exists": False, "error": f"{type(exc).__name__}: {str(exc)[:180]}"}
    config = getattr(endpoint, "config", None)
    served = getattr(config, "served_entities", None) or getattr(config, "served_models", None) or []
    served_entities = []
    for entity in served:
        served_entities.append(
            {
                "entity_name": getattr(entity, "entity_name", None),
                "entity_version": getattr(entity, "entity_version", None),
                "workload_type": str(getattr(entity, "workload_type", None)),
                "workload_size": str(getattr(entity, "workload_size", None)),
                "scale_to_zero_enabled": getattr(entity, "scale_to_zero_enabled", None),
            }
        )
    return {
        "exists": True,
        "ready": str(getattr(getattr(endpoint, "state", None), "ready", None)),
        "served_entities": served_entities,
    }


def summarize_eval(summary: dict[str, Any] | None) -> dict[str, Any]:
    if not summary:
        return {"present": False}
    return {
        "present": True,
        "clips": summary.get("clips") or summary.get("clip_count") or summary.get("rows"),
        "avg_wer": metric(summary, "avg_wer", "wer"),
        "avg_cer": metric(summary, "avg_cer", "cer"),
        "avg_entity_accuracy": metric(summary, "avg_entity_accuracy", "entity_accuracy"),
        "avg_critical_entity_accuracy": metric(
            summary,
            "avg_critical_entity_accuracy",
            "critical_entity_accuracy",
        ),
        "unsafe_for_resolution_rate": metric(summary, "unsafe_for_resolution_rate"),
        "empty_transcript_rate": metric(summary, "empty_transcript_rate"),
        "avg_latency_ms": metric(summary, "avg_latency_ms"),
        "p95_latency_ms": metric(summary, "p95_latency_ms"),
    }


def compare(base: dict[str, Any], lora: dict[str, Any]) -> dict[str, Any]:
    deltas: dict[str, float | None] = {}
    for key in (
        "avg_wer",
        "avg_cer",
        "avg_entity_accuracy",
        "avg_critical_entity_accuracy",
        "unsafe_for_resolution_rate",
        "avg_latency_ms",
        "p95_latency_ms",
    ):
        b = safe_float(base.get(key))
        l = safe_float(lora.get(key))
        deltas[key] = None if b is None or l is None else l - b
    return deltas


def decision_for(
    external_base: dict[str, Any],
    external_lora: dict[str, Any],
    business_base: dict[str, Any],
    business_lora: dict[str, Any],
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if require_business_holdout and (not business_base.get("present") or not business_lora.get("present")):
        reasons.append("missing_business_holdout_comparison")

    if not external_base.get("present") or not external_lora.get("present"):
        reasons.append("missing_external_holdout_comparison")

    comparison_source_base = business_base if business_base.get("present") else external_base
    comparison_source_lora = business_lora if business_lora.get("present") else external_lora

    base_entity = safe_float(comparison_source_base.get("avg_critical_entity_accuracy"))
    lora_entity = safe_float(comparison_source_lora.get("avg_critical_entity_accuracy"))
    if base_entity is None or lora_entity is None:
        base_entity = safe_float(comparison_source_base.get("avg_entity_accuracy"))
        lora_entity = safe_float(comparison_source_lora.get("avg_entity_accuracy"))
    if base_entity is None or lora_entity is None:
        reasons.append("missing_entity_accuracy")
    elif lora_entity - base_entity < entity_delta_min:
        reasons.append("lora_entity_gain_below_threshold")

    base_wer = safe_float(external_base.get("avg_wer"))
    lora_wer = safe_float(external_lora.get("avg_wer"))
    if base_wer is not None and lora_wer is not None and lora_wer - base_wer > wer_regression_max:
        reasons.append("lora_wer_regression")

    base_cer = safe_float(external_base.get("avg_cer"))
    lora_cer = safe_float(external_lora.get("avg_cer"))
    if base_cer is not None and lora_cer is not None and lora_cer - base_cer > cer_regression_max:
        reasons.append("lora_cer_regression")

    base_unsafe = safe_float(comparison_source_base.get("unsafe_for_resolution_rate"))
    lora_unsafe = safe_float(comparison_source_lora.get("unsafe_for_resolution_rate"))
    if base_unsafe is not None and lora_unsafe is not None and lora_unsafe > base_unsafe:
        reasons.append("lora_unsafe_rate_regression")

    base_latency = safe_float(external_base.get("p95_latency_ms")) or safe_float(external_base.get("avg_latency_ms"))
    lora_latency = safe_float(external_lora.get("p95_latency_ms")) or safe_float(external_lora.get("avg_latency_ms"))
    if (
        base_latency is not None
        and lora_latency is not None
        and lora_latency - base_latency > latency_regression_ms_max
    ):
        reasons.append("lora_latency_regression")

    if reasons:
        return "no-promotion", sorted(set(reasons))
    return "promote-lora", ["lora_passed_promotion_gate"]


def registered_candidate_eval(client: Any, registered_root: str, deployed_candidate: str) -> dict[str, Any]:
    path = f"{registered_root}/outputs/{deployed_candidate}/eval_summary.json"
    return summarize_eval(read_json(client, path))


def file_exists(client: Any, path: str) -> bool:
    parent = path.rsplit("/", 1)[0]
    name = path.rsplit("/", 1)[1]
    return any((entry.path.rstrip("/").rsplit("/", 1)[-1] == name) for entry in list_dir(client, parent))


settings = get_settings()
client = get_workspace_client(settings)
remote_root = os.environ.get("ASR_PROMOTION_REMOTE_ROOT") or (
    f"/Volumes/{settings.databricks.catalog}/{settings.databricks.schema_name}/"
    f"{settings.volume.streaming_name}/asr_model_training"
)
eval_root = f"{remote_root}/evaluations/multilingual"
registered_root = f"{remote_root}/registered_candidates/multilingual_asr"
run_root = f"{remote_root}/model_artifacts/multilingual_lora_runs"
manifest_root = f"{remote_root}/datasets/multilingual_gold/manifests"

language_reports = []
for plan in LANGUAGES:
    base_holdout = summarize_eval(
        read_json(client, f"{eval_root}/{plan.base_eval_prefix}_holdout/base_evaluation_summary.json")
    )
    base_validation = summarize_eval(
        read_json(client, f"{eval_root}/{plan.base_eval_prefix}_validation/base_evaluation_summary.json")
    )
    lora_summary = read_json(client, f"{eval_root}/{plan.latest_lora_run}/lora_evaluation_summary.rescored.json")
    if lora_summary is None:
        lora_summary = read_json(client, f"{eval_root}/{plan.latest_lora_run}/lora_evaluation_summary.json")
    lora_holdout = summarize_eval(lora_summary)

    business_base = summarize_eval(
        read_json(client, f"{eval_root}/base_{plan.lora_candidate}_holdout_business/base_evaluation_summary.json")
    )
    business_lora = summarize_eval(
        read_json(client, f"{eval_root}/{plan.latest_lora_run}_business/lora_evaluation_summary.rescored.json")
        or read_json(client, f"{eval_root}/{plan.latest_lora_run}_business/lora_evaluation_summary.json")
    )

    deployed_eval = registered_candidate_eval(client, registered_root, plan.deployed_candidate)
    endpoint = endpoint_status(client, plan.deployed_endpoint)
    adapter_path = f"{run_root}/{plan.latest_lora_run}/adapter/adapter_model.safetensors"
    adapter_present = file_exists(client, adapter_path)

    decision, reasons = decision_for(base_holdout, lora_holdout, business_base, business_lora)
    if not adapter_present:
        decision = "no-promotion"
        reasons = sorted(set([*reasons, "missing_lora_adapter_artifact"]))

    language_reports.append(
        {
            "language": plan.label,
            "language_code": plan.code,
            "app_language_code": plan.app_code,
            "decision": decision,
            "reasons": reasons,
            "lora_candidate": plan.lora_candidate,
            "latest_lora_run": plan.latest_lora_run,
            "lora_adapter_present": adapter_present,
            "current_deployed_endpoint": plan.deployed_endpoint,
            "current_deployed_endpoint_status": endpoint,
            "current_deployed_candidate_eval": deployed_eval,
            "external_holdout": {
                "base": base_holdout,
                "lora": lora_holdout,
                "delta_lora_minus_base": compare(base_holdout, lora_holdout),
            },
            "validation_or_synthetic": {
                "base": base_validation,
            },
            "business_holdout": {
                "base": business_base,
                "lora": business_lora,
                "delta_lora_minus_base": compare(business_base, business_lora),
                "manifest_present": file_exists(client, f"{manifest_root}/{plan.code}_business_holdout.built.jsonl"),
            },
            "final_naming_target": {
                "uc_model": plan.intended_final_model,
                "serving_endpoint": plan.intended_final_endpoint,
            },
        }
    )

report = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "remote_root": remote_root,
    "mode": "read_only_no_deploy",
    "thresholds": {
        "entity_delta_min": entity_delta_min,
        "wer_regression_max": wer_regression_max,
        "cer_regression_max": cer_regression_max,
        "latency_regression_ms_max": latency_regression_ms_max,
        "require_business_holdout": require_business_holdout,
    },
    "overall_decision": (
        "promote_some"
        if any(item["decision"] == "promote-lora" for item in language_reports)
        else "no_multilingual_lora_promotion"
    ),
    "languages": language_reports,
    "next_actions": [
        "Run or materialize real business holdout comparisons for each language.",
        "Promote LoRA only where the gate passes; otherwise keep the deployed OSS/base endpoint.",
        "Register winners under final non-misleading UC model names.",
        "Deploy one final serving endpoint per language and update app STT routes.",
        "Run /mic-transcribe smoke per language before enabling the UI route.",
    ],
}

json_path = out_dir / "multilingual_asr_promotion_report.json"
md_path = out_dir / "multilingual_asr_promotion_report.md"
json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

lines = [
    "# Multilingual ASR Promotion Report",
    "",
    f"Generated: `{report['generated_at']}`",
    f"Remote root: `{remote_root}`",
    f"Overall decision: **{report['overall_decision']}**",
    "",
    "## Decisions",
    "",
    "| Language | Decision | Reasons | LoRA run | Current endpoint |",
    "|---|---|---|---|---|",
]
for item in language_reports:
    reasons = ", ".join(item["reasons"])
    lines.append(
        "| {language} | {decision} | {reasons} | `{run}` | `{endpoint}` |".format(
            language=item["language"],
            decision=item["decision"],
            reasons=reasons,
            run=item["latest_lora_run"],
            endpoint=item["current_deployed_endpoint"],
        )
    )

lines.extend(["", "## Metric Snapshot", ""])
for item in language_reports:
    ext = item["external_holdout"]
    business = item["business_holdout"]
    lines.extend(
        [
            f"### {item['language']}",
            f"- External base: `{ext['base']}`",
            f"- External LoRA: `{ext['lora']}`",
            f"- External delta LoRA-base: `{ext['delta_lora_minus_base']}`",
            f"- Business holdout present: `{business['manifest_present']}`",
            f"- Business base: `{business['base']}`",
            f"- Business LoRA: `{business['lora']}`",
            "",
        ]
    )

lines.extend(["## Next Actions", ""])
for action in report["next_actions"]:
    lines.append(f"- {action}")

md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

print(f"JSON report: {json_path}")
print(f"Markdown report: {md_path}")
print(f"Overall decision: {report['overall_decision']}")
for item in language_reports:
    print(f"{item['language']}: {item['decision']} ({', '.join(item['reasons'])})")
PY
}

case "$COMMAND" in
  report|print)
    write_report
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
