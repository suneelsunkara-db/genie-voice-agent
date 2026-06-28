#!/usr/bin/env bash
# =============================================================================
# 07_deep_voice_model_eval.sh
#
# Deep, utterance-level voice model evaluation for Deepgram vs the Databricks
# fine-tuned Whisper serving endpoint. This excludes human review and streaming
# metrics by design; it focuses on offline ASR quality, business entity
# correctness, latency, reliability, and app-readiness signals.
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEFAULT_MANIFEST="$ROOT/.run/asr_model_training/manifests/asr_training_gold_v1.jsonl"
DEFAULT_OUTPUT_DIR="$ROOT/.run/asr_model_training/evaluations/voice_model_deep_eval"

ASR_EVAL_MANIFEST="${ASR_EVAL_MANIFEST:-$DEFAULT_MANIFEST}"
ASR_EVAL_OUTPUT_DIR="${ASR_EVAL_OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}"
ASR_EVAL_LIMIT="${ASR_EVAL_LIMIT:-}"
ASR_EVAL_SPLIT="${ASR_EVAL_SPLIT:-}"
ASR_EVAL_DEEPGRAM_MODEL="${ASR_EVAL_DEEPGRAM_MODEL:-nova-3}"
ASR_EVAL_DEEPGRAM_OUTPUT="${ASR_EVAL_DEEPGRAM_OUTPUT:-$ASR_EVAL_OUTPUT_DIR/deepgram_${ASR_EVAL_DEEPGRAM_MODEL}_deep_eval.jsonl}"
ASR_EVAL_DATABRICKS_ENDPOINT="${ASR_EVAL_DATABRICKS_ENDPOINT:-voice_finetuned_whisper_model}"
ASR_EVAL_DATABRICKS_OUTPUT="${ASR_EVAL_DATABRICKS_OUTPUT:-$ASR_EVAL_OUTPUT_DIR/databricks_finetuned_whisper_deep_eval.jsonl}"
ASR_EVAL_SUMMARY_OUTPUT="${ASR_EVAL_SUMMARY_OUTPUT:-$ASR_EVAL_OUTPUT_DIR/deepgram_vs_databricks_summary.json}"
ASR_EVAL_DATABRICKS_PROFILE="${ASR_EVAL_DATABRICKS_PROFILE:-${DATABRICKS_CONFIG_PROFILE:-fe-vm-vdm-classic-rcn6ip}}"

COMMAND="${1:-run}"
if [[ $# -gt 0 ]]; then
  shift
fi

log() { printf "\033[36m[asr-eval]\033[0m %s\n" "$*"; }
err() { printf "\033[31m[asr-eval]\033[0m %s\n" "$*" >&2; }

usage() {
  cat <<EOF
Deep voice model evaluation: Deepgram vs Databricks fine-tuned Whisper

Commands:
  preflight          Validate manifest, Python imports, Deepgram key, and Databricks endpoint access.
  run                Run Deepgram, run Databricks endpoint, then write comparison summary.
  deepgram           Run only the Deepgram offline ASR evaluation.
  databricks         Run only the Databricks endpoint ASR evaluation.
  compare-existing   Compare existing provider JSONL outputs and write summary JSON.
  help               Show this help.

Environment:
  ASR_EVAL_MANIFEST              Manifest JSONL. Default: $DEFAULT_MANIFEST
  ASR_EVAL_OUTPUT_DIR            Output directory. Default: $DEFAULT_OUTPUT_DIR
  ASR_EVAL_LIMIT                 Optional max clips, useful for smoke tests.
  ASR_EVAL_SPLIT                 Optional manifest split filter, e.g. test.
  ASR_EVAL_DEEPGRAM_MODEL        Deepgram model. Default: nova-3.
  ASR_EVAL_DATABRICKS_ENDPOINT   Serving endpoint. Default: voice_finetuned_whisper_model.
  ASR_EVAL_DATABRICKS_PROFILE    Databricks CLI/profile. Default: fe-vm-vdm-classic-rcn6ip.

Examples:
  ASR_EVAL_LIMIT=10 scripts/asr/07_deep_voice_model_eval.sh run
  ASR_EVAL_SPLIT=test scripts/asr/07_deep_voice_model_eval.sh run
  scripts/asr/07_deep_voice_model_eval.sh compare-existing

Outputs:
  Deepgram JSONL:    $ASR_EVAL_DEEPGRAM_OUTPUT
  Databricks JSONL: $ASR_EVAL_DATABRICKS_OUTPUT
  Summary JSON:     $ASR_EVAL_SUMMARY_OUTPUT

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
  if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
    python3 -m venv "$ROOT/.venv"
  fi
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
  export PYTHONPATH="$ROOT/backend${PYTHONPATH:+:$PYTHONPATH}"
  export DATABRICKS_CONFIG_PROFILE="$ASR_EVAL_DATABRICKS_PROFILE"
  export ASR_DATABRICKS_PROFILE="$ASR_EVAL_DATABRICKS_PROFILE"
}

python_args() {
  local args=(
    --manifest "$ASR_EVAL_MANIFEST"
    --deepgram-output "$ASR_EVAL_DEEPGRAM_OUTPUT"
    --databricks-output "$ASR_EVAL_DATABRICKS_OUTPUT"
    --summary-output "$ASR_EVAL_SUMMARY_OUTPUT"
    --deepgram-model "$ASR_EVAL_DEEPGRAM_MODEL"
    --databricks-endpoint "$ASR_EVAL_DATABRICKS_ENDPOINT"
  )
  if [[ -n "$ASR_EVAL_LIMIT" ]]; then
    args+=(--limit "$ASR_EVAL_LIMIT")
  fi
  if [[ -n "$ASR_EVAL_SPLIT" ]]; then
    args+=(--split "$ASR_EVAL_SPLIT")
  fi
  printf '%s\n' "${args[@]}"
}

run_python() {
  local action="$1"
  shift || true
  python - "$action" "$@" <<'PY'
from __future__ import annotations

import argparse
import base64
import json
import math
import mimetypes
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from genie_voice.asr_eval.deepgram_baseline import transcribe_clip
from genie_voice.asr_eval.manifest import ASRGoldClip, load_manifest
from genie_voice.asr_eval.metrics import score_transcript


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["preflight", "deepgram", "databricks", "compare"])
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--deepgram-output", required=True)
    parser.add_argument("--databricks-output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--deepgram-model", default="nova-3")
    parser.add_argument("--databricks-endpoint", default="voice_finetuned_whisper_model")
    parser.add_argument("--split", action="append", dest="splits")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def selected_clips(args: argparse.Namespace) -> list[ASRGoldClip]:
    clips = load_manifest(args.manifest, splits=args.splits)
    if args.limit is not None:
        clips = clips[: args.limit]
    if not clips:
        raise SystemExit(f"No clips selected from manifest: {args.manifest}")
    return clips


def main() -> None:
    args = parse_args()
    if args.action == "preflight":
        preflight(args)
    elif args.action == "deepgram":
        run_deepgram(args)
    elif args.action == "databricks":
        run_databricks(args)
    elif args.action == "compare":
        compare(args)


def preflight(args: argparse.Namespace) -> None:
    clips = selected_clips(args)
    first_audio = read_audio(clips[0].audio_path)
    checks: dict[str, Any] = {
        "manifest": str(args.manifest),
        "selected_clips": len(clips),
        "first_clip_id": clips[0].clip_id,
        "first_audio_bytes": len(first_audio),
        "deepgram_model": args.deepgram_model,
        "databricks_endpoint": args.databricks_endpoint,
    }
    try:
        from genie_voice.config import get_settings

        checks["deepgram_key_configured"] = bool(get_settings().secrets.deepgram_api_key.strip())
    except Exception:  # noqa: BLE001
        checks["deepgram_key_configured"] = False
    try:
        client = databricks_client()
        endpoint = client.serving_endpoints.get(args.databricks_endpoint)
        checks["databricks_endpoint_state"] = response_dict(endpoint).get("state")
    except Exception as exc:  # noqa: BLE001
        checks["databricks_endpoint_error"] = str(exc)
    print(json.dumps(checks, indent=2, default=str))


def run_deepgram(args: argparse.Namespace) -> None:
    clips = selected_clips(args)
    output = Path(args.deepgram_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with output.open("w", encoding="utf-8") as f:
        for clip in clips:
            result = transcribe_clip(
                clip,
                api_key=deepgram_api_key(),
                model=args.deepgram_model,
                language="en-US",
                smart_format=True,
                punctuate=True,
            )
            row = build_row(
                clip=clip,
                provider="deepgram",
                model=args.deepgram_model,
                transcript=result["transcript"],
                raw_transcript=result["transcript"],
                latency_ms=result["latency_ms"],
                confidence=result["confidence"],
                raw=result["raw"],
            )
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.append(row)
    print(json.dumps(summarize_provider(rows), indent=2))


def run_databricks(args: argparse.Namespace) -> None:
    clips = selected_clips(args)
    client = databricks_client()
    output = Path(args.databricks_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with output.open("w", encoding="utf-8") as f:
        for clip in clips:
            audio_b64 = base64.b64encode(read_audio(clip.audio_path)).decode("ascii")
            started = time.perf_counter()
            response = client.serving_endpoints.query(
                name=args.databricks_endpoint,
                dataframe_records=[
                    {
                        "audio_b64": audio_b64,
                        "mime_type": mime_type_for(clip.audio_path, clip.audio_format),
                        "speaker": speaker_number(clip.speaker),
                    }
                ],
            )
            latency_ms = round((time.perf_counter() - started) * 1000)
            payload = response_dict(response)
            prediction = first_prediction(payload)
            transcript = str(prediction.get("transcript") or prediction.get("raw_transcript") or "").strip()
            row = build_row(
                clip=clip,
                provider="databricks",
                model=str(prediction.get("model") or args.databricks_endpoint),
                transcript=transcript,
                raw_transcript=str(prediction.get("raw_transcript") or transcript).strip(),
                latency_ms=latency_ms,
                confidence=prediction.get("confidence"),
                raw=payload,
                extra={
                    "endpoint": args.databricks_endpoint,
                    "base_model": prediction.get("base_model"),
                    "lora_run_name": prediction.get("lora_run_name"),
                    "requires_invoice_postprocessing": prediction.get("requires_invoice_postprocessing"),
                },
            )
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.append(row)
    print(json.dumps(summarize_provider(rows), indent=2))


def compare(args: argparse.Namespace) -> None:
    deepgram_rows = read_jsonl(args.deepgram_output)
    databricks_rows = read_jsonl(args.databricks_output)
    by_provider = {
        "deepgram": summarize_provider(deepgram_rows),
        "databricks": summarize_provider(databricks_rows),
    }
    paired = pairwise_comparison(deepgram_rows, databricks_rows)
    summary = {
        "manifest": str(args.manifest),
        "deepgram_output": str(args.deepgram_output),
        "databricks_output": str(args.databricks_output),
        "providers": by_provider,
        "pairwise": paired,
        "promotion_read": promotion_read(by_provider, paired),
    }
    output = Path(args.summary_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def build_row(
    *,
    clip: ASRGoldClip,
    provider: str,
    model: str,
    transcript: str,
    raw_transcript: str,
    latency_ms: int,
    confidence: Any,
    raw: Any,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    score = score_transcript(clip.reference_transcript, transcript, clip.expected_entities)
    features = transcript_features(clip.reference_transcript, transcript)
    entity_scores = score.to_dict()["entity_scores"]
    app_readiness = app_readiness_signals(entity_scores, features, transcript)
    return {
        "clip_id": clip.clip_id,
        "call_id": clip.call_id,
        "speaker": clip.speaker,
        "audio_path": clip.audio_path,
        "duration_seconds": clip.duration_seconds,
        "scenario": clip.scenario,
        "split": clip.split,
        "dataset_version": clip.dataset_version,
        "provider": provider,
        "model": model,
        "transcript": transcript,
        "raw_transcript": raw_transcript,
        "reference_transcript": clip.reference_transcript,
        "latency_ms": latency_ms,
        "latency_per_audio_second": None
        if not clip.duration_seconds
        else latency_ms / max(float(clip.duration_seconds), 0.001),
        "confidence": confidence,
        "score": score.to_dict(),
        "transcript_features": features,
        "app_readiness": app_readiness,
        "raw": raw,
        **(extra or {}),
    }


def transcript_features(reference: str, hypothesis: str) -> dict[str, Any]:
    ref_tokens = token_counter(reference)
    hyp_tokens = token_counter(hypothesis)
    ref_numbers = numeric_tokens(reference)
    hyp_numbers = numeric_tokens(hypothesis)
    missing_numbers = [value for value in ref_numbers if value not in hyp_numbers]
    added_numbers = [value for value in hyp_numbers if value not in ref_numbers]
    ref_negations = negation_terms(reference)
    hyp_negations = negation_terms(hypothesis)
    return {
        "reference_word_count": sum(ref_tokens.values()),
        "hypothesis_word_count": sum(hyp_tokens.values()),
        "length_ratio": safe_div(sum(hyp_tokens.values()), sum(ref_tokens.values())),
        "numeric_reference": ref_numbers,
        "numeric_hypothesis": hyp_numbers,
        "numeric_missing": missing_numbers,
        "numeric_added": added_numbers,
        "numeric_recall": None if not ref_numbers else (len(ref_numbers) - len(missing_numbers)) / len(ref_numbers),
        "negations_reference": ref_negations,
        "negations_hypothesis": hyp_negations,
        "negation_match": Counter(ref_negations) == Counter(hyp_negations),
        "empty_transcript": not hypothesis.strip(),
    }


def app_readiness_signals(entity_scores: dict[str, Any], features: dict[str, Any], transcript: str) -> dict[str, Any]:
    critical_groups = ("invoice_ids", "amounts", "dates", "billing_actions", "confirmations", "refusals")
    critical_expected = 0
    critical_matched = 0
    missing: dict[str, list[str]] = {}
    for group in critical_groups:
        score = entity_scores.get(group) or {}
        critical_expected += int(score.get("expected") or 0)
        critical_matched += int(score.get("matched") or 0)
        if score.get("missing"):
            missing[group] = list(score["missing"])
    critical_accuracy = None if critical_expected == 0 else critical_matched / critical_expected
    unsafe_reasons = []
    if not transcript.strip():
        unsafe_reasons.append("empty_transcript")
    if features["numeric_missing"]:
        unsafe_reasons.append("missing_numeric_token")
    if not features["negation_match"]:
        unsafe_reasons.append("negation_mismatch")
    if missing.get("invoice_ids"):
        unsafe_reasons.append("missing_invoice_id")
    if missing.get("amounts"):
        unsafe_reasons.append("missing_amount")
    if missing.get("confirmations") or missing.get("refusals"):
        unsafe_reasons.append("missing_customer_decision_phrase")
    return {
        "critical_entity_accuracy": critical_accuracy,
        "critical_entities_expected": critical_expected,
        "critical_entities_matched": critical_matched,
        "critical_entities_missing": missing,
        "unsafe_for_resolution": bool(unsafe_reasons),
        "unsafe_reasons": unsafe_reasons,
    }


def summarize_provider(rows: list[dict[str, Any]]) -> dict[str, Any]:
    entity_totals: dict[str, dict[str, int]] = defaultdict(lambda: {"expected": 0, "matched": 0})
    unsafe_reasons = Counter()
    for row in rows:
        for group, score in (row.get("score", {}).get("entity_scores") or {}).items():
            entity_totals[group]["expected"] += int(score.get("expected") or 0)
            entity_totals[group]["matched"] += int(score.get("matched") or 0)
        unsafe_reasons.update(row.get("app_readiness", {}).get("unsafe_reasons") or [])
    latencies = [row.get("latency_ms") for row in rows if row.get("latency_ms") is not None]
    return {
        "clips": len(rows),
        "provider": sorted({row.get("provider") for row in rows if row.get("provider")}),
        "models": sorted({row.get("model") for row in rows if row.get("model")}),
        "avg_wer": avg(row.get("score", {}).get("wer") for row in rows),
        "avg_cer": avg(row.get("score", {}).get("cer") for row in rows),
        "avg_entity_accuracy": avg(row.get("score", {}).get("entity_accuracy") for row in rows),
        "avg_critical_entity_accuracy": avg(
            row.get("app_readiness", {}).get("critical_entity_accuracy") for row in rows
        ),
        "empty_transcript_rate": rate(row.get("transcript_features", {}).get("empty_transcript") for row in rows),
        "unsafe_for_resolution_rate": rate(row.get("app_readiness", {}).get("unsafe_for_resolution") for row in rows),
        "negation_mismatch_rate": rate(
            not row.get("transcript_features", {}).get("negation_match", True) for row in rows
        ),
        "numeric_recall": avg(row.get("transcript_features", {}).get("numeric_recall") for row in rows),
        "latency_ms": {
            "avg": avg(latencies),
            "p50": percentile(latencies, 50),
            "p90": percentile(latencies, 90),
            "p95": percentile(latencies, 95),
            "p99": percentile(latencies, 99),
        },
        "entity_groups": {
            group: {
                **counts,
                "accuracy": None if counts["expected"] == 0 else counts["matched"] / counts["expected"],
            }
            for group, counts in sorted(entity_totals.items())
        },
        "unsafe_reason_counts": dict(sorted(unsafe_reasons.items())),
        "worst_wer_examples": top_examples(rows, "wer"),
        "unsafe_examples": unsafe_examples(rows),
    }


def pairwise_comparison(deepgram_rows: list[dict[str, Any]], databricks_rows: list[dict[str, Any]]) -> dict[str, Any]:
    deepgram = {row["clip_id"]: row for row in deepgram_rows}
    databricks = {row["clip_id"]: row for row in databricks_rows}
    shared_ids = sorted(set(deepgram) & set(databricks))
    winners = Counter()
    deltas = []
    for clip_id in shared_ids:
        d = deepgram[clip_id]
        b = databricks[clip_id]
        d_score = d.get("score", {})
        b_score = b.get("score", {})
        d_business = d.get("app_readiness", {}).get("critical_entity_accuracy")
        b_business = b.get("app_readiness", {}).get("critical_entity_accuracy")
        if b_business is not None and d_business is not None and not math.isclose(b_business, d_business):
            winners["databricks_business" if b_business > d_business else "deepgram_business"] += 1
        if (b_score.get("wer") is not None) and (d_score.get("wer") is not None) and not math.isclose(
            b_score["wer"], d_score["wer"]
        ):
            winners["databricks_wer" if b_score["wer"] < d_score["wer"] else "deepgram_wer"] += 1
        deltas.append(
            {
                "clip_id": clip_id,
                "scenario": b.get("scenario") or d.get("scenario"),
                "wer_delta_databricks_minus_deepgram": none_safe_sub(b_score.get("wer"), d_score.get("wer")),
                "entity_accuracy_delta_databricks_minus_deepgram": none_safe_sub(
                    b_score.get("entity_accuracy"), d_score.get("entity_accuracy")
                ),
                "critical_entity_accuracy_delta_databricks_minus_deepgram": none_safe_sub(b_business, d_business),
                "latency_ms_delta_databricks_minus_deepgram": none_safe_sub(b.get("latency_ms"), d.get("latency_ms")),
                "deepgram_transcript": d.get("transcript"),
                "databricks_transcript": b.get("transcript"),
                "reference_transcript": b.get("reference_transcript") or d.get("reference_transcript"),
            }
        )
    return {
        "paired_clips": len(shared_ids),
        "winner_counts": dict(sorted(winners.items())),
        "largest_databricks_entity_wins": top_deltas(deltas, "critical_entity_accuracy_delta_databricks_minus_deepgram", True),
        "largest_deepgram_entity_wins": top_deltas(deltas, "critical_entity_accuracy_delta_databricks_minus_deepgram", False),
        "largest_databricks_latency_penalties": top_deltas(deltas, "latency_ms_delta_databricks_minus_deepgram", True),
    }


def promotion_read(by_provider: dict[str, Any], paired: dict[str, Any]) -> dict[str, Any]:
    dg = by_provider.get("deepgram", {})
    db = by_provider.get("databricks", {})
    return {
        "recommended_headline": (
            "Use Databricks fine-tuned ASR when business entity preservation improves enough to justify "
            "utterance-level latency; keep Deepgram for streaming UX."
        ),
        "databricks_business_delta": none_safe_sub(
            db.get("avg_critical_entity_accuracy"), dg.get("avg_critical_entity_accuracy")
        ),
        "databricks_wer_delta": none_safe_sub(db.get("avg_wer"), dg.get("avg_wer")),
        "databricks_p95_latency_delta_ms": none_safe_sub(
            (db.get("latency_ms") or {}).get("p95"), (dg.get("latency_ms") or {}).get("p95")
        ),
        "paired_clips": paired.get("paired_clips"),
    }


def read_audio(path: str) -> bytes:
    if path.startswith("data:"):
        _, _, encoded = path.partition(",")
        return base64.b64decode(encoded)
    if path.startswith("/Volumes/") or path.startswith("dbfs:/Volumes/"):
        from genie_voice.asr_eval.deepgram_baseline import _read_volume_audio

        return _read_volume_audio(path)
    if path.startswith("file://"):
        path = path.removeprefix("file://")
    return Path(path).read_bytes()


def mime_type_for(path: str, audio_format: str | None) -> str:
    if audio_format and "/" in audio_format:
        return audio_format
    return mimetypes.guess_type(path)[0] or "audio/wav"


def speaker_number(speaker: Any) -> int:
    text = str(speaker or "1").strip()
    return int(text) if text.isdigit() else 1


def databricks_client():
    from genie_voice.config import get_settings
    from genie_voice.databricks.client import get_workspace_client

    return get_workspace_client(get_settings())


def response_dict(response: Any) -> dict[str, Any]:
    if isinstance(response, dict):
        return response
    if hasattr(response, "as_dict"):
        return response.as_dict()
    predictions = getattr(response, "predictions", None)
    if predictions is not None:
        return {"predictions": predictions}
    return {}


def first_prediction(payload: dict[str, Any]) -> dict[str, Any]:
    predictions = payload.get("predictions") or []
    if not predictions:
        return {}
    first = predictions[0]
    return first if isinstance(first, dict) else dict(first)


def deepgram_api_key() -> str:
    from genie_voice.config import get_settings

    key = get_settings().secrets.deepgram_api_key.strip()
    if not key:
        raise RuntimeError("DEEPGRAM_API_KEY is not configured")
    return key


def token_counter(text: str) -> Counter[str]:
    return Counter(re.findall(r"[a-z0-9]+", text.lower()))


def numeric_tokens(text: str) -> list[str]:
    return re.findall(r"\b\d+(?:\.\d+)?\b", text.replace(",", ""))


def negation_terms(text: str) -> list[str]:
    return re.findall(r"\b(?:no|not|never|don't|dont|cannot|can't|cant|won't|wont|refuse|decline)\b", text.lower())


def safe_div(num: float, den: float) -> float | None:
    return None if den == 0 else num / den


def avg(values) -> float | None:
    present = [value for value in values if value is not None]
    return None if not present else sum(present) / len(present)


def rate(values) -> float | None:
    present = [bool(value) for value in values]
    return None if not present else sum(1 for value in present if value) / len(present)


def percentile(values, pct: int) -> float | None:
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


def none_safe_sub(left, right):
    if left is None or right is None:
        return None
    return left - right


def top_examples(rows: list[dict[str, Any]], metric: str, n: int = 5) -> list[dict[str, Any]]:
    scored = [
        row
        for row in rows
        if row.get("score", {}).get(metric) is not None
    ]
    scored.sort(key=lambda row: row["score"][metric], reverse=True)
    return [example_row(row) for row in scored[:n]]


def unsafe_examples(rows: list[dict[str, Any]], n: int = 5) -> list[dict[str, Any]]:
    unsafe = [row for row in rows if row.get("app_readiness", {}).get("unsafe_for_resolution")]
    unsafe.sort(key=lambda row: len(row.get("app_readiness", {}).get("unsafe_reasons") or []), reverse=True)
    return [example_row(row) for row in unsafe[:n]]


def top_deltas(rows: list[dict[str, Any]], key: str, reverse: bool, n: int = 5) -> list[dict[str, Any]]:
    present = [row for row in rows if row.get(key) is not None]
    present.sort(key=lambda row: row[key], reverse=reverse)
    return present[:n]


def example_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "clip_id": row.get("clip_id"),
        "scenario": row.get("scenario"),
        "wer": row.get("score", {}).get("wer"),
        "entity_accuracy": row.get("score", {}).get("entity_accuracy"),
        "critical_entity_accuracy": row.get("app_readiness", {}).get("critical_entity_accuracy"),
        "unsafe_reasons": row.get("app_readiness", {}).get("unsafe_reasons"),
        "reference_transcript": row.get("reference_transcript"),
        "transcript": row.get("transcript"),
    }


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"Missing result file: {p}")
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    main()
PY
}

load_python_args() {
  PY_ARGS=()
  while IFS= read -r arg; do
    PY_ARGS+=("$arg")
  done < <(python_args)
}

preflight() {
  setup_env
  mkdir -p "$ASR_EVAL_OUTPUT_DIR"
  load_python_args
  log "preflight"
  run_python preflight "${PY_ARGS[@]}"
}

run_deepgram() {
  setup_env
  mkdir -p "$ASR_EVAL_OUTPUT_DIR"
  load_python_args
  log "running Deepgram deep ASR eval"
  run_python deepgram "${PY_ARGS[@]}"
}

run_databricks() {
  setup_env
  mkdir -p "$ASR_EVAL_OUTPUT_DIR"
  load_python_args
  log "running Databricks fine-tuned ASR deep eval"
  run_python databricks "${PY_ARGS[@]}"
}

compare_existing() {
  setup_env
  mkdir -p "$ASR_EVAL_OUTPUT_DIR"
  load_python_args
  log "comparing existing ASR eval results"
  run_python compare "${PY_ARGS[@]}"
}

case "$COMMAND" in
  preflight)
    preflight
    ;;
  run)
    preflight
    run_deepgram
    run_databricks
    compare_existing
    ;;
  deepgram)
    run_deepgram
    ;;
  databricks)
    run_databricks
    ;;
  compare-existing)
    compare_existing
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
