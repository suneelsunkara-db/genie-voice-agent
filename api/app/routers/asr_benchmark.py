"""ASR benchmark result API for Deepgram vs fine-tuned Databricks ASR."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter

router = APIRouter(prefix="/asr-benchmark", tags=["asr-benchmark"])


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_eval_dir() -> Path:
    return _repo_root() / ".run" / "asr_model_training" / "evaluations" / "voice_model_deep_eval"


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path, *, limit: int = 500) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
        if len(rows) >= limit:
            break
    return rows


def _provider_paths(summary: dict[str, Any], base: Path) -> tuple[Path, Path]:
    deepgram = Path(str(summary.get("deepgram_output") or base / "deepgram_nova-3_deep_eval.jsonl"))
    databricks = Path(str(summary.get("databricks_output") or base / "databricks_finetuned_whisper_deep_eval.jsonl"))
    return deepgram, databricks


def _paired_examples(deepgram_rows: list[dict[str, Any]], databricks_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deepgram_by_clip = {str(row.get("clip_id")): row for row in deepgram_rows}
    databricks_by_clip = {str(row.get("clip_id")): row for row in databricks_rows}
    examples: list[dict[str, Any]] = []
    for clip_id in sorted(set(deepgram_by_clip) & set(databricks_by_clip)):
        dg = deepgram_by_clip[clip_id]
        db = databricks_by_clip[clip_id]
        dg_critical = (dg.get("app_readiness") or {}).get("critical_entity_accuracy")
        db_critical = (db.get("app_readiness") or {}).get("critical_entity_accuracy")
        dg_latency = dg.get("latency_ms")
        db_latency = db.get("latency_ms")
        examples.append(
            {
                "clip_id": clip_id,
                "scenario": db.get("scenario") or dg.get("scenario"),
                "reference_transcript": db.get("reference_transcript") or dg.get("reference_transcript"),
                "deepgram_transcript": dg.get("transcript"),
                "databricks_transcript": db.get("transcript"),
                "deepgram_wer": (dg.get("score") or {}).get("wer"),
                "databricks_wer": (db.get("score") or {}).get("wer"),
                "deepgram_critical_entity_accuracy": dg_critical,
                "databricks_critical_entity_accuracy": db_critical,
                "critical_entity_delta": None
                if dg_critical is None or db_critical is None
                else db_critical - dg_critical,
                "deepgram_latency_ms": dg_latency,
                "databricks_latency_ms": db_latency,
                "latency_delta_ms": None if dg_latency is None or db_latency is None else db_latency - dg_latency,
                "databricks_unsafe_reasons": (db.get("app_readiness") or {}).get("unsafe_reasons") or [],
                "deepgram_unsafe_reasons": (dg.get("app_readiness") or {}).get("unsafe_reasons") or [],
            }
        )
    examples.sort(
        key=lambda row: (
            len(row["databricks_unsafe_reasons"]) + len(row["deepgram_unsafe_reasons"]),
            abs(row["critical_entity_delta"] or 0),
            abs(row["latency_delta_ms"] or 0),
        ),
        reverse=True,
    )
    return examples[:20]


@router.get("")
def latest_asr_benchmark() -> dict[str, Any]:
    base = _default_eval_dir()
    summary_path = base / "deepgram_vs_databricks_summary.json"
    summary = _read_json(summary_path)
    if not summary:
        return {
            "available": False,
            "summary_path": str(summary_path),
            "message": "No ASR benchmark summary found. Run scripts/asr/07_deep_voice_model_eval.sh run first.",
        }

    deepgram_path, databricks_path = _provider_paths(summary, base)
    deepgram_rows = _read_jsonl(deepgram_path)
    databricks_rows = _read_jsonl(databricks_path)
    return {
        "available": True,
        "summary_path": str(summary_path),
        "deepgram_output": str(deepgram_path),
        "databricks_output": str(databricks_path),
        "summary": summary,
        "examples": _paired_examples(deepgram_rows, databricks_rows),
    }
