"""ASR benchmark result API for Deepgram vs fine-tuned Databricks ASR."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from genie_voice.i18n import LANGUAGE_SPECS, SUPPORTED_LANGUAGES, normalize_language

router = APIRouter(prefix="/asr-benchmark", tags=["asr-benchmark"])


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_eval_dir() -> Path:
    return _repo_root() / ".run" / "asr_model_training" / "evaluations" / "voice_model_deep_eval"


def _packaged_eval_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "asr_benchmarks"


def _summary_candidates(language: str) -> list[Path]:
    code = normalize_language(language)
    local_base = _default_eval_dir()
    packaged_base = _packaged_eval_dir()
    names = [code, code.split("-")[0]]
    candidates = [local_base / name / "deepgram_vs_databricks_summary.json" for name in names]
    candidates += [packaged_base / name / "deepgram_vs_databricks_summary.json" for name in names]
    if code == "en-US":
        candidates.insert(0, local_base / "deepgram_vs_databricks_summary.json")
        candidates.append(packaged_base / "deepgram_vs_databricks_summary.json")
        candidates.append(packaged_base / "en" / "deepgram_vs_databricks_summary.json")
    return candidates


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


def _resolve_provider_paths(summary: dict[str, Any], summary_path: Path) -> tuple[Path, Path]:
    deepgram_path, databricks_path = _provider_paths(summary, summary_path.parent)
    if not deepgram_path.exists():
        deepgram_path = summary_path.parent / deepgram_path.name
    if not databricks_path.exists():
        databricks_path = summary_path.parent / databricks_path.name
    return deepgram_path, databricks_path


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
def latest_asr_benchmark(language: str = "en-US") -> dict[str, Any]:
    language_code = normalize_language(language)
    summary_path = next((path for path in _summary_candidates(language_code) if path.exists()), None)
    summary = _read_json(summary_path) if summary_path else None
    if not summary:
        return {
            "available": False,
            "language": language_code,
            "available_languages": _available_languages(),
            "summary_path": str(_summary_candidates(language_code)[0]),
            "message": (
                "No ASR benchmark summary found for this language. "
                "Run scripts/asr/07_deep_voice_model_eval.sh run with ASR_EVAL_LANGUAGE set first."
            ),
        }

    deepgram_path, databricks_path = _resolve_provider_paths(summary, summary_path)
    deepgram_rows = _read_jsonl(deepgram_path)
    databricks_rows = _read_jsonl(databricks_path)
    return {
        "available": True,
        "language": language_code,
        "available_languages": _available_languages(),
        "summary_path": str(summary_path),
        "deepgram_output": str(deepgram_path),
        "databricks_output": str(databricks_path),
        "summary": summary,
        "examples": _paired_examples(deepgram_rows, databricks_rows),
    }


def _available_languages() -> list[dict[str, str | bool]]:
    out = []
    for code in SUPPORTED_LANGUAGES:
        spec = LANGUAGE_SPECS[code]
        out.append(
            {
                "code": code,
                "label": spec.label,
                "english_name": spec.english_name,
                "available": any(path.exists() for path in _summary_candidates(code)),
            }
        )
    return out
