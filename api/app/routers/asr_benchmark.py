"""ASR benchmark result API for Deepgram vs Databricks ASR routes."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter
from genie_voice.i18n import LANGUAGE_SPECS, SUPPORTED_LANGUAGES, normalize_language
from genie_voice.ml_asr.benchmark_export import (
    _compute_entity_winners,
    load_ml_asr_benchmark,
    load_ml_asr_overview,
    ml_asr_available_languages,
)

router = APIRouter(prefix="/asr-benchmark", tags=["asr-benchmark"])

BenchmarkSource = Literal["auto", "ml_asr", "legacy"]
BenchmarkTier = Literal["business", "acoustic"]


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


def _legacy_language_models(language_code: str) -> dict[str, dict[str, Any]] | None:
    summary_path = next((path for path in _summary_candidates(language_code) if path.exists()), None)
    summary = _read_json(summary_path) if summary_path else None
    if not summary:
        return None
    providers = summary.get("providers") or {}
    models: dict[str, dict[str, Any]] = {}
    deepgram = providers.get("deepgram") or {}
    databricks = providers.get("databricks") or {}
    if deepgram:
        models["deepgram_nova3"] = {
            "model_id": "deepgram_nova3",
            "model_label": "Deepgram Nova-3",
            "provider": "deepgram",
            "clips": deepgram.get("clips") or 0,
            "wer": deepgram.get("avg_wer"),
            "cer": deepgram.get("avg_cer"),
            "critical_entity_accuracy": deepgram.get("avg_critical_entity_accuracy"),
            "unsafe_for_resolution_rate": deepgram.get("unsafe_for_resolution_rate"),
            "p95_latency_ms": (deepgram.get("latency_ms") or {}).get("p95"),
            "entity_groups": deepgram.get("entity_groups") or {},
        }
    if databricks:
        db_models = databricks.get("models") or ["databricks"]
        model_label = ", ".join(str(name) for name in db_models)
        models["databricks_route"] = {
            "model_id": "databricks_route",
            "model_label": f"Databricks · {model_label}",
            "provider": "databricks",
            "clips": databricks.get("clips") or 0,
            "wer": databricks.get("avg_wer"),
            "cer": databricks.get("avg_cer"),
            "critical_entity_accuracy": databricks.get("avg_critical_entity_accuracy"),
            "unsafe_for_resolution_rate": databricks.get("unsafe_for_resolution_rate"),
            "p95_latency_ms": (databricks.get("latency_ms") or {}).get("p95"),
            "entity_groups": databricks.get("entity_groups") or {},
        }
    return models or None


def _legacy_overview() -> dict[str, Any] | None:
    from genie_voice.ml_asr.benchmark_export import _compute_entity_winners, _compute_metric_winners, _scoreboard

    languages: dict[str, Any] = {}
    for code in SUPPORTED_LANGUAGES:
        models = _legacy_language_models(code)
        if not models:
            continue
        languages[code] = {
            "code": code,
            "label": LANGUAGE_SPECS[code].label,
            "models": models,
            "winners": _compute_metric_winners(models, tier="business"),
            "entity_winners": _compute_entity_winners(models),
            "source": "legacy",
        }
    if not languages:
        return None
    return {
        "available": True,
        "source": "legacy",
        "tiers": {
            "business": {
                "dataset_id": "multilingual_gold_holdout",
                "languages": languages,
                "scoreboard": _scoreboard(languages),
            }
        },
    }


@router.get("/overview")
def asr_benchmark_overview(source: BenchmarkSource = "auto") -> dict[str, Any]:
    repo_root = _repo_root()
    overview: dict[str, Any] | None = None
    if source in {"auto", "ml_asr"}:
        overview = load_ml_asr_overview(repo_root=repo_root)

    if source == "ml_asr" and not overview:
        return {
            "available": False,
            "source": "ml_asr",
            "message": (
                "No ml_asr overview index found. Sync Volume index to "
                ".run/ml_asr_eval/index.json (./scripts/ml_asr/05_eval.sh sync-index)."
            ),
        }

    if source in {"auto", "legacy"} and not overview:
        overview = _legacy_overview()

    if not overview:
        return {
            "available": False,
            "source": source,
            "message": "No ASR benchmark overview found.",
        }

    overview["available_languages"] = _available_languages(tier="business")
    return overview


def _legacy_benchmark(language_code: str) -> dict[str, Any] | None:
    summary_path = next((path for path in _summary_candidates(language_code) if path.exists()), None)
    summary = _read_json(summary_path) if summary_path else None
    if not summary or not summary_path:
        return None

    deepgram_path, databricks_path = _resolve_provider_paths(summary, summary_path)
    deepgram_rows = _read_jsonl(deepgram_path)
    databricks_rows = _read_jsonl(databricks_path)
    return {
        "available": True,
        "source": "legacy",
        "tier": "business",
        "dataset_id": "multilingual_gold_holdout",
        "language": language_code,
        "available_languages": _available_languages(tier="business"),
        "summary_path": str(summary_path),
        "deepgram_output": str(deepgram_path),
        "databricks_output": str(databricks_path),
        "clip_count": (summary.get("providers") or {}).get("deepgram", {}).get("clips"),
        "summary": summary,
        "examples": _paired_examples(deepgram_rows, databricks_rows),
    }


def _available_languages(*, tier: BenchmarkTier = "business") -> list[dict[str, str | bool]]:
    ml_asr_langs = ml_asr_available_languages(tier=tier, repo_root=_repo_root())
    out = []
    for code in SUPPORTED_LANGUAGES:
        spec = LANGUAGE_SPECS[code]
        legacy = any(path.exists() for path in _summary_candidates(code))
        ml_asr = ml_asr_langs.get(code, False)
        out.append(
            {
                "code": code,
                "label": spec.label,
                "english_name": spec.english_name,
                "available": ml_asr or legacy,
                "ml_asr_available": ml_asr,
                "legacy_available": legacy,
            }
        )
    return out


@router.get("")
def latest_asr_benchmark(
    language: str = "en-US",
    tier: BenchmarkTier = "business",
    source: BenchmarkSource = "auto",
) -> dict[str, Any]:
    language_code = normalize_language(language)
    repo_root = _repo_root()

    ml_asr_payload: dict[str, Any] | None = None
    if source in {"auto", "ml_asr"}:
        ml_asr_payload = load_ml_asr_benchmark(language=language_code, tier=tier, repo_root=repo_root)

    if source == "ml_asr":
        if not ml_asr_payload:
            return {
                "available": False,
                "source": "ml_asr",
                "tier": tier,
                "language": language_code,
                "available_languages": _available_languages(tier=tier),
                "message": (
                    "No ml_asr benchmark index found. Sync Volume index to "
                    ".run/ml_asr_eval/index.json (see scripts/ml_asr.sh eval / summarize)."
                ),
            }
        ml_asr_payload["available_languages"] = _available_languages(tier=tier)
        return ml_asr_payload

    if source == "auto" and ml_asr_payload:
        ml_asr_payload["available_languages"] = _available_languages(tier=tier)
        return ml_asr_payload

    legacy_payload = _legacy_benchmark(language_code)
    if legacy_payload:
        return legacy_payload

    return {
        "available": False,
        "source": source,
        "tier": tier,
        "language": language_code,
        "available_languages": _available_languages(tier=tier),
        "summary_path": str(_summary_candidates(language_code)[0]),
        "message": (
            "No ASR benchmark results found. Run ./scripts/ml_asr.sh eval and sync "
            "evaluations/ml_asr_eval/results/index.json to .run/ml_asr_eval/index.json, "
            "or run scripts/asr/07_deep_voice_model_eval.sh for legacy holdout eval."
        ),
    }
