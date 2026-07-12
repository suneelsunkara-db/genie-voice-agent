"""Load ml_asr eval index for the ASR benchmark UI."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from genie_voice.i18n import LANGUAGE_SPECS, LanguageCode, SUPPORTED_LANGUAGES, normalize_language
from genie_voice.ml_asr.config import DEFAULT_CONFIG_PATH, load_config

_ML_ASR_EVAL_MARKER = "/evaluations/ml_asr_eval/"


def index_candidates(*, repo_root: Path | None = None) -> list[Path]:
    root = repo_root or Path(__file__).resolve().parents[3]
    return [
        root / ".run" / "ml_asr_eval" / "index.json",
        root / ".run" / "ml_asr_eval" / "results" / "index.json",
    ]


def resolve_index_path(*, repo_root: Path | None = None) -> Path | None:
    return next((path for path in index_candidates(repo_root=repo_root) if path.is_file()), None)


def dataset_id_for_tier(tier: str) -> str:
    if tier == "acoustic":
        return "fleurs_acoustic_v1"
    return "fleurs_business_v1"


def load_ml_asr_benchmark(
    *,
    language: str,
    tier: str = "business",
    repo_root: Path | None = None,
) -> dict[str, Any] | None:
    language_code = normalize_language(language)
    index_path = resolve_index_path(repo_root=repo_root)
    if not index_path:
        return None

    index = json.loads(index_path.read_text(encoding="utf-8"))
    dataset_id = dataset_id_for_tier(tier)
    dataset = (index.get("datasets") or {}).get(dataset_id) or {}
    language_block = (dataset.get("languages") or {}).get(language_code)
    if not language_block:
        return None

    models_raw: dict[str, dict[str, Any]] = language_block.get("models") or {}
    if not models_raw or all(entry.get("status") == "missing" for entry in models_raw.values()):
        return None

    config = load_config(config_path=DEFAULT_CONFIG_PATH, repo_root=repo_root or index_path.parents[2])
    eval_matrix = (index.get("catalog") or {}).get("eval_matrix") or config.eval_matrix
    matrix = list(eval_matrix.get(language_code) or [])
    deepgram_id = next((model_id for model_id in matrix if model_id.startswith("deepgram")), None)
    databricks_id = next((model_id for model_id in matrix if model_id.startswith("databricks")), None)

    models: dict[str, dict[str, Any]] = {}
    for model_id, raw in models_raw.items():
        if raw.get("status") == "missing":
            continue
        rows = _read_results_rows(raw.get("results_path"), repo_root=repo_root)
        models[model_id] = _to_provider_summary(model_id, raw, rows)

    deepgram = models.get(deepgram_id or "") if deepgram_id else None
    databricks = models.get(databricks_id or "") if databricks_id else None
    ranking = language_block.get("ranking") or []
    clip_count = max((summary.get("clips") or 0 for summary in models.values()), default=0)

    deepgram_rows = _read_results_rows(models_raw.get(deepgram_id or "", {}).get("results_path"), repo_root=repo_root)
    databricks_rows = _read_results_rows(models_raw.get(databricks_id or "", {}).get("results_path"), repo_root=repo_root)

    dg_critical = (deepgram or {}).get("avg_critical_entity_accuracy")
    db_critical = (databricks or {}).get("avg_critical_entity_accuracy")
    dg_p95 = ((deepgram or {}).get("latency_ms") or {}).get("p95")
    db_p95 = ((databricks or {}).get("latency_ms") or {}).get("p95")

    return {
        "available": True,
        "source": "ml_asr",
        "tier": tier,
        "dataset_id": dataset_id,
        "language": language_code,
        "index_path": str(index_path),
        "clip_count": clip_count,
        "models": models,
        "ranking": ranking,
        "summary_path": str(index_path),
        "deepgram_output": models_raw.get(deepgram_id or "", {}).get("results_path"),
        "databricks_output": models_raw.get(databricks_id or "", {}).get("results_path"),
        "summary": {
            "language": language_code,
            "manifest": _manifest_path(index, dataset_id, language_code),
            "providers": {
                "deepgram": deepgram,
                "databricks": databricks,
            },
            "pairwise": {"paired_clips": clip_count, "winner_counts": {}},
            "promotion_read": {
                "recommended_headline": f"ml_asr FLEURS {tier} smoke eval ({clip_count} clips max/lang).",
                "databricks_business_delta": None
                if dg_critical is None or db_critical is None
                else db_critical - dg_critical,
                "databricks_wer_delta": None
                if not deepgram or not databricks
                else (databricks.get("avg_wer") or 0) - (deepgram.get("avg_wer") or 0),
                "databricks_p95_latency_delta_ms": None
                if dg_p95 is None or db_p95 is None
                else db_p95 - dg_p95,
                "paired_clips": clip_count,
            },
        },
        "examples": _paired_examples_ml_asr(deepgram_rows, databricks_rows),
    }


def ml_asr_available_languages(*, tier: str = "business", repo_root: Path | None = None) -> dict[LanguageCode, bool]:
    index_path = resolve_index_path(repo_root=repo_root)
    if not index_path:
        return {code: False for code in SUPPORTED_LANGUAGES}

    index = json.loads(index_path.read_text(encoding="utf-8"))
    dataset_id = dataset_id_for_tier(tier)
    dataset = (index.get("datasets") or {}).get(dataset_id) or {}
    languages = dataset.get("languages") or {}
    out: dict[LanguageCode, bool] = {}
    for code in SUPPORTED_LANGUAGES:
        block = languages.get(code)
        if not block:
            out[code] = False
            continue
        models = block.get("models") or {}
        out[code] = any(entry.get("status") != "missing" for entry in models.values())
    return out


def _manifest_path(index: dict[str, Any], dataset_id: str, language: LanguageCode) -> str | None:
    catalog = index.get("catalog") or {}
    datasets = catalog.get("datasets") or {}
    entry = datasets.get(dataset_id) or {}
    languages = entry.get("languages") or {}
    lang = languages.get(language) or {}
    return lang.get("manifest")


def _local_results_path(results_path: str | None, *, repo_root: Path | None) -> Path | None:
    if not results_path:
        return None
    path = Path(results_path)
    if path.is_file():
        return path
    if _ML_ASR_EVAL_MARKER not in results_path:
        return None
    root = repo_root or Path(__file__).resolve().parents[3]
    suffix = results_path.split(_ML_ASR_EVAL_MARKER, 1)[1]
    local = root / ".run" / "ml_asr_eval" / suffix
    return local if local.is_file() else None


def _read_results_rows(results_path: str | None, *, repo_root: Path | None) -> list[dict[str, Any]]:
    local = _local_results_path(results_path, repo_root=repo_root)
    if not local:
        return []
    rows: list[dict[str, Any]] = []
    for line in local.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return [row for row in rows if not row.get("error")]


def _to_provider_summary(model_id: str, raw: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    provider = "deepgram" if model_id.startswith("deepgram") else "databricks"
    summary: dict[str, Any] = {
        "clips": raw.get("success_count") or raw.get("clip_count") or 0,
        "provider": [provider],
        "models": [raw.get("model_label") or model_id],
        "avg_wer": raw.get("avg_wer"),
        "avg_cer": raw.get("avg_cer"),
        "avg_entity_accuracy": raw.get("avg_entity_accuracy"),
        "avg_critical_entity_accuracy": raw.get("avg_critical_entity_accuracy"),
        "unsafe_for_resolution_rate": raw.get("unsafe_for_resolution_rate"),
        "latency_ms": {
            "p50": None,
            "p90": None,
            "p95": raw.get("p95_latency_ms"),
            "p99": None,
            "avg": raw.get("avg_latency_ms"),
        },
    }
    if rows:
        summary.update(_aggregate_row_details(rows))
    return summary


def _entity_groups_from_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    entity_totals: dict[str, dict[str, int]] = defaultdict(lambda: {"expected": 0, "matched": 0})
    for row in rows:
        for group, score in (row.get("score") or {}).get("entity_scores", {}).items():
            entity_totals[group]["expected"] += int(score.get("expected") or 0)
            entity_totals[group]["matched"] += int(score.get("matched") or 0)
    return {
        group: {
            **counts,
            "accuracy": None if counts["expected"] == 0 else counts["matched"] / counts["expected"],
        }
        for group, counts in sorted(entity_totals.items())
    }


def _aggregate_row_details(rows: list[dict[str, Any]]) -> dict[str, Any]:
    unsafe_reasons: Counter[str] = Counter()
    for row in rows:
        readiness = row.get("readiness") or row.get("app_readiness") or {}
        unsafe_reasons.update(readiness.get("unsafe_reasons") or [])
    return {
        "entity_groups": _entity_groups_from_rows(rows),
        "unsafe_reason_counts": dict(sorted(unsafe_reasons.items())),
    }


def _readiness(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("readiness") or row.get("app_readiness") or {}


def _paired_examples_ml_asr(
    deepgram_rows: list[dict[str, Any]],
    databricks_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    deepgram_by_clip = {str(row.get("clip_id")): row for row in deepgram_rows}
    databricks_by_clip = {str(row.get("clip_id")): row for row in databricks_rows}
    examples: list[dict[str, Any]] = []
    for clip_id in sorted(set(deepgram_by_clip) & set(databricks_by_clip)):
        dg = deepgram_by_clip[clip_id]
        db = databricks_by_clip[clip_id]
        dg_readiness = _readiness(dg)
        db_readiness = _readiness(db)
        dg_critical = dg_readiness.get("critical_entity_accuracy")
        db_critical = db_readiness.get("critical_entity_accuracy")
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
                "databricks_unsafe_reasons": db_readiness.get("unsafe_reasons") or [],
                "deepgram_unsafe_reasons": dg_readiness.get("unsafe_reasons") or [],
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


def load_ml_asr_overview(*, repo_root: Path | None = None) -> dict[str, Any] | None:
    index_path = resolve_index_path(repo_root=repo_root)
    if not index_path:
        return None

    index = json.loads(index_path.read_text(encoding="utf-8"))
    tiers: dict[str, Any] = {}
    for tier in ("business", "acoustic"):
        dataset_id = dataset_id_for_tier(tier)
        dataset = (index.get("datasets") or {}).get(dataset_id) or {}
        languages: dict[str, Any] = {}
        for code in SUPPORTED_LANGUAGES:
            block = (dataset.get("languages") or {}).get(code)
            if not block:
                continue
            models_raw = block.get("models") or {}
            models = {
                model_id: _model_metrics(model_id, raw, repo_root=repo_root, tier=tier)
                for model_id, raw in models_raw.items()
                if raw.get("status") != "missing"
            }
            if not models:
                continue
            if tier == "business":
                _merge_legacy_entity_groups(code, models, repo_root=repo_root)
            languages[code] = {
                "code": code,
                "label": LANGUAGE_SPECS[code].label,
                "models": models,
                "winners": _compute_metric_winners(models, tier=tier),
                "entity_winners": _compute_entity_winners(models) if tier == "business" else {},
            }
        if languages:
            tiers[tier] = {
                "dataset_id": dataset_id,
                "languages": languages,
                "scoreboard": _scoreboard(languages),
            }

    if not tiers:
        return None

    return {
        "available": True,
        "source": "ml_asr",
        "index_path": str(index_path),
        "tiers": tiers,
    }


def _model_metrics(
    model_id: str,
    raw: dict[str, Any],
    *,
    repo_root: Path | None = None,
    tier: str = "business",
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "model_id": model_id,
        "model_label": str(raw.get("model_label") or model_id),
        "provider": "deepgram" if model_id.startswith("deepgram") else "databricks",
        "clips": raw.get("success_count") or raw.get("clip_count") or 0,
        "wer": raw.get("avg_wer"),
        "cer": raw.get("avg_cer"),
        "critical_entity_accuracy": raw.get("avg_critical_entity_accuracy"),
        "unsafe_for_resolution_rate": raw.get("unsafe_for_resolution_rate"),
        "p95_latency_ms": raw.get("p95_latency_ms"),
    }
    if tier != "business":
        return metrics

    entity_groups = raw.get("entity_groups")
    if isinstance(entity_groups, dict) and entity_groups:
        metrics["entity_groups"] = entity_groups
        metrics["entity_groups_source"] = "ml_asr_summary"
    else:
        rows = _read_results_rows(raw.get("results_path"), repo_root=repo_root)
        if rows:
            groups = _entity_groups_from_rows(rows)
            if groups:
                metrics["entity_groups"] = groups
                metrics["entity_groups_source"] = "ml_asr_results"
    return metrics


def _merge_legacy_entity_groups(
    language_code: LanguageCode,
    models: dict[str, dict[str, Any]],
    *,
    repo_root: Path | None,
) -> None:
    legacy_models = _legacy_entity_models_for_language(language_code, repo_root=repo_root)
    if not legacy_models:
        return
    for model_id, model in models.items():
        if model.get("entity_groups"):
            continue
        legacy_key = "deepgram_nova3" if model["provider"] == "deepgram" else "databricks_route"
        legacy = legacy_models.get(legacy_key) or {}
        groups = legacy.get("entity_groups")
        if groups:
            model["entity_groups"] = groups
            model["entity_groups_source"] = "legacy_holdout"


def _compute_entity_winners(models: dict[str, dict[str, Any]]) -> dict[str, Any]:
    groups = sorted(
        {
            group
            for model in models.values()
            for group in (model.get("entity_groups") or {}).keys()
        }
    )
    winners: dict[str, Any] = {}
    for group in groups:
        candidates = []
        for model_id, model in models.items():
            accuracy = (model.get("entity_groups") or {}).get(group, {}).get("accuracy")
            if accuracy is not None:
                candidates.append((model_id, accuracy))
        if not candidates:
            continue
        best_value = max(value for _, value in candidates)
        best_ids = [model_id for model_id, value in candidates if abs(value - best_value) < 1e-9]
        if len(best_ids) == 1:
            model_id = best_ids[0]
            model = models[model_id]
            winners[group] = {
                "model_id": model_id,
                "model_label": model["model_label"],
                "provider": model["provider"],
                "tie": False,
            }
        else:
            winners[group] = {
                "tie": True,
                "model_ids": best_ids,
                "model_labels": [models[mid]["model_label"] for mid in best_ids],
            }
    return winners


def _metric_specs(tier: str) -> list[tuple[str, bool]]:
    specs = [("wer", True), ("cer", True), ("p95_latency_ms", True)]
    if tier == "business":
        specs += [("critical_entity_accuracy", False), ("unsafe_for_resolution_rate", True)]
    return specs


def _compute_metric_winners(models: dict[str, dict[str, Any]], *, tier: str) -> dict[str, Any]:
    winners: dict[str, Any] = {}
    for metric, lower_is_better in _metric_specs(tier):
        candidates = [(model_id, model[metric]) for model_id, model in models.items() if model.get(metric) is not None]
        if not candidates:
            continue
        if lower_is_better:
            best_value = min(value for _, value in candidates)
            best_ids = [model_id for model_id, value in candidates if abs(value - best_value) < 1e-9]
        else:
            best_value = max(value for _, value in candidates)
            best_ids = [model_id for model_id, value in candidates if abs(value - best_value) < 1e-9]
        if len(best_ids) == 1:
            model_id = best_ids[0]
            model = models[model_id]
            winners[metric] = {
                "model_id": model_id,
                "model_label": model["model_label"],
                "provider": model["provider"],
                "tie": False,
            }
        else:
            winners[metric] = {
                "tie": True,
                "model_ids": best_ids,
                "model_labels": [models[model_id]["model_label"] for model_id in best_ids],
            }
    return winners


def _scoreboard(languages: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for language in languages.values():
        for metric, winner in (language.get("winners") or {}).items():
            if winner.get("tie"):
                continue
            model_id = winner.get("model_id")
            if model_id:
                counts[metric][model_id] += 1
    scoreboard: dict[str, list[dict[str, Any]]] = {}
    for metric, counter in counts.items():
        scoreboard[metric] = [
            {"model_id": model_id, "wins": win_count}
            for model_id, win_count in counter.most_common()
        ]
    return scoreboard


def _legacy_summary_candidates(language: str, *, repo_root: Path | None) -> list[Path]:
    root = repo_root or Path(__file__).resolve().parents[3]
    code = normalize_language(language)
    local_base = root / ".run" / "asr_model_training" / "evaluations" / "voice_model_deep_eval"
    packaged_base = root / "api" / "app" / "data" / "asr_benchmarks"
    names = [code, code.split("-")[0]]
    candidates = [local_base / name / "deepgram_vs_databricks_summary.json" for name in names]
    candidates += [packaged_base / name / "deepgram_vs_databricks_summary.json" for name in names]
    if code == "en-US":
        candidates.insert(0, local_base / "deepgram_vs_databricks_summary.json")
        candidates.append(packaged_base / "deepgram_vs_databricks_summary.json")
        candidates.append(packaged_base / "en" / "deepgram_vs_databricks_summary.json")
    return candidates


def _legacy_entity_models_for_language(
    language_code: LanguageCode,
    *,
    repo_root: Path | None,
) -> dict[str, dict[str, Any]] | None:
    summary_path = next((path for path in _legacy_summary_candidates(language_code, repo_root=repo_root) if path.is_file()), None)
    if not summary_path:
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    providers = summary.get("providers") or {}
    models: dict[str, dict[str, Any]] = {}
    deepgram = providers.get("deepgram") or {}
    databricks = providers.get("databricks") or {}
    if deepgram.get("entity_groups"):
        models["deepgram_nova3"] = {"entity_groups": deepgram["entity_groups"]}
    if databricks.get("entity_groups"):
        models["databricks_route"] = {"entity_groups": databricks["entity_groups"]}
    return models or None
