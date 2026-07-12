from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Iterable

from genie_voice.asr_eval.metrics import score_transcript
from genie_voice.i18n import LanguageCode, normalize_language

from genie_voice.ml_asr.config import EvalConfig, load_config
from genie_voice.ml_asr.manifest import load_eval_manifest
from genie_voice.ml_asr.providers.factory import build_provider
from genie_voice.ml_asr.scoring.readiness import assess_readiness


def evaluate(
    *,
    config_path: str | None = None,
    languages: Iterable[str] | None = None,
    dataset_ids: Iterable[str] | None = None,
    tiers: Iterable[str] | None = None,
    model_ids: Iterable[str] | None = None,
    limit: int | None = None,
    use_remote_manifest: bool = True,
    skip_missing_audio: bool = True,
    volume_mode: bool | None = None,
) -> dict[str, dict]:
    from genie_voice.ml_asr.runtime import is_volume_mode

    config = load_config(config_path=config_path)
    volume_mode = is_volume_mode() if volume_mode is None else volume_mode
    selected_languages = _selected_languages(config, languages)
    selected_models = {str(model_id) for model_id in model_ids} if model_ids else None
    datasets = _datasets_to_evaluate(config, dataset_ids=dataset_ids, tiers=tiers)
    summary: dict[str, dict] = {}

    for dataset in datasets:
        dataset_summary: dict[str, dict] = {}
        for language in selected_languages:
            if language not in dataset.languages:
                continue
            manifest_path = config.manifest_path(dataset.dataset_id, language, volume_mode=volume_mode or use_remote_manifest)
            manifest = load_eval_manifest(manifest_path, splits=[dataset.split])
            clips = manifest.clips[:limit] if limit else manifest.clips
            if not clips:
                raise RuntimeError(f"No clips to evaluate for {dataset.dataset_id}/{language}: {manifest_path}")

            language_summary: dict[str, dict] = {}
            for model_spec in config.models_for_language(language):
                if selected_models and model_spec.model_id not in selected_models:
                    continue
                provider = build_provider(model_spec)
                result_dir = config.result_dir(language, dataset.dataset_id, model_spec.model_id, volume_mode=volume_mode)
                result_dir.mkdir(parents=True, exist_ok=True)
                output_path = result_dir / "results.jsonl"
                rows = _score_clips(
                    provider,
                    clips,
                    language=language,
                    eval_tier=dataset.eval_tier,
                    skip_missing_audio=skip_missing_audio,
                )
                if not rows:
                    language_summary[model_spec.model_id] = {
                        "status": "skipped",
                        "reason": "no readable audio clips",
                    }
                    continue
                _write_jsonl(output_path, rows)
                model_summary = _summarize_rows(rows, model_spec.model_id, model_spec.label, dataset.eval_tier)
                model_summary["status"] = "ok"
                model_summary["results_path"] = str(output_path)
                (result_dir / "summary.json").write_text(
                    json.dumps(model_summary, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                language_summary[model_spec.model_id] = model_summary
            dataset_summary[language] = language_summary
        summary[dataset.dataset_id] = dataset_summary
    return summary


def _datasets_to_evaluate(config: EvalConfig, *, dataset_ids: Iterable[str] | None, tiers: Iterable[str] | None):
    if dataset_ids:
        return [config.datasets[str(dataset_id)] for dataset_id in dataset_ids]
    if tiers:
        wanted = {str(tier) for tier in tiers}
        return [dataset for dataset in config.datasets.values() if dataset.eval_tier in wanted]
    return list(config.datasets.values())


def _selected_languages(config: EvalConfig, languages: Iterable[str] | None) -> list[LanguageCode]:
    if not languages:
        return list(config.eval_languages)
    return [normalize_language(str(language)) for language in languages]


def _score_clips(provider, clips, *, language: LanguageCode, eval_tier: str, skip_missing_audio: bool) -> list[dict]:
    rows: list[dict] = []
    for clip in clips:
        if skip_missing_audio:
            try:
                from genie_voice.ml_asr.audio import read_audio_bytes

                read_audio_bytes(clip.audio_path)
            except Exception as exc:  # noqa: BLE001
                rows.append(
                    {
                        "clip_id": clip.clip_id,
                        "language": language,
                        "eval_tier": eval_tier,
                        "model_id": provider.model_id,
                        "error": f"audio_unavailable: {exc}",
                    }
                )
                continue
        result = provider.transcribe(clip, language=language)
        score = score_transcript(
            clip.reference_transcript,
            result.transcript,
            clip.expected_entities,
        ).to_dict()
        readiness = assess_readiness(clip, result.transcript, score) if eval_tier == "business" else {}
        rows.append(
            {
                "clip_id": clip.clip_id,
                "language": language,
                "eval_tier": eval_tier,
                "scenario": clip.scenario,
                "model_id": provider.model_id,
                "model_label": provider.label,
                "audio_path": clip.audio_path,
                "reference_transcript": clip.reference_transcript,
                "transcript": result.transcript,
                "raw_transcript": result.raw_transcript,
                "latency_ms": result.latency_ms,
                "confidence": result.confidence,
                "error": result.error,
                "score": score,
                "readiness": readiness,
            }
        )
    return [row for row in rows if not row.get("error") or not str(row["error"]).startswith("audio_unavailable")]


def _summarize_rows(rows: list[dict], model_id: str, model_label: str, eval_tier: str) -> dict:
    ok_rows = [row for row in rows if not row.get("error")]
    summary = {
        "model_id": model_id,
        "model_label": model_label,
        "eval_tier": eval_tier,
        "clip_count": len(rows),
        "success_count": len(ok_rows),
        "error_count": len(rows) - len(ok_rows),
        "avg_wer": _avg([row["score"]["wer"] for row in ok_rows]),
        "avg_cer": _avg([row["score"]["cer"] for row in ok_rows]),
        "avg_entity_accuracy": _avg(
            [row["score"]["entity_accuracy"] for row in ok_rows if row["score"]["entity_accuracy"] is not None]
        ),
        "p95_latency_ms": _p95([row["latency_ms"] for row in ok_rows]),
        "avg_latency_ms": _avg([row["latency_ms"] for row in ok_rows]),
    }
    if eval_tier == "business":
        unsafe = [bool((row.get("readiness") or {}).get("unsafe_for_resolution")) for row in ok_rows]
        summary["unsafe_for_resolution_rate"] = _avg([1.0 if value else 0.0 for value in unsafe])
        summary["avg_critical_entity_accuracy"] = _avg(
            [
                (row.get("readiness") or {}).get("critical_entity_accuracy")
                for row in ok_rows
                if (row.get("readiness") or {}).get("critical_entity_accuracy") is not None
            ]
        )
        from genie_voice.ml_asr.benchmark_export import _entity_groups_from_rows

        summary["entity_groups"] = _entity_groups_from_rows(ok_rows)
    return summary


def _avg(values: list[float | int]) -> float | None:
    if not values:
        return None
    return round(mean(values), 4)


def _p95(values: list[int]) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, int(round(0.95 * (len(ordered) - 1))))
    return ordered[index]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def preflight(*, config_path: str | None = None, languages: Iterable[str] | None = None) -> dict:
    from genie_voice.config import get_settings

    config = load_config(config_path=config_path)
    selected = _selected_languages(config, languages)
    checks = {
        "deepgram_key_configured": bool(get_settings().secrets.deepgram_api_key.strip()),
        "datasets": {},
    }
    for dataset in config.datasets.values():
        dataset_checks = {"languages": {}}
        for language in selected:
            if language not in dataset.languages:
                continue
            lang_spec = dataset.languages[language]
            manifest = load_eval_manifest(lang_spec.local_manifest_path, splits=[dataset.split])
            dataset_checks["languages"][language] = {
                "clips": len(manifest),
                "eval_tier": dataset.eval_tier,
                "audio_mode": dataset.audio_mode,
            }
        checks["datasets"][dataset.dataset_id] = dataset_checks
    return checks
