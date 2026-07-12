from __future__ import annotations

import json
import re
import statistics
from pathlib import Path
from typing import Any, Iterable

from genie_voice.i18n import LanguageCode, normalize_language

from genie_voice.ml_asr.audio import read_audio_bytes
from genie_voice.ml_asr.config import EvalConfig, load_config
from genie_voice.ml_asr.manifest import load_eval_manifest

ENTITY_GROUPS = (
    "invoice_ids",
    "amounts",
    "dates",
    "billing_actions",
    "confirmations",
    "refusals",
    "account_terms",
)


def audit_dataset(
    *,
    config_path: str | None = None,
    languages: Iterable[str] | None = None,
    dataset_ids: Iterable[str] | None = None,
    tiers: Iterable[str] | None = None,
    use_remote_manifest: bool = False,
    read_audio: bool = True,
    audio_sample_limit: int = 10,
) -> dict[str, Any]:
    config = load_config(config_path=config_path)
    selected_languages = _selected_languages(config, languages)
    if dataset_ids:
        selected_dataset_ids = list(dataset_ids)
    elif tiers:
        wanted = {str(tier) for tier in tiers}
        selected_dataset_ids = [
            dataset_id
            for dataset_id, dataset in config.datasets.items()
            if dataset.eval_tier in wanted
        ]
    else:
        selected_dataset_ids = config.dataset_ids()

    tier_reports: dict[str, list[dict[str, Any]]] = {tier: [] for tier in config.eval_tiers}
    for dataset_id in selected_dataset_ids:
        dataset = config.datasets[dataset_id]
        for language in selected_languages:
            if language not in dataset.languages:
                continue
            lang_spec = dataset.languages[language]
            manifest_path = lang_spec.remote_manifest_path if use_remote_manifest else lang_spec.local_manifest_path
            tier_reports[dataset.eval_tier].append(
                _audit_language_dataset(
                    config,
                    dataset_id=dataset_id,
                    eval_tier=dataset.eval_tier,
                    split=dataset.split,
                    audio_mode=dataset.audio_mode,
                    language=language,
                    manifest_path=manifest_path,
                    read_audio=read_audio and dataset.audio_mode not in {"scaffold"},
                    audio_sample_limit=audio_sample_limit,
                )
            )

    gates = _holistic_gates(config, tier_reports)
    return {
        "eval_plan": {
            "tiers": list(config.eval_tiers),
            "languages": list(config.eval_languages),
            "min_clips_per_language": config.min_clips_per_language,
        },
        "tiers": tier_reports,
        "gates": gates,
    }


def _selected_languages(config: EvalConfig, languages: Iterable[str] | None) -> list[LanguageCode]:
    if not languages:
        return list(config.eval_languages)
    selected = [normalize_language(str(language)) for language in languages]
    return selected


def _audit_language_dataset(
    config: EvalConfig,
    *,
    dataset_id: str,
    eval_tier: str,
    split: str,
    audio_mode: str,
    language: LanguageCode,
    manifest_path: str,
    read_audio: bool,
    audio_sample_limit: int,
) -> dict[str, Any]:
    manifest = load_eval_manifest(manifest_path, splits=[split])
    clips = manifest.clips
    scenario_counts: dict[str, int] = {}
    entity_coverage = {group: 0 for group in ENTITY_GROUPS}
    missing_transcripts = 0
    pending_audio = 0

    for clip in clips:
        scenario = clip.scenario or "unknown"
        scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1
        if not (clip.reference_transcript or "").strip():
            missing_transcripts += 1
        for group in ENTITY_GROUPS:
            values = getattr(clip.expected_entities, group, [])
            if values:
                entity_coverage[group] += 1
        if audio_mode == "scaffold":
            pending_audio += 1

    audio_checks: list[dict[str, Any]] = []
    audio_failures = 0
    should_read_audio = read_audio and audio_mode not in {"scaffold"}
    if should_read_audio:
        for clip in clips[:audio_sample_limit]:
            try:
                payload = read_audio_bytes(clip.audio_path)
                audio_checks.append({"clip_id": clip.clip_id, "bytes": len(payload), "ok": len(payload) > 0})
            except Exception as exc:  # noqa: BLE001
                audio_failures += 1
                audio_checks.append({"clip_id": clip.clip_id, "ok": False, "error": str(exc)})

    return {
        "dataset_id": dataset_id,
        "language": language,
        "eval_tier": eval_tier,
        "audio_mode": audio_mode,
        "manifest_path": manifest_path,
        "clip_count": len(clips),
        "scenario_coverage": scenario_counts,
        "entity_group_rows": entity_coverage,
        "missing_transcripts": missing_transcripts,
        "pending_audio_rows": pending_audio,
        "transcript_words": _summary_stats([len((clip.reference_transcript or "").split()) for clip in clips]),
        "audio_sample": {
            "requested": min(audio_sample_limit, len(clips)) if should_read_audio else 0,
            "failures": audio_failures,
            "checks": audio_checks,
        },
    }


def _summary_stats(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "max": None, "mean": None, "p50": None}
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "max": ordered[-1],
        "mean": round(statistics.mean(ordered), 2),
        "p50": ordered[len(ordered) // 2],
    }


def _holistic_gates(config: EvalConfig, tier_reports: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    tier_status: dict[str, Any] = {}
    for tier in config.eval_tiers:
        reports = tier_reports.get(tier, [])
        min_clips_required = config.min_clips_per_language.get(tier, 10)
        min_clips = min((report["clip_count"] for report in reports), default=0) if reports else 0
        missing = max((report["missing_transcripts"] for report in reports), default=0)
        audio_failures = sum(report["audio_sample"]["failures"] for report in reports)
        pending_audio = sum(report.get("pending_audio_rows", 0) for report in reports)

        if tier == "acoustic":
            ready = bool(reports) and min_clips >= min_clips_required and missing == 0 and audio_failures == 0
            notes = "Licensed read speech for WER/CER ranking."
        elif tier == "business":
            entity_ok = all(
                (
                    report["entity_group_rows"].get("invoice_ids", 0) > 0
                    or report["entity_group_rows"].get("amounts", 0) > 0
                )
                and report["entity_group_rows"].get("billing_actions", 0) > 0
                for report in reports
            ) if reports else False
            scenario_ok = all(len(report["scenario_coverage"]) >= 5 for report in reports) if reports else False
            ready = (
                bool(reports)
                and min_clips >= min_clips_required
                and missing == 0
                and entity_ok
                and scenario_ok
            )
            notes = (
                "Business manifests ready for entity scoring. "
                "Recorded audio still required before promotion decisions."
                if pending_audio
                else "Business holdout ready — licensed human speech with mined entity labels."
            )
        else:
            ready = False
            notes = f"Unknown tier: {tier}"

        tier_status[tier] = {
            "ready": ready,
            "min_clips": min_clips,
            "min_clips_required": min_clips_required,
            "audio_failures": audio_failures,
            "pending_audio_rows": pending_audio,
            "notes": notes,
        }

    holistic_ready = all(tier_status.get(tier, {}).get("ready") for tier in config.eval_tiers)
    model_eval_ready = tier_status.get("acoustic", {}).get("ready") and (
        tier_status.get("business", {}).get("ready") or tier_status.get("business", {}).get("pending_audio_rows", 0) > 0
    )
    promotion_ready = holistic_ready and tier_status.get("business", {}).get("pending_audio_rows", 0) == 0

    return {
        "holistic_dataset_ready": holistic_ready,
        "model_eval_ready": model_eval_ready,
        "promotion_ready": promotion_ready,
        "recommended_next_step": _recommended_next_step(tier_status, promotion_ready),
        "tiers": tier_status,
    }


def _recommended_next_step(tier_status: dict[str, Any], promotion_ready: bool) -> str:
    if promotion_ready:
        return "run evaluate-all and compare business critical_entity_accuracy + unsafe_for_resolution"
    if not tier_status.get("acoustic", {}).get("ready"):
        return "finish acoustic dataset prep (FLEURS) and re-run audit-dataset"
    if tier_status.get("business", {}).get("pending_audio_rows", 0) > 0:
        return "upload/record business holdout audio, then re-run audit-dataset before promotion"
    if not tier_status.get("business", {}).get("ready"):
        return "finish business manifest prep and re-run audit-dataset"
    return "run evaluate-all on acoustic tier first, then business once audio is recorded"


def write_audit_report(report: dict[str, Any], *, config: EvalConfig, volume_mode: bool | None = None) -> Path:
    from genie_voice.ml_asr.runtime import is_volume_mode

    volume_mode = is_volume_mode() if volume_mode is None else volume_mode
    out = Path(config.remote_audit_path if volume_mode else Path(config.local_root) / "dataset_audit.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out
