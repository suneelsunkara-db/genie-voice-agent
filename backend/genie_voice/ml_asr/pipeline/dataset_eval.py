from __future__ import annotations

import io
import json
import re
import statistics
from pathlib import Path
from typing import Any, Iterable

from genie_voice.asr_eval.manifest import ASRGoldClip
from genie_voice.i18n import LanguageCode, normalize_language

from genie_voice.ml_asr.audio import read_audio_bytes
from genie_voice.ml_asr.config import EvalConfig, load_config
from genie_voice.ml_asr.datasets.entity_mining import (
    NUMERIC_ENTITY_SCENARIO,
    classify_scenario,
    entity_quality_score,
    is_business_relevant,
    scenario_keyword_hits,
)
from genie_voice.ml_asr.manifest import load_eval_manifest

STRICT_INVOICE_RE = re.compile(
    r"(?:\bINV[-\s]?\w+\b|\binvoice\s*#?\s*\w+\b|\bbill\s+(?:number\s+)?[A-Z0-9-]{4,}\b)",
    re.IGNORECASE,
)
YEAR_AMOUNT_RE = re.compile(r"^(?:1\d{3}|20\d{2})$")
GENERIC_NUMBER_RE = re.compile(r"^\d{1,3}$")
SUSPICIOUS_INVOICE_TOKENS = {
    "invasion",
    "in",
    "on",
    "no",
    "ms",
    "us",
    "id",
    "or",
    "an",
    "as",
    "at",
    "to",
    "of",
    "for",
    "the",
    "and",
}

DEFAULT_MIN_ENTITY_QUALITY = 3
DEFAULT_MIN_SCENARIO_CONSISTENCY = 0.6
DEFAULT_MAX_SUSPICIOUS_LABEL_RATE = 0.25
DEFAULT_MAX_DUPLICATE_TRANSCRIPT_RATE = 0.05
DEFAULT_MIN_AUDIO_SECONDS = 1.0
DEFAULT_MAX_AUDIO_SECONDS = 30.0


def evaluate_datasets(
    *,
    config_path: str | None = None,
    languages: Iterable[str] | None = None,
    dataset_ids: Iterable[str] | None = None,
    tiers: Iterable[str] | None = None,
    use_remote_manifest: bool = False,
    read_audio: bool = True,
    audio_sample_limit: int = 10,
    min_entity_quality: int = DEFAULT_MIN_ENTITY_QUALITY,
) -> dict[str, Any]:
    config = load_config(config_path=config_path)
    selected_languages = _selected_languages(config, languages)
    selected_dataset_ids = _selected_dataset_ids(config, dataset_ids, tiers)

    tier_reports: dict[str, list[dict[str, Any]]] = {tier: [] for tier in config.eval_tiers}
    for dataset_id in selected_dataset_ids:
        dataset = config.datasets[dataset_id]
        for language in selected_languages:
            if language not in dataset.languages:
                continue
            lang_spec = dataset.languages[language]
            manifest_path = lang_spec.remote_manifest_path if use_remote_manifest else lang_spec.local_manifest_path
            if dataset.eval_tier == "business":
                report = _evaluate_business_manifest(
                    dataset_id=dataset_id,
                    language=language,
                    split=dataset.split,
                    manifest_path=manifest_path,
                    read_audio=read_audio,
                    audio_sample_limit=audio_sample_limit,
                    min_entity_quality=min_entity_quality,
                )
            else:
                report = _evaluate_acoustic_manifest(
                    dataset_id=dataset_id,
                    language=language,
                    split=dataset.split,
                    manifest_path=manifest_path,
                    read_audio=read_audio,
                    audio_sample_limit=audio_sample_limit,
                )
            tier_reports[dataset.eval_tier].append(report)

    gates = _quality_gates(tier_reports)
    return {
        "eval_plan": {
            "tiers": list(config.eval_tiers),
            "languages": list(config.eval_languages),
        },
        "thresholds": {
            "min_entity_quality": min_entity_quality,
            "min_scenario_consistency": DEFAULT_MIN_SCENARIO_CONSISTENCY,
            "max_suspicious_label_rate": DEFAULT_MAX_SUSPICIOUS_LABEL_RATE,
            "max_duplicate_transcript_rate": DEFAULT_MAX_DUPLICATE_TRANSCRIPT_RATE,
            "audio_seconds": [DEFAULT_MIN_AUDIO_SECONDS, DEFAULT_MAX_AUDIO_SECONDS],
        },
        "tiers": tier_reports,
        "gates": gates,
    }


def write_dataset_eval_report(
    report: dict[str, Any],
    *,
    config: EvalConfig,
    volume_mode: bool | None = None,
) -> Path:
    from genie_voice.ml_asr.runtime import is_volume_mode

    volume_mode = is_volume_mode() if volume_mode is None else volume_mode
    out = Path(config.remote_results_dir if volume_mode else config.local_root) / "dataset_quality_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def _selected_languages(config: EvalConfig, languages: Iterable[str] | None) -> list[LanguageCode]:
    if not languages:
        return list(config.eval_languages)
    return [normalize_language(str(language)) for language in languages]


def _selected_dataset_ids(
    config: EvalConfig,
    dataset_ids: Iterable[str] | None,
    tiers: Iterable[str] | None,
) -> list[str]:
    if dataset_ids:
        return list(dataset_ids)
    if tiers:
        wanted = {str(tier) for tier in tiers}
        return [dataset_id for dataset_id, dataset in config.datasets.items() if dataset.eval_tier in wanted]
    return config.dataset_ids()


def _evaluate_business_manifest(
    *,
    dataset_id: str,
    language: LanguageCode,
    split: str,
    manifest_path: str,
    read_audio: bool,
    audio_sample_limit: int,
    min_entity_quality: int,
) -> dict[str, Any]:
    manifest = load_eval_manifest(manifest_path, splits=[split])
    clips = manifest.clips
    issues_by_clip: list[dict[str, Any]] = []
    quality_scores: list[int] = []
    scenario_consistent = 0
    suspicious_labels = 0
    strict_invoice_rows = 0
    billing_relevant_rows = 0
    transcripts: list[str] = []

    for clip in clips:
        transcript = (clip.reference_transcript or "").strip()
        transcripts.append(transcript.lower())
        entities = clip.expected_entities
        entity_dict = {
            "invoice_ids": list(entities.invoice_ids),
            "amounts": list(entities.amounts),
            "dates": list(entities.dates),
            "billing_actions": list(entities.billing_actions),
            "confirmations": list(entities.confirmations),
            "refusals": list(entities.refusals),
            "account_terms": list(entities.account_terms),
        }
        metadata_score = clip.metadata.get("entity_quality_score")
        computed_score = entity_quality_score(entity_dict)
        quality = int(metadata_score) if metadata_score is not None else computed_score
        quality_scores.append(quality)

        if is_business_relevant(transcript, language):
            billing_relevant_rows += 1

        clip_issues: list[str] = []
        clip_suspicious = False
        if quality < min_entity_quality:
            clip_issues.append("low_entity_quality_score")

        expected_scenario = clip.scenario or classify_scenario(transcript, language)
        if expected_scenario == NUMERIC_ENTITY_SCENARIO:
            if entities.amounts:
                scenario_consistent += 1
            else:
                clip_issues.append("scenario_keyword_mismatch")
        elif _scenario_keyword_hits(transcript, expected_scenario, language) > 0:
            scenario_consistent += 1
        else:
            clip_issues.append("scenario_keyword_mismatch")

        has_strict_invoice = False
        for invoice_id in entities.invoice_ids:
            token = invoice_id.strip().lower()
            if token in SUSPICIOUS_INVOICE_TOKENS or not STRICT_INVOICE_RE.search(invoice_id):
                clip_issues.append("suspicious_invoice_id")
                clip_suspicious = True
            else:
                has_strict_invoice = True
        if has_strict_invoice:
            strict_invoice_rows += 1
        elif entities.invoice_ids:
            clip_issues.append("weak_invoice_pattern")
            clip_suspicious = True

        for amount in entities.amounts:
            if YEAR_AMOUNT_RE.match(amount.strip()) or GENERIC_NUMBER_RE.match(amount.strip()):
                clip_issues.append("suspicious_amount")
                clip_suspicious = True
                break

        if clip_suspicious:
            suspicious_labels += 1

        if clip_issues:
            issues_by_clip.append(
                {
                    "clip_id": clip.clip_id,
                    "scenario": expected_scenario,
                    "entity_quality_score": quality,
                    "issues": sorted(set(clip_issues)),
                    "reference_transcript": _truncate(transcript, 160),
                    "expected_entities": entity_dict,
                }
            )

    duplicate_rate = _duplicate_transcript_rate(transcripts)
    audio_report = _sample_audio(clips, read_audio=read_audio, audio_sample_limit=audio_sample_limit)

    flagged = sorted(
        issues_by_clip,
        key=lambda row: (-len(row["issues"]), -row.get("entity_quality_score", 0)),
    )[:8]

    clip_count = len(clips)
    return {
        "dataset_id": dataset_id,
        "language": language,
        "eval_tier": "business",
        "manifest_path": manifest_path,
        "clip_count": clip_count,
        "entity_quality": _summary_stats(quality_scores),
        "billing_relevance_rate": round(billing_relevant_rows / clip_count, 3) if clip_count else 0.0,
        "scenario_consistency_rate": round(scenario_consistent / clip_count, 3) if clip_count else 0.0,
        "strict_invoice_row_rate": round(strict_invoice_rows / clip_count, 3) if clip_count else 0.0,
        "suspicious_label_rate": round(suspicious_labels / clip_count, 3) if clip_count else 0.0,
        "duplicate_transcript_rate": duplicate_rate,
        "clips_with_issues": len(issues_by_clip),
        "issue_breakdown": _issue_breakdown(issues_by_clip),
        "audio_sample": audio_report,
        "flagged_samples": flagged,
    }


def _evaluate_acoustic_manifest(
    *,
    dataset_id: str,
    language: LanguageCode,
    split: str,
    manifest_path: str,
    read_audio: bool,
    audio_sample_limit: int,
) -> dict[str, Any]:
    manifest = load_eval_manifest(manifest_path, splits=[split])
    clips = manifest.clips
    transcripts = [(clip.reference_transcript or "").strip().lower() for clip in clips]
    word_counts = [len(text.split()) for text in transcripts if text]
    empty_transcripts = sum(1 for text in transcripts if not text)
    duplicate_rate = _duplicate_transcript_rate(transcripts)
    audio_report = _sample_audio(clips, read_audio=read_audio, audio_sample_limit=audio_sample_limit)

    issues: list[dict[str, Any]] = []
    for clip in clips:
        text = (clip.reference_transcript or "").strip()
        clip_issues: list[str] = []
        if not text:
            clip_issues.append("empty_transcript")
        elif len(text.split()) < 3:
            clip_issues.append("very_short_transcript")
        if clip_issues:
            issues.append(
                {
                    "clip_id": clip.clip_id,
                    "issues": clip_issues,
                    "reference_transcript": _truncate(text, 160),
                }
            )

    clip_count = len(clips)
    return {
        "dataset_id": dataset_id,
        "language": language,
        "eval_tier": "acoustic",
        "manifest_path": manifest_path,
        "clip_count": clip_count,
        "empty_transcripts": empty_transcripts,
        "duplicate_transcript_rate": duplicate_rate,
        "transcript_words": _summary_stats(word_counts),
        "clips_with_issues": len(issues),
        "audio_sample": audio_report,
        "flagged_samples": issues[:5],
    }


def _scenario_keyword_hits(transcript: str, scenario: str, language: LanguageCode) -> int:
    return scenario_keyword_hits(transcript, scenario, language)


def _duplicate_transcript_rate(transcripts: list[str]) -> float:
    if not transcripts:
        return 0.0
    unique = len(set(transcripts))
    return round(1 - (unique / len(transcripts)), 3)


def _sample_audio(
    clips: tuple[ASRGoldClip, ...],
    *,
    read_audio: bool,
    audio_sample_limit: int,
) -> dict[str, Any]:
    if not read_audio or not clips:
        return {"requested": 0, "failures": 0, "duration_seconds": {}, "checks": []}

    import soundfile as sf

    checks: list[dict[str, Any]] = []
    failures = 0
    durations: list[float] = []
    for clip in clips[:audio_sample_limit]:
        try:
            payload = read_audio_bytes(clip.audio_path)
            with io.BytesIO(payload) as buffer:
                info = sf.info(buffer)
            duration = float(info.duration)
            durations.append(duration)
            ok = DEFAULT_MIN_AUDIO_SECONDS <= duration <= DEFAULT_MAX_AUDIO_SECONDS
            if not ok:
                failures += 1
            checks.append(
                {
                    "clip_id": clip.clip_id,
                    "bytes": len(payload),
                    "duration_seconds": round(duration, 2),
                    "sample_rate_hz": info.samplerate,
                    "ok": ok,
                }
            )
        except Exception as exc:  # noqa: BLE001
            failures += 1
            checks.append({"clip_id": clip.clip_id, "ok": False, "error": str(exc)})

    return {
        "requested": min(audio_sample_limit, len(clips)),
        "failures": failures,
        "duration_seconds": _summary_stats([round(value, 2) for value in durations]),
        "checks": checks,
    }


def _issue_breakdown(issues_by_clip: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in issues_by_clip:
        for issue in row.get("issues", []):
            counts[issue] = counts.get(issue, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _summary_stats(values: list[int | float]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "max": None, "mean": None, "p50": None}
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "max": ordered[-1],
        "mean": round(statistics.mean(ordered), 2),
        "p50": ordered[len(ordered) // 2],
    }


def _quality_gates(tier_reports: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    business_reports = tier_reports.get("business", [])
    acoustic_reports = tier_reports.get("acoustic", [])

    business_ready = bool(business_reports) and all(
        report["entity_quality"].get("min") is not None
        and float(report["entity_quality"]["min"]) >= DEFAULT_MIN_ENTITY_QUALITY
        and report["scenario_consistency_rate"] >= DEFAULT_MIN_SCENARIO_CONSISTENCY
        and report["suspicious_label_rate"] <= DEFAULT_MAX_SUSPICIOUS_LABEL_RATE
        and report["duplicate_transcript_rate"] <= DEFAULT_MAX_DUPLICATE_TRANSCRIPT_RATE
        and report["audio_sample"]["failures"] == 0
        for report in business_reports
    )

    acoustic_ready = bool(acoustic_reports) and all(
        report["empty_transcripts"] == 0
        and report["duplicate_transcript_rate"] <= DEFAULT_MAX_DUPLICATE_TRANSCRIPT_RATE
        and report["audio_sample"]["failures"] == 0
        for report in acoustic_reports
    )

    return {
        "dataset_quality_ready": business_ready and acoustic_ready,
        "business_semantic_ready": business_ready,
        "acoustic_semantic_ready": acoustic_ready,
        "recommended_next_step": _recommended_next_step(business_ready, acoustic_ready, business_reports),
        "notes": (
            "Semantic dataset eval checks label plausibility, scenario consistency, and audio duration — "
            "not just clip counts."
        ),
    }


def _recommended_next_step(
    business_ready: bool,
    acoustic_ready: bool,
    business_reports: list[dict[str, Any]],
) -> str:
    if business_ready and acoustic_ready:
        return "dataset labels look plausible — safe to scale clip counts and run ./scripts/ml_asr.sh eval"
    if not business_ready and business_reports:
        worst = min(business_reports, key=lambda report: report["scenario_consistency_rate"])
        return (
            f"tighten business mining for {worst['language']} "
            f"(scenario_consistency={worst['scenario_consistency_rate']}, "
            f"suspicious_label_rate={worst['suspicious_label_rate']}) then re-run dataset-eval"
        )
    if not acoustic_ready:
        return "fix acoustic manifest/audio issues and re-run dataset-eval"
    return "prepare datasets then re-run dataset-eval"


def _truncate(text: str, limit: int) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3] + "..."
