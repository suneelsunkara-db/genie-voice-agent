from __future__ import annotations

from pathlib import Path
from typing import Any

from genie_voice.i18n import LanguageCode

from genie_voice.ml_asr.config import DatasetSpec, EvalConfig
from genie_voice.ml_asr.datasets.entity_mining import passes_business_holdout_bar
from genie_voice.ml_asr.datasets.fleurs import _safe_json_value, _write_audio
from genie_voice.ml_asr.manifest import write_manifest_jsonl


def bootstrap_fleurs_business_language(
    config: EvalConfig,
    dataset: DatasetSpec,
    *,
    language: LanguageCode,
    limit: int,
    volume_mode: bool = False,
) -> dict[str, Any]:
    import pandas as pd
    from huggingface_hub import hf_hub_download, list_repo_files

    import soundfile as sf

    lang_spec = dataset.languages[language]
    if not lang_spec.fleurs_config:
        raise ValueError(f"Language {language} is missing fleurs_config")

    cache_root = Path(config.remote_training_root if volume_mode else config.local_root) / "hf_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    repo_files = list_repo_files(dataset.source, repo_type="dataset")
    parquet_candidates = sorted(
        name
        for name in repo_files
        if name.endswith(".parquet")
        and f"/{lang_spec.fleurs_config}/" in name
        and ("validation" in name or "train" in name)
    )
    parquet_candidates.sort(key=lambda name: (0 if "validation" in name else 1, name))
    if not parquet_candidates:
        raise RuntimeError(f"No FLEURS validation/train parquet found for {lang_spec.fleurs_config}")

    candidates: list[dict[str, Any]] = []
    seen_transcripts: set[str] = set()
    rows_scanned = 0
    for parquet_name in parquet_candidates:
        parquet_path = hf_hub_download(
            repo_id=dataset.source,
            repo_type="dataset",
            filename=parquet_name,
            cache_dir=str(cache_root / "hub"),
        )
        dataset_df = pd.read_parquet(parquet_path)
        rows_scanned += len(dataset_df)
        for _, item in dataset_df.iterrows():
            transcript = str(item.get("transcription") or item.get("raw_transcription") or "").strip()
            source_id = _safe_json_value(item.get("id"))
            normalized = " ".join(transcript.lower().split())
            if not transcript or normalized in seen_transcripts:
                continue

            ok, entities, scenario, quality = passes_business_holdout_bar(transcript, language)
            if not ok:
                continue

            seen_transcripts.add(normalized)
            candidates.append(
                {
                    "transcript": transcript,
                    "entities": entities,
                    "scenario": scenario,
                    "audio": item.get("audio"),
                    "source_id": source_id,
                    "quality": quality,
                    "fleurs_split": "validation" if "validation" in parquet_name else "train",
                }
            )

    candidates.sort(
        key=lambda row: (
            0 if row.get("fleurs_split") == "validation" else 1,
            -row["quality"],
            -len(row["transcript"]),
        )
    )
    selected = candidates[:limit]
    if not selected:
        raise RuntimeError(
            f"No business-relevant FLEURS clips found for {language} "
            f"(scanned {rows_scanned} rows). Tighten or broaden entity mining."
        )
    if len(selected) < limit:
        shortage_note = (
            f"Only {len(selected)}/{limit} clips available for {language} "
            f"({len(candidates)} passed holdout bar across validation+train)."
        )
    else:
        shortage_note = None

    rows: list[dict[str, Any]] = []
    uploads: list[dict[str, str]] = []
    for index, item in enumerate(selected, start=1):
        clip_id = f"{lang_spec.manifest_language}_fleurs_business_{index:04d}"
        if volume_mode:
            audio_path = Path(lang_spec.remote_audio_dir) / f"{clip_id}.wav"
        else:
            audio_path = (
                Path(config.local_audio_dir)
                / lang_spec.manifest_language
                / "fleurs_business_holdout"
                / f"{clip_id}.wav"
            )
        _write_audio(sf, item["audio"], audio_path)
        remote_audio = f"{lang_spec.remote_audio_dir}/{clip_id}.wav"
        rows.append(
            {
                "clip_id": clip_id,
                "call_id": f"FLEURS-BIZ-{lang_spec.manifest_language.upper()}-{index:04d}",
                "speaker": "customer",
                "audio_path": remote_audio,
                "audio_format": "audio/wav",
                "sample_rate_hz": 16000,
                "reference_transcript": item["transcript"],
                "language": lang_spec.manifest_language,
                "split": dataset.split,
                "scenario": item["scenario"],
                "domain": dataset.domain,
                "dataset_version": dataset.dataset_id,
                "expected_entities": item["entities"],
                "metadata": {
                    "source": dataset.source,
                    "license": "CC-BY-4.0",
                    "config": lang_spec.fleurs_config,
                    "eval_tier": dataset.eval_tier,
                    "dataset_id": dataset.dataset_id,
                    "audio_mode": dataset.audio_mode,
                    "audio_status": "licensed_human_speech",
                    "entity_quality_score": item["quality"],
                    "human_transcript_approved": False,
                    "business_holdout": True,
                    "id": item["source_id"],
                    "fleurs_split": item.get("fleurs_split"),
                    "mining_source": "fleurs_business_filter_v2",
                },
            }
        )
        if not volume_mode:
            uploads.append({"local": str(audio_path), "remote": remote_audio})

    manifest_path = lang_spec.remote_manifest_path if volume_mode else lang_spec.local_manifest_path
    write_manifest_jsonl(manifest_path, rows)
    return {
        "dataset_id": dataset.dataset_id,
        "language": language,
        "rows": len(rows),
        "candidates": len(candidates),
        "rows_scanned": rows_scanned,
        "shortage_note": shortage_note,
        "manifest": manifest_path,
        "remote_manifest": lang_spec.remote_manifest_path,
        "uploads": uploads,
        "volume_mode": volume_mode,
    }
