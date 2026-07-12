from __future__ import annotations

import io
from pathlib import Path
from typing import Any

from genie_voice.i18n import LanguageCode

from genie_voice.ml_asr.config import DatasetSpec, EvalConfig
from genie_voice.ml_asr.datasets.entity_mining import (
    classify_scenario,
    entity_quality_score,
    extract_entities,
    is_business_relevant,
)
from genie_voice.ml_asr.manifest import write_manifest_jsonl


def bootstrap_common_voice_business_language(
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
    hf_config = lang_spec.hf_config or lang_spec.fleurs_config
    if not hf_config:
        raise ValueError(f"Language {language} is missing hf_config for Common Voice")

    cache_root = Path(config.remote_training_root if volume_mode else config.local_root) / "hf_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    repo_files = list_repo_files(dataset.source, repo_type="dataset")
    parquet_candidates = [
        name
        for name in repo_files
        if name.endswith(".parquet")
        and name.startswith(f"{hf_config}/")
        and "validation" in name
    ]
    if not parquet_candidates:
        raise RuntimeError(f"No Common Voice validation parquet found for {hf_config}")

    min_upvotes = dataset.min_upvotes or 2
    candidates: list[dict[str, Any]] = []
    scanned = 0
    for parquet_name in sorted(parquet_candidates):
        parquet_path = hf_hub_download(
            repo_id=dataset.source,
            repo_type="dataset",
            filename=parquet_name,
            cache_dir=str(cache_root / "hub"),
        )
        frame = pd.read_parquet(parquet_path)
        for _, row in frame.iterrows():
            scanned += 1
            if scanned > max(limit * 300, 8000):
                break
            transcript = str(row.get("sentence") or "").strip()
            if not transcript or not is_business_relevant(transcript, language):
                continue
            up_votes = int(row.get("up_votes") or 0)
            down_votes = int(row.get("down_votes") or 0)
            if up_votes < min_upvotes or down_votes > up_votes:
                continue
            entities = extract_entities(transcript, language)
            if entity_quality_score(entities) < 2:
                continue
            candidates.append(
                {
                    "transcript": transcript,
                    "entities": entities,
                    "scenario": classify_scenario(transcript, language),
                    "audio": row.get("audio"),
                    "client_id": str(row.get("client_id") or ""),
                    "path": str(row.get("path") or ""),
                    "up_votes": up_votes,
                    "quality": entity_quality_score(entities),
                }
            )
        if scanned > max(limit * 300, 8000) or len(candidates) >= limit * 3:
            break

    candidates.sort(key=lambda item: (-item["quality"], -len(item["transcript"])))
    selected = candidates[:limit]
    if len(selected) < min(limit // 2, 10):
        raise RuntimeError(
            f"Only found {len(selected)} business-relevant Common Voice clips for {language} "
            f"(scanned {scanned}). Lower min_upvotes or clip_limit."
        )

    rows: list[dict[str, Any]] = []
    uploads: list[dict[str, str]] = []
    for index, item in enumerate(selected, start=1):
        clip_id = f"{lang_spec.manifest_language}_cv_business_{index:04d}"
        if volume_mode:
            audio_path = Path(lang_spec.remote_audio_dir) / f"{clip_id}.wav"
        else:
            audio_path = (
                Path(config.local_audio_dir)
                / lang_spec.manifest_language
                / "common_voice_business"
                / f"{clip_id}.wav"
            )
        duration = _write_audio_16k(sf, item["audio"], audio_path)
        remote_audio = f"{lang_spec.remote_audio_dir}/{clip_id}.wav"
        rows.append(
            {
                "clip_id": clip_id,
                "call_id": f"CV-{lang_spec.manifest_language.upper()}-{index:04d}",
                "speaker": "customer",
                "audio_path": remote_audio,
                "audio_format": "audio/wav",
                "sample_rate_hz": 16000,
                "duration_seconds": duration,
                "reference_transcript": item["transcript"],
                "language": lang_spec.manifest_language,
                "split": dataset.split,
                "scenario": item["scenario"],
                "domain": dataset.domain,
                "dataset_version": dataset.dataset_id,
                "expected_entities": item["entities"],
                "metadata": {
                    "source": dataset.source,
                    "license": "CC0-1.0",
                    "hf_config": hf_config,
                    "hf_split": "validation",
                    "eval_tier": dataset.eval_tier,
                    "dataset_id": dataset.dataset_id,
                    "audio_mode": dataset.audio_mode,
                    "audio_status": "licensed_human_speech",
                    "client_id": item["client_id"],
                    "original_path": item["path"],
                    "up_votes": item["up_votes"],
                    "entity_quality_score": item["quality"],
                    "human_transcript_approved": True,
                    "business_holdout": True,
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
        "scanned": scanned,
        "manifest": manifest_path,
        "remote_manifest": lang_spec.remote_manifest_path,
        "uploads": uploads,
        "volume_mode": volume_mode,
    }


def _write_audio_16k(sf: Any, audio: Any, path: Path) -> float:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload, sample_rate = _decode_audio_payload(audio)
    if sample_rate != 16000:
        payload, sample_rate = _resample_pcm16(payload, sample_rate, 16000)
    sf.write(path, payload, sample_rate, subtype="PCM_16")
    return round(len(payload) / 16000.0, 3)


def _decode_audio_payload(audio: Any) -> tuple[Any, int]:
    if isinstance(audio, dict):
        if audio.get("array") is not None:
            return audio["array"], int(audio.get("sampling_rate") or 48000)
        if audio.get("bytes"):
            import soundfile as sf

            data, sample_rate = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
            return data, int(sample_rate)
    if isinstance(audio, (bytes, bytearray, memoryview)):
        import soundfile as sf

        data, sample_rate = sf.read(io.BytesIO(bytes(audio)), dtype="float32")
        return data, int(sample_rate)
    raise RuntimeError(f"Unsupported Common Voice audio payload: {type(audio).__name__}")


def _resample_pcm16(samples: Any, source_rate: int, target_rate: int) -> tuple[Any, int]:
    import numpy as np

    array = np.asarray(samples, dtype=np.float32)
    if array.ndim > 1:
        array = array.mean(axis=1)
    if source_rate == target_rate:
        return array, target_rate
    target_length = max(1, int(round(len(array) * target_rate / source_rate)))
    source_times = np.linspace(0.0, len(array) / float(source_rate), num=len(array), endpoint=False)
    target_times = np.linspace(0.0, len(array) / float(source_rate), num=target_length, endpoint=False)
    resampled = np.interp(target_times, source_times, array)
    return resampled, target_rate
