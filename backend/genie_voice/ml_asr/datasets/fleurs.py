from __future__ import annotations

from pathlib import Path
from typing import Any

from genie_voice.i18n import LanguageCode

from genie_voice.ml_asr.config import DatasetSpec, EvalConfig
from genie_voice.ml_asr.manifest import empty_entities, write_manifest_jsonl


def bootstrap_fleurs_language(
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
    parquet_candidates = [
        name
        for name in repo_files
        if name.endswith(".parquet")
        and f"/{lang_spec.fleurs_config}/" in name
        and "validation" in name
    ]
    if not parquet_candidates:
        raise RuntimeError(f"No FLEURS validation parquet found for {lang_spec.fleurs_config}")

    parquet_path = hf_hub_download(
        repo_id=dataset.source,
        repo_type="dataset",
        filename=sorted(parquet_candidates)[0],
        cache_dir=str(cache_root / "hub"),
    )
    dataset_df = pd.read_parquet(parquet_path)

    rows: list[dict[str, Any]] = []
    uploads: list[dict[str, str]] = []
    written = 0
    for _, item in dataset_df.iterrows():
        if written >= limit:
            break
        transcript = str(item.get("transcription") or item.get("raw_transcription") or "").strip()
        if not transcript:
            continue
        clip_id = f"{lang_spec.manifest_language}_fleurs_holdout_{written + 1:04d}"
        if volume_mode:
            audio_path = Path(lang_spec.remote_audio_dir) / f"{clip_id}.wav"
        else:
            audio_path = (
                Path(config.local_audio_dir)
                / lang_spec.manifest_language
                / "external_fleurs_holdout"
                / f"{clip_id}.wav"
            )
        _write_audio(sf, item.get("audio"), audio_path)
        remote_audio = f"{lang_spec.remote_audio_dir}/{clip_id}.wav"
        rows.append(
            {
                "clip_id": clip_id,
                "audio_path": remote_audio,
                "audio_format": "audio/wav",
                "sample_rate_hz": 16000,
                "reference_transcript": transcript,
                "language": lang_spec.manifest_language,
                "split": dataset.split,
                "scenario": dataset.scenario,
                "speaker": "external_fleurs",
                "domain": dataset.domain,
                "dataset_version": dataset.dataset_id,
                "expected_entities": empty_entities(),
                "metadata": {
                    "source": dataset.source,
                    "license": "CC-BY-4.0",
                    "config": lang_spec.fleurs_config,
                    "eval_tier": dataset.eval_tier,
                    "dataset_id": dataset.dataset_id,
                    "id": _safe_json_value(item.get("id")),
                    "external_holdout_type": "acoustic_only",
                },
            }
        )
        if not volume_mode:
            uploads.append({"local": str(audio_path), "remote": remote_audio})
        written += 1

    manifest_path = lang_spec.remote_manifest_path if volume_mode else lang_spec.local_manifest_path
    write_manifest_jsonl(manifest_path, rows)
    return {
        "dataset_id": dataset.dataset_id,
        "language": language,
        "rows": len(rows),
        "manifest": manifest_path,
        "remote_manifest": lang_spec.remote_manifest_path,
        "uploads": uploads,
        "volume_mode": volume_mode,
    }


def _write_audio(sf: Any, audio: Any, path: Path) -> None:
    import io

    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(audio, dict):
        if audio.get("array") is not None:
            sf.write(path, audio["array"], int(audio.get("sampling_rate") or 16000))
            return
        if audio.get("bytes"):
            payload = audio["bytes"]
            if isinstance(payload, memoryview):
                payload = bytes(payload)
            data, sample_rate = sf.read(io.BytesIO(payload))
            if sample_rate != 16000:
                import numpy as np

                target_length = max(1, int(round(len(data) * 16000 / sample_rate)))
                source_times = np.linspace(0.0, len(data) / float(sample_rate), num=len(data), endpoint=False)
                target_times = np.linspace(0.0, len(data) / float(sample_rate), num=target_length, endpoint=False)
                if data.ndim > 1:
                    data = data.mean(axis=1)
                data = np.interp(target_times, source_times, data)
            sf.write(path, data, 16000, subtype="PCM_16")
            return
    if isinstance(audio, (bytes, bytearray, memoryview)):
        data, sample_rate = sf.read(io.BytesIO(bytes(audio)))
        sf.write(path, data, 16000, subtype="PCM_16")
        return
    raise RuntimeError(f"Unsupported FLEURS audio payload: {type(audio).__name__}")


def _safe_json_value(value: Any) -> Any:
    import pandas as pd

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return str(value)
