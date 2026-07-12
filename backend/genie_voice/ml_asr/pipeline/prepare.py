from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Iterable

from genie_voice.i18n import LanguageCode, normalize_language

from genie_voice.ml_asr.config import DatasetSpec, EvalConfig, load_config
from genie_voice.ml_asr.datasets.business_holdout import bootstrap_business_language
from genie_voice.ml_asr.datasets.common_voice_business import bootstrap_common_voice_business_language
from genie_voice.ml_asr.datasets.fleurs import bootstrap_fleurs_language
from genie_voice.ml_asr.datasets.fleurs_business import bootstrap_fleurs_business_language
from genie_voice.ml_asr.runtime import is_volume_mode


def prepare_dataset(
    *,
    config_path: str | None = None,
    languages: Iterable[str] | None = None,
    dataset_ids: Iterable[str] | None = None,
    limit: int | None = None,
    skip_upload: bool = False,
    volume_mode: bool | None = None,
) -> list[dict]:
    config = load_config(config_path=config_path)
    volume_mode = is_volume_mode() if volume_mode is None else volume_mode
    selected_languages = _selected_languages(config, languages)
    selected_datasets = _selected_datasets(config, dataset_ids)
    results: list[dict] = []
    all_uploads: list[dict[str, str]] = []

    if volume_mode:
        _ensure_remote_dirs(config, selected_datasets)

    for dataset in selected_datasets:
        if dataset.builder in {"fleurs", "fleurs_business", "common_voice"}:
            _ensure_hf_dataset_deps()
        for language in selected_languages:
            if language not in dataset.languages:
                continue
            if dataset.builder == "fleurs":
                clip_limit = limit or dataset.clip_limit or 100
                result = bootstrap_fleurs_language(
                    config,
                    dataset,
                    language=language,
                    limit=clip_limit,
                    volume_mode=volume_mode,
                )
            elif dataset.builder == "fleurs_business":
                clip_limit = limit or dataset.clip_limit or 150
                result = bootstrap_fleurs_business_language(
                    config,
                    dataset,
                    language=language,
                    limit=clip_limit,
                    volume_mode=volume_mode,
                )
            elif dataset.builder == "common_voice":
                clip_limit = limit or dataset.clip_limit or 150
                result = bootstrap_common_voice_business_language(
                    config,
                    dataset,
                    language=language,
                    limit=clip_limit,
                    volume_mode=volume_mode,
                )
            elif dataset.builder == "business":
                per_scenario = limit or dataset.clips_per_scenario or 4
                result = bootstrap_business_language(
                    config,
                    dataset,
                    language=language,
                    clips_per_scenario=per_scenario,
                    volume_mode=volume_mode,
                )
            else:
                raise ValueError(f"Unsupported dataset builder: {dataset.builder}")
            uploads = [item for item in result.pop("uploads", []) if item.get("local")]
            all_uploads.extend(uploads)
            results.append(result)

    uploads_path = Path(config.local_root) / "uploads.json"
    uploads_path.parent.mkdir(parents=True, exist_ok=True)
    uploads_path.write_text(json.dumps(all_uploads, indent=2), encoding="utf-8")

    if not volume_mode and not skip_upload:
        _upload_artifacts(config, all_uploads, selected_datasets, selected_languages)
    return results


def _ensure_remote_dirs(config: EvalConfig, datasets: list[DatasetSpec]) -> None:
    dirs = {config.remote_manifests_dir, config.remote_datasets_root, config.remote_results_dir}
    for dataset in datasets:
        for lang_spec in dataset.languages.values():
            dirs.add(lang_spec.remote_audio_dir)
    if is_volume_mode():
        for remote_dir in dirs:
            Path(remote_dir).mkdir(parents=True, exist_ok=True)
        return
    profile = os.environ.get("ML_ASR_DATABRICKS_PROFILE") or os.environ.get("DATABRICKS_CONFIG_PROFILE")
    base_cmd = ["databricks"]
    if profile:
        base_cmd.extend(["--profile", profile])
    for remote_dir in dirs:
        subprocess.run([*base_cmd, "fs", "mkdirs", f"dbfs:{remote_dir}"], check=False)


def _selected_languages(config: EvalConfig, languages: Iterable[str] | None) -> list[LanguageCode]:
    if not languages:
        return list(config.eval_languages)
    selected = [normalize_language(str(language)) for language in languages]
    unknown = [language for language in selected if language not in config.eval_languages]
    if unknown:
        raise ValueError(f"Unknown languages: {', '.join(unknown)}")
    return selected


def _selected_datasets(config: EvalConfig, dataset_ids: Iterable[str] | None) -> list[DatasetSpec]:
    if not dataset_ids:
        return [config.datasets[dataset_id] for dataset_id in config.dataset_ids()]
    selected = [str(dataset_id) for dataset_id in dataset_ids]
    unknown = [dataset_id for dataset_id in selected if dataset_id not in config.datasets]
    if unknown:
        raise ValueError(f"Unknown datasets: {', '.join(unknown)}")
    return [config.datasets[dataset_id] for dataset_id in selected]


def _ensure_hf_dataset_deps() -> None:
    try:
        import pandas  # noqa: F401
        import soundfile  # noqa: F401
        from huggingface_hub import hf_hub_download  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Dataset prep requires huggingface_hub, pandas, pyarrow, and soundfile."
        ) from exc


def _ensure_fleurs_deps() -> None:
    _ensure_hf_dataset_deps()
    try:
        from huggingface_hub import hf_hub_download  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("FLEURS dataset prep requires huggingface_hub.") from exc


def _upload_artifacts(
    config: EvalConfig,
    uploads: list[dict[str, str]],
    datasets: list[DatasetSpec],
    languages: list[LanguageCode],
) -> None:
    profile = os.environ.get("ML_ASR_DATABRICKS_PROFILE") or os.environ.get("DATABRICKS_CONFIG_PROFILE")
    base_cmd = ["databricks"]
    if profile:
        base_cmd.extend(["--profile", profile])

    for remote_dir in {config.remote_manifests_dir, config.remote_datasets_root}:
        subprocess.run([*base_cmd, "fs", "mkdirs", f"dbfs:{remote_dir}"], check=False)

    for item in uploads:
        local_path = item["local"]
        if not Path(local_path).is_file():
            continue
        remote_path = item["remote"]
        subprocess.run([*base_cmd, "fs", "mkdirs", f"dbfs:{Path(remote_path).parent}"], check=False)
        subprocess.run(
            [*base_cmd, "fs", "cp", local_path, f"dbfs:{remote_path}", "--overwrite"],
            check=True,
        )

    for dataset in datasets:
        for language in languages:
            if language not in dataset.languages:
                continue
            lang_spec = dataset.languages[language]
            subprocess.run(
                [
                    *base_cmd,
                    "fs",
                    "cp",
                    lang_spec.local_manifest_path,
                    f"dbfs:{lang_spec.remote_manifest_path}",
                    "--overwrite",
                ],
                check=True,
            )


def validate_dataset(
    *,
    config_path: str | None = None,
    local_only: bool = False,
    volume_mode: bool | None = None,
) -> dict[str, dict]:
    from genie_voice.ml_asr.manifest import load_eval_manifest

    config = load_config(config_path=config_path)
    volume_mode = is_volume_mode() if volume_mode is None else volume_mode
    report: dict[str, dict] = {}
    failed = False
    for dataset in config.datasets.values():
        for language, spec in dataset.languages.items():
            key_base = f"{dataset.dataset_id}:{language}"
            checks: list[tuple[str, str]] = []
            if volume_mode:
                checks.append(("remote", spec.remote_manifest_path))
            elif local_only:
                checks.append(("local", spec.local_manifest_path))
            else:
                checks.extend(
                    [
                        ("local", spec.local_manifest_path),
                        ("remote", spec.remote_manifest_path),
                    ]
                )
            for label, path in checks:
                key = f"{key_base}:{label}"
                try:
                    manifest = load_eval_manifest(path, splits=[dataset.split])
                    ok = len(manifest) > 0
                    report[key] = {"ok": ok, "clips": len(manifest), "path": path, "eval_tier": dataset.eval_tier}
                    if not ok:
                        failed = True
                except Exception as exc:  # noqa: BLE001
                    report[key] = {"ok": False, "path": path, "error": str(exc), "eval_tier": dataset.eval_tier}
                    failed = True
    if failed:
        raise SystemExit(2)
    return report
