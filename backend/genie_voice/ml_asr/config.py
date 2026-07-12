from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from genie_voice.config import get_settings
from genie_voice.i18n import LanguageCode, normalize_language

from genie_voice.ml_asr.runtime import training_root_from_config_path


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "ml_asr_eval.yaml"


@dataclass(frozen=True)
class DatasetLanguageSpec:
    language: LanguageCode
    manifest_language: str
    fleurs_config: str | None
    hf_config: str | None
    manifest_file: str
    local_manifest_path: str
    remote_manifest_path: str
    remote_audio_dir: str


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    eval_tier: str
    split: str
    builder: str
    source: str
    scenario: str
    domain: str
    audio_mode: str
    clip_limit: int | None
    clips_per_scenario: int | None
    min_upvotes: int | None
    languages: dict[LanguageCode, DatasetLanguageSpec]


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    provider: str
    label: str
    model: str | None = None
    endpoint: str | None = None
    languages: tuple[LanguageCode, ...] | None = None

    def supports(self, language: LanguageCode) -> bool:
        return self.languages is None or language in self.languages


@dataclass(frozen=True)
class EvalConfig:
    version: int
    remote_training_root: str
    remote_datasets_root: str
    remote_manifests_dir: str
    remote_results_dir: str
    remote_jobs_dir: str
    remote_state_path: str
    remote_audit_path: str
    remote_index_path: str
    serverless_environment_version: int
    serverless_deepgram_secret_scope: str
    serverless_deepgram_secret_key: str
    serverless_databricks_concurrency: int
    smoke_acoustic_clip_limit: int | None
    smoke_business_clip_limit: int | None
    smoke_eval_clip_limit: int | None
    local_root: str
    local_manifests_dir: str
    local_audio_dir: str
    local_results_dir: str
    datasets: dict[str, DatasetSpec]
    models: dict[str, ModelSpec]
    eval_matrix: dict[LanguageCode, tuple[str, ...]]
    eval_tiers: tuple[str, ...]
    eval_languages: tuple[LanguageCode, ...]
    min_clips_per_language: dict[str, int]
    config_path: str

    def language_codes(self) -> list[LanguageCode]:
        return list(self.eval_languages)

    def dataset_ids(self) -> list[str]:
        return list(self.datasets.keys())

    def datasets_for_tier(self, tier: str) -> list[DatasetSpec]:
        return [dataset for dataset in self.datasets.values() if dataset.eval_tier == tier]

    def language_spec(self, dataset_id: str, language: LanguageCode) -> DatasetLanguageSpec:
        dataset = self.datasets[dataset_id]
        if language not in dataset.languages:
            raise KeyError(f"Language {language} not in dataset {dataset_id}")
        return dataset.languages[language]

    def models_for_language(self, language: LanguageCode) -> list[ModelSpec]:
        ids = self.eval_matrix.get(language, ())
        out: list[ModelSpec] = []
        for model_id in ids:
            spec = self.models.get(model_id)
            if spec is None:
                raise KeyError(f"Unknown model id in eval_matrix: {model_id}")
            if not spec.supports(language):
                raise ValueError(f"Model {model_id} is not configured for {language}")
            out.append(spec)
        return out

    def result_dir(self, language: LanguageCode, dataset_id: str, model_id: str, *, volume_mode: bool = False) -> Path:
        base = self.remote_results_dir if volume_mode else self.local_results_dir
        return Path(base) / dataset_id / language / model_id

    def manifest_path(self, dataset_id: str, language: LanguageCode, *, volume_mode: bool = False) -> str:
        spec = self.language_spec(dataset_id, language)
        return spec.remote_manifest_path if volume_mode else spec.local_manifest_path


def resolve_training_root(remote_root: str | None = None) -> str:
    if remote_root:
        return remote_root.rstrip("/")
    settings = get_settings()
    catalog = settings.databricks.catalog
    schema = settings.databricks.schema_name
    volume = settings.volume.streaming_name
    if any("<" in str(value) or not str(value).strip() for value in (catalog, schema, volume)):
        raise ValueError("Databricks catalog/schema/streaming volume are not configured.")
    return f"/Volumes/{catalog}/{schema}/{volume}/asr_model_training"


def _resolve_training_root(path: Path, storage: dict[str, Any], remote_root: str | None) -> str:
    explicit = storage.get("training_root")
    if explicit:
        return str(explicit).rstrip("/")
    inferred = training_root_from_config_path(path)
    if inferred:
        return inferred
    return resolve_training_root(remote_root)


def load_config(
    *,
    config_path: str | Path | None = None,
    repo_root: str | Path | None = None,
    remote_root: str | None = None,
) -> EvalConfig:
    path = Path(config_path or DEFAULT_CONFIG_PATH)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid config: {path}")

    root = Path(repo_root or path.resolve().parents[1])
    storage = raw.get("storage") or {}
    serverless = raw.get("serverless") or {}
    datasets_subdir = str(storage.get("datasets_subdir") or "datasets/ml_asr_eval")
    results_subdir = str(storage.get("results_subdir") or "evaluations/ml_asr_eval")
    jobs_subdir = str(serverless.get("jobs_subdir") or "jobs/ml_asr_eval")

    training_root = _resolve_training_root(path, storage, remote_root)
    remote_datasets_root = f"{training_root}/{datasets_subdir}"
    remote_manifests_dir = f"{remote_datasets_root}/manifests"
    remote_results_dir = f"{training_root}/{results_subdir}"
    remote_jobs_dir = f"{training_root}/{jobs_subdir}"
    remote_state_path = f"{training_root}/{storage.get('state_file', f'{results_subdir}/pipeline_state.json')}"
    remote_audit_path = f"{training_root}/{storage.get('audit_file', f'{results_subdir}/dataset_audit.json')}"
    remote_index_path = f"{training_root}/{storage.get('index_file', f'{results_subdir}/results/index.json')}"

    local_root = root / ".run" / "ml_asr_eval"
    local_manifests_dir = local_root / "manifests"
    local_audio_dir = local_root / "audio"
    local_results_dir = local_root / "results"

    datasets: dict[str, DatasetSpec] = {}
    for dataset_id, entry in (raw.get("datasets") or {}).items():
        spec = entry if isinstance(entry, dict) else {}
        tier = str(spec.get("eval_tier") or "acoustic")
        builder = str(spec.get("builder") or "fleurs")
        if builder == "fleurs" or tier == "acoustic":
            audio_subdir = "external_fleurs_holdout"
        elif builder == "common_voice":
            audio_subdir = "common_voice_business"
        elif builder == "fleurs_business":
            audio_subdir = "fleurs_business_holdout"
        else:
            audio_subdir = "business_holdout"
        language_specs: dict[LanguageCode, DatasetLanguageSpec] = {}
        for language_code, lang_entry in (spec.get("languages") or {}).items():
            language = normalize_language(str(language_code))
            lang = lang_entry if isinstance(lang_entry, dict) else {}
            manifest_language = str(lang.get("manifest_language") or language.split("-")[0])
            manifest_file = f"{dataset_id}_{manifest_language}.jsonl"
            language_specs[language] = DatasetLanguageSpec(
                language=language,
                manifest_language=manifest_language,
                fleurs_config=_optional_str(lang.get("fleurs_config")),
                hf_config=_optional_str(lang.get("hf_config")),
                manifest_file=manifest_file,
                local_manifest_path=str(local_manifests_dir / manifest_file),
                remote_manifest_path=f"{remote_manifests_dir}/{manifest_file}",
                remote_audio_dir=f"{remote_datasets_root}/audio/{manifest_language}/{audio_subdir}",
            )
        datasets[str(dataset_id)] = DatasetSpec(
            dataset_id=str(dataset_id),
            eval_tier=tier,
            split=str(spec.get("split") or "holdout"),
            builder=builder,
            source=str(spec.get("source") or ""),
            scenario=str(spec.get("scenario") or dataset_id),
            domain=str(spec.get("domain") or "billing_support"),
            audio_mode=str(spec.get("audio_mode") or "scaffold"),
            clip_limit=_optional_int(spec.get("clip_limit")),
            clips_per_scenario=_optional_int(spec.get("clips_per_scenario")),
            min_upvotes=_optional_int(spec.get("min_upvotes")),
            languages=language_specs,
        )

    models: dict[str, ModelSpec] = {}
    for model_id, entry in (raw.get("models") or {}).items():
        spec = entry if isinstance(entry, dict) else {}
        lang_list = spec.get("languages")
        languages_tuple = None
        if lang_list:
            languages_tuple = tuple(normalize_language(str(code)) for code in lang_list)
        models[str(model_id)] = ModelSpec(
            model_id=str(model_id),
            provider=str(spec.get("provider") or ""),
            label=str(spec.get("label") or model_id),
            model=_optional_str(spec.get("model")),
            endpoint=_optional_str(spec.get("endpoint")),
            languages=languages_tuple,
        )

    eval_matrix: dict[LanguageCode, tuple[str, ...]] = {}
    for language_code, model_ids in (raw.get("eval_matrix") or {}).items():
        language = normalize_language(str(language_code))
        eval_matrix[language] = tuple(str(model_id) for model_id in (model_ids or []))

    plan = raw.get("eval_plan") or {}
    smoke = plan.get("smoke") or {}
    eval_tiers = tuple(str(tier) for tier in (plan.get("tiers") or ["acoustic", "business"]))
    eval_languages = tuple(
        normalize_language(str(code)) for code in (plan.get("languages") or list(eval_matrix.keys()))
    )
    min_clips = {
        str(tier): int((plan.get("min_clips_per_language") or {}).get(tier, 10))
        for tier in eval_tiers
    }

    if not datasets:
        raise ValueError(f"No datasets configured in {path}")
    if not models:
        raise ValueError(f"No models configured in {path}")
    if not eval_matrix:
        raise ValueError(f"No eval_matrix configured in {path}")

    return EvalConfig(
        version=int(raw.get("version") or 1),
        remote_training_root=training_root,
        remote_datasets_root=remote_datasets_root,
        remote_manifests_dir=remote_manifests_dir,
        remote_results_dir=remote_results_dir,
        remote_jobs_dir=remote_jobs_dir,
        remote_state_path=remote_state_path,
        remote_audit_path=remote_audit_path,
        remote_index_path=remote_index_path,
        serverless_environment_version=int(serverless.get("environment_version") or 2),
        serverless_deepgram_secret_scope=str(serverless.get("deepgram_secret_scope") or "genie-voice"),
        serverless_deepgram_secret_key=str(serverless.get("deepgram_secret_key") or "deepgram_api_key"),
        serverless_databricks_concurrency=int(serverless.get("databricks_concurrency") or 4),
        smoke_acoustic_clip_limit=_optional_int(smoke.get("acoustic_clip_limit")),
        smoke_business_clip_limit=_optional_int(smoke.get("business_clip_limit")),
        smoke_eval_clip_limit=_optional_int(smoke.get("eval_clip_limit")),
        local_root=str(local_root),
        local_manifests_dir=str(local_manifests_dir),
        local_audio_dir=str(local_audio_dir),
        local_results_dir=str(local_results_dir),
        datasets=datasets,
        models=models,
        eval_matrix=eval_matrix,
        eval_tiers=eval_tiers,
        eval_languages=eval_languages,
        min_clips_per_language=min_clips,
        config_path=str(path),
    )


def config_summary(config: EvalConfig) -> dict[str, Any]:
    return {
        "version": config.version,
        "config_path": config.config_path,
        "eval_plan": {
            "tiers": list(config.eval_tiers),
            "languages": list(config.eval_languages),
            "min_clips_per_language": config.min_clips_per_language,
        },
        "remote_datasets_root": config.remote_datasets_root,
        "remote_results_dir": config.remote_results_dir,
        "remote_jobs_dir": config.remote_jobs_dir,
        "remote_state_path": config.remote_state_path,
        "local_root": config.local_root,
        "datasets": {
            dataset_id: {
                "eval_tier": dataset.eval_tier,
                "builder": dataset.builder,
                "audio_mode": dataset.audio_mode,
                "split": dataset.split,
                "languages": {
                    language: {
                        "manifest": lang_spec.remote_manifest_path,
                        "audio_dir": lang_spec.remote_audio_dir,
                    }
                    for language, lang_spec in dataset.languages.items()
                },
            }
            for dataset_id, dataset in config.datasets.items()
        },
        "models": {
            model_id: {
                "provider": spec.provider,
                "label": spec.label,
                "model": spec.model,
                "endpoint": spec.endpoint,
                "languages": list(spec.languages or []),
            }
            for model_id, spec in config.models.items()
        },
        "eval_matrix": {language: list(model_ids) for language, model_ids in config.eval_matrix.items()},
    }


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)
