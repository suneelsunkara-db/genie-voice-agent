from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from genie_voice.config import get_settings
from genie_voice.ml_asr.config import DEFAULT_CONFIG_PATH, load_config


@dataclass(frozen=True)
class ServingSpec:
    model_id: str
    label: str
    endpoint: str
    registered_model_fqdn: str
    alias: str
    workload_type: str
    workload_size: str
    scale_to_zero: bool
    route_optimized: bool
    served_entity_name: str
    register_type: str | None = None
    register_candidate_id: str | None = None


def load_serving_specs(*, config_path: str | Path | None = None) -> list[ServingSpec]:
    path = Path(config_path or DEFAULT_CONFIG_PATH)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    eval_config = load_config(config_path=path)
    serving = raw.get("model_serving") or {}
    defaults = serving.get("defaults") or {}
    alias = str(serving.get("alias") or "candidate")
    models_cfg = serving.get("models") or {}
    settings = get_settings()
    prefix = f"{settings.databricks.catalog}.{settings.databricks.schema_name}."

    specs: list[ServingSpec] = []
    for model_id, model in eval_config.models.items():
        if model.provider != "databricks_serving" or not model.endpoint:
            continue
        entry = models_cfg.get(model_id) if isinstance(models_cfg.get(model_id), dict) else {}
        serve = entry.get("serve") if isinstance(entry.get("serve"), dict) else {}
        register = entry.get("register") if isinstance(entry.get("register"), dict) else {}
        leaf = str(entry.get("registered_model_leaf") or _default_registered_leaf(model_id))
        specs.append(
            ServingSpec(
                model_id=model_id,
                label=model.label,
                endpoint=model.endpoint,
                registered_model_fqdn=f"{prefix}{leaf}",
                alias=alias,
                workload_type=str(serve.get("workload_type") or defaults.get("workload_type") or "CPU"),
                workload_size=str(serve.get("workload_size") or defaults.get("workload_size") or "Medium"),
                scale_to_zero=bool(serve.get("scale_to_zero", defaults.get("scale_to_zero", False))),
                route_optimized=bool(serve.get("route_optimized", defaults.get("route_optimized", False))),
                served_entity_name=str(serve.get("served_entity_name") or model_id),
                register_type=_optional_str(register.get("type")),
                register_candidate_id=_optional_str(register.get("candidate_id")),
            )
        )
    return specs


def get_serving_spec(model_id: str, *, config_path: str | Path | None = None) -> ServingSpec:
    for spec in load_serving_specs(config_path=config_path):
        if spec.model_id == model_id:
            return spec
    known = ", ".join(spec.model_id for spec in load_serving_specs(config_path=config_path))
    raise KeyError(f"Unknown databricks serving model_id={model_id!r}. Known: {known}")


def _default_registered_leaf(model_id: str) -> str:
    if model_id.startswith("databricks_"):
        return "genie_asr_" + model_id.removeprefix("databricks_")
    return "genie_asr_" + model_id


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
