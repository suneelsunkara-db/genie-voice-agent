from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

from genie_voice.databricks.client import get_workspace_client
from genie_voice.ml_asr.audio import mime_type_for, read_audio_bytes
from genie_voice.ml_asr.manifest import load_eval_manifest
from genie_voice.ml_asr.serving.catalog import ServingSpec, get_serving_spec, load_serving_specs


def list_deployments(*, config_path: str | None = None) -> list[dict[str, Any]]:
    return [_spec_summary(spec) for spec in load_serving_specs(config_path=config_path)]


def preflight(model_id: str, *, config_path: str | None = None) -> dict[str, Any]:
    spec = get_serving_spec(model_id, config_path=config_path)
    client = get_workspace_client()
    version = _resolve_alias_version(client, spec.registered_model_fqdn, spec.alias)
    return {
        "model_id": spec.model_id,
        "endpoint": spec.endpoint,
        "registered_model": spec.registered_model_fqdn,
        "alias": spec.alias,
        "version": version,
        "workload_type": spec.workload_type,
        "workload_size": spec.workload_size,
        "scale_to_zero": spec.scale_to_zero,
        "ready": True,
    }


def deploy(model_id: str, *, config_path: str | None = None, recreate_on_route_change: bool = True) -> dict[str, Any]:
    spec = get_serving_spec(model_id, config_path=config_path)
    client = get_workspace_client()
    version = _resolve_alias_version(client, spec.registered_model_fqdn, spec.alias)
    config = _endpoint_config(spec, version)

    exists = _endpoint_exists(client, spec.endpoint)
    if exists and recreate_on_route_change:
        current = _route_optimized(client, spec.endpoint)
        if current is not None and current != spec.route_optimized:
            client.serving_endpoints.delete(name=spec.endpoint)
            exists = False

    if exists:
        client.api_client.do(
            "PUT",
            f"/api/2.0/serving-endpoints/{spec.endpoint}/config",
            body=config,
        )
        action = "update-config"
    else:
        client.api_client.do(
            "POST",
            "/api/2.0/serving-endpoints",
            body={"name": spec.endpoint, "config": config, "route_optimized": spec.route_optimized},
        )
        action = "create"

    status = endpoint_status(model_id, config_path=config_path)
    return {"model_id": spec.model_id, "action": action, "endpoint": spec.endpoint, "version": version, "status": status}


def deploy_all(*, config_path: str | None = None) -> list[dict[str, Any]]:
    return [deploy(spec.model_id, config_path=config_path) for spec in load_serving_specs(config_path=config_path)]


def endpoint_status(model_id: str, *, config_path: str | None = None) -> dict[str, Any]:
    spec = get_serving_spec(model_id, config_path=config_path)
    client = get_workspace_client()
    try:
        endpoint = client.serving_endpoints.get(name=spec.endpoint)
    except Exception as exc:  # noqa: BLE001
        return {"endpoint": spec.endpoint, "exists": False, "error": str(exc)}

    state = getattr(endpoint, "state", None)
    config = getattr(endpoint, "config", None)
    ready = getattr(state, "ready", None)
    config_update = getattr(state, "config_update", None)
    deployments = []
    for entity in getattr(config, "served_entities", None) or []:
        deployments.append(
            {
                "name": getattr(entity, "name", None),
                "entity_name": getattr(entity, "entity_name", None),
                "entity_version": getattr(entity, "entity_version", None),
                "state": str(getattr(getattr(entity, "state", None), "deployment", "")),
            }
        )
    return {
        "endpoint": spec.endpoint,
        "exists": True,
        "ready": str(ready) if ready is not None else None,
        "config_update": str(config_update) if config_update is not None else None,
        "route_optimized": _route_optimized(client, spec.endpoint),
        "deployments": deployments,
    }


def smoke(model_id: str, *, config_path: str | None = None, manifest_path: str | None = None) -> dict[str, Any]:
    from genie_voice.ml_asr.config import load_config

    spec = get_serving_spec(model_id, config_path=config_path)
    clip = _smoke_clip(spec, manifest_path=manifest_path, config_path=config_path)
    config = load_config(config_path=config_path)
    language = next(iter(config.models[spec.model_id].languages or config.eval_languages))
    body = {
        "dataframe_records": [
            {
                "audio_b64": base64.b64encode(read_audio_bytes(clip.audio_path)).decode("ascii"),
                "mime_type": mime_type_for(clip.audio_path, clip.audio_format),
                "speaker": 0,
                "language": language,
            }
        ]
    }
    client = get_workspace_client()
    response = client.serving_endpoints.query(name=spec.endpoint, **body)
    payload = response.as_dict() if hasattr(response, "as_dict") else dict(response)
    predictions = payload.get("predictions") or []
    first = predictions[0] if predictions else {}
    return {
        "model_id": spec.model_id,
        "endpoint": spec.endpoint,
        "clip_id": clip.clip_id,
        "transcript": first.get("transcript") or first.get("raw_transcript"),
        "confidence": first.get("confidence"),
        "raw": payload,
    }


def _smoke_clip(spec: ServingSpec, *, manifest_path: str | None, config_path: str | None):
    from genie_voice.ml_asr.config import load_config

    config = load_config(config_path=config_path)
    language = next(iter(config.models[spec.model_id].languages or config.eval_languages))
    if manifest_path:
        manifest = load_eval_manifest(manifest_path, splits=["holdout", "validation", "test"])
    else:
        dataset_id = next(
            ds.dataset_id for ds in config.datasets.values() if language in ds.languages and ds.eval_tier == "acoustic"
        )
        manifest = load_eval_manifest(
            config.manifest_path(dataset_id, language, volume_mode=False),
            splits=["holdout"],
        )
    if not manifest.clips:
        raise RuntimeError(f"No clips found for smoke test ({spec.model_id})")
    return manifest.clips[0]


def _endpoint_config(spec: ServingSpec, version: str) -> dict[str, Any]:
    served_entity = {
        "name": spec.served_entity_name,
        "entity_name": spec.registered_model_fqdn,
        "entity_version": version,
        "workload_type": spec.workload_type,
        "workload_size": spec.workload_size,
        "scale_to_zero_enabled": spec.scale_to_zero,
    }
    return {
        "served_entities": [served_entity],
        "traffic_config": {
            "routes": [{"served_model_name": spec.served_entity_name, "traffic_percentage": 100}],
        },
    }


def _resolve_alias_version(client, registered_model: str, alias: str) -> str:
    payload = client.api_client.do("GET", f"/api/2.1/unity-catalog/models/{registered_model}/aliases/{alias}")
    version = payload.get("version")
    if not version:
        raise RuntimeError(f"Alias {alias!r} on {registered_model} did not resolve to a version.")
    return str(version)


def _endpoint_exists(client, endpoint: str) -> bool:
    try:
        client.serving_endpoints.get(name=endpoint)
        return True
    except Exception:  # noqa: BLE001
        return False


def _route_optimized(client, endpoint: str) -> bool | None:
    try:
        endpoint_info = client.serving_endpoints.get(name=endpoint)
    except Exception:  # noqa: BLE001
        return None
    return bool(getattr(endpoint_info, "route_optimized", None))


def _spec_summary(spec: ServingSpec) -> dict[str, Any]:
    return {
        "model_id": spec.model_id,
        "label": spec.label,
        "endpoint": spec.endpoint,
        "registered_model": spec.registered_model_fqdn,
        "workload_type": spec.workload_type,
        "workload_size": spec.workload_size,
        "register_type": spec.register_type,
    }
