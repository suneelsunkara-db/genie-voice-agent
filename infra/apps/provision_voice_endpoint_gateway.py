"""Enable every AI Gateway feature supported by the voice agent endpoints.

The Qwen3-ASR and VoxCPM2 models are MLflow ResponsesAgent endpoints
(`task=agent/v1/responses`). Databricks currently supports only AI Gateway
inference tables for agent endpoints; rate limits, guardrails, usage tracking,
and fallback are not supported for this endpoint type.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from genie_voice.config import get_settings
from genie_voice.databricks.client import get_workspace_client


def _log(message: str) -> None:
    print(f"[voice-gateway] {message}")


def _configured_endpoints(raw: dict[str, Any]) -> list[tuple[str, str]]:
    voice = raw.get("realtime_voice") or {}
    gateway = ((voice.get("serving") or {}).get("ai_gateway") or {})
    prefixes = gateway.get("table_prefixes") or {}
    endpoints: list[tuple[str, str]] = []
    for group in ("stt_candidates", "tts_candidates"):
        for candidate_id, candidate in (voice.get(group) or {}).items():
            if not isinstance(candidate, dict) or not candidate.get("endpoint"):
                continue
            prefix = str(prefixes.get(candidate_id) or "").strip()
            if not prefix:
                raise ValueError(
                    f"realtime_voice.serving.ai_gateway.table_prefixes.{candidate_id} is required"
                )
            endpoints.append((str(candidate["endpoint"]), prefix))
    return endpoints


def _inference_config(
    *, enabled: bool, catalog: str, schema: str, prefix: str
) -> dict[str, Any]:
    config: dict[str, Any] = {"enabled": enabled}
    if enabled:
        config.update(
            {
                "catalog_name": catalog,
                "schema_name": schema,
                "table_name_prefix": prefix,
            }
        )
    return config


def reconcile(config_path: str) -> list[dict[str, Any]]:
    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    voice = raw.get("realtime_voice") or {}
    gateway = ((voice.get("serving") or {}).get("ai_gateway") or {})
    if not gateway.get("enabled", False):
        _log("disabled in config; skipping")
        return []

    db = raw.get("databricks") or {}
    catalog = str(db.get("catalog") or "").strip()
    schema = str(db.get("schema") or "").strip()
    if not catalog or not schema:
        raise ValueError("databricks.catalog and databricks.schema are required")

    client = get_workspace_client(get_settings())
    results: list[dict[str, Any]] = []
    for endpoint, prefix in _configured_endpoints(raw):
        path = f"/api/2.0/serving-endpoints/{endpoint}"
        current_endpoint = client.api_client.do("GET", path)
        task = str(current_endpoint.get("task") or "")
        if task != "agent/v1/responses":
            raise RuntimeError(
                f"{endpoint} has unexpected task {task!r}; expected agent/v1/responses"
            )
        current = ((current_endpoint.get("ai_gateway") or {}).get("inference_table_config") or {})
        desired = _inference_config(
            enabled=True, catalog=catalog, schema=schema, prefix=prefix
        )
        comparable = {key: current.get(key) for key in desired}
        if comparable == desired:
            _log(f"verified: {endpoint} -> {catalog}.{schema}.{prefix}_payload")
            results.append({"endpoint": endpoint, "action": "verified", "config": desired})
            continue

        # Databricks requires disabling an existing inference table before its
        # catalog, schema, or prefix can be changed.
        if current.get("enabled"):
            client.api_client.do(
                "PUT",
                f"{path}/ai-gateway",
                body={"inference_table_config": {"enabled": False}},
            )
            _log(f"disabled drifted inference table: {endpoint}")
        client.api_client.do(
            "PUT",
            f"{path}/ai-gateway",
            body={"inference_table_config": desired},
        )
        refreshed = client.api_client.do("GET", path)
        actual = ((refreshed.get("ai_gateway") or {}).get("inference_table_config") or {})
        if {key: actual.get(key) for key in desired} != desired:
            raise RuntimeError(f"{endpoint} AI Gateway inference-table verification failed")
        _log(f"enabled: {endpoint} -> {catalog}.{schema}.{prefix}_payload")
        results.append({"endpoint": endpoint, "action": "updated", "config": desired})
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    reconcile(args.config)


if __name__ == "__main__":
    main()
