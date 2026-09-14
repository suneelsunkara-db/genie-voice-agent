"""Provision the app-owned Unity AI Gateway model services.

Service creation, routing, rate limits, and inference tables have public APIs
and are safe to reconcile on every deploy. Service-policy writes remain UI-only
during the Databricks beta, but the public read API exposes their configuration.
Deployment therefore fails closed when an attachment is missing or drifts from
the manifest's handler, phase, or rank.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from genie_voice.config import get_settings
from genie_voice.databricks.client import get_workspace_client
from genie_voice.guardrails import get_policy_manifest


def _log(message: str) -> None:
    print(f"[gateway] {message}")


def _fqn_parts(fqn: str) -> tuple[str, str, str]:
    parts = fqn.split(".")
    if len(parts) != 3 or not all(parts):
        raise ValueError(f"model service must be catalog.schema.name, got {fqn!r}")
    return parts[0], parts[1], parts[2]


def _api(client: Any, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
    kwargs: dict[str, Any] = {}
    if body is not None:
        kwargs["body"] = body
    return client.api_client.do(method, path, **kwargs)


def _service_body(destination: str, comment: str) -> dict[str, Any]:
    return {
        "comment": comment,
        "config": {
            "routing": {
                "destinations": [
                    {
                        "name": destination,
                        "destination_type": "DESTINATION_TYPE_PAY_PER_TOKEN_FOUNDATION_MODEL",
                        "pay_per_token_config": {"model": f"models/{destination}"},
                        "traffic_percentage": 100,
                    }
                ]
            }
        },
    }


def _destination(service: dict[str, Any]) -> str:
    destinations = (((service.get("config") or {}).get("routing") or {}).get("destinations") or [])
    if not destinations:
        return ""
    return str(((destinations[0].get("pay_per_token_config") or {}).get("model") or "")).removeprefix(
        "models/"
    )


def _normalized_limits(limits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            **limit,
            "requests": int(limit["requests"]) if limit.get("requests") is not None else None,
            "tokens": int(limit["tokens"]) if limit.get("tokens") is not None else None,
        }
        for limit in limits
    ]


def _reconcile_service(
    client: Any,
    *,
    fqn: str,
    destination: str,
    roles: list[str],
    request_limit: int,
    token_limit: int,
    inference_parent: str,
    table_prefix: str,
) -> dict[str, Any]:
    catalog, schema, leaf = _fqn_parts(fqn)
    path = f"/api/2.1/unity-catalog/model-services/{fqn}"
    try:
        current = _api(client, "GET", path)
        _log(f"exists: {fqn}")
    except Exception as exc:  # noqa: BLE001
        if "does not exist" not in str(exc).lower() and "not found" not in str(exc).lower():
            raise
        parent = f"schemas/{catalog}.{schema}"
        body = _service_body(
            destination,
            f"Genie Voice guarded model service for: {', '.join(roles)}",
        )
        current = _api(
            client,
            "POST",
            (
                "/api/2.1/unity-catalog/model-services"
                f"?parent={parent}&model_service_id={leaf}"
            ),
            body,
        )
        _log(f"created: {fqn} -> {destination}")

    if _destination(current) != destination:
        routing = _service_body(destination, "")["config"]["routing"]
        current = _api(
            client,
            "PATCH",
            f"{path}?update_mask=config.routing.destinations",
            {"config": {"routing": routing}},
        )
        _log(f"updated route: {fqn} -> {destination}")

    limits = [
        {
            "key": "RATE_LIMIT_KEY_SERVICE",
            "renewal_period": "RATE_LIMIT_RENEWAL_PERIOD_MINUTE",
            "requests": request_limit,
            "tokens": token_limit,
        }
    ]
    current_limits = (current.get("config") or {}).get("rate_limits") or []
    if _normalized_limits(current_limits) != _normalized_limits(limits):
        current = _api(
            client,
            "PATCH",
            f"{path}?update_mask=config.rate_limits",
            {"config": {"rate_limits": limits}},
        )
        _log(f"rate limits: {fqn} = {request_limit} QPM / {token_limit} TPM")

    inference = (current.get("config") or {}).get("inference_table")
    if not inference:
        current = _api(
            client,
            "PATCH",
            f"{path}?update_mask=config.inference_table",
            {
                "config": {
                    "inference_table": {
                        "parent": inference_parent,
                        "table_name_prefix": table_prefix,
                    }
                }
            },
        )
        _log(f"inference table enabled: {table_prefix}_payload")
    elif inference.get("is_deleted"):
        raise RuntimeError(f"inference table for {fqn} was deleted; repair it before deployment")

    return current


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--guardrails-config", default="config/guardrails.yaml")
    args = parser.parse_args()

    raw = yaml.safe_load(Path(args.config).read_text()) or {}
    gateway = raw.get("ai_gateway") or {}
    if not gateway.get("enabled", False):
        _log("disabled in config; skipping")
        return

    services = gateway.get("model_services") or {}
    if not services:
        raise SystemExit("ai_gateway.enabled is true but model_services is empty")

    catalog = str((raw.get("databricks") or {}).get("catalog") or "")
    schema = str((raw.get("databricks") or {}).get("schema") or "")
    inference_parent = f"schemas/{catalog}.{schema}"
    request_limit = int(gateway.get("requests_per_minute") or 60)
    token_limit = int(gateway.get("tokens_per_minute") or 100_000)
    client = get_workspace_client(get_settings())
    manifest = get_policy_manifest(args.guardrails_config)

    policy_errors: list[str] = []
    for key, spec in services.items():
        fqn = str(spec.get("service") or "")
        destination = str(spec.get("destination") or "")
        roles = [str(role) for role in (spec.get("roles") or [])]
        if not fqn or not destination:
            raise SystemExit(f"ai_gateway.model_services.{key} needs service and destination")
        configured_bundle = str(spec.get("policy_bundle") or "")
        assigned_bundle = manifest.assignments.model_services.get(key)
        if not assigned_bundle or configured_bundle != assigned_bundle:
            raise SystemExit(
                f"model service {key!r} policy bundle mismatch: "
                f"config={configured_bundle!r}, manifest={assigned_bundle!r}"
            )
        current = _reconcile_service(
            client,
            fqn=fqn,
            destination=destination,
            roles=roles,
            request_limit=request_limit,
            token_limit=token_limit,
            inference_parent=inference_parent,
            table_prefix=f"ai_gateway_{key}",
        )
        table = (((current.get("config") or {}).get("inference_table") or {}).get("table") or "")
        if table:
            _log(f"logs: {str(table).removeprefix('tables/')}")
        policies, fully_configured = manifest.gateway_deployment(
            key, (current.get("config") or {}).get("service_policies") or []
        )
        missing = [
            f"{item['policy_id']} ({item['function']}, "
            f"{'+'.join(item['phases'])}, rank {item['rank']})"
            for item in policies
            if item["deployment_state"] != "configured"
        ]
        if fully_configured:
            _log(f"policies verified: {fqn}")
        else:
            policy_errors.append(f"{fqn}: {', '.join(missing)}")

    if policy_errors:
        raise SystemExit(
            "Gateway policy attachments are missing or drifted. "
            "Attachment writes are UI-only during the current Beta:\n- "
            + "\n- ".join(policy_errors)
        )


if __name__ == "__main__":
    main()
