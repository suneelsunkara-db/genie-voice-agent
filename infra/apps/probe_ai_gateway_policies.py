#!/usr/bin/env python3
"""Run the manifest's behavioral policy conformance matrix."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from genie_voice.config import get_settings
from genie_voice.databricks.ai_gateway import GatewayPolicyDenied, invoke
from genie_voice.databricks.client import get_workspace_client
from genie_voice.guardrails import get_policy_manifest


def _text(payload: dict) -> str:
    choices = payload.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return ""
    message = choices[0].get("message") or {}
    return str(message.get("content") or "")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--guardrails-config", default="config/guardrails.yaml")
    args = parser.parse_args()

    raw = yaml.safe_load(Path(args.config).read_text()) or {}
    services = ((raw.get("ai_gateway") or {}).get("model_services") or {})
    manifest = get_policy_manifest(args.guardrails_config)
    settings = get_settings()
    client = get_workspace_client(settings)
    failures: list[str] = []

    for probe in manifest.conformance:
        service = str((services.get(probe.service) or {}).get("service") or "")
        if not service:
            failures.append(f"{probe.id}: service {probe.service!r} is not configured")
            continue
        denied = False
        denial = ""
        output = ""
        try:
            payload = invoke(
                host=settings.databricks_host,
                authenticate=client.config.authenticate,
                endpoint=service,
                inputs={
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a concise assistant.",
                        },
                        {"role": "user", "content": probe.input},
                    ],
                    "max_tokens": 80,
                },
                timeout_s=120,
                request_tags={
                    "traffic_class": "deploy_probe",
                    "surface": "deployment",
                    "profile": "none",
                    "capability": "gateway_conformance",
                    "model_role": "policy_probe",
                    "probe_id": probe.id,
                },
            )
            output = _text(payload)
        except GatewayPolicyDenied as exc:
            denied = True
            denial = f"{exc.policy_name}: {exc.reason}"
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{probe.id}: request failed: {type(exc).__name__}: {exc}")
            continue

        if probe.expectation == "deny" and not denied:
            failures.append(f"{probe.id}: expected DENY but request was allowed")
        elif probe.expectation in {"allow", "redact"} and denied:
            failures.append(
                f"{probe.id}: expected {probe.expectation.upper()} but request was denied"
                + (f" ({denial})" if denial else "")
            )
        elif probe.expectation == "redact" and any(
            forbidden.lower() in output.lower()
            for forbidden in probe.forbidden_output
        ):
            failures.append(f"{probe.id}: raw sensitive value survived redaction")
        else:
            print(
                f"[gateway-probe] ok: {probe.id} "
                f"({probe.expectation.upper()})"
            )

    if failures:
        raise SystemExit(
            "AI Gateway policy conformance failed:\n- " + "\n- ".join(failures)
        )


if __name__ == "__main__":
    main()
