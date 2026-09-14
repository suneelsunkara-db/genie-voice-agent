"""Typed loader for the repository's guardrail policy manifest."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class PolicyDefinition(BaseModel):
    title: str
    owner: Literal["application", "gateway", "qwen"]
    family: str
    group: str
    stage: str
    kind: Literal["deterministic", "llm", "model-signal"]
    status: Literal["live", "delegated", "planned"]
    phase: str
    description: str


class GatewayPolicyDefinition(BaseModel):
    function: str
    phases: list[Literal["input", "output"]]
    rank: int = Field(ge=1)


class PolicyBundle(BaseModel):
    title: str
    enforcer: Literal["application", "unity_ai_gateway"]
    policies: list[str]


class BoundaryAssignment(BaseModel):
    bundle: str
    resources: list[str] = Field(default_factory=list)


class PolicyAssignments(BaseModel):
    model_services: dict[str, str] = Field(default_factory=dict)
    boundaries: dict[str, BoundaryAssignment] = Field(default_factory=dict)


class PolicyManifest(BaseModel):
    version: str
    policies: dict[str, PolicyDefinition]
    gateway_policies: dict[str, GatewayPolicyDefinition]
    bundles: dict[str, PolicyBundle]
    assignments: PolicyAssignments

    @model_validator(mode="after")
    def validate_references(self) -> "PolicyManifest":
        for bundle_id, bundle in self.bundles.items():
            source = self.gateway_policies if bundle.enforcer == "unity_ai_gateway" else self.policies
            missing = sorted(set(bundle.policies) - set(source))
            if missing:
                raise ValueError(f"bundle {bundle_id!r} references unknown policies: {missing}")
        for service_key, bundle_id in self.assignments.model_services.items():
            bundle = self.bundles.get(bundle_id)
            if bundle is None:
                raise ValueError(f"model service {service_key!r} references unknown bundle {bundle_id!r}")
            if bundle.enforcer != "unity_ai_gateway":
                raise ValueError(f"model service {service_key!r} must use a Unity AI Gateway bundle")
        for boundary, assignment in self.assignments.boundaries.items():
            bundle = self.bundles.get(assignment.bundle)
            if bundle is None:
                raise ValueError(f"boundary {boundary!r} references unknown bundle {assignment.bundle!r}")
            if bundle.enforcer != "application":
                raise ValueError(f"boundary {boundary!r} must use an application bundle")
        return self

    def gateway_functions_for_service(self, service_key: str) -> list[str]:
        bundle_id = self.assignments.model_services[service_key]
        bundle = self.bundles[bundle_id]
        return [self.gateway_policies[policy_id].function for policy_id in bundle.policies]

    def gateway_deployment(
        self, service_key: str, configured: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], bool]:
        """Compare observed policy handlers, phases, and ranks with the manifest."""
        bundle_id = self.assignments.model_services[service_key]
        active = [item for item in configured if not item.get("is_deleted")]
        phase_name = {"input": "pre_call", "output": "post_call"}
        details: list[dict[str, Any]] = []
        for policy_id in self.bundles[bundle_id].policies:
            expected = self.gateway_policies[policy_id]
            expected_phases = {phase_name[phase] for phase in expected.phases}
            match = next(
                (
                    item
                    for item in active
                    if item.get("handler") == expected.function
                    and int(item.get("rank") or 0) == expected.rank
                    and {
                        phase.strip()
                        for phase in str((item.get("options") or {}).get("phases") or "").split(",")
                        if phase.strip()
                    }
                    == expected_phases
                ),
                None,
            )
            details.append(
                {
                    "policy_id": policy_id,
                    **expected.model_dump(),
                    "deployment_state": "configured" if match else "missing_or_drifted",
                    "deployed_name": match.get("name") if match else None,
                }
            )
        return details, all(item["deployment_state"] == "configured" for item in details)

    def catalog(self) -> list[dict]:
        return [
            {"guard_id": policy_id, **definition.model_dump()}
            for policy_id, definition in self.policies.items()
        ]


def _default_path() -> Path:
    return Path(__file__).resolve().parents[3] / "config" / "guardrails.yaml"


@lru_cache(maxsize=1)
def get_policy_manifest(path: str | Path | None = None) -> PolicyManifest:
    resolved = Path(path or os.environ.get("GENIE_GUARDRAILS_CONFIG") or _default_path())
    with resolved.open() as handle:
        return PolicyManifest.model_validate(yaml.safe_load(handle) or {})
