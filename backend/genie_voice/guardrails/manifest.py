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
    mode: Literal["enforce", "log"] = "enforce"
    action: Literal["block", "redact"] | None = None
    categories: list[str] = Field(default_factory=list)
    evaluator: str | None = None
    max_turns: int | None = Field(default=None, ge=1)


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


class ConformanceProbe(BaseModel):
    id: str
    service: str
    expectation: Literal["allow", "deny", "redact"]
    input: str
    forbidden_output: list[str] = Field(default_factory=list)


class PolicyManifest(BaseModel):
    version: str
    policies: dict[str, PolicyDefinition]
    gateway_policies: dict[str, GatewayPolicyDefinition]
    bundles: dict[str, PolicyBundle]
    assignments: PolicyAssignments
    conformance: list[ConformanceProbe] = Field(default_factory=list)

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
        unknown_probe_services = sorted(
            {probe.service for probe in self.conformance}
            - set(self.assignments.model_services)
        )
        if unknown_probe_services:
            raise ValueError(
                f"conformance probes reference unknown services: {unknown_probe_services}"
            )
        return self

    def gateway_functions_for_service(self, service_key: str) -> list[str]:
        bundle_id = self.assignments.model_services[service_key]
        bundle = self.bundles[bundle_id]
        return [self.gateway_policies[policy_id].function for policy_id in bundle.policies]

    def gateway_deployment(
        self, service_key: str, configured: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], bool]:
        """Compare the complete observed policy set with the manifest."""
        bundle_id = self.assignments.model_services[service_key]
        active = [item for item in configured if not item.get("is_deleted")]
        phase_name = {"input": "pre_call", "output": "post_call"}
        details: list[dict[str, Any]] = []
        matched_items: set[int] = set()

        def options_match(
            expected: GatewayPolicyDefinition, item: dict[str, Any]
        ) -> bool:
            options = item.get("options") or {}
            actual_phases = {
                phase.strip()
                for phase in str(options.get("phases") or "").split(",")
                if phase.strip()
            }
            if actual_phases != {phase_name[phase] for phase in expected.phases}:
                return False
            if str(options.get("dry_run") or "false").lower() != str(
                expected.mode == "log"
            ).lower():
                return False
            actual_action = str(options.get("action") or "").lower()
            if expected.action:
                aliases = {"redact": {"redact", "transform"}, "block": {"block"}}
                if actual_action not in aliases.get(expected.action, {expected.action}):
                    return False
            if expected.categories:
                actual_categories = {
                    category.strip()
                    for category in str(options.get("categories") or "").split(",")
                    if category.strip()
                }
                required = set(expected.categories)
                # Block attachments may add extra categories; redaction must stay exact.
                if expected.action == "block":
                    if not required <= actual_categories:
                        return False
                elif actual_categories != required:
                    return False
            if expected.evaluator:
                actual_evaluator = str(options.get("model_service") or "").removeprefix(
                    "model-services/"
                )
                if actual_evaluator != expected.evaluator:
                    return False
            if expected.max_turns is not None and int(options.get("max_turns") or 0) != expected.max_turns:
                return False
            return True

        for policy_id in self.bundles[bundle_id].policies:
            expected = self.gateway_policies[policy_id]
            match = next(
                (
                    item
                    for item in active
                    if item.get("handler") == expected.function
                    and int(item.get("rank") or 0) == expected.rank
                    and options_match(expected, item)
                ),
                None,
            )
            if match:
                matched_items.add(id(match))
            details.append(
                {
                    "policy_id": policy_id,
                    **expected.model_dump(),
                    "deployment_state": "configured" if match else "missing_or_drifted",
                    "deployed_name": match.get("name") if match else None,
                }
            )
        for item in active:
            if id(item) in matched_items:
                continue
            public_phase = {"pre_call": "input", "post_call": "output"}
            phases = [
                public_phase.get(phase.strip(), phase.strip())
                for phase in str((item.get("options") or {}).get("phases") or "").split(",")
                if phase.strip()
            ]
            details.append(
                {
                    "policy_id": str(item.get("name") or item.get("handler") or "unknown"),
                    "function": str(item.get("handler") or ""),
                    "phases": phases,
                    "rank": int(item.get("rank") or 0),
                    "mode": (
                        "log"
                        if str((item.get("options") or {}).get("dry_run") or "false").lower()
                        == "true"
                        else "enforce"
                    ),
                    "action": (item.get("options") or {}).get("action"),
                    "categories": [
                        category.strip()
                        for category in str(
                            (item.get("options") or {}).get("categories") or ""
                        ).split(",")
                        if category.strip()
                    ],
                    "deployment_state": "unexpected",
                    "deployed_name": item.get("name"),
                }
            )
        return details, all(
            item["deployment_state"] == "configured" for item in details
        )

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
