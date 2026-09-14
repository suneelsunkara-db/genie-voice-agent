from __future__ import annotations

from genie_voice.guardrails import get_policy_manifest


def test_policy_manifest_references_are_valid() -> None:
    manifest = get_policy_manifest()
    assert manifest.version == "1.0"
    assert manifest.assignments.model_services == {
        "qwen": "governed_foundation_model",
        "gpt55": "governed_foundation_model",
    }
    assert manifest.gateway_functions_for_service("qwen") == [
        "system.ai.block_unsafe_content",
        "system.ai.block_jailbreak",
        "system.ai.detect_sensitive_data",
        "system.ai.block_hallucination",
    ]


def test_catalog_is_api_ready_and_has_boundary_policies() -> None:
    manifest = get_policy_manifest()
    catalog = {entry["guard_id"]: entry for entry in manifest.catalog()}
    assert catalog["language_id"]["family"] == "speech_input"
    assert catalog["speech_output_boundary"]["stage"] == "Pre-TTS"


def test_gateway_deployment_requires_exact_handler_phase_and_rank() -> None:
    manifest = get_policy_manifest()
    configured = [
        {
            "name": "block-jailbreak",
            "handler": "system.ai.block_jailbreak",
            "rank": 2,
            "options": {"phases": "pre_call"},
        }
    ]
    policies, complete = manifest.gateway_deployment("qwen", configured)
    by_id = {policy["policy_id"]: policy for policy in policies}
    assert by_id["jailbreak"]["deployment_state"] == "configured"
    assert by_id["unsafe_content"]["deployment_state"] == "missing_or_drifted"
    assert complete is False
