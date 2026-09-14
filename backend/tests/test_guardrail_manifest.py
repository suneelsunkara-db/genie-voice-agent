from __future__ import annotations

from genie_voice.guardrails import get_policy_manifest


def test_policy_manifest_references_are_valid() -> None:
    manifest = get_policy_manifest()
    assert manifest.version == "2.0"
    assert manifest.assignments.model_services == {
        "qwen": "interactive_model_safety",
        "gpt55": "transform_model_safety",
    }
    assert manifest.gateway_functions_for_service("qwen") == [
        "system.ai.block_unsafe_content",
        "system.ai.block_jailbreak",
        "system.ai.detect_sensitive_data",
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
            "options": {
                "phases": "pre_call",
                "dry_run": "false",
                "model_service": "model-services/system.ai.gpt-5-2",
                "max_turns": "10",
            },
        }
    ]
    policies, complete = manifest.gateway_deployment("qwen", configured)
    by_id = {policy["policy_id"]: policy for policy in policies}
    assert by_id["jailbreak"]["deployment_state"] == "configured"
    assert by_id["unsafe_content"]["deployment_state"] == "missing_or_drifted"
    assert complete is False


def test_gateway_deployment_rejects_unassigned_policy() -> None:
    manifest = get_policy_manifest()
    configured = [
        {
            "name": "block-unsafe",
            "handler": "system.ai.block_unsafe_content",
            "rank": 1,
            "options": {
                "phases": "pre_call,post_call",
                "dry_run": "false",
                "model_service": "model-services/system.ai.gpt-5-2",
                "max_turns": "10",
            },
        },
        {
            "name": "block-jailbreak",
            "handler": "system.ai.block_jailbreak",
            "rank": 2,
            "options": {
                "phases": "pre_call",
                "dry_run": "false",
                "model_service": "model-services/system.ai.gpt-5-2",
                "max_turns": "10",
            },
        },
        {
            "name": "block-hallucination",
            "handler": "system.ai.block_hallucination",
            "rank": 4,
            "options": {
                "phases": "post_call",
                "dry_run": "false",
                "model_service": "model-services/system.ai.gpt-5-2",
            },
        },
    ]
    policies, complete = manifest.gateway_deployment("qwen", configured)
    unexpected = [
        policy for policy in policies if policy["deployment_state"] == "unexpected"
    ]
    assert [policy["function"] for policy in unexpected] == [
        "system.ai.block_hallucination"
    ]
    assert complete is False


def test_sensitive_policy_requires_exact_action_and_categories() -> None:
    manifest = get_policy_manifest()
    configured = [
        {
            "name": "contact-data",
            "handler": "system.ai.detect_sensitive_data",
            "rank": 3,
            "options": {
                "phases": "pre_call,post_call",
                "dry_run": "false",
                "action": "block",
                "categories": "class.email_address,class.phone_number",
            },
        }
    ]
    policies, complete = manifest.gateway_deployment("qwen", configured)
    by_id = {policy["policy_id"]: policy for policy in policies}
    assert by_id["contact_data_redaction"]["deployment_state"] == "missing_or_drifted"
    assert complete is False


def test_redact_matches_databricks_transform_action() -> None:
    manifest = get_policy_manifest()
    configured = [
        {
            "name": "contact-data",
            "handler": "system.ai.detect_sensitive_data",
            "rank": 3,
            "options": {
                "phases": "pre_call,post_call",
                "dry_run": "false",
                "action": "transform",
                "categories": "class.email_address,class.phone_number",
            },
        }
    ]
    policies, _complete = manifest.gateway_deployment("qwen", configured)
    by_id = {policy["policy_id"]: policy for policy in policies}
    assert by_id["contact_data_redaction"]["deployment_state"] == "configured"
