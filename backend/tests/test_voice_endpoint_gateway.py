import sys
from pathlib import Path
from types import SimpleNamespace

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "infra" / "apps"))

import provision_voice_endpoint_gateway as gateway  # noqa: E402


class _Api:
    def __init__(self) -> None:
        self.configs: dict[str, dict] = {}
        self.puts: list[tuple[str, dict]] = []

    def do(self, method: str, path: str, body=None):
        endpoint = path.split("/serving-endpoints/", 1)[1].split("/", 1)[0]
        if method == "GET":
            return {
                "task": "agent/v1/responses",
                "ai_gateway": self.configs.get(endpoint),
            }
        assert method == "PUT"
        self.puts.append((path, body))
        self.configs[endpoint] = {
            "inference_table_config": body["inference_table_config"]
        }
        return self.configs[endpoint]


def test_reconcile_voice_gateway_is_idempotent(tmp_path, monkeypatch) -> None:
    config = {
        "databricks": {"catalog": "catalog", "schema": "schema"},
        "realtime_voice": {
            "serving": {
                "ai_gateway": {
                    "enabled": True,
                    "table_prefixes": {"qwen": "voice_qwen", "vox": "voice_vox"},
                }
            },
            "stt_candidates": {"qwen": {"endpoint": "stt"}},
            "tts_candidates": {"vox": {"endpoint": "tts"}},
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    api = _Api()
    client = SimpleNamespace(api_client=api)
    monkeypatch.setattr(gateway, "get_settings", lambda: object())
    monkeypatch.setattr(gateway, "get_workspace_client", lambda _settings: client)

    first = gateway.reconcile(str(path))
    second = gateway.reconcile(str(path))

    assert [item["action"] for item in first] == ["updated", "updated"]
    assert [item["action"] for item in second] == ["verified", "verified"]
    assert len(api.puts) == 2
    assert api.configs["stt"]["inference_table_config"] == {
        "enabled": True,
        "catalog_name": "catalog",
        "schema_name": "schema",
        "table_name_prefix": "voice_qwen",
    }


def test_agent_endpoint_configuration_only_claims_supported_feature() -> None:
    source = (REPO / "infra/apps/provision_voice_endpoint_gateway.py").read_text(
        encoding="utf-8"
    )
    assert "inference_table_config" in source
    assert '"rate_limits"' not in source
    assert '"guardrails"' not in source
    assert '"usage_tracking_config"' not in source
    assert '"fallback_config"' not in source
