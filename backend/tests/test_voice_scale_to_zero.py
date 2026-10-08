from types import SimpleNamespace


class _Api:
    def __init__(self, payloads):
        self.payloads = payloads
        self.puts = []

    def do(self, method, path, body=None):
        name = path.split("/serving-endpoints/", 1)[1].split("/", 1)[0]
        if method == "GET":
            return self.payloads[name]
        if method == "PUT":
            self.puts.append((name, body))
            target = body["served_entities"][0]["scale_to_zero_enabled"]
            for entity in self.payloads[name]["config"]["served_entities"]:
                entity["scale_to_zero_enabled"] = target
            self.payloads[name]["state"]["config_update"] = "IN_PROGRESS"
            return {}
        raise AssertionError((method, path))


class _Client:
    def __init__(self, payloads):
        self.api_client = _Api(payloads)


def _payload(scale_to_zero):
    return {
        "state": {"ready": "READY", "config_update": "NOT_UPDATING"},
        "config": {
            "served_entities": [
                {
                    "name": "candidate",
                    "entity_name": "cat.schema.model",
                    "entity_version": "1",
                    "workload_type": "GPU_MEDIUM",
                    "workload_size": "Small",
                    "scale_to_zero_enabled": scale_to_zero,
                }
            ],
            "traffic_config": {
                "routes": [
                    {"served_model_name": "candidate", "traffic_percentage": 100}
                ]
            },
        },
    }


def _candidates():
    return [
        (
            "stt:qwen",
            {
                "endpoint": "qwen-endpoint",
                "base_model": "Qwen/Qwen3-ASR-1.7B",
            },
        ),
        (
            "tts:vox",
            {
                "endpoint": "vox-endpoint",
                "base_model": "openbmb/VoxCPM2",
            },
        ),
    ]


def test_voice_serving_status_reports_scale_to_zero(monkeypatch):
    from genie_voice import readiness

    monkeypatch.setattr(readiness, "_voice_candidates", _candidates)
    client = _Client(
        {"qwen-endpoint": _payload(True), "vox-endpoint": _payload(False)}
    )

    status = readiness.voice_serving_status(SimpleNamespace(), client=client)

    assert status["requires_action"] is True
    assert status["compliant"] is False
    assert status["metrics"] == {
        "total": 2,
        "scale_to_zero_enabled": 1,
        "scale_to_zero_disabled": 1,
        "ready": 2,
        "updating": 0,
        "failed": 0,
    }


def test_disable_voice_scale_to_zero_updates_only_drifted_endpoint(monkeypatch):
    from genie_voice import readiness
    from genie_voice.databricks import client as client_module

    monkeypatch.setattr(readiness, "_voice_candidates", _candidates)
    fake = _Client(
        {"qwen-endpoint": _payload(True), "vox-endpoint": _payload(False)}
    )
    monkeypatch.setattr(client_module, "get_workspace_client", lambda settings: fake)

    result = readiness.disable_voice_scale_to_zero(SimpleNamespace())

    assert result["accepted"] == ["qwen-endpoint"]
    assert result["unchanged"] == ["vox-endpoint"]
    assert fake.api_client.puts[0][1]["served_entities"][0][
        "scale_to_zero_enabled"
    ] is False
    assert result["status"]["updating"] is True


def test_enable_voice_scale_to_zero_updates_only_warm_endpoint(monkeypatch):
    from genie_voice import readiness
    from genie_voice.databricks import client as client_module

    monkeypatch.setattr(readiness, "_voice_candidates", _candidates)
    fake = _Client(
        {"qwen-endpoint": _payload(False), "vox-endpoint": _payload(True)}
    )
    monkeypatch.setattr(client_module, "get_workspace_client", lambda settings: fake)

    result = readiness.enable_voice_scale_to_zero(SimpleNamespace())

    assert result["accepted"] == ["qwen-endpoint"]
    assert result["unchanged"] == ["vox-endpoint"]
    assert fake.api_client.puts[0][1]["served_entities"][0][
        "scale_to_zero_enabled"
    ] is True
    assert result["status"]["updating"] is True


def test_scale_to_zero_retries_failed_update(monkeypatch):
    from genie_voice import readiness
    from genie_voice.databricks import client as client_module

    monkeypatch.setattr(readiness, "_voice_candidates", _candidates)
    qwen = _payload(False)
    qwen["state"]["config_update"] = "UPDATE_FAILED"
    fake = _Client({"qwen-endpoint": qwen, "vox-endpoint": _payload(False)})
    monkeypatch.setattr(client_module, "get_workspace_client", lambda settings: fake)

    result = readiness.disable_voice_scale_to_zero(SimpleNamespace())

    assert result["accepted"] == ["qwen-endpoint"]
    assert result["unchanged"] == ["vox-endpoint"]
