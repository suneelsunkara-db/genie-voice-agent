"""Guard tests for the voice framework seams.

These lock the invariants the framework refactor established so they can't quietly
regress:

  1. Runtime knobs come from CONFIG (turn-taking timings + deep-dive timeout),
     not hardcoded constants.
  2. The supported-language set has ONE canonical source.
  3. No caller pokes the private ``GenieClient._space_id`` (they pass ``space_name``).
  4. The realtime serving is built in ONE place (no second ``DatabricksServing``);
     there is no mlflow ``from_workspace`` serving path.
  5. Deploy-identifying config values fail fast when omitted (no wrong-target default).
  6. Deepgram STT params come from config options, not a hardcoded ``nova-3``.
  7. Cockpit mic transcription uses the shared ResponsesAgent contract.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _write(dir_: Path, text: str) -> None:
    (dir_ / "config.yaml").write_text(text, encoding="utf-8")


def test_cockpit_mic_uses_shared_responses_agent() -> None:
    router = (REPO / "api/app/routers/agent_assist.py").read_text(encoding="utf-8")
    config = (REPO / "config/config.yaml").read_text(encoding="utf-8")

    assert '"custom_inputs"' in router
    assert "custom_outputs" in router
    assert "dataframe_records" not in router
    assert "endpoint: realtime_voice_stt_qwen3_asr_1_7b" in config
    assert "voice_asr_en_finetuned_whisper_lora" not in config


def test_realtime_settings_honors_config_overrides(tmp_path):
    """turn_taking + deep_dive keys in config drive the settings (not code defaults)."""
    from realtime_api.config import RealtimeSettings

    _write(
        tmp_path,
        """
realtime_voice:
  llm_endpoint: dummy_llm
  conversion_endpoint: dummy_convert
  stt_candidates:
    a: { endpoint: dummy_stt, supported_languages: [en, es, fr] }
  tts_candidates:
    b: { endpoint: dummy_tts, supported_languages: [en, es] }
  turn_taking:
    vad_silence_ms: 1234
    max_turn_seconds: 42
    min_speech_ms: 111
    barge_in_ms: 999
    sample_rate_hz: 8000
  timeouts:
    predict_s: 33
    tts_stream_s: 111
    llm_turn_s: 44
  deep_dive:
    read_timeout_s: 77
    summary_temperature: 0.9
    summary_max_tokens: 42
""",
    )
    s = RealtimeSettings.from_config(tmp_path)
    assert s.vad_silence_ms == 1234
    assert s.max_turn_seconds == 42
    assert s.min_speech_ms == 111
    assert s.barge_in_ms == 999
    assert s.sample_rate_hz == 8000
    assert s.deep_dive_read_timeout_s == 77.0
    assert s.predict_timeout_s == 33.0
    assert s.tts_stream_timeout_s == 111.0
    assert s.llm_turn_timeout_s == 44.0
    assert s.deep_dive_summary_temperature == 0.9
    assert s.deep_dive_summary_max_tokens == 42
    # Runtime text->text conversion model is split from the voice-turn llm_endpoint.
    assert s.conversion_endpoint == "dummy_convert"
    assert s.llm_endpoint == "dummy_llm"
    # Supported languages = STT ∩ TTS (fr is STT-only, so excluded).
    assert set(s.supported_languages) == {"en", "es"}


def test_realtime_settings_fall_back_to_code_defaults(tmp_path):
    """Omitting the optional blocks keeps the built-in defaults (backward compatible)."""
    from realtime_api.config import RealtimeSettings

    _write(
        tmp_path,
        """
realtime_voice:
  llm_endpoint: dummy_llm
  stt_candidates:
    a: { endpoint: dummy_stt, supported_languages: [en] }
  tts_candidates:
    b: { endpoint: dummy_tts, supported_languages: [en] }
""",
    )
    s = RealtimeSettings.from_config(tmp_path)
    assert s.vad_silence_ms == 2500
    assert s.barge_in_ms == 700
    assert s.deep_dive_read_timeout_s == 420.0
    assert s.predict_timeout_s == 45.0
    assert s.tts_stream_timeout_s == 180.0
    assert s.llm_turn_timeout_s == 50.0


def _py_files(*rel_dirs: str) -> list[Path]:
    out: list[Path] = []
    for rel in rel_dirs:
        out.extend((REPO / rel).rglob("*.py"))
    return [p for p in out if "__pycache__" not in p.parts]


def test_no_private_space_id_poke():
    """Nobody assigns GenieClient._space_id directly except the client itself.

    The card path must pass ``space_name=`` to the constructor so the client's own
    stale-space retry re-resolves the RIGHT space.
    """
    offenders: list[str] = []
    allowed = REPO / "backend/genie_voice/genie/client.py"
    pattern = re.compile(r"\._space_id\s*=")
    for path in _py_files("realtime_api", "api", "backend/genie_voice"):
        if path == allowed:
            continue
        if pattern.search(path.read_text(encoding="utf-8")):
            offenders.append(str(path.relative_to(REPO)))
    assert not offenders, f"private _space_id assignment outside client.py: {offenders}"


def test_no_mlflow_serving_path():
    """There is no mlflow-based DatabricksServing path (from_workspace is removed).

    Serving is SDK-only through serving_factory, so no code may define or call a
    ``from_workspace`` serving constructor.
    """
    pattern = re.compile(r"\bfrom_workspace\b")
    offenders = [
        str(p.relative_to(REPO))
        for p in _py_files("realtime_api", "api")
        if pattern.search(p.read_text(encoding="utf-8"))
    ]
    assert not offenders, f"mlflow from_workspace serving path resurfaced in: {offenders}"


def test_single_serving_construction_site():
    """DatabricksServing.from_sdk(...) is CALLED from exactly one place.

    The shared serving_factory is the single construction site so the WS loop and
    the deep-dive summarizer share one instance + one set of config knobs.
    """
    call = re.compile(r"DatabricksServing\.from_sdk\s*\(")
    callers = [
        str(p.relative_to(REPO))
        for p in _py_files("realtime_api", "api")
        if call.search(p.read_text(encoding="utf-8"))
    ]
    assert callers == ["realtime_api/serving_factory.py"], callers


def test_config_requires_deploy_identifying_values():
    """Missing deploy-identifying keys fail fast (no silent wrong-target default)."""
    from genie_voice.config.settings import ConfigError, _validate_required

    ok = {
        "databricks": {"catalog": "c", "schema": "s", "genie_space_name": "g"},
        "enrichment": {"model_endpoint": "m"},
        "lakebase": {"enabled": True, "instance": "i", "schema": "ls"},
        "card_issuer": {"enabled": True, "schema": "cs", "genie_space_name": "cg"},
    }
    _validate_required(ok)  # complete config: no raise

    # Drop the core catalog -> must raise, naming the missing key.
    bad = {**ok, "databricks": {"schema": "s", "genie_space_name": "g"}}
    with pytest.raises(ConfigError, match="databricks.catalog"):
        _validate_required(bad)

    # Disabled features don't require their target keys.
    _validate_required(
        {
            "databricks": {"catalog": "c", "schema": "s", "genie_space_name": "g"},
            "enrichment": {"model_endpoint": "m"},
            "lakebase": {"enabled": False},
            "card_issuer": {"enabled": False},
        }
    )


def test_deepgram_params_come_from_config():
    """Deepgram query params read model/flags from config options (no hardcoded nova-3)."""
    from genie_voice.providers.stt.deepgram import deepgram_query_params

    params = deepgram_query_params(
        {"model": "nova-42", "smart_format": False, "endpointing_ms": 25},
        language="es",
        streaming=True,
        sample_rate=24000,
    )
    assert params["model"] == "nova-42"
    assert params["smart_format"] == "false"
    assert params["endpointing"] == "25"
    assert params["sample_rate"] == "24000"
    # Prerecorded upload omits transport-shape params.
    rec = deepgram_query_params({"model": "nova-42"}, language="es", streaming=False)
    assert "sample_rate" not in rec and rec["model"] == "nova-42"


@pytest.mark.parametrize("rel", ["api/app/routers/card.py", "realtime_api/app.py"])
def test_language_payload_is_single_source(rel):
    """Both language endpoints go through the canonical language_payload helper."""
    text = (REPO / rel).read_text(encoding="utf-8")
    assert "language_payload" in text, f"{rel} should use the canonical language_payload"


def test_cdf_history_allows_empty_app_written_tables(monkeypatch) -> None:
    """Empty optional feeds and unchanged immutable rows do not block redeploy."""
    from genie_voice.lakebase.cdf import (
        _optional_tables,
        _required_tables,
        history_blockers,
    )
    from genie_voice.config import get_settings

    counts = {
        "call_facts": {"total_rows": 60, "fresh_rows": 60, "max_updated_at": "2026-09-19"},
        # Immutable utterances correctly produce no event on an unchanged rerun.
        "live_call_utterances": {"total_rows": 60, "fresh_rows": 0, "max_updated_at": "2026-09-18"},
        "billing_adjustments": {"total_rows": 0, "fresh_rows": 0, "max_updated_at": None},
    }
    sources = {
        "call_facts": 60,
        "live_call_utterances": 60,
        "billing_adjustments": 0,
    }
    missing, empty, stale = history_blockers(counts, sources, None)
    assert missing == []
    assert empty == []
    assert stale == []

    counts["call_facts"] = {"total_rows": 0, "fresh_rows": 0, "max_updated_at": None}
    missing, empty, stale = history_blockers(counts, sources, None)
    assert "call_facts" in empty
    assert stale == []

    monkeypatch.setenv("GENIE_CONFIG", str(REPO / "config/config.yaml"))
    get_settings.cache_clear()
    settings = get_settings()
    assert _required_tables(settings) == ["call_facts", "live_call_utterances"]
    assert _optional_tables(settings) == ["billing_adjustments"]
    assert "call_state" not in _required_tables(settings)
    assert "resolution_events" not in _optional_tables(settings)


# --- 8. Deployment readiness (Setup page + installer automation) ------------
def test_readiness_validate_config_flags_bad_values(monkeypatch):
    """validate_config catches unset/placeholder values and mistargeted FQNs."""
    from types import SimpleNamespace

    from genie_voice import readiness

    # A model-service FQN that lives outside the deployed catalog.schema is flagged.
    monkeypatch.setattr(
        readiness,
        "_app_owned_model_services",
        lambda s: [("realtime_voice.llm_endpoint", "other_cat.other_sch.model")],
    )
    good = SimpleNamespace(
        databricks_host="https://x.cloud.databricks.com",
        databricks=SimpleNamespace(catalog="cat", schema_name="sch", sql_warehouse_id="wh1"),
        lakebase=SimpleNamespace(enabled=True, instance="lb"),
    )
    problems = readiness.validate_config(good)
    assert any("realtime_voice.llm_endpoint" in p for p in problems)

    monkeypatch.setattr(readiness, "_app_owned_model_services", lambda s: [])
    bad = SimpleNamespace(
        databricks_host="<your-workspace>",
        databricks=SimpleNamespace(catalog="", schema_name="sch", sql_warehouse_id=""),
        lakebase=SimpleNamespace(enabled=False, instance=""),
    )
    problems = readiness.validate_config(bad)
    assert "databricks.host" in problems
    assert "databricks.catalog" in problems
    assert "databricks.sql_warehouse_id" in problems


def test_readiness_serialization_shape():
    from genie_voice.readiness import FAIL, OK, Check, Fix, Readiness

    plain = Check(id="a", title="A", category="scripted", status=OK, detail="d")
    assert "fix" not in plain.to_dict()

    linked = Check(
        id="b", title="B", category="manual", status=FAIL, detail="d",
        fix=Fix(kind="link", label="Open", href="https://h"),
    )
    assert linked.to_dict()["fix"]["href"] == "https://h"

    payload = Readiness(ready=False, checks=[plain, linked]).to_dict()
    assert payload["ready"] is False
    assert payload["summary"] == {"ok": 1, "warn": 0, "fail": 1, "total": 2}


def test_readiness_covers_full_customer_install_chain():
    from genie_voice import readiness

    quick = {fn.__name__ for fn in readiness._CHECKS}
    full = {fn.__name__ for fn in readiness._FULL_CHECKS}
    assert {
        "check_uc_storage",
        "check_uc_data",
        "check_pipeline_job",
        "check_model_registration",
        "check_model_endpoints",
        "check_lakebase_cdf",
        "check_reference_cache",
        "check_gateway_services",
        "check_gateway_policies",
        "check_obo",
    } <= quick
    assert {"check_voice_contract", "check_viewer_genie", "check_agent_mode"} <= full


def test_readiness_names_exact_objects_and_setup_has_two_sections(monkeypatch):
    from genie_voice import readiness
    from genie_voice.config import get_settings

    monkeypatch.setenv("GENIE_CONFIG", str(REPO / "config/config.yaml"))
    get_settings.cache_clear()
    settings = get_settings()

    model_objects = readiness._readiness_objects(settings, "model_registration")
    assert "Hugging Face model: Qwen/Qwen3-ASR-1.7B" in model_objects
    assert (
        "UC model: partner_demo_catalog.genie_voice_contact_center."
        "realtime_voice_stt_qwen3_asr_1_7b (alias: candidate)"
    ) in model_objects
    cdf_objects = readiness._readiness_objects(settings, "lakebase_cdf")
    assert any("lb_call_facts_history" in item for item in cdf_objects)
    assert any("billing_adjustments" in item for item in cdf_objects)
    required, actions = readiness._manual_guidance(settings, "gateway_policies")
    assert any("interactive_model_safety" in item for item in required)
    assert any("Serving → AI Gateway" in item for item in actions)
    required, actions = readiness._manual_guidance(settings, "agent_mode")
    assert any("Agent Mode APIs for Genie Agents" in item for item in required)
    assert any("Settings → Previews" in item for item in actions)

    setup = (REPO / "frontend/src/components/SetupPage.tsx").read_text(encoding="utf-8")
    assert "1. Automated deploy checks" in setup
    assert "2. Manual checks needed" in setup
    assert "Exact objects checked" in setup
    assert 'role="progressbar"' in setup
    assert "setAutomatedOpen(automatedIssues > 0)" in setup
    assert "setManualOpen(manualIssues > 0)" in setup
    readiness_source = (REPO / "backend/genie_voice/readiness.py").read_text(
        encoding="utf-8"
    )
    assert "effective_user_api_scopes" in readiness_source
    assert 'missing = {"genie", "sql"} - effective' in readiness_source

    gateway = (REPO / "infra/apps/provision_ai_gateway.py").read_text(encoding="utf-8")
    policy_block = gateway.split("if policy_errors:", 1)[1]
    assert "_log(" in policy_block
    assert "raise SystemExit(" not in policy_block


def test_realtime_candidates_name_hugging_face_models(monkeypatch):
    from genie_voice import readiness

    monkeypatch.setenv("GENIE_CONFIG", str(REPO / "config/config.yaml"))
    candidates = {name: cfg for name, cfg in readiness._voice_candidates()}
    assert candidates["stt:qwen3_asr_1_7b_multilingual"]["base_model"] == "Qwen/Qwen3-ASR-1.7B"
    assert candidates["tts:voxcpm2_multilingual"]["base_model"] == "openbmb/VoxCPM2"
    assert all(cfg.get("registered_model") and cfg.get("endpoint") for cfg in candidates.values())


def test_readiness_router_wired_and_reads_obo():
    main = (REPO / "api/app/main.py").read_text(encoding="utf-8")
    assert "readiness.router" in main
    router = (REPO / "api/app/routers/readiness.py").read_text(encoding="utf-8")
    assert "x-forwarded-access-token" in router


def test_genie_obo_resolution_falls_back_to_app_sp(monkeypatch):
    """OBO may invoke a known space even when it cannot enumerate spaces."""
    from types import SimpleNamespace

    import genie_voice.genie.client as module

    viewer_client = object()
    app_client = object()
    client = module.GenieClient(
        SimpleNamespace(databricks=SimpleNamespace(genie_space_name="Voice Space"))
    )
    monkeypatch.setattr(
        client,
        "_workspace_client",
        lambda *, access_token=None: viewer_client if access_token else app_client,
    )
    monkeypatch.setattr(
        module,
        "find_space_ids",
        lambda workspace, name: [] if workspace is viewer_client else ["space-1"],
    )
    client._space_id = None

    assert client._resolve_space_id(access_token="obo-token") == "space-1"
    assert 'auth_type="pat"' in (REPO / "backend/genie_voice/genie/client.py").read_text(
        encoding="utf-8"
    )


def test_deploy_ip_allow_list_is_additive_and_refuses_block_lists():
    from types import SimpleNamespace

    sys.path.insert(0, str(REPO / "infra" / "apps"))
    import ensure_deploy_ip as helper  # noqa: E402

    ip = "182.19.249.101"
    allow = SimpleNamespace(
        label="office",
        list_type="ALLOW",
        enabled=True,
        ip_addresses=["10.0.0.0/8"],
        list_id="allow-1",
    )
    owned = SimpleNamespace(
        label="genie-voice-deploy",
        list_type="ALLOW",
        enabled=True,
        ip_addresses=["1.2.3.4/32"],
        list_id="owned-1",
    )
    deny = SimpleNamespace(
        label="bad",
        list_type="BLOCK",
        enabled=True,
        ip_addresses=[f"{ip}/32"],
        list_id="deny-1",
    )
    assert helper.cidr_covers("182.19.249.0/24", ip)
    assert not helper.already_allowed([allow], ip)
    assert helper.already_allowed([allow, SimpleNamespace(
        label="office", list_type="ALLOW", enabled=True, ip_addresses=[f"{ip}/32"]
    )], ip)
    assert helper.find_owned_allow_list([allow, owned]) is owned
    assert helper.merged_addresses(owned.ip_addresses, ip) == ["1.2.3.4/32", f"{ip}/32"]
    assert helper.is_blocked_by_deny([deny], ip) is deny
    assert helper.is_blocked_by_deny([allow], ip) is None


def test_deploy_orders_gateway_before_gpu_and_grants_viewers():
    sh = (REPO / "deploy_app.sh").read_text(encoding="utf-8")
    assert "infra/apps/ensure_deploy_ip.py" in sh
    # AI Gateway provision/probe must run before the (billable) GPU model job.
    assert sh.index("provision_ai_gateway.py") < sh.index("submit_realtime_voice_jobs.py")
    # Endpoint AI Gateway is reconciled after the voice endpoints exist.
    assert sh.index("submit_realtime_voice_jobs.py") < sh.index(
        "provision_voice_endpoint_gateway.py"
    )
    # Viewer Genie CAN_RUN is granted from the installer, not left manual.
    assert "--run-users" in sh
    # Group viewers receive both App CAN_USE and Genie CAN_RUN.
    assert 'grant_app_can_use group_name "$_group"' in sh
    # Setup can prove the endpoint's UC registered-model alias and inspect the job.
    assert "--registered-models" in sh
    grants = (REPO / "infra/apps/grant_app_sp.py").read_text(encoding="utf-8")
    assert "grant_pipeline_job(settings, sp)" in grants
    assert "GRANT EXECUTE ON FUNCTION" in grants
    # The installer prints the readiness checklist from the deployed app.
    assert "/readiness?full=false" in sh
    # The standalone Story Deck ships with the app and is checked after deploy.
    assert '--include "story_deck/**"' in sh
    assert 'base + "/story/"' in sh
    main = (REPO / "api/app/main.py").read_text(encoding="utf-8")
    home = (REPO / "frontend/src/components/HomePage.tsx").read_text(encoding="utf-8")
    assert "_mount_story_deck(app)" in main
    assert 'app.mount("/story"' in main
    assert 'href="/story/"' in home
    assert "Story Deck" in home
    # UI-only gates must not prevent the app/Setup page from being deployed.
    assert 'if ! DATABRICKS_CONFIG_PROFILE="$DATABRICKS_PROFILE"' in sh
    assert "Continuing so the app and Setup page are deployed" in sh
