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
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _write(dir_: Path, text: str) -> None:
    (dir_ / "config.yaml").write_text(text, encoding="utf-8")


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
