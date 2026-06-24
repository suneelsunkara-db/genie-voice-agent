"""Configuration loader.

Loads `config/config.yaml` (committed template), deep-merges gitignored
`config/config.local.yaml` when present (local wins — use a complete file for
local development), then `.env` secrets and `GENIE_*` env overrides.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field


def _repo_root() -> Path:
    # backend/genie_voice/config/settings.py -> repo root is 4 levels up.
    return Path(__file__).resolve().parents[3]


def _config_path() -> Path:
    override = os.environ.get("GENIE_CONFIG")
    if override:
        return Path(override)
    return _repo_root() / "config" / "config.yaml"


def _local_config_path() -> Path:
    return _repo_root() / "config" / "config.local.yaml"


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge overlay into base (overlay wins)."""
    out = dict(base)
    for key, value in overlay.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


# --------------------------------------------------------------------------- #
# Config models (non-secret)
# --------------------------------------------------------------------------- #
class DatabricksConfig(BaseModel):
    host: str = ""
    # default = SDK unified-auth credential chain (OAuth U2M from `databricks
    #           auth login`, env vars, or a profile). No secrets in .env.
    # pat     = personal access token (DATABRICKS_TOKEN).
    # oauth   = service principal M2M (DATABRICKS_CLIENT_ID/SECRET).
    auth_type: Literal["default", "pat", "oauth"] = "default"
    # Optional named profile from ~/.databrickscfg (used with auth_type=default).
    profile: str = ""
    catalog: str = "genie_voice"
    schema_name: str = Field("contact_center", alias="schema")
    # When false we never CREATE CATALOG (use an existing one we have rights to).
    create_catalog: bool = False
    sql_warehouse_id: str = ""
    # Genie space is created/resolved dynamically BY NAME (never a hardcoded id).
    genie_space_name: str = "Genie Voice - Contact Center"
    # The identity the app runs as (used for GRANTs / Lakebase Postgres role).
    run_as: str = ""

    model_config = {"populate_by_name": True}


class VolumeConfig(BaseModel):
    batch_name: str = "raw_batch_data"
    streaming_name: str = "raw_streaming_data"
    # Reference/customer/billing source files land in the batch Volume.
    reference_path: str = "/Volumes/{catalog}/{schema}/{batch_volume}/reference"
    # Voice/call events and artifacts land in the streaming Volume.
    raw_stt_path: str = "/Volumes/{catalog}/{schema}/{streaming_volume}/call_streaming_data/raw_stt"
    call_facts_path: str = "/Volumes/{catalog}/{schema}/{streaming_volume}/call_streaming_data/call_facts"
    audio_path: str = "/Volumes/{catalog}/{schema}/{streaming_volume}/call_streaming_data/audio"
    transcript_path: str = "/Volumes/{catalog}/{schema}/{streaming_volume}/call_streaming_data/transcripts"
    # Auto Loader checkpoints + inferred-schema location. MUST be OUTSIDE the
    # ingest input path (raw_stt) so the stream never tries to read its own state.
    checkpoint_path: str = "/Volumes/{catalog}/{schema}/{streaming_volume}/_pipeline_state"


class MedallionConfig(BaseModel):
    gold_call_insights: str = "gold_call_insights"


class LakebaseConfig(BaseModel):
    model_config = {"populate_by_name": True}

    enabled: bool = False                    # use real Lakebase (else in-memory fallback)
    instance: str = "genie_voice_lakebase"   # Lakebase database instance name
    database: str = "databricks_postgres"    # Postgres database inside the instance
    port: int = 5432                         # Postgres port for the compute endpoint
    schema_name: str = Field("genie_voice_contact_center", alias="schema")  # Postgres serving schema
    capacity: str = "CU_1"                   # autoscaling capacity unit
    serving_table: str = "call_state"
    live_utterances_table: str = "live_call_utterances"
    cdf_required: bool = True
    cdf_history_prefix: str = "lb_"
    cdf_history_suffix: str = "_history"
    cdf_wait_timeout_seconds: int = 600
    cdf_poll_seconds: int = 15
    # Seeded non-empty Lakebase tables whose CDF history tables must exist before
    # UC analytics/Genie runs. Live tables can be empty and are not required here.
    cdf_required_tables: list[str] = Field(default_factory=list)
    # Lakebase-native serving/source tables loaded under `schema` using primary
    # names. No duplicate *_serving managed-sync tables are created.
    sync_tables: list[str] = Field(default_factory=list)


class ProviderSlot(BaseModel):
    # logical_name -> "module.path:ClassName" (resolved dynamically; the core
    # never imports a vendor adapter directly).
    adapters: dict[str, str] = Field(default_factory=dict)
    active: str
    options: dict[str, dict[str, Any]] = Field(default_factory=dict)

    def active_options(self) -> dict[str, Any]:
        return self.options.get(self.active, {})


class ProvidersConfig(BaseModel):
    stt: ProviderSlot
    tts: ProviderSlot


class MockConfig(BaseModel):
    interim_words_step: int = 2
    realtime_pacing: bool = True
    inject_low_confidence: bool = True
    channels: dict[str, int] = Field(default_factory=lambda: {"agent": 0, "customer": 1})


class ApiConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173"])


class PipelineConfig(BaseModel):
    autoloader_format: str = "cloudFiles"
    source_format: str = "json"
    # Lakebase-first orchestration: seed serving tables, refresh UC analytics,
    # then reconcile Genie.
    # Legacy Lakeflow pipeline name retained only so reset can delete old deploys.
    analytics_pipeline_name: str = "Genie Voice - UC Analytics Pipeline"
    orchestration_job_name: str = "Genie Voice - Lakebase First Orchestration"
    # Databricks WORKSPACE folder the deployer copies the job source (wheel) +
    # config into. Empty -> "/Users/<run_as-or-current-user>/genie_voice_pipeline".
    # The serverless jobs install the wheel from this workspace path.
    workspace_dir: str = ""
    # Serverless environment version (Python + base libs). "client" is deprecated.
    environment_version: str = "3"


class EnrichmentConfig(BaseModel):
    """How conversation insights (intent/sentiment/NBA/summary) are produced.

    The Databricks Foundation Model API is the SOLE engine. Live (agent-assist)
    enrichment calls the serving endpoint per utterance; batch gold uses the
    `ai_query` SQL function with structured (json_schema) output - same contract.
    There is no heuristic/rules fallback: when the FM is unavailable (offline or
    an endpoint outage) the insight is reported as unavailable, not faked.
    """
    # `model_endpoint` would otherwise collide with pydantic's protected `model_`
    # namespace; we use it deliberately, so opt out of the namespace guard.
    model_config = {"protected_namespaces": ()}

    # A Databricks model serving endpoint (pay-per-token FM, provisioned-throughput,
    # external, or custom). Stronger models improve accuracy; smaller/faster ones
    # lower latency + cost on the per-utterance live path.
    model_endpoint: str = "databricks-claude-opus-4-8"
    max_tokens: int = 512
    # Optional: some reasoning models (Claude Opus 4.x) reject `temperature`. Set
    # to null to omit it; the engine also retries without it if rejected.
    temperature: float | None = None


class DatagenConfig(BaseModel):
    seed: int = 42
    num_agents: int = 6
    num_customers: int = 40
    num_calls: int = 60
    months_history: int = 4


class Secrets(BaseModel):
    """Vendor API keys for live STT/TTS.

    Loaded from config `secrets:` (typically config.local.yaml) with environment
    variables overriding when set (DEEPGRAM_API_KEY, etc.).
    """
    deepgram_api_key: str = ""
    elevenlabs_api_key: str = ""


class Settings(BaseModel):
    # The ONLY deployment switch. It selects who PRODUCES the data on this host:
    #   local = synthetic producer (datagen) generates vendor-shaped payloads.
    #   live  = real Deepgram/ElevenLabs capture produces payloads.
    # The Databricks ingestion (streaming voice job + batch job) is identical for
    # both - only the source of the landed files differs.
    deployment: Literal["local", "live"] = "local"
    databricks: DatabricksConfig
    volume: VolumeConfig
    medallion: MedallionConfig
    lakebase: LakebaseConfig
    providers: ProvidersConfig
    mock: MockConfig
    api: ApiConfig
    pipeline: PipelineConfig
    enrichment: EnrichmentConfig
    datagen: DatagenConfig
    secrets: Secrets

    # ---- deployment helpers ----
    @property
    def is_live(self) -> bool:
        """True when real vendor capture (Deepgram/ElevenLabs) should be used."""
        return self.deployment == "live"

    @property
    def mode(self) -> str:
        """Back-compat transport label derived from `deployment`: the providers,
        health, and UI speak 'mock' | 'live'. `local` -> mock, `live` -> live."""
        return "live" if self.deployment == "live" else "mock"

    # ---- convenience resolvers (keep templating in one place) ----
    def resolve_volume_path(self, template: str) -> str:
        return template.format(
            catalog=self.databricks.catalog,
            schema=self.databricks.schema_name,
            batch_volume=self.volume.batch_name,
            streaming_volume=self.volume.streaming_name,
        )

    @property
    def raw_stt_path(self) -> str:
        return self.resolve_volume_path(self.volume.raw_stt_path)

    @property
    def call_facts_path(self) -> str:
        return self.resolve_volume_path(self.volume.call_facts_path)

    @property
    def reference_path(self) -> str:
        return self.resolve_volume_path(self.volume.reference_path)

    @property
    def checkpoint_path(self) -> str:
        """Auto Loader checkpoint/schema root (outside the ingest input path)."""
        return self.resolve_volume_path(self.volume.checkpoint_path)

    def reference_table_path(self, table: str) -> str:
        """Volume sub-dir where the producer lands a reference table's files."""
        return f"{self.reference_path}/{table}"

    def fqtn(self, table: str) -> str:
        """Fully qualified table name catalog.schema.table."""
        return f"{self.databricks.catalog}.{self.databricks.schema_name}.{table}"

    def lakebase_synced_table_name(self, source_table: str) -> str:
        """Legacy managed-sync target name used only for cleanup."""
        return f"{source_table}_serving"

    def lakebase_table_name(self, table: str) -> str:
        """Two-part Postgres table name for Lakebase-native serving tables."""
        return f"{self.lakebase.schema_name}.{table}"

    def lakebase_synced_fqtn(self, source_table: str) -> str:
        """Fully qualified UC synced-table object name for Lakebase managed sync."""
        return self.fqtn(self.lakebase_synced_table_name(source_table))

    @property
    def databricks_host(self) -> str:
        # env wins over yaml for host.
        return os.environ.get("DATABRICKS_HOST") or self.databricks.host


def _load_yaml() -> dict[str, Any]:
    """Load config.yaml, then overlay config.local.yaml if it exists (local wins)."""
    if os.environ.get("GENIE_CONFIG"):
        path = _config_path()
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with path.open() as fh:
            return yaml.safe_load(fh) or {}

    base_path = _config_path()
    if not base_path.exists():
        raise FileNotFoundError(f"Config file not found: {base_path}")
    with base_path.open() as fh:
        raw = yaml.safe_load(fh) or {}

    local_path = _local_config_path()
    if local_path.exists():
        with local_path.open() as fh:
            local = yaml.safe_load(fh) or {}
        if local:
            raw = _deep_merge(raw, local)
    elif _looks_like_template_config(raw):
        import warnings

        warnings.warn(
            "config/config.local.yaml is missing. Copy config/config.local.yaml.example "
            "to config/config.local.yaml and set your workspace values for local dev.",
            stacklevel=2,
        )
    return raw


def _looks_like_template_config(raw: dict[str, Any]) -> bool:
    host = str((raw.get("databricks") or {}).get("host") or "")
    return "<your-workspace>" in host


def _apply_env_overrides(raw: dict[str, Any]) -> None:
    """Override any config value via GENIE_<SECTION>__<KEY>=value (YAML-parsed).

    Example: GENIE_LAKEBASE__ENABLED=false, GENIE_DATABRICKS__CATALOG=my_cat.
    """
    for key, val in os.environ.items():
        if not key.startswith("GENIE_") or "__" not in key:
            continue
        path = key[len("GENIE_"):].lower().split("__")
        node = raw
        ok = True
        for p in path[:-1]:
            nxt = node.setdefault(p, {})
            if not isinstance(nxt, dict):
                ok = False
                break
            node = nxt
        if ok:
            node[path[-1]] = yaml.safe_load(val)


def _load_secrets(yaml_secrets: dict[str, Any] | None = None) -> Secrets:
    sec = yaml_secrets or {}

    def _pick(env_key: str, yaml_key: str, default: str = "") -> str:
        return os.environ.get(env_key) or str(sec.get(yaml_key) or default)

    return Secrets(
        deepgram_api_key=_pick("DEEPGRAM_API_KEY", "deepgram_api_key"),
        elevenlabs_api_key=_pick("ELEVENLABS_API_KEY", "elevenlabs_api_key"),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    # Load .env from repo root if present.
    env_file = _repo_root() / ".env"
    if env_file.exists():
        load_dotenv(env_file)
    else:
        load_dotenv()  # fall back to default search

    raw = _load_yaml()

    # Simple top-level override: GENIE_DEPLOYMENT=local|live.
    if os.environ.get("GENIE_DEPLOYMENT"):
        raw["deployment"] = os.environ["GENIE_DEPLOYMENT"]
    _apply_env_overrides(raw)

    yaml_secrets = raw.pop("secrets", None) or {}

    return Settings(
        deployment=raw.get("deployment", "local"),
        databricks=DatabricksConfig(**raw.get("databricks", {})),
        volume=VolumeConfig(**raw.get("volume", {})),
        medallion=MedallionConfig(**raw.get("medallion", {})),
        lakebase=LakebaseConfig(**raw.get("lakebase", {})),
        providers=ProvidersConfig(**raw["providers"]),
        mock=MockConfig(**raw.get("mock", {})),
        api=ApiConfig(**raw.get("api", {})),
        pipeline=PipelineConfig(**raw.get("pipeline", {})),
        enrichment=EnrichmentConfig(**raw.get("enrichment", {})),
        datagen=DatagenConfig(**raw.get("datagen", {})),
        secrets=_load_secrets(yaml_secrets),
    )
