"""Deployment readiness checks shared by the CLI and the in-app Setup page.

This module answers one question: *is a fresh install actually usable yet?* It
reuses the existing probes (config validation, AI Gateway conformance, Lakebase
CDF status, serving-endpoint state, Genie space existence) rather than inventing
parallel logic, and it classifies every check so the caller can tell apart:

  - ``prereq``   customer-supplied workspace prerequisites (catalog, warehouse,
                 GPU quota) — the installer cannot create these.
  - ``scripted`` things ``deploy_app.sh`` fully owns (data, models, SP grants);
                 a failure here means the installer did not finish or drifted.
  - ``manual``   genuinely UI-only Databricks actions with no public write API
                 (Lakebase CDF start, AI Gateway policy attach, Apps User
                 Authorization, Agent Mode preview) — the page links out and
                 re-checks.

Each check is best-effort and self-contained: one failing probe never masks the
others. Network calls are fine here — this runs on demand (a Setup page poll or
the end of a deploy), not on a hot path.

The only *fix* actions exposed are cheap and idempotent AND performable as the
app's own identity (e.g. re-snapshotting the Lakebase reference cache). Grants
must run as the object owner (the deployer), and GPU model deployment is long and
billable, so both stay in ``deploy_app.sh`` — never triggered from a page load.
"""
from __future__ import annotations

import base64
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml

from genie_voice.config import Settings, get_settings

OK = "ok"
WARN = "warn"
FAIL = "fail"

CATEGORY_PREREQ = "prereq"
CATEGORY_SCRIPTED = "scripted"
CATEGORY_MANUAL = "manual"


@dataclass
class Fix:
    """A remediation hint attached to a check.

    ``kind`` is one of:
      - ``auto``   the Setup page may call ``POST /readiness/fix`` with ``action``.
      - ``link``   open ``href`` in the workspace UI and re-check.
      - ``cli``    re-run ``deploy_app.sh`` (owner-only / expensive action).
    """

    kind: str
    label: str
    action: str | None = None
    href: str | None = None


@dataclass
class Check:
    id: str
    title: str
    category: str
    status: str
    detail: str
    fix: Fix | None = None
    explanation: str | None = None
    resolution_steps: list[str] = field(default_factory=list)
    technical_detail: str | None = None
    objects: list[str] = field(default_factory=list)
    required_configuration: list[str] = field(default_factory=list)
    workspace_actions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.fix is None:
            data.pop("fix", None)
        if self.explanation is None:
            data.pop("explanation", None)
        if not self.resolution_steps:
            data.pop("resolution_steps", None)
        if self.technical_detail is None:
            data.pop("technical_detail", None)
        if not self.objects:
            data.pop("objects", None)
        if not self.required_configuration:
            data.pop("required_configuration", None)
        if not self.workspace_actions:
            data.pop("workspace_actions", None)
        return data


@dataclass
class Readiness:
    ready: bool
    checks: list[Check] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "summary": {
                "ok": sum(1 for c in self.checks if c.status == OK),
                "warn": sum(1 for c in self.checks if c.status == WARN),
                "fail": sum(1 for c in self.checks if c.status == FAIL),
                "total": len(self.checks),
            },
            "checks": [c.to_dict() for c in self.checks],
        }


def _host_url(settings: Settings, path: str) -> str:
    host = (settings.databricks_host or "").rstrip("/")
    return f"{host}{path}" if host else path


def _deployment_config() -> dict[str, Any]:
    explicit = (os.environ.get("GENIE_CONFIG") or "").strip()
    path = Path(explicit) if explicit else Path(__file__).resolve().parents[2] / "config" / "config.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _voice_candidates() -> list[tuple[str, dict[str, Any]]]:
    rv = _deployment_config().get("realtime_voice") or {}
    out: list[tuple[str, dict[str, Any]]] = []
    for modality, group in (("stt", "stt_candidates"), ("tts", "tts_candidates")):
        for candidate_id, candidate in (rv.get(group) or {}).items():
            if isinstance(candidate, dict):
                out.append((f"{modality}:{candidate_id}", candidate))
    return out


def _error_text(exc: Exception) -> str:
    text = f"{type(exc).__name__}: {exc}".strip()
    return text[:1500]


# --------------------------------------------------------------------------- #
# Individual checks. Each takes (settings, obo_token) and returns a Check.
# --------------------------------------------------------------------------- #
def validate_config(settings: Settings) -> list[str]:
    """Return a list of human-readable config problems (empty when clean).

    Shared with ``deploy_app.sh`` intent: required deployment values must be set,
    not placeholders, and the app-owned model-service FQNs must live in the
    deployed catalog/schema.
    """
    db = settings.databricks
    problems: list[str] = []
    required = {
        "databricks.host": settings.databricks_host,
        "databricks.catalog": db.catalog,
        "databricks.schema": db.schema_name,
        "databricks.sql_warehouse_id": db.sql_warehouse_id,
        "lakebase.instance": settings.lakebase.instance if settings.lakebase.enabled else "n/a",
    }
    for name, value in required.items():
        text = str(value or "").strip()
        if not text or "<" in text or "your-" in text.lower():
            problems.append(name)

    prefix = f"{db.catalog}.{db.schema_name}."
    for label, fqn in _app_owned_model_services(settings):
        if fqn and fqn.count(".") == 2 and not fqn.startswith(prefix):
            problems.append(f"{label} must be under {prefix}* (got {fqn})")
    return problems


def _realtime_settings():
    """Best-effort RealtimeSettings (endpoints + candidates). None if unavailable."""
    try:
        from realtime_api.config import RealtimeSettings

        return RealtimeSettings.resolve()
    except Exception:  # noqa: BLE001
        return None


def _app_owned_model_services(settings: Settings) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    rt = _realtime_settings()
    if rt is not None:
        for key in ("llm_endpoint", "i18n_endpoint", "conversion_endpoint"):
            out.append((f"realtime_voice.{key}", str(getattr(rt, key, "") or "")))
    try:
        out.append(("enrichment.model_endpoint", str(settings.enrichment.model_endpoint or "")))
    except Exception:  # noqa: BLE001
        pass
    return out


def _readiness_objects(settings: Settings, check_id: str) -> list[str]:
    """Configured objects inspected by a readiness check, with exact names."""
    catalog = settings.databricks.catalog
    schema = settings.databricks.schema_name
    telco = f"{catalog}.{schema}"
    card_enabled = bool(
        getattr(settings, "card_issuer", None) and settings.card_issuer.enabled
    )
    card_schema = settings.card_issuer.schema_name if card_enabled else ""
    card = f"{catalog}.{card_schema}" if card_enabled else ""
    app_name = os.environ.get("DATABRICKS_APP_NAME", "genie-voice-agent")
    app_sp = os.environ.get("DATABRICKS_CLIENT_ID", "resolved when the App is created")
    rt = _realtime_settings()
    endpoints = [
        endpoint
        for endpoint in (
            getattr(rt, "stt_endpoint", "") if rt else "",
            getattr(rt, "tts_endpoint", "") if rt else "",
        )
        if endpoint
    ]
    candidates = _voice_candidates()

    if check_id == "config":
        return [
            "Config file: config/config.yaml",
            f"Workspace host: {settings.databricks_host}",
            f"Unity Catalog namespace: {telco}",
            f"Lakebase project: {settings.lakebase.instance}",
            f"App service principal: {app_sp}",
        ]
    if check_id == "warehouse":
        return [
            f"SQL warehouse ID: {settings.databricks.sql_warehouse_id}",
            f"App resource grant: {app_sp} → CAN_USE",
        ]
    if check_id == "uc_storage":
        objects = [
            f"UC volume: {telco}.{settings.volume.batch_name}",
            f"UC volume: {telco}.{settings.volume.streaming_name}",
        ]
        if card_enabled:
            objects.append(f"UC volume: {card}.{settings.card_issuer.batch_volume}")
        objects.append(
            f"App SP grants on volumes: {app_sp} → USE CATALOG, USE SCHEMA, READ VOLUME"
        )
        return objects
    if check_id == "uc_data":
        objects = [
            f"UC table: {telco}.customers",
            f"UC table: {telco}.invoices",
            f"UC table: {telco}.payments",
        ]
        if card_enabled:
            objects.extend(
                [f"UC table: {card}.cardholders", f"UC table: {card}.statements"]
            )
        objects.append(
            f"App SP grants on schemas: {app_sp} → USE CATALOG, USE SCHEMA, SELECT"
        )
        return objects
    if check_id == "pipeline_job":
        return [
            f"Workflow job: {settings.pipeline.orchestration_job_name}",
            f"App SP job permission: {app_sp} → CAN_VIEW",
        ]
    if check_id == "model_registration":
        objects: list[str] = []
        for _, candidate in candidates:
            base = str(candidate.get("base_model") or "")
            registered = str(candidate.get("registered_model") or "")
            endpoint = str(candidate.get("endpoint") or "")
            if base:
                objects.append(f"Hugging Face model: {base}")
            if registered:
                objects.append(f"UC model: {telco}.{registered} (alias: candidate)")
                objects.append(
                    f"App SP UC model permission: {app_sp} → EXECUTE ON FUNCTION"
                )
            if endpoint:
                objects.append(f"Serving endpoint: {endpoint}")
        return objects
    if check_id in {"model_endpoints", "voice_contract"}:
        objects = [f"Serving endpoint: {endpoint}" for endpoint in endpoints]
        objects.append(f"App resource grants: {app_sp} → CAN_QUERY")
        return objects
    if check_id in {"genie_spaces", "viewer_genie"}:
        objects = [f"Genie space: {settings.databricks.genie_space_name}"]
        if card_enabled:
            objects.append(f"Genie space: {settings.card_issuer.genie_space_name}")
        if check_id == "genie_spaces":
            objects.append(f"App SP Genie permission: {app_sp} → CAN_RUN")
        else:
            objects.extend(
                [
                    f"Databricks App: {app_name} → viewer requires CAN_USE",
                    "Signed-in viewer → CAN_RUN on every Genie space above",
                ]
            )
        return objects
    if check_id == "lakebase_cdf":
        tables = [
            *settings.lakebase.cdf_required_tables,
            *settings.lakebase.cdf_optional_tables,
        ]
        objects = [
            (
                "Required Lakebase CDF source: "
                f"{settings.lakebase.instance}/{settings.lakebase.database}."
                f"{settings.lakebase.schema_name}.{table}"
            )
            for table in settings.lakebase.cdf_required_tables
        ]
        objects.extend(
            (
                "Optional Lakebase CDF source: "
                f"{settings.lakebase.instance}/{settings.lakebase.database}."
                f"{settings.lakebase.schema_name}.{table}"
            )
            for table in settings.lakebase.cdf_optional_tables
        )
        objects.extend(
            (
                f"UC history table: {telco}."
                f"{settings.lakebase.cdf_history_prefix}{table}"
                f"{settings.lakebase.cdf_history_suffix}"
            )
            for table in tables
        )
        return objects
    if check_id == "reference_cache":
        objects = [
            f"Lakebase table: {settings.lakebase.database}.{settings.lakebase.schema_name}.customers",
            f"Lakebase table: {settings.lakebase.database}.{settings.lakebase.schema_name}.invoices",
            f"Lakebase table: {settings.lakebase.database}.{settings.lakebase.schema_name}.payments",
        ]
        if card_enabled:
            objects.extend(
                [
                    f"Lakebase table: {settings.lakebase.database}.{card_schema}.cardholders",
                    f"Lakebase table: {settings.lakebase.database}.{card_schema}.statements",
                ]
            )
        objects.extend(
            [
                f"App SP entitlement: {app_sp} → workspace-access",
                (
                    f"Lakebase role grants: {app_sp} → CONNECT database, "
                    "USAGE schema, SELECT/INSERT/UPDATE/DELETE tables"
                ),
            ]
        )
        return objects
    if check_id in {"gateway_services", "gateway_policies"}:
        objects = []
        for _, spec in settings.ai_gateway.model_services.items():
            objects.append(
                f"AI Gateway service: {spec.service} → {spec.destination}"
            )
            if check_id == "gateway_services":
                objects.append(
                    f"App SP model-service permission: {app_sp} → EXECUTE"
                )
            if check_id == "gateway_policies":
                objects.append(
                    f"AI Gateway policy bundle: {spec.policy_bundle} on {spec.service}"
                )
        return objects
    if check_id == "obo":
        return [
            f"Databricks App: {app_name}",
            "User Authorization scope: genie",
            "User Authorization scope: sql",
        ]
    if check_id == "agent_mode" and card_enabled:
        return [
            f"Genie space: {settings.card_issuer.genie_space_name}",
            "Workspace preview: Genie Agent Mode",
        ]
    return []


def _manual_guidance(settings: Settings, check_id: str) -> tuple[list[str], list[str]]:
    """Required state and exact UI actions for checks deploy_app.sh cannot finish."""
    catalog = settings.databricks.catalog
    schema = settings.databricks.schema_name
    app_name = os.environ.get("DATABRICKS_APP_NAME", "genie-voice-agent")
    card = getattr(settings, "card_issuer", None)
    if check_id == "lakebase_cdf":
        required = [
            (
                f"{table}: STREAMING from "
                f"{settings.lakebase.database}.{settings.lakebase.schema_name}.{table} "
                f"to {catalog}.{schema}.{settings.lakebase.cdf_history_prefix}"
                f"{table}{settings.lakebase.cdf_history_suffix}"
            )
            for table in settings.lakebase.cdf_required_tables
        ]
        required.extend(
            (
                f"{table}: optional audit feed to "
                f"{catalog}.{schema}.{settings.lakebase.cdf_history_prefix}"
                f"{table}{settings.lakebase.cdf_history_suffix}"
            )
            for table in settings.lakebase.cdf_optional_tables
        )
        return required, [
            (
                f"Open Lakebase → project '{settings.lakebase.instance}' → the branch "
                "shown in Current validation → Change Data Feed."
            ),
            (
                f"For each source above, set target catalog/schema to {catalog}.{schema}, "
                f"prefix '{settings.lakebase.cdf_history_prefix}', and suffix "
                f"'{settings.lakebase.cdf_history_suffix}'."
            ),
            "Start each feed with an initial snapshot; wait for STREAMING, then select Re-check.",
        ]
    if check_id == "gateway_policies":
        required = [
            f"{spec.service}: attach policy bundle '{spec.policy_bundle}'"
            for spec in settings.ai_gateway.model_services.values()
        ]
        return required, [
            "Open Serving → AI Gateway and select each model service listed above.",
            "Open Policies, attach the named bundle from config/guardrails.yaml, and save.",
            "Select Re-check; readiness sends live allow, deny, and PII-redaction probes.",
        ]
    if check_id == "obo":
        return [
            f"App '{app_name}': User Authorization enabled",
            "Requested/effective user scope: genie",
            "Requested/effective user scope: sql",
            "Each viewer has accepted the current consent prompt",
        ], [
            f"Open Compute → Apps → '{app_name}' → Authorization.",
            "Enable User Authorization and add scopes 'genie' and 'sql'.",
            "Restart/redeploy the App after a scope change; have each viewer revoke old consent and consent again.",
        ]
    if check_id == "viewer_genie":
        spaces = [settings.databricks.genie_space_name]
        if card is not None and card.enabled:
            spaces.append(card.genie_space_name)
        return [
            f"Every intended viewer/group: CAN_USE on App '{app_name}'",
            *[f"Every intended viewer/group: CAN_RUN on Genie space '{name}'" for name in spaces],
            "Users/groups already provisioned in the Databricks account (SCIM or JIT)",
        ], [
            (
                f"Open Compute → Apps → '{app_name}' → Permissions and grant CAN_USE, "
                "or rerun with APP_EXTERNAL_USERS / APP_EXTERNAL_GROUPS / APP_EXTERNAL_SPS."
            ),
            (
                "Open each Genie space → Share and grant CAN_RUN, or rerun with "
                "APP_EXTERNAL_USERS / APP_EXTERNAL_GROUPS."
            ),
            "Re-open Setup as the target viewer; this check asks a real question in every configured space.",
        ]
    if check_id == "agent_mode" and card is not None and card.enabled:
        return [
            "Workspace Preview: Agent Mode APIs for Genie Agents = Enabled",
            f"Viewer CAN_RUN on Genie space '{card.genie_space_name}'",
            "App User Authorization scope 'genie' consented",
        ], [
            "As a workspace admin, open Settings → Previews.",
            "Enable 'Agent Mode APIs for Genie Agents' for the workspace.",
            "Return to Setup and select Re-check; readiness runs a real governed Agent Mode investigation.",
        ]
    return [], []


def check_config(settings: Settings, obo_token: str | None) -> Check:
    problems = validate_config(settings)
    if problems:
        return Check(
            id="config",
            title="Deployment config",
            category=CATEGORY_SCRIPTED,
            status=FAIL,
            detail="Unset or mistargeted values: " + ", ".join(problems),
            fix=Fix(kind="cli", label="Edit config/config.yaml and redeploy"),
            resolution_steps=[
                "Open config/config.yaml and replace every placeholder with this customer's workspace resources.",
                "Keep all app-owned model-service FQNs under the configured catalog and schema.",
                "Re-run deploy_app.sh; hosted deployment intentionally ignores config.local.yaml.",
            ],
        )
    return Check(
        id="config",
        title="Deployment config",
        category=CATEGORY_SCRIPTED,
        status=OK,
        detail=f"catalog {settings.databricks.catalog}, schema {settings.databricks.schema_name}",
    )


def check_warehouse(settings: Settings, obo_token: str | None) -> Check:
    wh = settings.databricks.sql_warehouse_id
    if not wh:
        return Check(
            id="warehouse", title="SQL warehouse", category=CATEGORY_PREREQ,
            status=FAIL, detail="databricks.sql_warehouse_id is not set",
            fix=Fix(kind="cli", label="Set databricks.sql_warehouse_id and redeploy"),
        )
    try:
        from genie_voice.databricks.client import get_workspace_client

        client = get_workspace_client(settings)
        client.warehouses.get(wh)
        return Check(
            id="warehouse", title="SQL warehouse", category=CATEGORY_PREREQ,
            status=OK, detail=f"warehouse {wh} reachable",
        )
    except Exception as exc:  # noqa: BLE001
        return Check(
            id="warehouse", title="SQL warehouse", category=CATEGORY_PREREQ,
            status=FAIL, detail=f"warehouse {wh} not accessible: {exc}",
            fix=Fix(kind="link", label="Open SQL Warehouses", href=_host_url(settings, "/sql/warehouses")),
            technical_detail=_error_text(exc),
            resolution_steps=[
                "Confirm the warehouse ID in config/config.yaml belongs to this workspace.",
                "Start the warehouse and grant the deployer/app resource CAN_USE.",
                "Re-run deploy_app.sh so the warehouse is attached to the app.",
            ],
        )


def _sql_scalar(settings: Settings, statement: str) -> Any:
    from genie_voice.databricks.client import get_workspace_client

    response = get_workspace_client(settings).statement_execution.execute_statement(
        warehouse_id=settings.databricks.sql_warehouse_id,
        statement=statement,
        wait_timeout="30s",
    )
    rows = (response.result.data_array if response.result else None) or []
    return rows[0][0] if rows and rows[0] else None


def check_uc_data(settings: Settings, obo_token: str | None) -> Check:
    """Verify the minimum governed datasets needed by both product lanes."""
    catalog = settings.databricks.catalog.replace("`", "``")
    telco_schema = settings.databricks.schema_name.replace("`", "``")
    targets = [
        ("telco customers", f"`{catalog}`.`{telco_schema}`.`customers`"),
        ("telco invoices", f"`{catalog}`.`{telco_schema}`.`invoices`"),
        ("telco payments", f"`{catalog}`.`{telco_schema}`.`payments`"),
    ]
    if getattr(settings, "card_issuer", None) and settings.card_issuer.enabled:
        card_schema = settings.card_issuer.schema_name.replace("`", "``")
        targets.extend(
            [
                ("cardholders", f"`{catalog}`.`{card_schema}`.`cardholders`"),
                ("card statements", f"`{catalog}`.`{card_schema}`.`statements`"),
            ]
        )
    counts: list[str] = []
    failures: list[str] = []
    for label, table in targets:
        try:
            count = int(_sql_scalar(settings, f"SELECT count(*) FROM {table}") or 0)
            if count <= 0:
                failures.append(f"{label}: table is empty")
            else:
                counts.append(f"{label}={count}")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{label}: {_error_text(exc)}")
    if failures:
        return Check(
            id="uc_data", title="Unity Catalog reference data",
            category=CATEGORY_SCRIPTED, status=FAIL,
            detail="Required governed data is missing or empty.",
            technical_detail="; ".join(failures),
            resolution_steps=[
                "Confirm the configured catalog/schema and SQL warehouse are correct.",
                "Re-run deploy_app.sh with DEPLOY_DATA=1.",
                "Inspect the orchestration job's batch-reference-ingest task if tables remain missing.",
            ],
            fix=Fix(kind="cli", label="Re-run data deployment (DEPLOY_DATA=1)"),
        )
    return Check(
        id="uc_data", title="Unity Catalog reference data",
        category=CATEGORY_SCRIPTED, status=OK,
        detail="Required telco/card reference tables are populated.",
        explanation=", ".join(counts),
    )


def check_uc_storage(settings: Settings, obo_token: str | None) -> Check:
    """Verify the UC volumes required for landing, artifacts, and checkpoints."""
    from genie_voice.databricks.client import get_workspace_client

    client = get_workspace_client(settings)
    catalog = settings.databricks.catalog
    schema = settings.databricks.schema_name
    volumes = [
        f"{catalog}.{schema}.{settings.volume.batch_name}",
        f"{catalog}.{schema}.{settings.volume.streaming_name}",
    ]
    if getattr(settings, "card_issuer", None) and settings.card_issuer.enabled:
        volumes.append(
            f"{catalog}.{settings.card_issuer.schema_name}.{settings.card_issuer.batch_volume}"
        )
    failures: list[str] = []
    for fqn in volumes:
        try:
            client.api_client.do("GET", f"/api/2.1/unity-catalog/volumes/{fqn}")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{fqn}: {_error_text(exc)}")
    if failures:
        return Check(
            id="uc_storage", title="Unity Catalog landing volumes",
            category=CATEGORY_SCRIPTED, status=FAIL,
            detail="One or more required UC volumes are absent or invisible to the app.",
            technical_detail="; ".join(failures),
            resolution_steps=[
                "Confirm catalog/schema names and the two volume names in config/config.yaml.",
                "Re-run deploy_app.sh with DEPLOY_DATA=1 as a catalog/schema owner.",
                "If volumes exist, re-run grants so the app receives READ VOLUME.",
            ],
            fix=Fix(kind="cli", label="Re-run UC bootstrap and grants"),
        )
    return Check(
        id="uc_storage", title="Unity Catalog landing volumes",
        category=CATEGORY_SCRIPTED, status=OK,
        detail=f"{len(volumes)} required landing/artifact volume(s) are accessible.",
        explanation="; ".join(volumes),
    )


def check_pipeline_job(settings: Settings, obo_token: str | None) -> Check:
    """Confirm the orchestration definition exists and its latest run succeeded."""
    from genie_voice.databricks.client import get_workspace_client

    client = get_workspace_client(settings)
    name = settings.pipeline.orchestration_job_name
    try:
        job = next(iter(client.jobs.list(name=name, limit=2)), None)
        if job is None or job.job_id is None:
            raise RuntimeError("job not found")
        run = next(iter(client.jobs.list_runs(job_id=job.job_id, limit=1)), None)
        if run is None:
            raise RuntimeError("job exists but has never run")
        state = run.state
        life = getattr(getattr(state, "life_cycle_state", None), "value", None) or str(
            getattr(state, "life_cycle_state", "")
        )
        result = getattr(getattr(state, "result_state", None), "value", None) or str(
            getattr(state, "result_state", "")
        )
        if life != "TERMINATED" or result != "SUCCESS":
            raise RuntimeError(f"latest run lifecycle={life or '?'} result={result or '?'}")
    except Exception as exc:  # noqa: BLE001
        return Check(
            id="pipeline_job", title="Data orchestration job",
            category=CATEGORY_SCRIPTED, status=FAIL,
            detail=f"'{name}' has no successful latest run.",
            technical_detail=_error_text(exc),
            resolution_steps=[
                "Open Workflows and inspect the failed task and its repair-run option.",
                "If wait_for_lakebase_cdf failed, start CDF from the Lakebase UI first.",
                "Re-run deploy_app.sh with DEPLOY_DATA=1 after the prerequisite is fixed.",
            ],
            fix=Fix(kind="link", label="Open Workflows", href=_host_url(settings, "/jobs")),
        )
    return Check(
        id="pipeline_job", title="Data orchestration job",
        category=CATEGORY_SCRIPTED, status=OK,
        detail=f"Latest run of '{name}' succeeded.",
    )


def check_model_endpoints(settings: Settings, obo_token: str | None) -> Check:
    rt = _realtime_settings()
    endpoints: list[str] = []
    if rt is not None:
        endpoints = [e for e in (rt.stt_endpoint, rt.tts_endpoint) if e]
    if not endpoints:
        return Check(
            id="model_endpoints", title="Voice model endpoints", category=CATEGORY_SCRIPTED,
            status=FAIL, detail="No realtime STT/TTS endpoints configured",
        )
    try:
        from genie_voice.databricks.client import get_workspace_client

        client = get_workspace_client(settings)
    except Exception as exc:  # noqa: BLE001
        return Check(
            id="model_endpoints", title="Voice model endpoints", category=CATEGORY_SCRIPTED,
            status=FAIL, detail=f"cannot reach serving API: {exc}",
        )
    not_ready: list[str] = []
    for name in endpoints:
        try:
            ep = client.serving_endpoints.get(name)
            state = getattr(getattr(ep, "state", None), "ready", None)
            ready = str(getattr(state, "value", state) or "")
            if ready != "READY":
                not_ready.append(f"{name}={ready or 'MISSING'}")
        except Exception:  # noqa: BLE001
            not_ready.append(f"{name}=MISSING")
    if not_ready:
        return Check(
            id="model_endpoints", title="Voice model endpoints", category=CATEGORY_SCRIPTED,
            status=FAIL, detail="Not READY: " + ", ".join(not_ready),
            fix=Fix(kind="cli", label="Re-run deploy (FORCE_REALTIME_MODELS=1 if needed)"),
            resolution_steps=[
                "Open Serving and inspect the endpoint state/build logs.",
                "Confirm GPU serving capacity is available for the configured workload type.",
                "Re-run FORCE_REALTIME_MODELS=1 ./deploy_app.sh if the endpoint model/config drifted.",
            ],
        )
    return Check(
        id="model_endpoints", title="Voice model endpoints", category=CATEGORY_SCRIPTED,
        status=OK, detail="Qwen3-ASR + VoxCPM2 READY",
        explanation="Both GPU serving endpoints exist and Databricks reports them ready.",
    )


def check_model_registration(settings: Settings, obo_token: str | None) -> Check:
    """Verify HF candidates reached UC registration and the endpoint serves the alias."""
    candidates = _voice_candidates()
    if not candidates:
        return Check(
            id="model_registration", title="Hugging Face → Unity Catalog models",
            category=CATEGORY_SCRIPTED, status=FAIL,
            detail="No realtime voice candidates are configured.",
            fix=Fix(kind="cli", label="Configure realtime_voice candidates and redeploy"),
        )
    from genie_voice.databricks.client import get_workspace_client

    client = get_workspace_client(settings)
    failures: list[str] = []
    verified: list[str] = []
    for candidate_id, candidate in candidates:
        registered = str(candidate.get("registered_model") or "").strip()
        endpoint = str(candidate.get("endpoint") or "").strip()
        base_model = str(candidate.get("base_model") or "").strip()
        if not registered or not endpoint or not base_model:
            failures.append(f"{candidate_id}: base_model/registered_model/endpoint incomplete")
            continue
        fqn = f"{settings.databricks.catalog}.{settings.databricks.schema_name}.{registered}"
        try:
            alias = client.api_client.do(
                "GET", f"/api/2.1/unity-catalog/models/{fqn}/aliases/candidate"
            )
            version = str(alias.get("version") or "")
            if not version:
                raise RuntimeError("candidate alias has no version")
            payload = client.api_client.do("GET", f"/api/2.0/serving-endpoints/{endpoint}")
            config = payload.get("config") or {}
            served = config.get("served_entities") or config.get("served_models") or []
            matches = [
                entity for entity in served
                if str(entity.get("entity_name") or entity.get("model_name") or "") == fqn
                and str(entity.get("entity_version") or entity.get("model_version") or "") == version
            ]
            if not matches:
                raise RuntimeError(
                    f"endpoint is not serving {fqn}@candidate(version {version})"
                )
            verified.append(f"{candidate_id}={base_model} → {fqn}@{version} → {endpoint}")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{candidate_id}: {_error_text(exc)}")
    if failures:
        return Check(
            id="model_registration", title="Hugging Face → Unity Catalog models",
            category=CATEGORY_SCRIPTED, status=FAIL,
            detail="One or more registration/deployment links are broken.",
            explanation=(
                "The installer must download each configured Hugging Face repository, "
                "log a ResponsesAgent version in Unity Catalog, assign the candidate "
                "alias, and route the serving endpoint to that exact version."
            ),
            technical_detail="; ".join(failures),
            resolution_steps=[
                "Confirm outbound access to huggingface.co and provide HF_TOKEN for gated repositories.",
                "Confirm GPU serving capacity is available in this workspace/region.",
                "Re-run: FORCE_REALTIME_MODELS=1 ./deploy_app.sh",
                "Open the failed submitted run from Workflows and inspect its register/deploy task output.",
            ],
            fix=Fix(kind="cli", label="Re-run forced voice-model deployment"),
        )
    return Check(
        id="model_registration", title="Hugging Face → Unity Catalog models",
        category=CATEGORY_SCRIPTED, status=OK,
        detail=f"{len(verified)} model(s) registered, aliased, and routed correctly.",
        explanation=" | ".join(verified),
    )


def check_voice_contract(settings: Settings, obo_token: str | None) -> Check:
    """Invoke TTS then feed its output to STT using the production contract."""
    rt = _realtime_settings()
    if rt is None or not rt.stt_endpoint or not rt.tts_endpoint:
        return Check(
            id="voice_contract", title="Live STT/TTS contract",
            category=CATEGORY_SCRIPTED, status=FAIL,
            detail="STT/TTS endpoints could not be resolved from config.",
            fix=Fix(kind="cli", label="Fix realtime_voice config and redeploy"),
        )
    from genie_voice.databricks.client import get_workspace_client

    client = get_workspace_client(settings)
    text = "Readiness check."
    try:
        tts = client.api_client.do(
            "POST",
            f"/serving-endpoints/{rt.tts_endpoint}/invocations",
            body={
                "input": [{"role": "user", "content": text}],
                "custom_inputs": {"text": text, "language": "en-US"},
            },
        )
        tts_custom = (tts or {}).get("custom_outputs") or {}
        audio = str(tts_custom.get("audio_b64") or "")
        sample_rate = int(tts_custom.get("sample_rate_hz") or 16000)
        if not audio:
            raise RuntimeError("TTS returned no custom_outputs.audio_b64")
        # Validate base64 before sending it on; catches truncated/corrupt responses.
        if not base64.b64decode(audio, validate=True):
            raise RuntimeError("TTS returned empty audio")
        stt = client.api_client.do(
            "POST",
            f"/serving-endpoints/{rt.stt_endpoint}/invocations",
            body={
                "input": [{"role": "user", "content": "transcribe"}],
                "custom_inputs": {
                    "audio_b64": audio,
                    "language": "en-US",
                    "sample_rate_hz": sample_rate,
                },
            },
        )
        stt_custom = (stt or {}).get("custom_outputs") or {}
        if "transcript" not in stt_custom:
            raise RuntimeError("STT returned no custom_outputs.transcript")
    except Exception as exc:  # noqa: BLE001
        return Check(
            id="voice_contract", title="Live STT/TTS contract",
            category=CATEGORY_SCRIPTED, status=FAIL,
            detail="A real TTS → STT round trip failed.",
            explanation=(
                "Endpoint READY alone is insufficient. This sends the same "
                "ResponsesAgent custom_inputs/custom_outputs shape used by the app."
            ),
            technical_detail=_error_text(exc),
            resolution_steps=[
                "Open Serving and inspect endpoint build/runtime logs.",
                "Confirm the app service principal has CAN_QUERY on both endpoints.",
                "Re-run deploy_app.sh; use FORCE_REALTIME_MODELS=1 if the served model environment is broken.",
            ],
            fix=Fix(kind="link", label="Open Serving endpoints", href=_host_url(settings, "/ml/endpoints")),
        )
    transcript = str(stt_custom.get("transcript") or "").strip()
    return Check(
        id="voice_contract", title="Live STT/TTS contract",
        category=CATEGORY_SCRIPTED, status=OK,
        detail=f"Production TTS → STT round trip succeeded ({len(base64.b64decode(audio))} audio bytes).",
        explanation=f"STT returned transcript: {transcript[:120] or '(empty but contract-valid)'}",
    )


def check_gateway_services(settings: Settings, obo_token: str | None) -> Check:
    """Verify every Gateway service exists and routes to its configured model."""
    if not settings.ai_gateway.enabled:
        return Check(
            id="gateway_services", title="Unity AI Gateway services",
            category=CATEGORY_SCRIPTED, status=OK, detail="AI Gateway disabled by config.",
        )
    from genie_voice.databricks.client import get_workspace_client

    client = get_workspace_client(settings)
    failures: list[str] = []
    routes: list[str] = []
    for label, spec in settings.ai_gateway.model_services.items():
        fqn = str(spec.service)
        expected = str(spec.destination).removeprefix("models/")
        try:
            payload = client.api_client.do(
                "GET", f"/api/2.1/unity-catalog/model-services/{fqn}"
            )
            config = payload.get("config") or {}
            destinations = (config.get("routing") or {}).get("destinations") or []
            actual = ""
            if destinations:
                actual = str(
                    (destinations[0].get("pay_per_token_config") or {}).get("model") or ""
                ).removeprefix("models/")
            if actual != expected:
                raise RuntimeError(f"destination={actual or 'missing'}, expected={expected}")
            inference = config.get("inference_table") or {}
            if not inference or inference.get("is_deleted"):
                raise RuntimeError("inference table is absent or deleted")
            routes.append(f"{label}: {fqn} → {actual}")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{label}/{fqn}: {_error_text(exc)}")
    if failures:
        return Check(
            id="gateway_services", title="Unity AI Gateway services",
            category=CATEGORY_SCRIPTED, status=FAIL,
            detail="A Gateway service is missing, misrouted, or lacks its inference table.",
            technical_detail="; ".join(failures),
            resolution_steps=[
                "Re-run deploy_app.sh to reconcile model-service routes, rate limits, and inference tables.",
                "Confirm each configured system.ai destination is available in this workspace/region.",
                "Make this check green before diagnosing policy-conformance failures.",
            ],
            fix=Fix(kind="cli", label="Reconcile Gateway services"),
        )
    return Check(
        id="gateway_services", title="Unity AI Gateway services",
        category=CATEGORY_SCRIPTED, status=OK,
        detail=f"{len(routes)} Gateway service(s) exist and route correctly.",
        explanation="; ".join(routes),
    )


def check_gateway_policies(settings: Settings, obo_token: str | None) -> Check:
    try:
        failures = gateway_conformance_failures(settings)
    except Exception as exc:  # noqa: BLE001
        return Check(
            id="gateway_policies", title="AI Gateway guardrails", category=CATEGORY_MANUAL,
            status=FAIL, detail=f"could not probe Gateway services: {exc}",
            fix=Fix(kind="link", label="Open Serving / AI Gateway", href=_host_url(settings, "/ml/endpoints")),
            technical_detail=_error_text(exc),
            resolution_steps=[
                "Open each app-owned model service and verify its configured destination is healthy.",
                "Attach the policy bundles named by config/guardrails.yaml in the AI Gateway UI.",
                "Re-check; the page sends allow/deny/redaction behavioral probes.",
            ],
        )
    if failures:
        return Check(
            id="gateway_policies", title="AI Gateway guardrails", category=CATEGORY_MANUAL,
            status=FAIL,
            detail="Gateway services exist, but the required policy behavior is not active.",
            fix=Fix(kind="link", label="Open Serving / AI Gateway", href=_host_url(settings, "/ml/endpoints")),
            technical_detail="; ".join(failures),
            resolution_steps=[
                "Open each app-owned Unity AI Gateway model service.",
                "Attach the matching rate-limit, safety, and PII policy bundle from config/guardrails.yaml.",
                "Save the policy attachment and Re-check until allow/deny/redaction probes pass.",
            ],
        )
    return Check(
        id="gateway_policies", title="AI Gateway guardrails", category=CATEGORY_MANUAL,
        status=OK,
        detail="; ".join(
            f"{spec.service}: bundle={spec.policy_bundle}, behavioral probes=PASS"
            for spec in settings.ai_gateway.model_services.values()
        ),
    )


def gateway_conformance_failures(settings: Settings) -> list[str]:
    """Run the behavioral policy conformance matrix; return failure strings."""
    from genie_voice.databricks.ai_gateway import GatewayPolicyDenied, invoke
    from genie_voice.databricks.client import get_workspace_client
    from genie_voice.guardrails import get_policy_manifest

    services = {label: fqn for label, fqn in _gateway_service_map(settings).items()}
    manifest = get_policy_manifest()
    client = get_workspace_client(settings)
    failures: list[str] = []
    for probe in manifest.conformance:
        service = services.get(probe.service, "")
        if not service:
            failures.append(f"{probe.id}: service {probe.service!r} not configured")
            continue
        denied, output = False, ""
        try:
            payload = invoke(
                host=settings.databricks_host,
                authenticate=client.config.authenticate,
                endpoint=service,
                inputs={
                    "messages": [
                        {"role": "system", "content": "You are a concise assistant."},
                        {"role": "user", "content": probe.input},
                    ],
                    "max_tokens": 80,
                },
                timeout_s=120,
            )
            choices = payload.get("choices") or []
            if choices and isinstance(choices[0], dict):
                output = str((choices[0].get("message") or {}).get("content") or "")
        except GatewayPolicyDenied:
            denied = True
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{probe.id}: request failed: {exc}")
            continue
        if probe.expectation == "deny" and not denied:
            failures.append(f"{probe.id}: expected DENY but allowed")
        elif probe.expectation in {"allow", "redact"} and denied:
            failures.append(f"{probe.id}: expected {probe.expectation.upper()} but denied")
        elif probe.expectation == "redact" and any(
            f.lower() in output.lower() for f in probe.forbidden_output
        ):
            failures.append(f"{probe.id}: sensitive value survived redaction")
    return failures


def _gateway_service_map(settings: Settings) -> dict[str, str]:
    out: dict[str, str] = {}
    for label, item in (settings.ai_gateway.model_services or {}).items():
        service = getattr(item, "service", None)
        if service:
            out[label] = str(service)
    return out


def check_lakebase_cdf(settings: Settings, obo_token: str | None) -> Check:
    if not settings.lakebase.enabled or not settings.lakebase.cdf_required:
        return Check(
            id="lakebase_cdf", title="Lakebase CDF", category=CATEGORY_MANUAL,
            status=OK, detail="CDF not required by config",
        )
    try:
        from genie_voice.databricks.client import get_workspace_client
        from genie_voice.lakebase.cdf import (
            _optional_tables,
            _replica_identity_ok,
            _required_tables,
            _resolve_lakebase_branch,
            _source_row_counts,
            _uc_history_status,
            _wal2delta_status,
            history_blockers,
        )

        required = _required_tables(settings)
        optional = _optional_tables(settings)
        client = get_workspace_client(settings)
        resolved = _resolve_lakebase_branch(client.api_client, settings.lakebase.instance)
        branch = resolved["branch"]
        _replica_identity_ok(settings, required)
        raw_statuses = _wal2delta_status(settings, required)
        statuses = {
            table: str((raw_statuses.get(table) or {}).get("status") or "MISSING")
            for table in required
        }
        history = _uc_history_status(settings, required, None)
        source_counts = _source_row_counts(settings, required)
        missing, empty, stale = history_blockers(history, source_counts, None)
    except Exception as exc:  # noqa: BLE001
        return Check(
            id="lakebase_cdf", title="Lakebase CDF", category=CATEGORY_MANUAL,
            status=FAIL,
            detail=f"CDF not readable — start it in the Lakebase UI: {exc}",
            fix=Fix(kind="link", label="Open Lakebase", href=_host_url(settings, "/compute/lakebase")),
            technical_detail=_error_text(exc),
            resolution_steps=[
                "Open the configured Lakebase project and its default branch.",
                "Start Change Data Feed for the Gold inputs in lakebase.cdf_required_tables.",
                "Target the configured Unity Catalog catalog/schema and wait for STREAMING.",
                "Re-run deploy_app.sh so the orchestration job can finish, then Re-check.",
            ],
        )
    bad = [f"{t}={s}" for t, s in statuses.items() if s not in {"STREAMING", "SNAPSHOTTING"}]
    if bad:
        return Check(
            id="lakebase_cdf", title="Lakebase CDF", category=CATEGORY_MANUAL,
            status=FAIL, detail="Not streaming: " + ", ".join(bad),
            fix=Fix(kind="link", label="Open Lakebase", href=_host_url(settings, "/compute/lakebase")),
            resolution_steps=[
                "Open Lakebase Change Data Feed for the configured branch.",
                "Start or repair each listed table until wal2delta reports STREAMING or SNAPSHOTTING.",
                "Re-run the orchestration job, then Re-check.",
            ],
        )
    history_bad = missing + empty + stale
    if history_bad:
        return Check(
            id="lakebase_cdf", title="Lakebase CDF", category=CATEGORY_MANUAL,
            status=FAIL,
            detail="CDF is running, but Unity Catalog history is missing for tables that have Lakebase rows.",
            technical_detail="; ".join(
                f"{table}: {history.get(table, {}).get('error') or 'no published rows'}"
                for table in history_bad
            ),
            resolution_steps=[
                "Wait for CDF to publish call facts and immutable utterance history.",
                "Confirm the CDF target catalog/schema matches config/config.yaml.",
                "Run the orchestration job after history tables contain rows, then Re-check.",
            ],
            fix=Fix(kind="link", label="Open Lakebase", href=_host_url(settings, "/compute/lakebase")),
        )

    optional_issues: list[str] = []
    optional_statuses: dict[str, Any] = {}
    optional_history: dict[str, dict[str, Any]] = {}
    if optional:
        try:
            _replica_identity_ok(settings, optional)
            optional_statuses = _wal2delta_status(settings, optional)
            optional_issues.extend(
                f"{table}={str((optional_statuses.get(table) or {}).get('status') or 'MISSING')}"
                for table in optional
                if str((optional_statuses.get(table) or {}).get("status") or "MISSING")
                not in {"STREAMING", "SNAPSHOTTING"}
            )
            optional_history = _uc_history_status(settings, optional, None)
            optional_sources = _source_row_counts(settings, optional)
            opt_missing, opt_empty, opt_stale = history_blockers(
                optional_history, optional_sources, None
            )
            optional_issues.extend(
                f"{table}: {optional_history.get(table, {}).get('error') or 'no published rows'}"
                for table in opt_missing + opt_empty + opt_stale
            )
        except Exception as exc:  # noqa: BLE001
            optional_issues.append(_error_text(exc))

    branch_name = str(branch.get("name") or branch.get("branch_id") or "default")
    if optional_issues:
        return Check(
            id="lakebase_cdf", title="Lakebase CDF", category=CATEGORY_MANUAL,
            status=WARN,
            detail=(
                f"Required feeds on branch '{branch_name}' are STREAMING; "
                "optional feed failure: " + "; ".join(optional_issues)
            ),
            technical_detail="; ".join(optional_issues),
            explanation=(
                f"branch={branch_name}; required="
                + ", ".join(
                    f"{table}={history[table].get('total_rows')} rows"
                    for table in required
                )
            ),
            resolution_steps=[
                "The live voice workflow and Gold pipeline can run without this optional feed.",
                "Open Lakebase CDF and repair the optional table listed in technical details.",
                "For billing adjustments that predate CDF, recreate the feed with a snapshot or generate a new test adjustment.",
                "Re-check to enable complete Genie billing-adjustment analytics.",
            ],
            fix=Fix(kind="link", label="Open Lakebase", href=_host_url(settings, "/compute/lakebase")),
        )
    return Check(
        id="lakebase_cdf", title="Lakebase CDF", category=CATEGORY_MANUAL,
        status=OK,
        detail="; ".join(
            [
                f"branch={branch_name}",
                *[
                    (
                        f"{table}={statuses[table]}, "
                        f"{settings.databricks.catalog}.{settings.databricks.schema_name}."
                        f"{settings.lakebase.cdf_history_prefix}{table}"
                        f"{settings.lakebase.cdf_history_suffix}="
                        f"{history[table].get('total_rows')} rows"
                    )
                    for table in required
                ],
                *[
                    (
                        f"{table}="
                        f"{str((optional_statuses.get(table) or {}).get('status') or 'MISSING')}, "
                        f"{settings.databricks.catalog}.{settings.databricks.schema_name}."
                        f"{settings.lakebase.cdf_history_prefix}{table}"
                        f"{settings.lakebase.cdf_history_suffix}="
                        f"{optional_history[table].get('total_rows')} rows"
                    )
                    for table in optional
                ],
            ]
        ),
        explanation=(
            f"branch={branch_name}; "
            + ", ".join(
                f"{table}={meta.get('total_rows')} rows"
                for table, meta in {**history, **optional_history}.items()
            )
        ),
    )


def cdf_status(settings: Settings) -> dict[str, str]:
    """Non-blocking per-table wal2delta status for the required CDF tables."""
    from genie_voice.lakebase.cdf import _expected_tables, _wal2delta_status

    tables = _expected_tables(settings)
    raw = _wal2delta_status(settings, tables)
    out = {t: str((raw.get(t) or {}).get("status") or "MISSING") for t in tables}
    return out


def check_genie_spaces(settings: Settings, obo_token: str | None) -> Check:
    names = [settings.databricks.genie_space_name]
    if getattr(settings, "card_issuer", None) and settings.card_issuer.enabled:
        names.append(settings.card_issuer.genie_space_name)
    try:
        from genie_voice.databricks.client import get_workspace_client
        from genie_voice.genie.space import find_space_id

        client = get_workspace_client(settings)
        missing = [n for n in names if not find_space_id(client, n)]
    except Exception as exc:  # noqa: BLE001
        return Check(
            id="genie_spaces", title="Genie spaces", category=CATEGORY_SCRIPTED,
            status=FAIL, detail=f"could not list Genie spaces: {exc}",
        )
    if missing:
        return Check(
            id="genie_spaces", title="Genie spaces", category=CATEGORY_SCRIPTED,
            status=FAIL, detail="Missing: " + ", ".join(missing),
            fix=Fix(kind="cli", label="Re-run deploy (creates the Genie spaces)"),
            resolution_steps=[
                "Make the data, CDF, and orchestration-job checks green first.",
                "Re-run deploy_app.sh with DEPLOY_DATA=1 as the catalog/schema/space owner.",
                "Populate APP_EXTERNAL_USERS or APP_EXTERNAL_GROUPS to grant viewer CAN_RUN.",
            ],
        )
    return Check(
        id="genie_spaces", title="Genie spaces", category=CATEGORY_SCRIPTED,
        status=OK, detail="; ".join(names),
    )


def check_viewer_genie(settings: Settings, obo_token: str | None) -> Check:
    """Run a real question as the signed-in viewer against every app space."""
    if not obo_token:
        return Check(
            id="viewer_genie", title="App viewer + Genie permissions", category=CATEGORY_MANUAL,
            status=WARN,
            detail="Cannot test viewer CAN_RUN until Apps User Authorization supplies a user token.",
            explanation="Open this page through the Databricks App UI after enabling genie + sql scopes.",
            fix=Fix(kind="link", label="Open App settings", href=_host_url(settings, "/apps")),
        )
    from genie_voice.genie.client import GenieClient

    spaces = [settings.databricks.genie_space_name]
    if getattr(settings, "card_issuer", None) and settings.card_issuer.enabled:
        spaces.append(settings.card_issuer.genie_space_name)
    failures: list[str] = []
    passed: list[str] = []
    for name in spaces:
        try:
            result = GenieClient(settings, space_name=name).ask(
                "How many records are in the primary customer table? Return the total.",
                language="en-US",
                access_token=obo_token,
            )
            if not any(result.get(key) for key in ("answer", "description", "rows")):
                raise RuntimeError("Genie completed but returned no answer, description, or rows")
            passed.append(name)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{name}: {_error_text(exc)}")
    if failures:
        return Check(
            id="viewer_genie", title="App viewer + Genie permissions", category=CATEGORY_MANUAL,
            status=FAIL,
            detail="The signed-in viewer cannot successfully query every configured Genie space.",
            technical_detail="; ".join(failures),
            resolution_steps=[
                "Grant this user (or their group) CAN_RUN on both configured Genie spaces.",
                "Confirm Apps User Authorization requests both genie and sql scopes.",
                "Sign out/in or revoke and re-consent if scopes changed after the first app visit.",
                "Re-run deploy_app.sh with APP_EXTERNAL_USERS or APP_EXTERNAL_GROUPS populated.",
            ],
            fix=Fix(kind="link", label="Open Apps / permissions", href=_host_url(settings, "/apps")),
        )
    return Check(
        id="viewer_genie", title="App viewer + Genie permissions", category=CATEGORY_MANUAL,
        status=OK,
        detail="Real viewer-scoped query succeeded: " + "; ".join(passed),
        explanation="; ".join(passed),
    )


def check_agent_mode(settings: Settings, obo_token: str | None) -> Check:
    """Run a small Agent Mode question to verify the Beta preview and OBO path."""
    card = getattr(settings, "card_issuer", None)
    if card is None or not card.enabled:
        return Check(
            id="agent_mode", title="Genie Agent Mode preview", category=CATEGORY_MANUAL,
            status=OK, detail="Card issuer lane disabled; Agent Mode is not required.",
        )
    if not obo_token:
        return Check(
            id="agent_mode", title="Genie Agent Mode preview", category=CATEGORY_MANUAL,
            status=WARN,
            detail="Cannot validate Agent Mode without the signed-in user's OBO token.",
            fix=Fix(kind="link", label="Open App settings", href=_host_url(settings, "/apps")),
        )
    try:
        from genie_voice.genie.agent_mode import GenieAgentModeClient
        from realtime_api.runtime.identity import SessionPrincipal, workspace_client_for_principal

        principal = SessionPrincipal(access_token=obo_token, from_forwarded_header=True)
        client = workspace_client_for_principal(principal, settings)
        read_timeout = float(
            (((_deployment_config().get("realtime_voice") or {}).get("deep_dive") or {}).get("read_timeout_s"))
            or 420
        )
        result = GenieAgentModeClient(settings, workspace_client=client).ask(
            "How many cardholders are in the configured data? Return the total with one supporting table.",
            space_name=card.genie_space_name,
            read_timeout_s=read_timeout,
        )
        if result.status != "completed":
            raise RuntimeError(f"status={result.status}; error={result.error}")
        if not result.report_text and not result.tables:
            raise RuntimeError("completed without a report or supporting table")
    except Exception as exc:  # noqa: BLE001
        detail = _error_text(exc)
        return Check(
            id="agent_mode", title="Genie Agent Mode preview", category=CATEGORY_MANUAL,
            status=FAIL,
            detail="A real viewer-scoped Agent Mode run did not complete.",
            explanation=(
                "This validates the workspace Preview flag, the card Genie Agent, "
                "viewer authorization, SSE response stream, and governed SQL execution."
            ),
            technical_detail=detail,
            resolution_steps=[
                "As a workspace admin, open Previews and enable 'Agent Mode APIs for Genie Agents'.",
                "Grant the viewer CAN_RUN on the card-issuer Genie space.",
                "Confirm the card reference-data and Viewer Genie checks are green first.",
                "Re-check; transient Agent Mode runs can take several minutes.",
            ],
            fix=Fix(kind="link", label="Open workspace Previews", href=_host_url(settings, "/settings/workspace/previews")),
        )
    return Check(
        id="agent_mode", title="Genie Agent Mode preview", category=CATEGORY_MANUAL,
        status=OK,
        detail=(
            f"Preview 'Agent Mode APIs for Genie Agents'=ENABLED; "
            f"viewer-scoped run on '{card.genie_space_name}'=COMPLETED with governed evidence."
        ),
    )


def check_obo(settings: Settings, obo_token: str | None) -> Check:
    """Whether the caller carried a forwarded user token (Apps User Authorization).

    A present token means User Authorization is enabled and the viewer consented;
    Genie/SQL calls can run as the viewer. Absent → those paths fail closed.
    """
    app_name = os.environ.get("DATABRICKS_APP_NAME", "genie-voice-agent")
    fix = Fix(kind="link", label="Open App settings", href=_host_url(settings, "/apps"))
    steps = [
        "Open the Databricks App settings and enable User Authorization.",
        "Grant/request the genie and sql scopes declared in app.yaml.",
        "Restart/redeploy the App, then re-open it and approve the user-consent prompt.",
        "If consent predates the scope change, revoke it and consent again.",
    ]
    if not obo_token:
        return Check(
            id="obo", title="Apps User Authorization (OBO)", category=CATEGORY_MANUAL,
            status=WARN,
            detail=(
                "x-forwarded-access-token=MISSING; effective scopes and viewer "
                "identity cannot be validated."
            ),
            fix=fix,
            resolution_steps=steps,
        )
    try:
        from databricks.sdk import WorkspaceClient

        from genie_voice.databricks.client import get_workspace_client

        app = get_workspace_client(settings).apps.get(app_name)
        effective = {
            str(scope)
            for scope in (getattr(app, "effective_user_api_scopes", None) or [])
        }
        missing = {"genie", "sql"} - effective
        if missing:
            raise RuntimeError(
                "missing effective App scope(s): " + ", ".join(sorted(missing))
            )
        viewer = WorkspaceClient(
            host=settings.databricks_host,
            token=obo_token,
            auth_type="pat",
        ).current_user.me()
        viewer_name = (
            getattr(viewer, "user_name", None)
            or getattr(viewer, "display_name", None)
            or "authenticated viewer"
        )
    except Exception as exc:  # noqa: BLE001
        return Check(
            id="obo", title="Apps User Authorization (OBO)", category=CATEGORY_MANUAL,
            status=FAIL,
            detail=(
                f"App '{app_name}' User Authorization is not effective for both "
                f"required scopes: {_error_text(exc)}"
            ),
            fix=fix,
            technical_detail=_error_text(exc),
            resolution_steps=steps,
        )
    return Check(
        id="obo", title="Apps User Authorization (OBO)", category=CATEGORY_MANUAL,
        status=OK,
        detail=(
            f"App '{app_name}': effective scopes=genie,sql; "
            f"x-forwarded-access-token=PRESENT; viewer={viewer_name}."
        ),
    )


def check_reference_cache(settings: Settings, obo_token: str | None) -> Check:
    if not settings.lakebase.enabled:
        return Check(
            id="reference_cache", title="Lakebase reference cache", category=CATEGORY_SCRIPTED,
            status=OK, detail="Lakebase disabled; served from datagen",
        )
    try:
        from genie_voice.serve import LakebaseServing

        lb = LakebaseServing(settings)
        with lb._conn() as conn, conn.cursor() as cur:  # noqa: SLF001
            schema = settings.lakebase.schema_name.replace('"', '""')
            counts: list[str] = []
            for table in ("customers", "invoices", "payments"):
                cur.execute(f'SELECT count(*) FROM "{schema}"."{table}"')
                count = int(cur.fetchone()[0] or 0)
                if count <= 0:
                    raise RuntimeError(f"{schema}.{table} is empty")
                counts.append(f"{table}={count}")
            if getattr(settings, "card_issuer", None) and settings.card_issuer.enabled:
                card_schema = settings.card_issuer.schema_name.replace('"', '""')
                for table in ("cardholders", "statements"):
                    cur.execute(f'SELECT count(*) FROM "{card_schema}"."{table}"')
                    count = int(cur.fetchone()[0] or 0)
                    if count <= 0:
                        raise RuntimeError(f"{card_schema}.{table} is empty")
                    counts.append(f"{table}={count}")
    except Exception as exc:  # noqa: BLE001
        return Check(
            id="reference_cache", title="Lakebase serving data", category=CATEGORY_SCRIPTED,
            status=FAIL, detail="Lakebase is unreachable or its serving cache is empty.",
            technical_detail=_error_text(exc),
            resolution_steps=[
                "Confirm the app service principal has workspace-access and Lakebase CONNECT/USAGE grants.",
                "Use the repair button to resnapshot telco reference tables.",
                "For card data, rerun deploy_app.sh after the UC reference-data check is green.",
            ],
            fix=Fix(kind="auto", label="Snapshot reference tables", action="snapshot_reference"),
        )
    return Check(
        id="reference_cache", title="Lakebase serving data", category=CATEGORY_SCRIPTED,
        status=OK, detail="Low-latency telco/card serving caches are populated.",
        explanation=", ".join(counts),
    )


_CHECKS: list[Callable[[Settings, str | None], Check]] = [
    check_config,
    check_warehouse,
    check_uc_storage,
    check_uc_data,
    check_pipeline_job,
    check_model_registration,
    check_model_endpoints,
    check_genie_spaces,
    check_lakebase_cdf,
    check_reference_cache,
    check_gateway_services,
    check_gateway_policies,
    check_obo,
]

_FULL_CHECKS: list[Callable[[Settings, str | None], Check]] = [
    check_voice_contract,
    check_viewer_genie,
    check_agent_mode,
]


def run_checks(
    settings: Settings | None = None,
    obo_token: str | None = None,
    *,
    full: bool = True,
) -> Readiness:
    """Run independent probes concurrently.

    Full mode adds real STT/TTS, Genie Conversation, and Agent Mode calls. Those
    are definitive but may take minutes on cold endpoints, so the deploy script
    can request ``full=False`` while the Setup page uses the complete suite.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    settings = settings or get_settings()
    functions = [*_CHECKS, *(_FULL_CHECKS if full else [])]
    results: dict[str, Check] = {}

    def _run(fn: Callable[[Settings, str | None], Check]) -> Check:
        try:
            return fn(settings, obo_token)
        except Exception as exc:  # noqa: BLE001
            return Check(
                id=getattr(fn, "__name__", "check"),
                title=getattr(fn, "__name__", "check"),
                category=CATEGORY_SCRIPTED,
                status=FAIL,
                detail="The readiness probe itself failed unexpectedly.",
                technical_detail=_error_text(exc),
                resolution_steps=[
                    "Retry once to rule out a transient workspace/API error.",
                    "If it persists, inspect the Databricks App logs for this check id.",
                ],
            )
    with ThreadPoolExecutor(max_workers=min(8, len(functions))) as pool:
        future_to_fn = {pool.submit(_run, fn): fn for fn in functions}
        for future in as_completed(future_to_fn):
            fn = future_to_fn[future]
            results[fn.__name__] = future.result()

    checks = [results[fn.__name__] for fn in functions]
    for check in checks:
        if not check.objects:
            check.objects = _readiness_objects(settings, check.id)
        if check.category == CATEGORY_MANUAL:
            required, actions = _manual_guidance(settings, check.id)
            check.required_configuration = required
            check.workspace_actions = actions
    ready = all(c.status == OK for c in checks)
    return Readiness(ready=ready, checks=checks)


# --------------------------------------------------------------------------- #
# Cheap, idempotent fixes performable as the app's own identity.
# --------------------------------------------------------------------------- #
def apply_fix(action: str, settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or get_settings()
    if action == "snapshot_reference":
        from genie_voice.serve import LakebaseServing

        counts: dict[str, Any] = {
            "telco": LakebaseServing(settings).snapshot_reference_tables()
        }
        if getattr(settings, "card_issuer", None) and settings.card_issuer.enabled:
            from genie_voice.serve.card_lakebase import CardLakebaseServing

            counts["card"] = CardLakebaseServing(settings).snapshot_card_reference_tables()
        return {"action": action, "ok": True, "detail": counts}
    raise ValueError(f"unknown or unsupported fix action: {action!r}")
