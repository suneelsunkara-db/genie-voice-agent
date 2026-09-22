#!/usr/bin/env bash
# =============================================================================
# Genie Voice Agent - deploy to Databricks Apps.
#
# Turns the local two-process app (Vite + uvicorn) into a single Databricks App
# web process: the built React SPA is served by FastAPI from api/app/static, and
# the app authenticates as its injected service principal.
#
# Steps:
#   1. validate the committed deployment config, build the frontend, and run tests
#   2. bootstrap UC/Lakebase, land demo data, run the orchestration job, and
#      reconcile both Genie spaces
#   3. register/deploy Qwen3-ASR and VoxCPM2, reconcile their supported AI
#      Gateway configuration, and smoke their ResponsesAgent contracts
#   4. push optional vendor API keys into a Databricks secret scope
#   5. reconcile app-owned Gateway model services (routing, rate limits, inference
#      tables), attach app resources (warehouse + secrets + STT/TTS serving
#      endpoints), and grant EXECUTE on model services; then re-assert CAN_USE
#      for external callers (see APP_EXTERNAL_* below) so cross-workspace access
#      survives every redeploy
#   6. sync source to a workspace folder (respects .gitignore) — this includes
#      the mcp_server/ package, which the app hosts in-process at /realtime/mcp
#      (a remote MCP endpoint; `mcp` is declared in requirements.txt). No extra
#      deploy step is needed: syncing the source + the mount in api/app/main.py
#      deploy and update the MCP server together with the app.
#   7. create the app (first run) and deploy it
#
# Config: edit config/config.yaml; environment variables only control installer
# behavior, the CLI profile, optional secrets, and caller ACLs.
# Auth:   uses your Databricks CLI profile (U2M) to DEPLOY; the running app uses
#         its own service principal for hosting / serving. Genie workspace Q&A
#         prefers the viewer's OBO token (x-forwarded-access-token) when Apps
#         user authorization is enabled with genie (+ sql) scopes — see app.yaml
#         and README "User authorization (OBO)".
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# ---- config (runtime and jobs use this exact committed file) ----------------
APP_NAME="${APP_NAME:-genie-voice-agent}"
DATABRICKS_PROFILE="${DATABRICKS_PROFILE:-${DATABRICKS_CONFIG_PROFILE:-}}" # ~/.databrickscfg profile
SECRET_SCOPE="${SECRET_SCOPE:-genie-voice}"           # scope holding vendor keys
SQL_WAREHOUSE_ID_OVERRIDE="${SQL_WAREHOUSE_ID:-}"     # must match config/config.yaml
SQL_WAREHOUSE_ID=""
ENRICHMENT_MODEL_SERVICE=""
WORKSPACE_DIR="${WORKSPACE_DIR:-}"                    # empty -> /Workspace/Users/<me>/<app>
ASR_ENDPOINTS=""
DEPLOY_DATA="${DEPLOY_DATA:-1}"                        # 0 = skip UC/Lakebase/data/jobs
DEPLOY_REALTIME_MODELS="${DEPLOY_REALTIME_MODELS:-auto}" # auto | 1 | 0
FORCE_REALTIME_MODELS="${FORCE_REALTIME_MODELS:-0}"

# External principals (re-)granted CAN_USE on the app every deploy so external /
# cross-workspace API callers keep access across redeploys. Comma-separated;
# leave empty to skip. Grants are ADDITIVE (owner/admins/account-users untouched).
#   APP_EXTERNAL_USERS  -> user_name (email) of an account user
#   APP_EXTERNAL_SPS    -> service_principal_name (OAuth client / application id)
#   APP_EXTERNAL_GROUPS -> account group names granted App CAN_USE + Genie CAN_RUN
APP_EXTERNAL_USERS="${APP_EXTERNAL_USERS:-}"
APP_EXTERNAL_SPS="${APP_EXTERNAL_SPS:-}"
APP_EXTERNAL_GROUPS="${APP_EXTERNAL_GROUPS:-}"

log()  { printf "\033[36m[app-deploy]\033[0m %s\n" "$*"; }
warn() { printf "\033[33m[app-deploy]\033[0m %s\n" "$*"; }
die()  { printf "\033[31m[app-deploy]\033[0m %s\n" "$*"; exit 1; }

# Prefer the repo venv python (has backend deps installed) over system python3.
PYBIN="python3"
[[ -x "$ROOT/.venv/bin/python" ]] && PYBIN="$ROOT/.venv/bin/python"
"$PYBIN" -c "import yaml, databricks.sdk" >/dev/null 2>&1 \
  || die "Python deploy dependencies are missing; create .venv and install backend/api requirements."
DEPLOY_CONFIG="$ROOT/config/config.yaml"
[[ -f "$DEPLOY_CONFIG" ]] || die "Missing deployment config: $DEPLOY_CONFIG"

# The app, serverless jobs, model jobs, and deploy-time helpers must all read the
# same file. This intentionally disables config.local.yaml overlays for deploys.
export GENIE_CONFIG="$DEPLOY_CONFIG"

IFS='|' read -r _CFG_PROFILE _CFG_WAREHOUSE < <(
  "$PYBIN" - <<'PY'
import yaml
with open("config/config.yaml", encoding="utf-8") as handle:
    config = yaml.safe_load(handle) or {}
db = config.get("databricks") or {}
print(f"{db.get('profile') or ''}|{db.get('sql_warehouse_id') or ''}")
PY
)
DATABRICKS_PROFILE="${DATABRICKS_PROFILE:-$_CFG_PROFILE}"
SQL_WAREHOUSE_ID="$_CFG_WAREHOUSE"
if [[ -n "$SQL_WAREHOUSE_ID_OVERRIDE" && "$SQL_WAREHOUSE_ID_OVERRIDE" != "$SQL_WAREHOUSE_ID" ]]; then
  die "SQL_WAREHOUSE_ID must match databricks.sql_warehouse_id in config/config.yaml."
fi
if [[ -n "$DATABRICKS_PROFILE" ]]; then
  export DATABRICKS_CONFIG_PROFILE="$DATABRICKS_PROFILE"
fi

dbx() { if [[ -n "$DATABRICKS_PROFILE" ]]; then databricks "$@" -p "$DATABRICKS_PROFILE"; else databricks "$@"; fi; }

command -v databricks >/dev/null 2>&1 || die "Databricks CLI not found (brew install databricks)."
command -v npm >/dev/null 2>&1 || die "npm not found (needed to build the frontend)."
[[ -n "$SQL_WAREHOUSE_ID" ]] || die "Set databricks.sql_warehouse_id in config/config.yaml."

log "validating deployment config"
"$PYBIN" - <<'PY'
import yaml

with open("config/config.yaml", encoding="utf-8") as handle:
    config = yaml.safe_load(handle) or {}
db = config.get("databricks") or {}
required = {
    "databricks.host": db.get("host"),
    "databricks.catalog": db.get("catalog"),
    "databricks.schema": db.get("schema"),
    "databricks.sql_warehouse_id": db.get("sql_warehouse_id"),
    "lakebase.instance": (config.get("lakebase") or {}).get("instance"),
}
invalid = [
    name for name, value in required.items()
    if not str(value or "").strip()
    or "<" in str(value)
    or "your-" in str(value).lower()
]
if invalid:
    raise SystemExit(
        "config/config.yaml has unset deployment values: "
        + ", ".join(sorted(invalid))
    )

prefix = f"{db['catalog']}.{db['schema']}."
services = (config.get("ai_gateway") or {}).get("model_services") or {}
wrong = [
    item.get("service")
    for item in services.values()
    if item.get("service") and not str(item["service"]).startswith(prefix)
]
rv = config.get("realtime_voice") or {}
app_routes = [
    (config.get("enrichment") or {}).get("model_endpoint"),
    rv.get("llm_endpoint"),
    rv.get("i18n_endpoint"),
    rv.get("conversion_endpoint"),
]
wrong.extend(
    route for route in app_routes
    if route and str(route).count(".") == 2 and not str(route).startswith(prefix)
)
if wrong:
    raise SystemExit(
        "AI Gateway model-service FQNs must use the deployed catalog/schema "
        f"({prefix}*): {', '.join(wrong)}"
    )
PY

# Workspace IP ACLs also block the IP Access Lists API, so this can only add
# the current /32 while we are still allowed. If this machine is already
# blocked, the helper fails with the exact admin command to run from VPN/office.
log "checking workspace reachability and IP access lists"
PYTHONPATH=backend "$PYBIN" infra/apps/ensure_deploy_ip.py \
  || die "Workspace is unreachable from this machine's public IP."
_CLI_HOST="$(dbx auth env -o json 2>/dev/null | "$PYBIN" -c 'import json,sys; print((json.load(sys.stdin).get("env") or {}).get("DATABRICKS_HOST") or "")' 2>/dev/null || true)"
_CONFIG_HOST="$("$PYBIN" -c 'import yaml; print((yaml.safe_load(open("config/config.yaml")) or {}).get("databricks", {}).get("host", ""))')"
if [[ -n "$_CLI_HOST" && "${_CLI_HOST%/}" != "${_CONFIG_HOST%/}" ]]; then
  die "CLI profile host ($_CLI_HOST) does not match config/config.yaml ($_CONFIG_HOST)."
fi
ME="$(dbx current-user me -o json | python3 -c 'import sys,json;print(json.load(sys.stdin)["userName"])')"
WORKSPACE_DIR="${WORKSPACE_DIR:-/Workspace/Users/$ME/$APP_NAME}"
log "deploying as: $ME  ->  app '$APP_NAME'  (source: $WORKSPACE_DIR)"

# ---- 1. build the frontend for same-origin ---------------------------------
log "building frontend (same-origin) -> api/app/static"
( cd frontend && npm ci --silent && npm test -- --run && VITE_API_BASE_URL="" npm run build )
rm -rf api/app/static
mkdir -p api/app/static
cp -R frontend/dist/. api/app/static/
[[ -f api/app/static/index.html ]] || die "frontend build missing index.html"

log "running deploy-gate Python tests"
"$PYBIN" -c "import pytest" >/dev/null 2>&1 \
  || die "pytest is missing from $PYBIN; install the repo dev dependencies before deploy."
PYTHONPATH="$ROOT/backend:$ROOT" "$PYBIN" -m pytest \
  realtime_api/tests backend/tests/test_framework_seams.py -q

# Databricks Apps rejects any single source file over 10 MiB. Check the exact
# git-visible working tree plus the explicitly included generated SPA.
log "checking Databricks Apps 10 MiB per-file limit"
ROOT="$ROOT" "$PYBIN" - <<'PY'
import os
import pathlib
import subprocess

root = pathlib.Path(os.environ["ROOT"])
listed = subprocess.check_output(
    ["git", "ls-files", "-co", "--exclude-standard", "-z"], cwd=root
).decode().split("\0")
listed = [item for item in listed if item]
# `git ls-files -c` includes tracked presentation files even when .gitignore
# excludes them; `databricks sync` excludes those paths, so mirror that behavior.
ignored_raw = subprocess.run(
    ["git", "check-ignore", "--no-index", "-z", "--stdin"],
    cwd=root,
    input=("\0".join(listed) + "\0").encode(),
    stdout=subprocess.PIPE,
    check=False,
).stdout.decode()
ignored = {item for item in ignored_raw.split("\0") if item}
paths = {root / item for item in listed if item not in ignored}
static = root / "api" / "app" / "static"
if static.exists():
    paths.update(p for p in static.rglob("*") if p.is_file())
limit = 10 * 1024 * 1024
oversized = sorted((p.stat().st_size, p) for p in paths if p.is_file() and p.stat().st_size > limit)
if oversized:
    for size, path in oversized:
        print(f"{path.relative_to(root)}: {size} bytes")
    raise SystemExit("source contains files over the Databricks Apps 10 MiB limit")
PY

log "verifying warehouse and required serving / model-service targets"
dbx warehouses get "$SQL_WAREHOUSE_ID" >/dev/null 2>&1 \
  || die "SQL warehouse '$SQL_WAREHOUSE_ID' does not exist or is not accessible."

# ---- 2. UC, Lakebase, data, jobs, and Genie spaces -------------------------
case "$(printf '%s' "$DEPLOY_DATA" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes)
    log "bootstrapping UC schemas and Volumes"
    PYTHONPATH=backend "$PYBIN" -m genie_voice.databricks.bootstrap --skip-tables
    PYTHONPATH=backend "$PYBIN" -m genie_voice.databricks.bootstrap_card --skip-tables

    log "landing deterministic telco and card reference data"
    PYTHONPATH=backend "$PYBIN" -m genie_voice.datagen.loader
    PYTHONPATH=backend "$PYBIN" -m genie_voice.datagen.card.loader_card
    log "landing deterministic call-stream events"
    PYTHONPATH=backend "$PYBIN" -m genie_voice.ingest.producer

    log "provisioning Lakebase project and serving schemas"
    GENIE_LAKEBASE_AUTOCREATE="${GENIE_LAKEBASE_AUTOCREATE:-true}" \
      PYTHONPATH=backend "$PYBIN" infra/lakebase/setup_lakebase.py \
        --skip-reference-snapshot

    log "ingesting card reference data into Unity Catalog"
    PYTHONPATH=backend "$PYBIN" -m genie_voice.databricks.reference_ingest_card

    log "deploying and running the Lakebase-first orchestration job"
    if ! PYTHONPATH=backend "$PYBIN" infra/jobs/deploy_pipeline.py; then
      warn "The orchestration job did not complete."
      warn "For a fresh Lakebase project, start CDF for the required tables as documented in docs/CUSTOMER_INSTALL.md, then rerun this command."
      warn "Continuing so the app and Setup page are deployed; readiness will stay red until the job succeeds."
    fi

    log "snapshotting UC reference data into Lakebase serving"
    if ! GENIE_LAKEBASE_AUTOCREATE=false PYTHONPATH=backend \
      "$PYBIN" infra/lakebase/setup_lakebase.py; then
      warn "Telco reference snapshot is not ready; the Setup page will show the repair action."
    fi
    if ! PYTHONPATH=backend "$PYBIN" -m genie_voice.serve.card_lakebase; then
      warn "Card reference snapshot is not ready; rerun deploy after resolving the reported prerequisite."
    fi

    log "reconciling the card-issuer Genie space"
    if ! PYTHONPATH=backend "$PYBIN" -m genie_voice.genie.space_card; then
      warn "Card Genie space reconciliation is not ready; Setup will report it and the required repair."
    fi
    ;;
  0|false|no)
    warn "DEPLOY_DATA=0: skipping UC, Lakebase, data, orchestration, and Genie setup."
    ;;
  *)
    die "DEPLOY_DATA must be 1 or 0 (received '$DEPLOY_DATA')."
    ;;
esac

# ---- 2b. app-owned AI Gateway services (BEFORE any GPU work) ---------------
# Provisioning + the policy conformance probe run first so a fresh install stops
# at the UI-only policy-attach checkpoint BEFORE paying for GPU model registration
# and serving. Model services must also exist before their FQNs are validated.
log "reconciling app-owned Unity AI Gateway model services"
DATABRICKS_CONFIG_PROFILE="$DATABRICKS_PROFILE" PYTHONPATH=backend \
  "$PYBIN" infra/apps/provision_ai_gateway.py \
    --config config/config.yaml \
    --guardrails-config config/guardrails.yaml

log "running AI Gateway policy conformance matrix"
if ! DATABRICKS_CONFIG_PROFILE="$DATABRICKS_PROFILE" PYTHONPATH=backend \
  "$PYBIN" infra/apps/probe_ai_gateway_policies.py \
    --config config/config.yaml \
    --guardrails-config config/guardrails.yaml; then
  warn "Gateway policies are not conformant yet (policy attachment is UI-only)."
  warn "Continuing with model + app deployment so the Setup page can guide and verify this step."
fi

# ---- 3. realtime Qwen3-ASR + VoxCPM2 endpoints -----------------------------
_MODEL_ENDPOINTS="$(
  "$PYBIN" - <<'PY'
import yaml
with open("config/config.yaml", encoding="utf-8") as handle:
    rv = (yaml.safe_load(handle) or {}).get("realtime_voice") or {}
for group in ("stt_candidates", "tts_candidates"):
    for candidate in (rv.get(group) or {}).values():
        endpoint = candidate.get("endpoint") if isinstance(candidate, dict) else None
        if endpoint:
            print(endpoint)
PY
)"
[[ -n "$_MODEL_ENDPOINTS" ]] || die "No realtime voice model endpoints are configured."
_REGISTERED_VOICE_MODELS="$(
  "$PYBIN" - <<'PY'
import yaml
with open("config/config.yaml", encoding="utf-8") as handle:
    rv = (yaml.safe_load(handle) or {}).get("realtime_voice") or {}
seen = set()
for group in ("stt_candidates", "tts_candidates"):
    for candidate in (rv.get(group) or {}).values():
        model = candidate.get("registered_model") if isinstance(candidate, dict) else None
        if model and model not in seen:
            seen.add(model)
            print(model)
PY
)"
REGISTERED_VOICE_MODELS="$(printf '%s\n' "$_REGISTERED_VOICE_MODELS" | paste -sd, -)"
[[ -n "$REGISTERED_VOICE_MODELS" ]] || die "No registered realtime voice models are configured."

_ALL_MODELS_READY=1
while IFS= read -r _ep; do
  [[ -z "$_ep" ]] && continue
  _state="$(dbx serving-endpoints get "$_ep" -o json 2>/dev/null \
    | "$PYBIN" -c 'import json,sys; print((json.load(sys.stdin).get("state") or {}).get("ready") or "")' \
    2>/dev/null || true)"
  if [[ "$_state" != "READY" ]]; then
    _ALL_MODELS_READY=0
  fi
done <<< "$_MODEL_ENDPOINTS"

# READY is not enough: an existing endpoint may still route an old/wrong UC
# model version. Verify HF candidate -> UC registered model -> candidate alias ->
# served entity as one chain. Any drift makes auto mode redeploy it.
if [[ "$_ALL_MODELS_READY" -eq 1 ]]; then
  _MODEL_CHAIN_STATUS="$(
    PYTHONPATH="$ROOT/backend:$ROOT" "$PYBIN" - <<'PY' 2>/dev/null || true
from genie_voice.config import get_settings
from genie_voice.readiness import check_model_registration

print(check_model_registration(get_settings(), None).status)
PY
  )"
  if [[ "$_MODEL_CHAIN_STATUS" != "ok" ]]; then
    warn "Voice endpoints are READY but their UC model alias/routing chain drifted; auto mode will redeploy."
    _ALL_MODELS_READY=0
  fi
fi

case "$(printf '%s' "$DEPLOY_REALTIME_MODELS" | tr '[:upper:]' '[:lower:]')" in
  auto)
    _DEPLOY_MODELS=$((1 - _ALL_MODELS_READY))
    ;;
  1|true|yes) _DEPLOY_MODELS=1 ;;
  0|false|no) _DEPLOY_MODELS=0 ;;
  *) die "DEPLOY_REALTIME_MODELS must be auto, 1, or 0." ;;
esac
case "$(printf '%s' "$FORCE_REALTIME_MODELS" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes) _DEPLOY_MODELS=1 ;;
esac

if [[ "$_DEPLOY_MODELS" -eq 1 ]]; then
  # Preflight: HF weights are downloaded on the GPU registration job. Resolve a
  # token from env or the secret scope so gated repos don't fail mid-job (public
  # repos still work without one, so this is a warning, not a hard stop).
  HF_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"
  if [[ -z "$HF_TOKEN" ]]; then
    HF_TOKEN="$(dbx secrets get-secret "$SECRET_SCOPE" hf_token -o json 2>/dev/null \
      | "$PYBIN" -c 'import base64,json,sys; d=json.load(sys.stdin); print(base64.b64decode(d.get("value","")).decode() if d.get("value") else "")' 2>/dev/null || true)"
  fi
  if [[ -n "$HF_TOKEN" ]]; then
    export HF_TOKEN HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
    dbx secrets create-scope "$SECRET_SCOPE" >/dev/null 2>&1 || true
    dbx secrets put-secret "$SECRET_SCOPE" hf_token --string-value "$HF_TOKEN" >/dev/null 2>&1 || true
  else
    warn "No HF token (HF_TOKEN / secret '$SECRET_SCOPE/hf_token'); gated model downloads will fail."
  fi
  log "registering and deploying realtime Qwen3-ASR + VoxCPM2 agents"
  PYTHONPATH=scripts/ml_asr "$PYBIN" scripts/ml_asr/submit_realtime_voice_jobs.py \
    --config "$DEPLOY_CONFIG" --wait
elif [[ "$_ALL_MODELS_READY" -eq 1 ]]; then
  log "realtime voice endpoints are already READY; skipping registration"
else
  die "Realtime model deployment was disabled but one or more configured endpoints are not READY."
fi

log "reconciling AI Gateway inference tables for voice agent endpoints"
DATABRICKS_CONFIG_PROFILE="$DATABRICKS_PROFILE" PYTHONPATH=backend \
  "$PYBIN" infra/apps/provision_voice_endpoint_gateway.py \
    --config "$DEPLOY_CONFIG"

log "smoke-testing realtime voice model contracts"
PYTHONPATH=scripts/ml_asr "$PYBIN" scripts/ml_asr/smoke_realtime_voice_agents.py \
  --config "$DEPLOY_CONFIG"

# ---- 4. vendor keys -> secret scope (evals/benchmarks; not injected into the app)
DEEPGRAM_API_KEY="${DEEPGRAM_API_KEY:-$(PYTHONPATH=backend "$PYBIN" -c 'from genie_voice.config import get_settings;print(get_settings().secrets.deepgram_api_key)' 2>/dev/null || true)}"
ELEVENLABS_API_KEY="${ELEVENLABS_API_KEY:-$(PYTHONPATH=backend "$PYBIN" -c 'from genie_voice.config import get_settings;print(get_settings().secrets.elevenlabs_api_key)' 2>/dev/null || true)}"

log "ensuring secret scope '$SECRET_SCOPE'"
dbx secrets create-scope "$SECRET_SCOPE" >/dev/null 2>&1 || true
if [[ -n "$DEEPGRAM_API_KEY" ]]; then
  dbx secrets put-secret "$SECRET_SCOPE" deepgram_api_key --string-value "$DEEPGRAM_API_KEY"
  log "stored deepgram_api_key (evals/benchmarks only; not attached to the app)"
else
  warn "DEEPGRAM_API_KEY empty - skipping (live app STT is Databricks; evals that need Deepgram will fail)."
fi
if [[ -n "$ELEVENLABS_API_KEY" ]]; then
  dbx secrets put-secret "$SECRET_SCOPE" elevenlabs_api_key --string-value "$ELEVENLABS_API_KEY"
  log "stored elevenlabs_api_key"
else
  warn "ELEVENLABS_API_KEY empty - skipping (TTS optional; not in the live flow)."
fi

# ---- 5. app resource spec (auto-grants the app SP on deploy) ----------------
APP_JSON="$(mktemp)"
INCLUDE_EL="$([[ -n "$ELEVENLABS_API_KEY" ]] && echo 1 || echo 0)"
# Derive the ASR endpoints to attach as app resources from the exact deployed
# config. Local overrides must never leak into the app's resource grants.
ASR_ENDPOINTS="$(
    CONFIG_YAML="$ROOT/config/config.yaml" "$PYBIN" - <<'PY' 2>/dev/null || true
import os, yaml
with open(os.environ["CONFIG_YAML"]) as fh:
    cfg = yaml.safe_load(fh) or {}
opts = ((((cfg.get("providers") or {}).get("stt") or {}).get("options") or {}).get("databricks")) or {}
endpoints = []
base = opts.get("endpoint")
if base:
    endpoints.append(base)
for route in (opts.get("routes") or opts.get("language_routes") or {}).values():
    if isinstance(route, dict) and route.get("endpoint"):
        endpoints.append(route["endpoint"])
seen = []
for endpoint in endpoints:
    if endpoint and endpoint not in seen:
        seen.append(endpoint)
print(",".join(seen))
PY
)"

# Every configured language route is a hard dependency. Silently dropping one
# creates an app that deploys successfully but fails only for that language.
_FILTERED=""
IFS=',' read -ra _ASR_EPS <<< "$ASR_ENDPOINTS"
for _ep in "${_ASR_EPS[@]}"; do
  _ep="$(printf '%s' "$_ep" | xargs)"
  [[ -z "$_ep" ]] && continue
  if dbx serving-endpoints get "$_ep" >/dev/null 2>&1; then
    _FILTERED="${_FILTERED:+$_FILTERED,}$_ep"
  else
    die "Required ASR endpoint '$_ep' does not exist or is not accessible."
  fi
done
[[ -n "$_FILTERED" ]] || die "No valid ASR serving endpoints resolved to attach. Check config/config.yaml routes."
ASR_ENDPOINTS="$_FILTERED"
log "ASR endpoints to attach: $ASR_ENDPOINTS"

# ---- 3a. realtime voice endpoints (STT/TTS serving + FM model services) ------
# STT/TTS stay on serving endpoints (CAN_QUERY app resources). Foundation-model
# chat (llm / conversion / enrichment) uses Unity Catalog model services via Unity
# AI Gateway; those are EXECUTE-granted in grant_app_sp.py, not attached as
# serving_endpoint resources. Read from the DEPLOYED config (never config.local.yaml).
_VOICE_JSON="$(
  CONFIG_YAML="$ROOT/config/config.yaml" \
  PYTHONPATH="$ROOT/backend:$ROOT" "$PYBIN" - <<'PY'
import json, os, yaml
from genie_voice.databricks.ai_gateway import is_unity_model_service

with open(os.environ["CONFIG_YAML"]) as fh:
    cfg = yaml.safe_load(fh) or {}
rv = cfg.get("realtime_voice") or {}
enrichment = ((cfg.get("enrichment") or {}).get("model_endpoint") or "").strip()

serving, models = [], []
for name in (rv.get("llm_endpoint"), rv.get("conversion_endpoint"), rv.get("i18n_endpoint"), enrichment):
    if not name:
        continue
    (models if is_unity_model_service(name) else serving).append(name)
for group in ("stt_candidates", "tts_candidates"):
    for cand in (rv.get(group) or {}).values():
        if isinstance(cand, dict) and cand.get("endpoint"):
            serving.append(cand["endpoint"])

def _dedupe(items):
    seen, out = [], []
    for item in items:
        if item and item not in seen:
            seen.append(item)
            out.append(item)
    return out

print(json.dumps({
    "enrichment": enrichment,
    "serving": _dedupe(serving),
    "model_services": _dedupe(models),
}))
PY
)"
ENRICHMENT_MODEL_SERVICE="$(printf '%s' "$_VOICE_JSON" | "$PYBIN" -c 'import json,sys;print(json.load(sys.stdin).get("enrichment") or "")')"
MODEL_SERVICES="$(printf '%s' "$_VOICE_JSON" | "$PYBIN" -c 'import json,sys;print(",".join(json.load(sys.stdin).get("model_services") or []))')"
REALTIME_ENDPOINTS="$(printf '%s' "$_VOICE_JSON" | "$PYBIN" -c 'import json,sys;print(",".join(json.load(sys.stdin).get("serving") or []))')"

if [[ -n "$ENRICHMENT_MODEL_SERVICE" ]]; then
  if [[ "$ENRICHMENT_MODEL_SERVICE" == *.*.* ]]; then
    dbx api get "/api/2.1/unity-catalog/model-services/$ENRICHMENT_MODEL_SERVICE" >/dev/null 2>&1 \
      || die "Required enrichment model service '$ENRICHMENT_MODEL_SERVICE' does not exist or is not accessible."
  else
    dbx serving-endpoints get "$ENRICHMENT_MODEL_SERVICE" >/dev/null 2>&1 \
      || die "Required enrichment endpoint '$ENRICHMENT_MODEL_SERVICE' does not exist or is not accessible."
  fi
fi

IFS=',' read -ra _MS <<< "$MODEL_SERVICES"
for _ep in "${_MS[@]}"; do
  _ep="$(printf '%s' "$_ep" | xargs)"
  [[ -z "$_ep" ]] && continue
  dbx api get "/api/2.1/unity-catalog/model-services/$_ep" >/dev/null 2>&1 \
    || die "Required Unity model service '$_ep' does not exist or is not accessible."
done
[[ -n "$MODEL_SERVICES" ]] && log "Unity model services (Gateway): $MODEL_SERVICES"

_RT_FILTERED=""
IFS=',' read -ra _RT_EPS <<< "$REALTIME_ENDPOINTS"
for _ep in "${_RT_EPS[@]}"; do
  _ep="$(printf '%s' "$_ep" | xargs)"
  [[ -z "$_ep" ]] && continue
  if dbx serving-endpoints get "$_ep" >/dev/null 2>&1; then
    _RT_FILTERED="${_RT_FILTERED:+$_RT_FILTERED,}$_ep"
  else
    die "Required realtime serving endpoint '$_ep' does not exist or is not accessible."
  fi
done
REALTIME_ENDPOINTS="$_RT_FILTERED"
[[ -n "$REALTIME_ENDPOINTS" ]] || die "No realtime STT/TTS serving endpoints resolved from config/config.yaml."
log "Realtime serving endpoints to attach: $REALTIME_ENDPOINTS"

APP_NAME="$APP_NAME" SECRET_SCOPE="$SECRET_SCOPE" SQL_WAREHOUSE_ID="$SQL_WAREHOUSE_ID" \
ENRICHMENT_MODEL_SERVICE="$ENRICHMENT_MODEL_SERVICE" ASR_ENDPOINTS="$ASR_ENDPOINTS" \
REALTIME_ENDPOINTS="$REALTIME_ENDPOINTS" INCLUDE_EL="$INCLUDE_EL" \
PYTHONPATH="$ROOT/backend:$ROOT" "$PYBIN" - > "$APP_JSON" <<'PY'
import json, os
from genie_voice.databricks.ai_gateway import is_unity_model_service
res = [
    {"name": "sql-warehouse",    "sql_warehouse":    {"id": os.environ["SQL_WAREHOUSE_ID"], "permission": "CAN_USE"}},
]
enrichment = (os.environ.get("ENRICHMENT_MODEL_SERVICE") or "").strip()
if enrichment and not is_unity_model_service(enrichment):
    res.append({"name": "enrichment-model", "serving_endpoint": {"name": enrichment, "permission": "CAN_QUERY"}})

seen = set()
for idx, endpoint in enumerate(os.environ["ASR_ENDPOINTS"].split(","), start=1):
    endpoint = endpoint.strip()
    if not endpoint or endpoint in seen:
        continue
    seen.add(endpoint)
    res.append({
        "name": f"asr-endpoint-{idx}",
        "serving_endpoint": {"name": endpoint, "permission": "CAN_QUERY"},
    })
for idx, endpoint in enumerate(os.environ.get("REALTIME_ENDPOINTS", "").split(","), start=1):
    endpoint = endpoint.strip()
    if not endpoint or endpoint in seen:
        continue
    seen.add(endpoint)
    res.append({
        "name": f"realtime-endpoint-{idx}",
        "serving_endpoint": {"name": endpoint, "permission": "CAN_QUERY"},
    })
if os.environ.get("INCLUDE_EL") == "1":
    res.append({"name": "elevenlabs-api-key", "secret": {"scope": os.environ["SECRET_SCOPE"], "key": "elevenlabs_api_key", "permission": "READ"}})
print(json.dumps({
    "name": os.environ["APP_NAME"],
    "description": "Genie Voice Agent - contact-center voice intelligence",
    "resources": res,
    # app.yaml requests these at runtime; the Apps control plane must also grant
    # them explicitly or x-forwarded-access-token contains only default IAM scopes.
    "user_api_scopes": ["genie", "sql"],
}))
PY

if dbx apps get "$APP_NAME" >/dev/null 2>&1; then
  log "app exists - updating resources"
  dbx apps update "$APP_NAME" --json "@$APP_JSON" >/dev/null
else
  log "creating app '$APP_NAME' (provisions its service principal + compute)"
  dbx apps create --json "@$APP_JSON"
fi
rm -f "$APP_JSON"

# ---- 3b. grant the app's service principal its runtime access ----------------
# Resolve the app SP (client id + SCIM id), retrying while it provisions.
SP_CLIENT_ID=""; SP_ID=""
for _ in 1 2 3 4 5 6; do
  read -r SP_CLIENT_ID SP_ID < <(dbx apps get "$APP_NAME" -o json | python3 -c 'import sys,json;d=json.load(sys.stdin);print(d.get("service_principal_client_id") or "", d.get("service_principal_id") or "")' 2>/dev/null || true)
  [[ -n "$SP_CLIENT_ID" ]] && break
  sleep 5
done

# Ensure the SP has the 'workspace-access' entitlement (required for it to mint
# Lakebase Postgres OAuth tokens at runtime). Idempotent SCIM PATCH.
if [[ -n "$SP_ID" ]]; then
  log "ensuring 'workspace-access' entitlement on app SP (scim id $SP_ID)"
  ENT_JSON="$(mktemp)"
  printf '%s' '{"schemas":["urn:ietf:params:scim:api:messages:2.0:PatchOp"],"Operations":[{"op":"add","path":"entitlements","value":[{"value":"workspace-access"}]}]}' > "$ENT_JSON"
  if dbx api patch "/api/2.0/preview/scim/v2/ServicePrincipals/$SP_ID" --json "@$ENT_JSON" >/dev/null 2>&1; then
    log "workspace-access entitlement ok"
  else
    rm -f "$ENT_JSON"
    die "Could not set workspace-access on app SP $SP_CLIENT_ID. Grant it in Settings > Identity & access > Service principals, then rerun."
  fi
  rm -f "$ENT_JSON"
fi

# Grant UC + Lakebase + Genie access AS YOU (catalog/instance/space owner).
if [[ -n "$SP_CLIENT_ID" ]]; then
  log "granting app service principal ($SP_CLIENT_ID): UC + Lakebase + Genie + model services"
  PYTHONPATH=backend "$PYBIN" infra/apps/grant_app_sp.py --sp-client-id "$SP_CLIENT_ID" \
    --model-services "$MODEL_SERVICES" \
    --registered-models "$REGISTERED_VOICE_MODELS" \
    --run-users "$APP_EXTERNAL_USERS" \
    --run-groups "$APP_EXTERNAL_GROUPS" \
    || die "required app service-principal grants failed"
else
  die "Could not resolve the app service principal after app creation."
fi

# ---- 3c. re-assert external caller access (survives redeploys) --------------
# App permissions are stored separately from `apps update`/`deploy`, so a normal
# redeploy does NOT drop them. We still re-assert them here so external access is
# declared in this script and self-heals if an ACL edit ever removes it. We use
# `update-permissions` (an ADDITIVE PATCH) on purpose: `set-permissions` REPLACES
# the whole ACL and would wipe owner/admins/account-users.
grant_app_can_use() {  # $1 = ACL key (user_name|group_name|service_principal_name), $2 = value
  local acl
  acl="$(printf '{"access_control_list":[{"%s":"%s","permission_level":"CAN_USE"}]}' "$1" "$2")"
  if dbx apps update-permissions "$APP_NAME" --json "$acl" >/dev/null 2>&1; then
    log "granted CAN_USE on app: $2"
  else
    die "Could not grant CAN_USE to $2 (the deployer needs CAN_MANAGE on the app)."
  fi
}
if [[ -n "$APP_EXTERNAL_USERS" ]]; then
  IFS=',' read -ra _EXT_USERS <<< "$APP_EXTERNAL_USERS"
  for _u in "${_EXT_USERS[@]}"; do
    _u="$(printf '%s' "$_u" | xargs)"
    [[ -n "$_u" ]] && grant_app_can_use user_name "$_u"
  done
fi
if [[ -n "$APP_EXTERNAL_SPS" ]]; then
  IFS=',' read -ra _EXT_SPS <<< "$APP_EXTERNAL_SPS"
  for _sp in "${_EXT_SPS[@]}"; do
    _sp="$(printf '%s' "$_sp" | xargs)"
    [[ -n "$_sp" ]] && grant_app_can_use service_principal_name "$_sp"
  done
fi
if [[ -n "$APP_EXTERNAL_GROUPS" ]]; then
  IFS=',' read -ra _EXT_GROUPS <<< "$APP_EXTERNAL_GROUPS"
  for _group in "${_EXT_GROUPS[@]}"; do
    _group="$(printf '%s' "$_group" | xargs)"
    [[ -n "$_group" ]] && grant_app_can_use group_name "$_group"
  done
fi

# ---- 6. sync source to the workspace ---------------------------------------
log "syncing source -> $WORKSPACE_DIR (includes built SPA + Story Deck)"
dbx sync . "$WORKSPACE_DIR" \
  --include "api/app/static/**" \
  --include "story_deck/**"

# ---- 7. deploy --------------------------------------------------------------
log "deploying app version"
dbx apps deploy "$APP_NAME" --source-code-path "$WORKSPACE_DIR" --mode SNAPSHOT

APP_STATE="$(dbx apps get "$APP_NAME" -o json)"
APP_URL="$(printf '%s' "$APP_STATE" | python3 -c 'import sys,json;print(json.load(sys.stdin).get("url",""))')"
printf '%s' "$APP_STATE" | "$PYBIN" -c '
import json, sys
d = json.load(sys.stdin)
effective = set(d.get("effective_user_api_scopes") or [])
missing = {"genie", "sql"} - effective
if missing:
    print(
        "[app-deploy] WARNING: deployed app is missing effective OBO scopes: "
        + ", ".join(sorted(missing))
        + ". Enable User Authorization in the app settings; Setup will verify user consent."
    )
status = (d.get("app_status") or {}).get("state")
if status != "RUNNING":
    raise SystemExit(f"deployed app is not RUNNING (status={status!r})")
'
[[ -n "$APP_URL" ]] || die "deployment succeeded but the app URL is empty."

# Authenticated smoke test catches source-sync omissions, SPA catch-all masking a
# missing API route, dependency/import failures, and a broken realtime mount.
log "smoke-testing deployed HTTP surfaces"
APP_URL="$APP_URL" APP_TOKEN="$(dbx auth token -o json | "$PYBIN" -c 'import json,sys;print(json.load(sys.stdin)["access_token"])')" \
  "$PYBIN" - <<'PY'
import json
import os
import urllib.error
import urllib.request

base = os.environ["APP_URL"].rstrip("/")
headers = {"Authorization": f"Bearer {os.environ['APP_TOKEN']}"}
checks = {
    "/health": lambda body: body.get("status") == "ok",
    "/realtime/healthz": lambda body: body.get("status") == "ok",
    "/knowledge/corpus": lambda body: isinstance(body.get("topics"), list),
    "/realtime/v1/capabilities": lambda body: "speech-llm-toolassist-speech" in body,
    "/me": lambda body: body.get("authenticated") is True,
}
for route, valid in checks.items():
    request = urllib.request.Request(base + route, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            content_type = response.headers.get("content-type", "")
            if response.status != 200 or "json" not in content_type:
                raise RuntimeError(
                    f"{route}: status={response.status}, content-type={content_type!r}"
                )
            body = json.load(response)
    except (urllib.error.URLError, json.JSONDecodeError) as exc:
        raise SystemExit(f"deployed smoke test failed for {route}: {exc}") from exc
    if not valid(body):
        raise SystemExit(f"deployed smoke test returned an invalid payload for {route}")
    print(f"[app-deploy] smoke ok: {route}")

story_request = urllib.request.Request(base + "/story/", headers=headers)
try:
    with urllib.request.urlopen(story_request, timeout=30) as response:
        content_type = response.headers.get("content-type", "")
        page = response.read().decode("utf-8")
        if (
            response.status != 200
            or "text/html" not in content_type
            or "Genie for Voice" not in page
        ):
            raise RuntimeError(
                "story deck: "
                f"status={response.status}, content-type={content_type!r}"
            )
except (urllib.error.URLError, UnicodeDecodeError) as exc:
    raise SystemExit(f"deployed smoke test failed for /story/: {exc}") from exc
print("[app-deploy] smoke ok: /story/")
PY

# Final readiness snapshot from the DEPLOYED app: one source of truth shared with
# the in-app Setup page. Prints the same checklist so the operator sees exactly
# which UI-only steps (CDF start, Gateway policy attach, User Authorization) remain.
log "checking deployment readiness (GET /readiness)"
APP_URL="$APP_URL" APP_TOKEN="$(dbx auth token -o json | "$PYBIN" -c 'import json,sys;print(json.load(sys.stdin)["access_token"])')" \
  "$PYBIN" - <<'PY' || warn "readiness endpoint not reachable yet; open the Setup page in the app to check."
import json, os, urllib.request

base = os.environ["APP_URL"].rstrip("/")
req = urllib.request.Request(base + "/readiness?full=false", headers={"Authorization": f"Bearer {os.environ['APP_TOKEN']}"})
with urllib.request.urlopen(req, timeout=60) as resp:
    data = json.load(resp)
mark = {"ok": "OK  ", "warn": "WARN", "fail": "FAIL"}
for c in data.get("checks", []):
    print(f"  [{mark.get(c['status'], c['status'])}] {c['title']}: {c['detail']}")
s = data.get("summary", {})
print(f"[app-deploy] readiness: {s.get('ok',0)} ok / {s.get('warn',0)} warn / {s.get('fail',0)} fail"
      + ("  -> READY TO USE" if data.get("ready") else "  -> finish the WARN/FAIL items above, then re-check on the Setup page"))
PY

log "------------------------------------------------------------------"
log "deployed. app URL: ${APP_URL:-<see: databricks apps get $APP_NAME>}"
if [[ -n "$APP_URL" ]]; then
  log "realtime voice API: ${APP_URL%/}/realtime"
  log "MCP endpoint (remote MCP over HTTP): ${APP_URL%/}/realtime/mcp"
  log "  connect an MCP client with that URL + header 'Authorization: Bearer <databricks-token>'"
fi
if [[ -n "$APP_URL" ]]; then
  log "readiness / setup page: ${APP_URL%/}/#/setup"
fi
log "logs: Compute -> Apps -> $APP_NAME -> Logs"
log "------------------------------------------------------------------"
