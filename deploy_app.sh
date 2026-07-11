#!/usr/bin/env bash
# =============================================================================
# Genie Voice Agent - deploy to Databricks Apps.
#
# Turns the local two-process app (Vite + uvicorn) into a single Databricks App
# web process: the built React SPA is served by FastAPI from api/app/static, and
# the app authenticates as its injected service principal.
#
# Steps:
#   1. build the frontend for same-origin (VITE_API_BASE_URL="") -> api/app/static
#   2. push vendor API keys into a Databricks secret scope
#   3. attach app resources (warehouse + secrets + serving endpoints) so the app
#      service principal is auto-granted access on deploy
#   4. sync source to a workspace folder (respects .gitignore)
#   5. create the app (first run) and deploy it
#
# Config: edit the block below OR export the same vars before running.
# Auth:   uses your Databricks CLI profile (U2M) to DEPLOY; the running app uses
#         its own service principal.
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# ---- config (defaults from config/config.local.yaml; override via env) ------
APP_NAME="${APP_NAME:-genie-voice-agent}"
DATABRICKS_PROFILE="${DATABRICKS_PROFILE:-fe-vm-vdm-classic-rcn6ip}"  # ~/.databrickscfg profile
SECRET_SCOPE="${SECRET_SCOPE:-genie-voice}"           # scope holding vendor keys
SQL_WAREHOUSE_ID="${SQL_WAREHOUSE_ID:-d0a0a25efd015c58}"  # serving warehouse
CLAUDE_ENDPOINT="${CLAUDE_ENDPOINT:-databricks-claude-opus-4-8}"
WHISPER_ENDPOINT="${WHISPER_ENDPOINT:-voice_asr_en_finetuned_whisper_lora}"
WORKSPACE_DIR="${WORKSPACE_DIR:-}"                    # empty -> /Workspace/Users/<me>/<app>
# Optional comma-separated override. By default this is derived from the
# configured Databricks STT endpoint + per-language routes.
ASR_ENDPOINTS="${ASR_ENDPOINTS:-}"

log()  { printf "\033[36m[app-deploy]\033[0m %s\n" "$*"; }
warn() { printf "\033[33m[app-deploy]\033[0m %s\n" "$*"; }
die()  { printf "\033[31m[app-deploy]\033[0m %s\n" "$*"; exit 1; }

dbx() { if [[ -n "$DATABRICKS_PROFILE" ]]; then databricks "$@" -p "$DATABRICKS_PROFILE"; else databricks "$@"; fi; }

# Prefer the repo venv python (has backend deps installed) over system python3.
PYBIN="python3"
[[ -x "$ROOT/.venv/bin/python" ]] && PYBIN="$ROOT/.venv/bin/python"

command -v databricks >/dev/null 2>&1 || die "Databricks CLI not found (brew install databricks)."
command -v npm >/dev/null 2>&1 || die "npm not found (needed to build the frontend)."
[[ -n "$SQL_WAREHOUSE_ID" ]] || die "Set SQL_WAREHOUSE_ID (the serving SQL warehouse id)."

dbx current-user me >/dev/null 2>&1 || die "Not authenticated. Run: databricks auth login ${DATABRICKS_PROFILE:+--profile $DATABRICKS_PROFILE}"
ME="$(dbx current-user me -o json | python3 -c 'import sys,json;print(json.load(sys.stdin)["userName"])')"
WORKSPACE_DIR="${WORKSPACE_DIR:-/Workspace/Users/$ME/$APP_NAME}"
log "deploying as: $ME  ->  app '$APP_NAME'  (source: $WORKSPACE_DIR)"

# ---- 1. build the frontend for same-origin ---------------------------------
log "building frontend (same-origin) -> api/app/static"
( cd frontend && npm install --silent && VITE_API_BASE_URL="" npm run build )
rm -rf api/app/static
mkdir -p api/app/static
cp -R frontend/dist/. api/app/static/
[[ -f api/app/static/index.html ]] || die "frontend build missing index.html"

# ---- 2. vendor keys -> secret scope ----------------------------------------
DEEPGRAM_API_KEY="${DEEPGRAM_API_KEY:-$(PYTHONPATH=backend "$PYBIN" -c 'from genie_voice.config import get_settings;print(get_settings().secrets.deepgram_api_key)' 2>/dev/null || true)}"
ELEVENLABS_API_KEY="${ELEVENLABS_API_KEY:-$(PYTHONPATH=backend "$PYBIN" -c 'from genie_voice.config import get_settings;print(get_settings().secrets.elevenlabs_api_key)' 2>/dev/null || true)}"
[[ -n "$DEEPGRAM_API_KEY" ]] || die "DEEPGRAM_API_KEY not found (env or config.local.yaml). Required for mic STT."

log "ensuring secret scope '$SECRET_SCOPE'"
dbx secrets create-scope "$SECRET_SCOPE" >/dev/null 2>&1 || true
dbx secrets put-secret "$SECRET_SCOPE" deepgram_api_key --string-value "$DEEPGRAM_API_KEY"
if [[ -n "$ELEVENLABS_API_KEY" ]]; then
  dbx secrets put-secret "$SECRET_SCOPE" elevenlabs_api_key --string-value "$ELEVENLABS_API_KEY"
  log "stored elevenlabs_api_key"
else
  warn "ELEVENLABS_API_KEY empty - skipping (TTS optional; not in the live flow)."
fi

# ---- 3. app resource spec (auto-grants the app SP on deploy) ----------------
APP_JSON="$(mktemp)"
INCLUDE_EL="$([[ -n "$ELEVENLABS_API_KEY" ]] && echo 1 || echo 0)"
if [[ -z "$ASR_ENDPOINTS" ]]; then
  ASR_ENDPOINTS="$(
    PYTHONPATH=backend "$PYBIN" - <<PY 2>/dev/null || true
from genie_voice.config import get_settings

s = get_settings()
opts = s.providers.stt.options.get("databricks", {})
endpoints = []
base = opts.get("endpoint") or "$WHISPER_ENDPOINT"
if base:
    endpoints.append(base)
for route in (opts.get("routes") or opts.get("language_routes") or {}).values():
    if isinstance(route, dict) and route.get("endpoint"):
        endpoints.append(route["endpoint"])
seen = []
for endpoint in endpoints:
    if endpoint not in seen:
        seen.append(endpoint)
print(",".join(seen))
PY
  )"
fi
ASR_ENDPOINTS="${ASR_ENDPOINTS:-$WHISPER_ENDPOINT}"

APP_NAME="$APP_NAME" SECRET_SCOPE="$SECRET_SCOPE" SQL_WAREHOUSE_ID="$SQL_WAREHOUSE_ID" \
CLAUDE_ENDPOINT="$CLAUDE_ENDPOINT" WHISPER_ENDPOINT="$WHISPER_ENDPOINT" ASR_ENDPOINTS="$ASR_ENDPOINTS" INCLUDE_EL="$INCLUDE_EL" \
"$PYBIN" - > "$APP_JSON" <<'PY'
import json, os
res = [
    {"name": "sql-warehouse",    "sql_warehouse":    {"id": os.environ["SQL_WAREHOUSE_ID"], "permission": "CAN_USE"}},
    {"name": "deepgram-api-key", "secret":           {"scope": os.environ["SECRET_SCOPE"], "key": "deepgram_api_key", "permission": "READ"}},
    {"name": "claude-endpoint",  "serving_endpoint": {"name": os.environ["CLAUDE_ENDPOINT"], "permission": "CAN_QUERY"}},
]
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
if os.environ.get("INCLUDE_EL") == "1":
    res.append({"name": "elevenlabs-api-key", "secret": {"scope": os.environ["SECRET_SCOPE"], "key": "elevenlabs_api_key", "permission": "READ"}})
print(json.dumps({
    "name": os.environ["APP_NAME"],
    "description": "Genie Voice Agent - contact-center voice intelligence",
    "resources": res,
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
    warn "could not set workspace-access entitlement automatically."
    warn "  If Lakebase auth fails, enable it: Settings > Identity & access > Service principals > $SP_CLIENT_ID > Configurations."
  fi
  rm -f "$ENT_JSON"
fi

# Grant UC + Lakebase + Genie access AS YOU (catalog/instance/space owner).
if [[ -n "$SP_CLIENT_ID" ]]; then
  log "granting app service principal ($SP_CLIENT_ID): UC + Lakebase + Genie"
  PYTHONPATH=backend "$PYBIN" infra/apps/grant_app_sp.py --sp-client-id "$SP_CLIENT_ID" \
    || warn "some grants failed - review output above and re-run infra/apps/grant_app_sp.py"
else
  warn "could not resolve app service principal id; after deploy run:"
  warn "  PYTHONPATH=backend python3 infra/apps/grant_app_sp.py --sp-client-id <id>"
fi

# ---- 4. sync source to the workspace ---------------------------------------
log "syncing source -> $WORKSPACE_DIR (respects .gitignore, includes built SPA)"
dbx sync . "$WORKSPACE_DIR" --include "api/app/static/**"

# ---- 5. deploy --------------------------------------------------------------
log "deploying app version"
dbx apps deploy "$APP_NAME" --source-code-path "$WORKSPACE_DIR" --mode SNAPSHOT

APP_URL="$(dbx apps get "$APP_NAME" -o json | python3 -c 'import sys,json;print(json.load(sys.stdin).get("url",""))' 2>/dev/null || true)"
log "------------------------------------------------------------------"
log "deployed. app URL: ${APP_URL:-<see: databricks apps get $APP_NAME>}"
log "logs: Compute -> Apps -> $APP_NAME -> Logs"
log "------------------------------------------------------------------"
