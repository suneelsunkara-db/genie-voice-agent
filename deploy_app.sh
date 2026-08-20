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
#      service principal is auto-granted access on deploy, then re-assert CAN_USE
#      for external callers (see APP_EXTERNAL_* below) so cross-workspace access
#      survives every redeploy
#   4. sync source to a workspace folder (respects .gitignore) — this includes
#      the mcp_server/ package, which the app hosts in-process at /realtime/mcp
#      (a remote MCP endpoint; `mcp` is declared in requirements.txt). No extra
#      deploy step is needed: syncing the source + the mount in api/app/main.py
#      deploy and update the MCP server together with the app.
#   5. create the app (first run) and deploy it
#
# Config: edit the block below OR export the same vars before running.
# Auth:   uses your Databricks CLI profile (U2M) to DEPLOY; the running app uses
#         its own service principal for hosting / serving. Genie workspace Q&A
#         prefers the viewer's OBO token (x-forwarded-access-token) when Apps
#         user authorization is enabled with genie (+ sql) scopes — see app.yaml
#         and README "User authorization (OBO)".
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

# External principals (re-)granted CAN_USE on the app every deploy so external /
# cross-workspace API callers keep access across redeploys. Comma-separated;
# leave empty to skip. Grants are ADDITIVE (owner/admins/account-users untouched).
#   APP_EXTERNAL_USERS -> user_name (email) of an account user
#   APP_EXTERNAL_SPS   -> service_principal_name (OAuth client / application id)
APP_EXTERNAL_USERS="${APP_EXTERNAL_USERS:-j.wang@databricks.com}"
APP_EXTERNAL_SPS="${APP_EXTERNAL_SPS:-705947df-7bea-415f-af6a-4642a43ba1be}"

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

log "verifying warehouse and required serving endpoints"
dbx warehouses get "$SQL_WAREHOUSE_ID" >/dev/null 2>&1 \
  || die "SQL warehouse '$SQL_WAREHOUSE_ID' does not exist or is not accessible."
dbx serving-endpoints get "$CLAUDE_ENDPOINT" >/dev/null 2>&1 \
  || die "Required enrichment endpoint '$CLAUDE_ENDPOINT' does not exist or is not accessible."

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
# Derive the ASR endpoints to attach as app resources from the DEPLOYED config
# (config/config.yaml) - NOT get_settings(), which deep-merges the gitignored
# config.local.yaml. Local dev overrides (e.g. a personal base endpoint) must not
# leak into the app's granted resources, since config.local.yaml is never synced.
if [[ -z "$ASR_ENDPOINTS" ]]; then
  ASR_ENDPOINTS="$(
    CONFIG_YAML="$ROOT/config/config.yaml" WHISPER_ENDPOINT="$WHISPER_ENDPOINT" "$PYBIN" - <<'PY' 2>/dev/null || true
import os, yaml
with open(os.environ["CONFIG_YAML"]) as fh:
    cfg = yaml.safe_load(fh) or {}
opts = ((((cfg.get("providers") or {}).get("stt") or {}).get("options") or {}).get("databricks")) or {}
endpoints = []
base = opts.get("endpoint") or os.environ.get("WHISPER_ENDPOINT")
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
fi
ASR_ENDPOINTS="${ASR_ENDPOINTS:-$WHISPER_ENDPOINT}"

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

# ---- 3a. realtime voice endpoints (STT/LLM/TTS) as app resources ------------
# The Realtime Voice API is mounted at /realtime in the app; its Databricks
# serving endpoints (from the realtime_voice: block in config/config.yaml) need
# CAN_QUERY granted to the app service principal. Read from the DEPLOYED config
# (never config.local.yaml, which is not synced). All configured endpoints are
# required: a partial voice deployment is not a successful deployment.
REALTIME_ENDPOINTS="${REALTIME_ENDPOINTS:-}"
if [[ -z "$REALTIME_ENDPOINTS" ]]; then
  REALTIME_ENDPOINTS="$(
    CONFIG_YAML="$ROOT/config/config.yaml" "$PYBIN" - <<'PY' 2>/dev/null || true
import os, yaml
with open(os.environ["CONFIG_YAML"]) as fh:
    cfg = yaml.safe_load(fh) or {}
rv = cfg.get("realtime_voice") or {}
endpoints = []
if rv.get("llm_endpoint"):
    endpoints.append(rv["llm_endpoint"])
# Runtime text->text conversion lane (Agent-Mode deep-dive spoken "why" summary +
# on-screen report translation). A DISTINCT endpoint (e.g. gpt-5-5) the app SP must
# be able to CAN_QUERY -- without this grant deep dives can neither localize the
# report nor speak the summary (both silently fail on the deployed app).
if rv.get("conversion_endpoint"):
    endpoints.append(rv["conversion_endpoint"])
for group in ("stt_candidates", "tts_candidates"):
    for cand in (rv.get(group) or {}).values():
        if isinstance(cand, dict) and cand.get("endpoint"):
            endpoints.append(cand["endpoint"])
seen = []
for e in endpoints:
    if e and e not in seen:
        seen.append(e)
print(",".join(seen))
PY
  )"
fi
_RT_FILTERED=""
IFS=',' read -ra _RT_EPS <<< "$REALTIME_ENDPOINTS"
for _ep in "${_RT_EPS[@]}"; do
  _ep="$(printf '%s' "$_ep" | xargs)"
  [[ -z "$_ep" ]] && continue
  if dbx serving-endpoints get "$_ep" >/dev/null 2>&1; then
    _RT_FILTERED="${_RT_FILTERED:+$_RT_FILTERED,}$_ep"
  else
    die "Required realtime endpoint '$_ep' does not exist or is not accessible."
  fi
done
REALTIME_ENDPOINTS="$_RT_FILTERED"
[[ -n "$REALTIME_ENDPOINTS" ]] || die "No realtime voice endpoints resolved from config/config.yaml."
log "Realtime endpoints to attach: $REALTIME_ENDPOINTS"

APP_NAME="$APP_NAME" SECRET_SCOPE="$SECRET_SCOPE" SQL_WAREHOUSE_ID="$SQL_WAREHOUSE_ID" \
CLAUDE_ENDPOINT="$CLAUDE_ENDPOINT" WHISPER_ENDPOINT="$WHISPER_ENDPOINT" ASR_ENDPOINTS="$ASR_ENDPOINTS" \
REALTIME_ENDPOINTS="$REALTIME_ENDPOINTS" INCLUDE_EL="$INCLUDE_EL" \
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

# ---- 3c. re-assert external caller access (survives redeploys) --------------
# App permissions are stored separately from `apps update`/`deploy`, so a normal
# redeploy does NOT drop them. We still re-assert them here so external access is
# declared in this script and self-heals if an ACL edit ever removes it. We use
# `update-permissions` (an ADDITIVE PATCH) on purpose: `set-permissions` REPLACES
# the whole ACL and would wipe owner/admins/account-users.
grant_app_can_use() {  # $1 = ACL key (user_name|service_principal_name), $2 = value
  local acl
  acl="$(printf '{"access_control_list":[{"%s":"%s","permission_level":"CAN_USE"}]}' "$1" "$2")"
  if dbx apps update-permissions "$APP_NAME" --json "$acl" >/dev/null 2>&1; then
    log "granted CAN_USE on app: $2"
  else
    warn "could not grant CAN_USE to $2 (need CAN_MANAGE on the app; skipping)."
  fi
}
IFS=',' read -ra _EXT_USERS <<< "$APP_EXTERNAL_USERS"
for _u in "${_EXT_USERS[@]}"; do
  _u="$(printf '%s' "$_u" | xargs)"
  [[ -n "$_u" ]] && grant_app_can_use user_name "$_u"
done
IFS=',' read -ra _EXT_SPS <<< "$APP_EXTERNAL_SPS"
for _sp in "${_EXT_SPS[@]}"; do
  _sp="$(printf '%s' "$_sp" | xargs)"
  [[ -n "$_sp" ]] && grant_app_can_use service_principal_name "$_sp"
done

# ---- 4. sync source to the workspace ---------------------------------------
log "syncing source -> $WORKSPACE_DIR (respects .gitignore, includes built SPA)"
dbx sync . "$WORKSPACE_DIR" --include "api/app/static/**"

# ---- 5. deploy --------------------------------------------------------------
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
    raise SystemExit(
        "deployed app is missing effective OBO scopes: "
        + ", ".join(sorted(missing))
        + ". Enable User Authorization in the workspace/app, then redeploy."
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
PY

log "------------------------------------------------------------------"
log "deployed. app URL: ${APP_URL:-<see: databricks apps get $APP_NAME>}"
if [[ -n "$APP_URL" ]]; then
  log "realtime voice API: ${APP_URL%/}/realtime"
  log "MCP endpoint (remote MCP over HTTP): ${APP_URL%/}/realtime/mcp"
  log "  connect an MCP client with that URL + header 'Authorization: Bearer <databricks-token>'"
fi
log "logs: Compute -> Apps -> $APP_NAME -> Logs"
log "------------------------------------------------------------------"
