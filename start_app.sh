#!/usr/bin/env bash
# =============================================================================
# start_app.sh
#
# Start API + UI only (no deploy jobs, no Databricks orchestration).
# Loads Deepgram key, runs a cheap auth check, then launches app services.
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
RUN_DIR="$ROOT/.run"
mkdir -p "$RUN_DIR"

log() { printf "\033[35m[start-app]\033[0m %s\n" "$*"; }
warn() { printf "\033[33m[start-app]\033[0m %s\n" "$*"; }
err() { printf "\033[31m[start-app]\033[0m %s\n" "$*"; }

# Minimal env setup.
if [[ ! -d .venv ]]; then
  log "creating virtualenv"
  python3 -m venv .venv
fi
source .venv/bin/activate
# Public PyPI is not reachable from this network; install from the internal
# Databricks PyPI proxy. Override by exporting PIP_INDEX_URL before running.
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi-proxy.cloud.databricks.com/simple}"
if python -c "import genie_voice, fastapi, uvicorn, psycopg_pool" >/dev/null 2>&1; then
  :
else
  log "installing backend + api deps (index: $PIP_INDEX_URL)"
  pip install -q --upgrade pip || true
  pip install -q -e backend
  pip install -q -r api/requirements.txt
fi

# Load .env if present (env overrides config.local.yaml secrets).
if [[ -f .env ]]; then
  set -a; source .env; set +a
fi
if [[ -z "${DEEPGRAM_API_KEY:-}" ]]; then
  DEEPGRAM_API_KEY="$(PYTHONPATH="$ROOT/backend" python -c "
from genie_voice.config import get_settings
get_settings.cache_clear()
print(get_settings().secrets.deepgram_api_key or '')
" 2>/dev/null || true)"
  export DEEPGRAM_API_KEY
fi

# Live mode so mic -> Deepgram path is active.
if [[ -n "${DEEPGRAM_API_KEY:-}" ]]; then
  export GENIE_DEPLOYMENT=live
  code="$(curl -sS -o /tmp/dg_projects.json -w "%{http_code}" \
    -H "Authorization: Token ${DEEPGRAM_API_KEY}" \
    "https://api.deepgram.com/v1/projects" || true)"
  if [[ "$code" != "200" ]]; then
    err "Deepgram key check failed (HTTP $code)."
    exit 1
  fi
  log "Deepgram auth check passed."
else
  warn "DEEPGRAM_API_KEY not found; mic transcription will fail."
fi

# --- Databricks auth (re-auth if the cached OAuth U2M token is invalid) ------
# The cockpit's data comes from Databricks (Lakebase / UC / Genie). A stale OAuth
# token makes those calls 500, which the browser surfaces as "Failed to fetch".
# Probe the same credential chain the app uses and, for U2M (auth_type=default),
# launch an interactive 'databricks auth login' when the token is expired.
probe_databricks_auth() {
  PYTHONPATH="$ROOT/backend" python - <<'PY' >/dev/null 2>&1
from genie_voice.config import get_settings
from genie_voice.databricks.client import get_workspace_client
get_settings.cache_clear()
get_workspace_client(get_settings()).current_user.me()
PY
}

ensure_databricks_auth() {
  local cfg auth_type profile host
  cfg="$(PYTHONPATH="$ROOT/backend" python - <<'PY' 2>/dev/null || true
from genie_voice.config import get_settings
get_settings.cache_clear()
s = get_settings()
print(s.databricks.auth_type or "default")
print(s.databricks.profile or "")
print(s.databricks_host or "")
PY
)"
  auth_type="$(printf '%s\n' "$cfg" | sed -n 1p)"
  profile="$(printf '%s\n' "$cfg" | sed -n 2p)"
  host="$(printf '%s\n' "$cfg" | sed -n 3p)"

  # Only U2M OAuth ("default") uses an interactive browser login; pat / M2M oauth
  # carry their own credentials and need no login step here.
  if [[ "$auth_type" != "default" ]]; then
    return 0
  fi

  if probe_databricks_auth; then
    log "Databricks auth check passed."
    return 0
  fi

  warn "Databricks auth invalid/expired - launching 'databricks auth login'."
  if ! command -v databricks >/dev/null 2>&1; then
    warn "databricks CLI not found; run 'databricks auth login' manually, then re-run."
    return 0
  fi
  if [[ -n "$profile" ]]; then
    databricks auth login --profile "$profile" || true
  elif [[ -n "$host" ]]; then
    databricks auth login --host "$host" || true
  else
    databricks auth login || true
  fi

  if probe_databricks_auth; then
    log "Databricks auth re-established."
  else
    warn "Databricks auth still failing; Databricks-backed views may error until fixed."
  fi
  return 0
}
ensure_databricks_auth

# Stop previous API/UI if our pid files exist.
stop_pid_file() {
  local f="$1"
  if [[ -f "$f" ]]; then
    local p
    p="$(tr -d '[:space:]' < "$f")"
    if [[ -n "$p" ]] && ps -p "$p" >/dev/null 2>&1; then
      kill "$p" || true
    fi
    rm -f "$f"
  fi
}
stop_pid_file "$RUN_DIR/api.pid"
stop_pid_file "$RUN_DIR/frontend.pid"

API_PORT="${GENIE_API_PORT:-8000}"
UI_PORT="${GENIE_UI_PORT:-5173}"

# Launch each service as a real background child of THIS shell so `wait` can
# track it and Ctrl+C tears it down. `( cd … && exec … ) &` backgrounds the
# subshell; `exec` replaces it with the server, so `$!` is the server's own pid
# (kill/wait target). No nohup/disown: this launcher is foreground by design
# (see "press Ctrl+C to stop" below) — detaching broke `wait` and made the EXIT
# trap kill the servers the instant they started.
log "starting API on :$API_PORT"
( cd "$ROOT/api" && exec "$ROOT/.venv/bin/uvicorn" app.main:app --host 0.0.0.0 --port "$API_PORT" ) \
  > "$RUN_DIR/api.log" 2>&1 &
API_PID=$!
echo "$API_PID" > "$RUN_DIR/api.pid"

log "starting frontend on :$UI_PORT"
( cd "$ROOT/frontend" && exec env VITE_API_BASE_URL="http://localhost:$API_PORT" npx vite --host 0.0.0.0 --port "$UI_PORT" --strictPort ) \
  > "$RUN_DIR/frontend.log" 2>&1 &
FRONTEND_PID=$!
echo "$FRONTEND_PID" > "$RUN_DIR/frontend.pid"

# Register cleanup now (before the readiness waits) so a failed health check or
# any early exit also tears down whatever already started.
cleanup() {
  [[ -n "${API_PID:-}" ]] && kill "$API_PID" >/dev/null 2>&1 || true
  [[ -n "${FRONTEND_PID:-}" ]] && kill "$FRONTEND_PID" >/dev/null 2>&1 || true
  rm -f "$RUN_DIR/api.pid" "$RUN_DIR/frontend.pid"
}
trap cleanup INT TERM EXIT

wait_http_ok() {
  local url="$1"
  local label="$2"
  local timeout_s="${3:-45}"
  local i=0
  while (( i < timeout_s )); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      log "$label: $url"
      return 0
    fi
    sleep 1
    ((i+=1))
  done
  err "$label did not become reachable within ${timeout_s}s."
  return 1
}

wait_http_ok "http://localhost:$API_PORT/health" "API" 60 || {
  err "Check $RUN_DIR/api.log"
  exit 1
}
wait_http_ok "http://localhost:$UI_PORT" "UI" 30 || {
  err "Check $RUN_DIR/frontend.log"
  exit 1
}
log "logs: $RUN_DIR/{api,frontend}.log"

log "services running (api pid=$API_PID, frontend pid=$FRONTEND_PID). press Ctrl+C to stop."
wait "$API_PID" "$FRONTEND_PID"

