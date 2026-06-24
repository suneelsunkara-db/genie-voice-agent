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

# Load .env first; fallback to .env.example for convenience.
if [[ -f .env ]]; then
  set -a; source .env; set +a
fi
if [[ -z "${DEEPGRAM_API_KEY:-}" && -f config/.env.example ]]; then
  DEEPGRAM_API_KEY="$(python3 - <<'PY'
from pathlib import Path
for line in Path("config/.env.example").read_text().splitlines():
    if line.startswith("DEEPGRAM_API_KEY="):
        print(line.split("=", 1)[1].strip())
        break
PY
)"
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

# Minimal env setup.
if [[ ! -d .venv ]]; then
  log "creating virtualenv"
  python3 -m venv .venv
fi
source .venv/bin/activate
if python -c "import genie_voice, fastapi, uvicorn" >/dev/null 2>&1; then
  :
else
  log "installing backend + api deps"
  pip install -q --upgrade pip || true
  pip install -q -e backend
  pip install -q -r api/requirements.txt
fi

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
log "starting API on :$API_PORT"
( cd "$ROOT/api" && "$ROOT/.venv/bin/uvicorn" app.main:app --host 0.0.0.0 --port "$API_PORT" \
    > "$RUN_DIR/api.log" 2>&1 ) &
echo $! > "$RUN_DIR/api.pid"

log "starting frontend on :5173"
( cd "$ROOT/frontend" && VITE_API_BASE_URL="http://localhost:$API_PORT" npm run dev -- --host 0.0.0.0 --port 5173 \
    > "$RUN_DIR/frontend.log" 2>&1 ) &
echo $! > "$RUN_DIR/frontend.pid"

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
wait_http_ok "http://localhost:5173" "UI" 30 || {
  err "Check $RUN_DIR/frontend.log"
  exit 1
}
log "logs: $RUN_DIR/{api,frontend}.log"

API_PID="$(tr -d '[:space:]' < "$RUN_DIR/api.pid")"
FRONTEND_PID="$(tr -d '[:space:]' < "$RUN_DIR/frontend.pid")"

cleanup() {
  [[ -n "${API_PID:-}" ]] && kill "$API_PID" >/dev/null 2>&1 || true
  [[ -n "${FRONTEND_PID:-}" ]] && kill "$FRONTEND_PID" >/dev/null 2>&1 || true
}
trap cleanup INT TERM EXIT

log "services running (api pid=$API_PID, frontend pid=$FRONTEND_PID). press Ctrl+C to stop."
wait "$API_PID" "$FRONTEND_PID"

