#!/usr/bin/env bash
# =============================================================================
# Genie Voice Agent - local end-to-end deploy.
#
# Auth model: OAuth U2M (runs AS you, e.g. suneel.sunkara@databricks.com). No
# PAT, no secrets in .env. Lakebase is the low-latency serving path; UC is the
# asynchronous analytics path. Steps:
#   1. install backend + api (venv) and frontend (npm)
#   2. databricks auth login (U2M) if not already authenticated
#   3. optional --reset: drop generated UC/Lakebase tables while keeping Volume data
#   4. bootstrap: schema + Volume + typed tables (DDL) + GRANTs
#   5. produce reference table files for UC analytics
#   6. produce voice STT events for UC analytics
#   7. prepare Lakebase serving project + schema + seed primary serving tables
#   8. ingest:  ONLINE -> deploy + run reference UC ingest + Lakebase ingest,
#               CDF sync check, gold refresh, constraints, DQ, Genie
#               OFFLINE -> emulate the whole pipeline in-process (enrich.derive)
#   9. start UI
#
# Everything is config-driven (config/config.yaml + .env). No hardcoded vendors.
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
RUN_DIR="$ROOT/.run"
mkdir -p "$RUN_DIR"

log() { printf "\033[36m[deploy]\033[0m %s\n" "$*"; }

HOST="${DATABRICKS_HOST:-https://fe-vm-vdm-classic-rcn6ip.cloud.databricks.com}"
RESET="${GENIE_RESET:-0}"

usage() {
  cat <<'EOF'
Usage: ./local-deploy.sh [--reset]

Options:
  --reset   Drop all UC tables/views in the configured demo schema and Lakebase
            serving tables. UC Volume data is preserved and reused.

Environment:
  GENIE_RESET=1  Same as --reset.
EOF
}

for arg in "$@"; do
  case "$arg" in
    --reset) RESET=1 ;;
    -h|--help) usage; exit 0 ;;
    *) log "unknown argument: $arg"; usage; exit 2 ;;
  esac
done

case "$(printf '%s' "$RESET" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes) RESET=1 ;;
  *) RESET=0 ;;
esac

# ---- 0. env -----------------------------------------------------------------
if [[ -f .env ]]; then
  log "loading .env"
  set -a; source .env; set +a
else
  log ".env not found - that's OK for U2M (no secrets needed)"
fi

# ---- 1. python env ----------------------------------------------------------
if [[ ! -d .venv ]]; then
  log "creating virtualenv"
  python3 -m venv .venv
fi
source .venv/bin/activate

# Install deps only when they're missing. In locked-down environments PyPI is
# unreachable; if the packages are already importable we skip install so the rest
# of the deploy still runs (and we don't abort the whole script on a pip failure).
if python -c "import genie_voice, fastapi, uvicorn, databricks.sdk" >/dev/null 2>&1; then
  log "python deps already present - skipping install"
else
  log "installing backend + api"
  pip install -q --upgrade pip || log "pip upgrade skipped (no PyPI?)"
  pip install -q -e backend || log "backend install failed (no PyPI?) - continuing if importable"
  pip install -q -r api/requirements.txt || log "api deps install failed (no PyPI?)"
  python -c "import genie_voice" >/dev/null 2>&1 \
    || { log "FATAL: genie_voice not importable and cannot install (no PyPI access)"; exit 1; }
fi

# ---- 2. Databricks CLI + U2M login -----------------------------------------
# Pin the SAME ~/.databrickscfg profile the Python code uses (config.databricks
# .profile). Multiple profiles can match one host, which otherwise makes the CLI
# fail with "must specify --profile"; the wrapper below appends it everywhere.
PROFILE="$(python -c "import sys;sys.path.insert(0,'backend');from genie_voice.config import get_settings;print(get_settings().databricks.profile or '')" 2>/dev/null || true)"
dbx() { if [[ -n "$PROFILE" ]]; then databricks "$@" -p "$PROFILE"; else databricks "$@"; fi; }

ONLINE=0
if ! command -v databricks >/dev/null 2>&1; then
  log "Databricks CLI not found - attempting 'brew install databricks'"
  brew install databricks >/dev/null 2>&1 || log "could not auto-install CLI"
fi

if command -v databricks >/dev/null 2>&1; then
  if dbx current-user me >/dev/null 2>&1; then
    ONLINE=1
  else
    log "not authenticated - launching OAuth U2M login (browser) for $HOST (profile ${PROFILE:-default})"
    databricks auth login --host "$HOST" ${PROFILE:+--profile "$PROFILE"} || log "auth login failed/cancelled"
    if dbx current-user me >/dev/null 2>&1; then ONLINE=1; fi
  fi
fi

if [[ "$ONLINE" -eq 1 ]]; then
  WHOAMI="$(dbx current-user me 2>/dev/null | python -c 'import sys,json;print(json.load(sys.stdin).get("userName",""))' 2>/dev/null || true)"
  log "authenticated as: ${WHOAMI:-unknown}"

  # Auto-discover a SQL warehouse if one isn't configured (or env-provided).
  CFG_WH="$(python -c "import sys;sys.path.insert(0,'backend');from genie_voice.config import get_settings;print(get_settings().databricks.sql_warehouse_id)" 2>/dev/null || true)"
  if [[ -z "$CFG_WH" && -z "${GENIE_DATABRICKS__SQL_WAREHOUSE_ID:-}" ]]; then
    WH_ID="$(dbx warehouses list -o json 2>/dev/null | python -c "import sys,json; d=json.load(sys.stdin); ws=d if isinstance(d,list) else d.get('warehouses',[]); run=[w for w in ws if str(w.get('state','')).upper()=='RUNNING']; pick=(run or ws); print(pick[0]['id'] if pick else '')" 2>/dev/null || true)"
    if [[ -n "$WH_ID" ]]; then
      export GENIE_DATABRICKS__SQL_WAREHOUSE_ID="$WH_ID"
      log "auto-selected SQL warehouse: $WH_ID"
    else
      log "WARNING: no SQL warehouse configured/found - Databricks steps will be skipped"
    fi
  fi
else
  log "no Databricks auth -> OFFLINE mode (local volume dir + in-memory Lakebase)"
  export GENIE_LOCAL_VOLUME_DIR="$RUN_DIR/volume/raw_stt"
  export GENIE_LAKEBASE__ENABLED=false
fi

# ---- 3. optional clean reset -------------------------------------------------
if [[ "$RESET" -eq 1 ]]; then
  if [[ "$ONLINE" -eq 1 ]]; then
    log "RESET: dropping demo UC/Lakebase tables while keeping UC Volume data"
    python -m genie_voice.databricks.reset --keep-volumes
  else
    log "RESET: offline mode - clearing local generated data"
    rm -rf "$RUN_DIR/volume"
    mkdir -p "$RUN_DIR"
  fi
fi

# ---- 4. bootstrap UC objects + permissions ---------------------------------
if [[ "$ONLINE" -eq 1 ]]; then
  log "bootstrapping schema + raw_batch_data/raw_streaming_data Volumes + GRANTs (existing catalog)"
  GENIE_SKIP_TABLE_BOOTSTRAP=true python -m genie_voice.databricks.bootstrap || log "bootstrap failed (check sql_warehouse_id)"
fi

# ---- 5. produce demo/audit files into raw_batch_data -----------------------
# Reference/customer/billing files are batch-ingested into UC Delta.
if python -c "import sys;sys.path.insert(0,'backend');from genie_voice.databricks.volume_state import reference_inputs_present; raise SystemExit(0 if reference_inputs_present() else 1)" >/dev/null 2>&1; then
  log "reference files already present in raw_batch_data - skipping producer"
else
  log "producing reference files -> raw_batch_data (customers/agents/invoices/payments)"
  python -m genie_voice.datagen.loader || log "reference produce skipped/failed"
fi

# ---- 6. produce voice STT events into raw_streaming_data -------------------
if python -c "import sys;sys.path.insert(0,'backend');from genie_voice.databricks.volume_state import streaming_inputs_present; raise SystemExit(0 if streaming_inputs_present() else 1)" >/dev/null 2>&1; then
  log "streaming files already present in raw_streaming_data - skipping producer"
else
  log "producing voice STT events -> raw_streaming_data"
  python -m genie_voice.ingest.producer
fi

# ---- 7. prepare Lakebase serving project ------------------------------------
if [[ "$ONLINE" -eq 1 ]]; then
  log "provisioning Lakebase project + serving schema"
  python infra/lakebase/setup_lakebase.py || log "lakebase setup skipped/failed"
fi

# ---- 8. ingest the landed files --------------------------------------------
if [[ "$ONLINE" -eq 1 ]]; then
  # Deploy + run one serverless orchestration job:
  #   Reference UC ingest: raw_batch_data -> customers/agents/invoices/payments
  #   Call Lakebase ingest: raw_streaming_data -> operational call tables
  #   Lakebase CDF sync check: wal2delta.tables is running + call lb_<table>_history exists in UC
  #   Gold insights refresh: call history + utterance history -> gold_call_insights
  #   UC constraints + data quality checks before Genie
  #   Downstream job task: Genie space reconcile
  log "deploying + running Lakebase-first orchestration job"
  if [[ "$RESET" -eq 1 ]]; then
    python infra/jobs/deploy_pipeline.py --full-refresh || log "pipeline job deploy/run skipped/failed"
  else
    python infra/jobs/deploy_pipeline.py || log "pipeline job deploy/run skipped/failed"
  fi
  # Reconcile Genie space explicitly as a post-step too. This avoids stale/trashed
  # space state when users rerun jobs manually from the Databricks UI. ensure_space()
  # is name-based and idempotent (trashes same-title spaces before creating one), so
  # this does not create duplicates.
  log "reconciling Genie space (name-based, no duplicates)"
  python -m genie_voice.genie.space || log "genie space reconcile skipped/failed"
else
  # OFFLINE: emulate both jobs in-process (no Spark / no workspace).
  log "OFFLINE: emulating the pipeline in-process -> local silver + gold exports"
  python -m genie_voice.enrich.derive || log "offline derive failed"
fi

# ---- 9a. offline live-state emulation ---------------------------------------
if [[ "$ONLINE" -eq 0 ]]; then
  log "OFFLINE: serving live call state in-process"
  python -m genie_voice.enrich.local_runner || log "serving step failed (UI still starts; call list may be empty)"
fi

# ---- 9b. start API ----------------------------------------------------------
API_PORT="${GENIE_API_PORT:-8000}"
log "starting API on :$API_PORT"
( cd api && uvicorn app.main:app --host 0.0.0.0 --port "$API_PORT" \
    > "$RUN_DIR/api.log" 2>&1 & echo $! > "$RUN_DIR/api.pid" )

# ---- 9c. start frontend -----------------------------------------------------
log "installing + starting frontend UI"
( cd frontend && npm install --silent \
    && VITE_API_BASE_URL="http://localhost:$API_PORT" npm run dev \
    > "$RUN_DIR/frontend.log" 2>&1 & echo $! > "$RUN_DIR/frontend.pid" )

sleep 2
log "------------------------------------------------------------------"
log "API:  http://localhost:$API_PORT/health"
log "UI:   http://localhost:5173"
log "logs: $RUN_DIR/{api,frontend}.log"
log "stop: ./local-undeploy.sh"
log "------------------------------------------------------------------"
