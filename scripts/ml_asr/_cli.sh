#!/usr/bin/env bash
# Internal CLI for genie_voice.ml_asr — use scripts/ml_asr.sh or numbered step scripts.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  python3 -m venv "$ROOT/.venv"
fi
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
export PYTHONPATH="$ROOT/backend${PYTHONPATH:+:$PYTHONPATH}"

if ! python -c "import genie_voice" >/dev/null 2>&1; then
  python -m pip install -q --upgrade pip
  python -m pip install -q -e "$ROOT/backend[ml-asr]" || python -m pip install -q -e "$ROOT/backend"
fi

# Databricks profile is read from config (databricks.profile in
# config.local.yaml) by the Python code — no env profile is set here.
exec python -m genie_voice.ml_asr.cli "$@"
