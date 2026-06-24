#!/usr/bin/env bash
# Stop the locally deployed API + UI started by local-deploy.sh.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$ROOT/.run"

for svc in api frontend; do
  if [[ -f "$RUN_DIR/$svc.pid" ]]; then
    pid="$(cat "$RUN_DIR/$svc.pid")"
    if kill "$pid" 2>/dev/null; then
      echo "[undeploy] stopped $svc (pid $pid)"
    fi
    rm -f "$RUN_DIR/$svc.pid"
  fi
done
echo "[undeploy] done"
