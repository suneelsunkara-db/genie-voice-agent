"""Verify/prepare the Lakebase Autoscaling (Projects) database for serving.

Lakebase Autoscaling is organized as PROJECTS (Project -> Branch -> Compute
endpoint -> Database), managed via the `/api/2.0/postgres/` API - NOT the older
`database/instances` API. `config.lakebase.instance` is matched against a
project's `project_id` or `display_name`.

Idempotent and config-driven. Steps:
  1. Resolve the Lakebase project by name (create it via the Projects API if it
     is missing and auto-create is allowed).
  2. Verify connectivity by minting a Postgres OAuth token (scoped to the
     read-write endpoint) and ensuring the configured serving schema exists.

Lakebase serving uses primary Postgres table names under the configured schema
for operational call data (`call_state`, `call_facts`, `live_call_utterances`),
not duplicate `*_serving` managed-sync tables.

Auth: runs AS the U2M user (no PAT). The app mints a short-lived Postgres token
at runtime, so there is no password to copy into .env.

Run:  python infra/lakebase/setup_lakebase.py
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, "backend")

from genie_voice.config import get_settings  # noqa: E402
from genie_voice.databricks.client import current_user, get_workspace_client  # noqa: E402

_PG_BASE = "/api/2.0/postgres"


def _find_project(ac, instance: str) -> dict | None:
    projects = ac.do("GET", f"{_PG_BASE}/projects").get("projects", []) or []
    return next(
        (
            p for p in projects
            if instance in (p.get("project_id"), (p.get("status") or {}).get("display_name"))
            or p.get("project_id") == instance.replace("_", "-")
        ),
        None,
    )


def _wait_project_ready(ac, project_id: str, timeout_s: int = 900) -> bool:
    """Poll the default branch's read-write endpoint until it has a host."""
    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        try:
            branches = ac.do(
                "GET", f"{_PG_BASE}/projects/{project_id}/branches"
            ).get("branches", []) or []
            branch = next((b for b in branches if (b.get("status") or {}).get("default")), None) \
                or (branches[0] if branches else None)
            if branch:
                bid = branch["branch_id"]
                eps = ac.do(
                    "GET", f"{_PG_BASE}/projects/{project_id}/branches/{bid}/endpoints"
                ).get("endpoints", []) or []
                ep = next(
                    (e for e in eps
                     if (e.get("status") or {}).get("endpoint_type") == "ENDPOINT_TYPE_READ_WRITE"),
                    None,
                ) or (eps[0] if eps else None)
                host = (((ep or {}).get("status") or {}).get("hosts") or {}).get("host")
                if host:
                    print(f"  read-write endpoint ready: {host}")
                    return True
                state = ((ep or {}).get("status") or {}).get("current_state")
                if state != last:
                    print(f"  endpoint state: {state} ... waiting")
                    last = state
        except Exception:  # noqa: BLE001 - resources still being created
            pass
        time.sleep(10)
    print(f"  WARNING: project '{project_id}' endpoint not ready within {timeout_s}s")
    return False


def main() -> None:
    s = get_settings()
    if not s.lakebase.enabled:
        print("lakebase.enabled=false -> skipping (app uses in-memory fallback).")
        return

    client = get_workspace_client(s)
    ac = client.api_client
    inst = s.lakebase.instance
    print(f"Authenticated as: {current_user(client)}")
    print(f"Resolving Lakebase project '{inst}' via {_PG_BASE}/projects ...")

    project = _find_project(ac, inst)
    if project is None:
        # Auto-create is opt-in: provisioning a project is a long, billable op and
        # most environments create it once in the UI.
        if os.environ.get("GENIE_LAKEBASE_AUTOCREATE", "false").lower() in ("1", "true", "yes"):
            project_id = inst.replace("_", "-")
            print(f"  not found - creating project '{project_id}' (pg 17) ...")
            try:
                from databricks.sdk.service.postgres import Project, ProjectSpec

                op = client.postgres.create_project(
                    project=Project(spec=ProjectSpec(pg_version=17)), project_id=project_id
                )
                op.wait()
                print(f"  created project {project_id}")
                project = _find_project(ac, inst)
            except Exception as exc:  # noqa: BLE001
                print(f"  (could not create project: {exc})")
                return
        else:
            print(
                f"  Lakebase project '{inst}' not found.\n"
                "  Create it in the UI (Compute > Lakebase > Create project) with display\n"
                f"  name '{inst}', or re-run with GENIE_LAKEBASE_AUTOCREATE=true to create it."
            )
            return

    project_id = project["project_id"]
    print(f"  found project: {project_id} "
          f"(display '{(project.get('status') or {}).get('display_name')}')")

    if not _wait_project_ready(ac, project_id):
        print("  endpoint not ready yet - serving will retry at runtime.")
        return

    # Verify end to end: mint a token, connect, ensure the serving table exists.
    print("Verifying connectivity + Lakebase serving schema ...")
    try:
        from genie_voice.serve import LakebaseServing

        LakebaseServing(s).ensure_schema()
        print(f"  ok: connected and ensured schema {s.lakebase.schema_name}")
    except Exception as exc:  # noqa: BLE001
        print(f"  WARNING: could not verify serving connectivity: {exc}")
        return

    print(
        "Done. The app connects via runtime-minted Postgres tokens "
        "(no LAKEBASE_PASSWORD needed)."
    )


if __name__ == "__main__":
    main()
