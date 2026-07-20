"""Create an OAuth (M2M) secret for the benchmark's app service principal.

The multilingual voice benchmark authenticates to the deployed app's OAuth
ingress with a service principal via client-credentials. By default it reuses
the app's own service principal (which already has CAN_USE on the app), so no
extra grant is needed — this script just mints a client secret for it.

Usage:
    python infra/apps/create_benchmark_sp_secret.py            # uses APP_NAME below
    python infra/apps/create_benchmark_sp_secret.py --app-name genie-voice-agent
    python infra/apps/create_benchmark_sp_secret.py --sp-id 76102321904006

Paste the printed client_id / client_secret into config.local.yaml under
realtime_voice.benchmark.auth (gitignored).
"""
from __future__ import annotations

import argparse
import json
import subprocess


def _dbx(profile: str | None) -> list[str]:
    return ["databricks"] + (["--profile", profile] if profile else [])


def _app_service_principal(dbx: list[str], app_name: str) -> tuple[str, str]:
    proc = subprocess.run(
        [*dbx, "apps", "get", app_name, "-o", "json"],
        check=True, capture_output=True, text=True,
    )
    data = json.loads(proc.stdout)
    client_id = data.get("service_principal_client_id")
    sp_id = data.get("service_principal_id")
    if not client_id or not sp_id:
        raise SystemExit(f"App {app_name!r} has no service principal yet (is it deployed?)")
    return str(client_id), str(sp_id)


def main() -> None:
    ap = argparse.ArgumentParser(description="Mint an OAuth secret for the benchmark app SP")
    ap.add_argument("--app-name", default="genie-voice-agent")
    ap.add_argument("--sp-id", default=None, help="Service principal SCIM id (skip app lookup)")
    ap.add_argument("--profile", default="fe-vm-vdm-classic-rcn6ip")
    args = ap.parse_args()

    dbx = _dbx(args.profile)
    if args.sp_id:
        sp_id = args.sp_id
        client_id = None
    else:
        client_id, sp_id = _app_service_principal(dbx, args.app_name)

    proc = subprocess.run(
        [*dbx, "service-principal-secrets-proxy", "create", sp_id, "-o", "json"],
        check=True, capture_output=True, text=True,
    )
    secret = json.loads(proc.stdout).get("secret")
    if not secret:
        raise SystemExit(f"Secret creation returned no secret: {proc.stdout}")

    print("OAuth secret created. Add to config.local.yaml:\n")
    print("  realtime_voice:")
    print("    benchmark:")
    print("      auth:")
    if client_id:
        print(f'        client_id: "{client_id}"')
    else:
        print('        client_id: "<app service_principal_client_id>"')
    print(f'        client_secret: "{secret}"')


if __name__ == "__main__":
    main()
