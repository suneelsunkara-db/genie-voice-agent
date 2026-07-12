from __future__ import annotations

import argparse
import json

from genie_voice.ml_asr.serving import deploy, register


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="genie_voice.ml_asr.serving")
    parser.add_argument("--config", default=None, help="Path to config/ml_asr_eval.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    reg = sub.add_parser("register", help="Register UC model candidates (delegates to scripts/asr).")
    reg_sub = reg.add_subparsers(dest="register_command", required=True)
    reg_sub.add_parser("list", help="List registerable databricks models from config.")
    one = reg_sub.add_parser("one", help="Register one model by model_id.")
    one.add_argument("model_id")
    reg_sub.add_parser("all", help="Register all configured databricks models.")

    serve = sub.add_parser("serve", help="Deploy Model Serving endpoints from config.")
    serve_sub = serve.add_subparsers(dest="serve_command", required=True)
    serve_sub.add_parser("list", help="List serving specs from config.")
    pf = serve_sub.add_parser("preflight", help="Validate UC alias + serving spec.")
    pf.add_argument("model_id")
    dep = serve_sub.add_parser("deploy", help="Create or update one endpoint.")
    dep.add_argument("model_id")
    serve_sub.add_parser("deploy-all", help="Deploy all configured endpoints.")
    st = serve_sub.add_parser("status", help="Show endpoint status.")
    st.add_argument("model_id")
    sm = serve_sub.add_parser("smoke", help="Query endpoint with one holdout clip.")
    sm.add_argument("model_id")
    sm.add_argument("--manifest", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = args.config

    if args.command == "register":
        if args.register_command == "list":
            print(json.dumps({"models": register.list_registrations(config_path=config_path)}, indent=2))
            return 0
        if args.register_command == "one":
            print(json.dumps(register.register(args.model_id, config_path=config_path), indent=2))
            return 0
        if args.register_command == "all":
            print(json.dumps({"results": register.register_all(config_path=config_path)}, indent=2))
            return 0

    if args.command == "serve":
        if args.serve_command == "list":
            print(json.dumps({"endpoints": deploy.list_deployments(config_path=config_path)}, indent=2))
            return 0
        if args.serve_command == "preflight":
            print(json.dumps(deploy.preflight(args.model_id, config_path=config_path), indent=2))
            return 0
        if args.serve_command == "deploy":
            print(json.dumps(deploy.deploy(args.model_id, config_path=config_path), indent=2))
            return 0
        if args.serve_command == "deploy-all":
            print(json.dumps({"results": deploy.deploy_all(config_path=config_path)}, indent=2))
            return 0
        if args.serve_command == "status":
            print(json.dumps(deploy.endpoint_status(args.model_id, config_path=config_path), indent=2))
            return 0
        if args.serve_command == "smoke":
            print(json.dumps(deploy.smoke(args.model_id, config_path=config_path, manifest_path=args.manifest), indent=2))
            return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
