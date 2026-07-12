from __future__ import annotations

import argparse
import json
import os

from genie_voice.ml_asr.config import config_summary, load_config
from genie_voice.ml_asr.pipeline.iterate import build_iterative_plan, reset_state, sync_state_from_volume
from genie_voice.ml_asr.pipeline.orchestrate import run_action, run_iterate_next, status_report
from genie_voice.ml_asr.runtime import is_volume_mode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="genie_voice.ml_asr",
        description="Holistic multilingual ASR evaluation (acoustic + business tiers).",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config/ml_asr_eval.yaml (default: repo config/ml_asr_eval.yaml)",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run on this machine instead of submitting a Databricks serverless job.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use smoke limits from eval_plan.smoke for iterative runs.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("plan", help="Show pipeline overview and resolved config.")
    subparsers.add_parser("status", help="Show iterative pipeline state and quality gates.")

    step = subparsers.add_parser("step", help="Run one pipeline action via serverless (or --local).")
    step.add_argument("action", choices=[
        "prepare", "validate", "audit-dataset", "dataset-eval", "preflight", "evaluate", "evaluate-all", "summarize",
    ])
    step.add_argument("--language", action="append")
    step.add_argument("--dataset", action="append")
    step.add_argument("--tier", action="append")
    step.add_argument("--model", action="append")
    step.add_argument("--limit", type=int)
    step.add_argument("--no-audio", action="store_true")
    step.add_argument("--audio-sample-limit", type=int, default=10)
    step.add_argument("--no-wait", action="store_true")

    iterate = subparsers.add_parser("iterate", help="Iterative quality-gated pipeline on UC Volume.")
    iterate_sub = iterate.add_subparsers(dest="iterate_command", required=True)
    iterate_sub.add_parser("plan", help="Show ordered iterative steps and completion state.")
    iterate_sub.add_parser("next", help="Run the next pending iterative step on serverless.")
    iterate_sub.add_parser("reset", help="Reset pipeline state on the Volume.")

    prepare = subparsers.add_parser("prepare", help="Build all configured dataset tiers.")
    prepare.add_argument("--language", action="append")
    prepare.add_argument("--dataset", action="append", help="Limit to dataset id(s), e.g. fleurs_acoustic_v1.")
    prepare.add_argument("--limit", type=int, help="Acoustic clip limit or business clips/scenario override.")
    prepare.add_argument("--skip-upload", action="store_true", help="Deprecated: volume mode writes directly.")

    validate = subparsers.add_parser("validate", help="Validate manifests.")
    validate.add_argument("--local-only", action="store_true")

    audit = subparsers.add_parser("audit-dataset", help="Audit holistic dataset quality before model eval.")
    audit.add_argument("--language", action="append")
    audit.add_argument("--dataset", action="append")
    audit.add_argument("--tier", action="append")
    audit.add_argument("--remote-manifest", action="store_true")
    audit.add_argument("--no-audio", action="store_true")
    audit.add_argument("--audio-sample-limit", type=int, default=10)

    dataset_eval = subparsers.add_parser("dataset-eval", help="Semantic dataset quality eval (labels, scenarios, audio).")
    dataset_eval.add_argument("--language", action="append")
    dataset_eval.add_argument("--dataset", action="append")
    dataset_eval.add_argument("--tier", action="append")
    dataset_eval.add_argument("--remote-manifest", action="store_true")
    dataset_eval.add_argument("--no-audio", action="store_true")
    dataset_eval.add_argument("--audio-sample-limit", type=int, default=10)
    dataset_eval.add_argument("--min-entity-quality", type=int, default=3)

    pre = subparsers.add_parser("preflight", help="Check manifests and provider configuration.")
    pre.add_argument("--language", action="append")

    ev = subparsers.add_parser("evaluate", help="Score models for selected datasets/tiers.")
    ev.add_argument("--language", action="append")
    ev.add_argument("--dataset", action="append")
    ev.add_argument("--tier", action="append", help="Limit to eval tiers: acoustic, business.")
    ev.add_argument("--model", action="append")
    ev.add_argument("--limit", type=int)
    ev.add_argument("--local-manifest", action="store_true")

    ev_all = subparsers.add_parser("evaluate-all", help="Score all datasets in eval_plan.")
    ev_all.add_argument("--limit", type=int)
    ev_all.add_argument("--tier", action="append")
    ev_all.add_argument("--local-manifest", action="store_true")

    subparsers.add_parser("summarize", help="Write holistic results/index.json.")

    run = subparsers.add_parser("run", help="Full pipeline via iterative serverless steps.")
    run.add_argument("--limit", type=int)
    run.add_argument("--skip-eval", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config_path = args.config
    local = args.local or is_volume_mode()

    if args.command == "plan":
        config = load_config(config_path=config_path)
        print(
            "Holistic multilingual ASR eval pipeline (iterative, UC Volume + serverless)\n"
            "  Datasets — licensed human speech only:\n"
            "    business  Google FLEURS (validation+train), mined for billing entities\n"
            "    acoustic  Google FLEURS validation holdout\n"
            "  1. prepare        download audio + manifests to Volume\n"
            "  2. audit-dataset   tier quality gates before model scoring\n"
            "  3. evaluate        Deepgram + Databricks routes per tier\n"
            "  4. summarize       rankings on Volume index.json\n"
            "\n"
            "Orchestration:\n"
            "  status            pipeline state + gates from Volume\n"
            "  step <action>     one serverless job\n"
            "  iterate next      next quality-gated step\n"
            "\n"
            "Promotion rule:\n"
            "  - acoustic tier ranks models on WER/CER\n"
            "  - business tier ranks on critical_entity_accuracy + unsafe_for_resolution\n"
            "  - business audio must be recorded/uploaded before promotion decisions\n"
        )
        print(json.dumps(config_summary(config), indent=2, ensure_ascii=False))
        return 0

    if args.command == "status":
        print(json.dumps(status_report(config_path=config_path, smoke=args.smoke), indent=2, ensure_ascii=False))
        return 0

    if args.command == "step":
        params = _step_params(args)
        result = run_action(
            args.action,
            config_path=config_path,
            params=params,
            local=local,
            wait=not args.no_wait,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    if args.command == "iterate":
        if args.iterate_command == "plan":
            config = load_config(config_path=config_path)
            state = sync_state_from_volume(config)
            completed = set(state.get("completed_steps") or [])
            plan = [
                {
                    "step_id": step.step_id,
                    "action": step.action,
                    "status": "done" if step.step_id in completed else "pending",
                }
                for step in build_iterative_plan(config, smoke=args.smoke)
            ]
            print(json.dumps({"steps": plan, "completed": list(completed)}, indent=2, ensure_ascii=False))
            return 0
        if args.iterate_command == "reset":
            config = load_config(config_path=config_path)
            reset_state(config)
            print(json.dumps({"status": "reset", "remote_state_path": config.remote_state_path}, indent=2))
            return 0
        if args.iterate_command == "next":
            result = run_iterate_next(config_path=config_path, local=local, smoke=args.smoke)
            print(json.dumps(result, indent=2, ensure_ascii=False))
            if result.get("status") == "complete":
                return 0
            gates = (result.get("gates") or {})
            if (
                not args.smoke
                and gates
                and not gates.get("model_eval_ready")
                and result.get("step", "").startswith("audit:")
            ):
                return 2
            return 0

    if args.command in {"prepare", "validate", "audit-dataset", "dataset-eval", "preflight", "evaluate", "evaluate-all", "summarize"}:
        params = _legacy_params(args)
        result = run_action(args.command, config_path=config_path, params=params, local=local)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        if args.command == "audit-dataset":
            gates = _report_gates(result)
            if _should_fail_on_gate(args, gates, "model_eval_ready"):
                return 2
        if args.command == "dataset-eval":
            gates = _report_gates(result)
            if _should_fail_on_gate(args, gates, "dataset_quality_ready"):
                return 2
        return 0

    if args.command == "run":
        while True:
            outcome = run_iterate_next(config_path=config_path, local=local, smoke=args.smoke)
            print(json.dumps(outcome, indent=2, ensure_ascii=False))
            if outcome.get("status") == "complete":
                return 0
            if args.skip_eval and str(outcome.get("step", "")).startswith("evaluate:"):
                config = load_config(config_path=config_path)
                from genie_voice.ml_asr.pipeline.iterate import mark_step_complete, next_step

                step = next_step(config, smoke=args.smoke)
                if step and step.step_id.startswith("evaluate:"):
                    mark_step_complete(config, step, {"status": "skipped"})
                continue
            gates = outcome.get("gates") or {}
            if (
                not args.smoke
                and gates
                and not gates.get("model_eval_ready")
                and str(outcome.get("step", "")).startswith("audit:")
            ):
                return 2
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


def _step_params(args: argparse.Namespace) -> dict:
    local = getattr(args, "local", False)
    params: dict = {"volume_mode": not local}
    if local:
        params.setdefault("remote_manifest", False)
    else:
        params.setdefault("remote_manifest", True)
    if getattr(args, "language", None):
        params["language"] = args.language
    if getattr(args, "dataset", None):
        params["dataset"] = args.dataset
    if getattr(args, "tier", None):
        params["tier"] = args.tier
    if getattr(args, "model", None):
        params["model"] = args.model
    if getattr(args, "limit", None) is not None:
        params["limit"] = args.limit
    if getattr(args, "action", None) in {"audit-dataset", "dataset-eval"}:
        if getattr(args, "no_audio", False):
            params["no_audio"] = True
        if getattr(args, "audio_sample_limit", None) is not None:
            params["audio_sample_limit"] = args.audio_sample_limit
        if getattr(args, "min_entity_quality", None) is not None:
            params["min_entity_quality"] = args.min_entity_quality
    if getattr(args, "local_manifest", False):
        params["remote_manifest"] = False
    return params


def _report_gates(result: dict) -> dict:
    report = result.get("report") or result.get("action_result") or {}
    gates = report.get("gates") if isinstance(report, dict) else {}
    return gates if isinstance(gates, dict) else {}


def _should_fail_on_gate(args: argparse.Namespace, gates: dict, gate_key: str) -> bool:
    if not gates or gates.get(gate_key):
        return False
    if os.environ.get("ML_ASR_RUN_MODE") == "serverless":
        return False
    return bool(args.local)


def _legacy_params(args: argparse.Namespace) -> dict:
    params = _step_params(args)
    if args.command == "validate":
        params["local_only"] = getattr(args, "local_only", False)
    if args.command in {"audit-dataset", "dataset-eval"}:
        if getattr(args, "no_audio", False):
            params["no_audio"] = True
        if getattr(args, "audio_sample_limit", None) is not None:
            params["audio_sample_limit"] = args.audio_sample_limit
        if getattr(args, "min_entity_quality", None) is not None:
            params["min_entity_quality"] = args.min_entity_quality
    if args.command in {"evaluate", "evaluate-all"} and not getattr(args, "local_manifest", False):
        params["remote_manifest"] = True
    return params


if __name__ == "__main__":
    raise SystemExit(main())
