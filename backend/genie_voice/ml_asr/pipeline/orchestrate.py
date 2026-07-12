from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from genie_voice.ml_asr.config import load_config
from genie_voice.ml_asr.jobs.serverless_submit import submit_serverless_step
from genie_voice.ml_asr.pipeline.audit import audit_dataset, write_audit_report
from genie_voice.ml_asr.pipeline.dataset_eval import evaluate_datasets, write_dataset_eval_report
from genie_voice.ml_asr.pipeline.evaluate import evaluate, preflight
from genie_voice.ml_asr.pipeline.iterate import (
    PipelineStep,
    build_iterative_plan,
    mark_step_complete,
    next_step,
    sync_state_from_volume,
)
from genie_voice.ml_asr.pipeline.prepare import prepare_dataset, validate_dataset
from genie_voice.ml_asr.pipeline.summarize import summarize
from genie_voice.ml_asr.runtime import is_volume_mode


def run_action(
    action: str,
    *,
    config_path: str | None = None,
    params: dict[str, Any] | None = None,
    local: bool = False,
    wait: bool = True,
) -> dict[str, Any]:
    if is_volume_mode() or local:
        return _run_local(action, config_path=config_path, params=params or {})
    result = submit_serverless_step(
        action=action,
        config_path=config_path,
        params=_serverless_params(action, params or {}),
        wait=wait,
    )
    if wait and action in {"audit-dataset", "dataset-eval", "prepare", "evaluate", "summarize", "validate"}:
        artifact = _pull_action_artifacts(action, config_path=config_path)
        result["action_result"] = artifact
        if action in {"audit-dataset", "dataset-eval"} and artifact:
            result["report"] = artifact
    return result


def run_step(
    step: PipelineStep,
    *,
    config_path: str | None = None,
    local: bool = False,
    smoke: bool = False,
) -> dict[str, Any]:
    config = load_config(config_path=config_path)
    result = run_action(step.action, config_path=config_path, params=step.params, local=local)
    if is_volume_mode() or local:
        action_result = result
    else:
        sync_state_from_volume(config)
        action_result = result.get("action_result") or result
    mark_step_complete(config, step, action_result if isinstance(action_result, dict) else {"status": "ok"})
    return {"step": step.step_id, "action": step.action, "result": result}


def run_iterate_next(
    *,
    config_path: str | None = None,
    local: bool = False,
    smoke: bool = False,
) -> dict[str, Any]:
    config = load_config(config_path=config_path)
    sync_state_from_volume(config)
    step = next_step(config, smoke=smoke)
    if step is None:
        return {"status": "complete", "message": "All iterative steps finished."}
    outcome = run_step(step, config_path=config_path, local=local, smoke=smoke)
    gates = None
    if step.action in {"prepare", "audit-dataset"}:
        try:
            gates = audit_dataset(config_path=config_path, use_remote_manifest=True, read_audio=False).get("gates")
        except Exception as exc:  # noqa: BLE001
            gates = {"error": str(exc)}
    return {
        "status": "ran",
        "step": step.step_id,
        "outcome": outcome,
        "gates": gates,
        "next": _step_summary(next_step(config, smoke=smoke)),
    }


def status_report(*, config_path: str | None = None, smoke: bool = False) -> dict[str, Any]:
    config = load_config(config_path=config_path)
    state = sync_state_from_volume(config)
    completed = set(state.get("completed_steps") or [])
    plan = build_iterative_plan(config, smoke=smoke)
    pending = [step for step in plan if step.step_id not in completed]
    gates: dict[str, Any] | None = None
    try:
        report = audit_dataset(config_path=config_path, use_remote_manifest=True, read_audio=False)
        gates = report.get("gates")
    except Exception as exc:  # noqa: BLE001
        gates = {"error": str(exc)}
    return {
        "mode": "serverless" if not is_volume_mode() else "worker",
        "completed_steps": list(completed),
        "pending_steps": [_step_summary(step) for step in pending],
        "next_step": _step_summary(pending[0]) if pending else None,
        "gates": gates,
        "remote_state_path": config.remote_state_path,
        "remote_audit_path": config.remote_audit_path,
        "remote_index_path": config.remote_index_path,
    }


def _run_local(action: str, *, config_path: str | None, params: dict[str, Any]) -> dict[str, Any]:
    volume_mode = is_volume_mode() or bool(params.pop("volume_mode", True))
    if action == "prepare":
        return {
            "results": prepare_dataset(
                config_path=config_path,
                languages=params.get("language"),
                dataset_ids=params.get("dataset"),
                limit=params.get("limit"),
                volume_mode=volume_mode,
            )
        }
    if action == "validate":
        return {
            "report": validate_dataset(
                config_path=config_path,
                volume_mode=volume_mode,
            )
        }
    if action == "audit-dataset":
        config = load_config(config_path=config_path)
        report = audit_dataset(
            config_path=config_path,
            languages=params.get("language"),
            dataset_ids=params.get("dataset"),
            tiers=params.get("tier"),
            use_remote_manifest=params.get("remote_manifest", True),
            read_audio=not params.get("no_audio", False),
            audio_sample_limit=int(params.get("audio_sample_limit", 10)),
        )
        out = write_audit_report(report, config=config, volume_mode=volume_mode)
        return {"report": report, "audit_path": str(out)}
    if action == "dataset-eval":
        config = load_config(config_path=config_path)
        report = evaluate_datasets(
            config_path=config_path,
            languages=params.get("language"),
            dataset_ids=params.get("dataset"),
            tiers=params.get("tier"),
            use_remote_manifest=params.get("remote_manifest", True),
            read_audio=not params.get("no_audio", False),
            audio_sample_limit=int(params.get("audio_sample_limit", 10)),
            min_entity_quality=int(params.get("min_entity_quality", 3)),
        )
        out = write_dataset_eval_report(report, config=config, volume_mode=volume_mode)
        return {"report": report, "eval_path": str(out)}
    if action == "preflight":
        return {"report": preflight(config_path=config_path, languages=params.get("language"))}
    if action in {"evaluate", "evaluate-all"}:
        return {
            "summary": evaluate(
                config_path=config_path,
                languages=params.get("language"),
                dataset_ids=params.get("dataset"),
                tiers=params.get("tier"),
                model_ids=params.get("model"),
                limit=params.get("limit"),
                use_remote_manifest=params.get("remote_manifest", True),
                volume_mode=volume_mode,
            )
        }
    if action == "summarize":
        index_path = summarize(config_path=config_path, volume_mode=volume_mode)
        return {"index_path": str(index_path)}
    raise ValueError(f"Unsupported action: {action}")


_SERVERLESS_CLI_PARAMS: dict[str, set[str]] = {
    "prepare": {"language", "dataset", "limit"},
    "audit-dataset": {"language", "dataset", "tier", "no_audio", "audio_sample_limit"},
    "dataset-eval": {"language", "dataset", "tier", "no_audio", "audio_sample_limit", "min_entity_quality"},
    "evaluate": {"language", "dataset", "tier", "model", "limit"},
    "evaluate-all": {"tier", "limit"},
    "validate": {"local_only"},
    "summarize": set(),
    "preflight": {"language"},
}


def _serverless_params(action: str, params: dict[str, Any]) -> dict[str, Any]:
    allowed = _SERVERLESS_CLI_PARAMS.get(action, set())
    return {key: value for key, value in params.items() if key in allowed}


def _pull_action_artifacts(action: str, *, config_path: str | None) -> dict[str, Any]:
    config = load_config(config_path=config_path)
    if action == "audit-dataset":
        local_audit = Path(config.local_root) / "dataset_audit.json"
        _sync_from_volume(config.remote_audit_path, local_audit)
        if local_audit.is_file():
            return json.loads(local_audit.read_text(encoding="utf-8"))
    if action == "dataset-eval":
        local_report = Path(config.local_root) / "dataset_quality_eval.json"
        remote_report = f"{config.remote_results_dir}/dataset_quality_eval.json"
        _sync_from_volume(remote_report, local_report)
        if local_report.is_file():
            return json.loads(local_report.read_text(encoding="utf-8"))
    if action == "summarize":
        local_index = Path(config.local_root) / "results" / "index.json"
        _sync_from_volume(config.remote_index_path, local_index)
        if local_index.is_file():
            return json.loads(local_index.read_text(encoding="utf-8"))
    return {}


def _sync_from_volume(remote_path: str, local_path: Path) -> None:
    if is_volume_mode():
        if Path(remote_path).is_file():
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_bytes(Path(remote_path).read_bytes())
        return
    import subprocess

    profile = os.environ.get("ML_ASR_DATABRICKS_PROFILE") or os.environ.get("DATABRICKS_CONFIG_PROFILE")
    dbx = ["databricks"]
    if profile:
        dbx = ["databricks", "--profile", profile]
    local_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [*dbx, "fs", "cp", f"dbfs:{remote_path}", str(local_path), "--overwrite"],
        check=False,
        capture_output=True,
        text=True,
    )


def _step_summary(step: PipelineStep | None) -> dict[str, Any] | None:
    if step is None:
        return None
    return {"step_id": step.step_id, "action": step.action, "params": step.params}
