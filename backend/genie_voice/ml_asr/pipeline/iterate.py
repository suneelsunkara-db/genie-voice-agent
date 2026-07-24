from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from genie_voice.ml_asr.config import EvalConfig, load_config
from genie_voice.ml_asr.runtime import databricks_profile, is_volume_mode


@dataclass(frozen=True)
class PipelineStep:
    step_id: str
    action: str
    params: dict[str, Any]


def build_iterative_plan(config: EvalConfig, *, smoke: bool = False) -> list[PipelineStep]:
    steps: list[PipelineStep] = []
    for tier in config.eval_tiers:
        for dataset in config.datasets_for_tier(tier):
            limit = None
            if smoke:
                if tier == "acoustic":
                    limit = config.smoke_acoustic_clip_limit or 25
                elif tier == "business":
                    limit = config.smoke_business_clip_limit or 25
            steps.append(
                PipelineStep(
                    step_id=f"prepare:{dataset.dataset_id}",
                    action="prepare",
                    params={"dataset": [dataset.dataset_id], "limit": limit, "volume_mode": True},
                )
            )
        steps.append(
            PipelineStep(
                step_id=f"audit:{tier}",
                action="audit-dataset",
                params={"tier": [tier], "remote_manifest": True, "no_audio": True, "volume_mode": True},
            )
        )
        eval_limit = config.smoke_eval_clip_limit if smoke else None
        steps.append(
            PipelineStep(
                step_id=f"evaluate:{tier}",
                action="evaluate",
                params={"tier": [tier], "limit": eval_limit, "remote_manifest": True, "volume_mode": True},
            )
        )
    steps.append(PipelineStep("summarize", "summarize", {"volume_mode": True}))
    return steps


def load_state(config: EvalConfig) -> dict[str, Any]:
    path = _state_path(config)
    if not path.is_file():
        return {"completed_steps": [], "history": []}
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(config: EvalConfig, state: dict[str, Any]) -> None:
    path = _state_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not is_volume_mode():
        _upload_state(config, path)


def mark_step_complete(config: EvalConfig, step: PipelineStep, result: dict[str, Any]) -> dict[str, Any]:
    state = load_state(config)
    completed = list(state.get("completed_steps") or [])
    if step.step_id not in completed:
        completed.append(step.step_id)
    history = list(state.get("history") or [])
    history.append({"step_id": step.step_id, "action": step.action, "result": result})
    state["completed_steps"] = completed
    state["history"] = history[-50:]
    save_state(config, state)
    return state


def next_step(config: EvalConfig, *, smoke: bool = False) -> PipelineStep | None:
    completed = set(load_state(config).get("completed_steps") or [])
    for step in build_iterative_plan(config, smoke=smoke):
        if step.step_id not in completed:
            return step
    return None


def reset_state(config: EvalConfig) -> None:
    save_state(config, {"completed_steps": [], "history": []})


def sync_state_from_volume(config: EvalConfig) -> dict[str, Any]:
    if is_volume_mode():
        return load_state(config)
    profile = databricks_profile()
    dbx = ["databricks"]
    if profile:
        dbx = ["databricks", "--profile", profile]
    local_copy = Path(config.local_root) / "pipeline_state.json"
    local_copy.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [*dbx, "fs", "cp", f"dbfs:{config.remote_state_path}", str(local_copy), "--overwrite"],
        check=False,
    )
    if local_copy.is_file():
        state = json.loads(local_copy.read_text(encoding="utf-8"))
        save_state(config, state)
        return state
    return {"completed_steps": [], "history": []}


def _state_path(config: EvalConfig) -> Path:
    if is_volume_mode():
        return Path(config.remote_state_path)
    return Path(config.local_root) / "pipeline_state.json"


def _upload_state(config: EvalConfig, local_path: Path) -> None:
    profile = databricks_profile()
    dbx = ["databricks"]
    if profile:
        dbx = ["databricks", "--profile", profile]
    subprocess.run(
        [*dbx, "fs", "mkdirs", f"dbfs:{Path(config.remote_state_path).parent}", "--overwrite"],
        check=False,
    )
    subprocess.run(
        [*dbx, "fs", "cp", str(local_path), f"dbfs:{config.remote_state_path}", "--overwrite"],
        check=True,
    )
