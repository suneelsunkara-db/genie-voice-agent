from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from genie_voice.ml_asr.serving.catalog import ServingSpec, get_serving_spec, load_serving_specs


def list_registrations(*, config_path: str | None = None) -> list[dict[str, Any]]:
    rows = []
    for spec in load_serving_specs(config_path=config_path):
        if not spec.register_type:
            continue
        rows.append(
            {
                "model_id": spec.model_id,
                "label": spec.label,
                "registered_model": spec.registered_model_fqdn,
                "register_type": spec.register_type,
                "candidate_id": spec.register_candidate_id,
            }
        )
    return rows


def register(model_id: str, *, config_path: str | None = None, wait: bool = True) -> dict[str, Any]:
    spec = get_serving_spec(model_id, config_path=config_path)
    if spec.register_type == "oss":
        if not spec.register_candidate_id:
            raise ValueError(f"{model_id} is missing model_serving.models.{model_id}.register.candidate_id")
        return _run_repo_script(
            "scripts/asr/10_register_multilingual_asr_candidates.sh",
            ["register-one"],
            {"ASR_ML_REGISTER_CANDIDATE": spec.register_candidate_id},
            wait=wait,
        )
    if spec.register_type == "finetuned_whisper":
        leaf = spec.registered_model_fqdn.rsplit(".", 1)[-1]
        return _run_repo_script(
            "scripts/asr/05_register_asr_model_candidate.sh",
            ["register-candidate"],
            {"ASR_REGISTERED_MODEL_NAME": leaf},
            wait=wait,
        )
    raise ValueError(f"{model_id} has no register.type in config/ml_asr_eval.yaml model_serving.models")


def register_all(*, config_path: str | None = None, wait: bool = True) -> list[dict[str, Any]]:
    results = []
    for spec in load_serving_specs(config_path=config_path):
        if not spec.register_type:
            continue
        results.append(register(spec.model_id, config_path=config_path, wait=wait))
    return results


def _run_repo_script(
    script_relpath: str,
    args: list[str],
    env: dict[str, str],
    *,
    wait: bool,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[4]
    script = root / script_relpath
    if not script.is_file():
        raise FileNotFoundError(f"Registration script not found: {script}")
    merged = os.environ.copy()
    merged.update(env)
    profile = merged.get("ML_ASR_DATABRICKS_PROFILE") or merged.get("DATABRICKS_CONFIG_PROFILE")
    if profile:
        merged.setdefault("ASR_DATABRICKS_PROFILE", profile)
        merged.setdefault("ASR_ML_REGISTER_PROFILE", profile)
        merged.setdefault("DATABRICKS_CONFIG_PROFILE", profile)
    cmd = [str(script), *args]
    if wait:
        subprocess.run(cmd, cwd=root, env=merged, check=True)
        return {"script": script_relpath, "args": args, "status": "completed"}
    proc = subprocess.Popen(cmd, cwd=root, env=merged)
    return {"script": script_relpath, "args": args, "status": "started", "pid": proc.pid}
