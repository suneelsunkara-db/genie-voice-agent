from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from genie_voice.ml_asr.serving.catalog import get_serving_spec, load_serving_specs
from genie_voice.ml_asr.serving.register_oss import register_oss


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
        return register_oss(model_id, config_path=config_path, wait=wait)
    if spec.register_type == "finetuned_whisper":
        return _run_legacy_script(
            "scripts/asr/05_register_asr_model_candidate.sh",
            ["register-candidate"],
            {"ASR_REGISTERED_MODEL_NAME": spec.registered_model_fqdn.rsplit(".", 1)[-1]},
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


def _run_legacy_script(
    script_relpath: str,
    args: list[str],
    env: dict[str, str],
    *,
    wait: bool,
) -> dict[str, Any]:
    """Temporary bridge for finetuned EN until that path is migrated off scripts/asr."""
    root = Path(__file__).resolve().parents[4]
    script = root / script_relpath
    if not script.is_file():
        raise FileNotFoundError(f"Registration script not found: {script}")
    merged = os.environ.copy()
    merged.update(env)
    profile = merged.get("ML_ASR_DATABRICKS_PROFILE") or merged.get("DATABRICKS_CONFIG_PROFILE")
    if profile:
        merged.setdefault("ASR_DATABRICKS_PROFILE", profile)
        merged.setdefault("DATABRICKS_CONFIG_PROFILE", profile)
    cmd = [str(script), *args]
    if wait:
        subprocess.run(cmd, cwd=root, env=merged, check=True)
        return {"script": script_relpath, "args": args, "status": "completed"}
    proc = subprocess.Popen(cmd, cwd=root, env=merged)
    return {"script": script_relpath, "args": args, "status": "started", "pid": proc.pid}
