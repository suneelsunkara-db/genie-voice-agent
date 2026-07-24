from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import yaml

from genie_voice.ml_asr.config import DEFAULT_CONFIG_PATH, load_config
from genie_voice.ml_asr.serving.catalog import ServingSpec, get_serving_spec


def register_oss(model_id: str, *, config_path: str | None = None, wait: bool = True) -> dict[str, Any]:
    spec = get_serving_spec(model_id, config_path=config_path)
    register_cfg = _register_config(model_id, config_path=config_path)
    if register_cfg.get("type") != "oss":
        raise ValueError(f"{model_id} is not an OSS registration target")

    eval_config = load_config(config_path=config_path)
    remote_root = f"{eval_config.remote_training_root}/evaluations/ml_asr_eval/registered_models"
    local_root = Path(eval_config.local_root) / "registered_models"
    candidate_id = str(register_cfg.get("candidate_id") or spec.register_candidate_id or "")
    if not candidate_id:
        raise ValueError(f"{model_id} is missing register.candidate_id in config/ml_asr_eval.yaml")

    profile = _databricks_profile()
    dbx = ["databricks", "--profile", profile] if profile else ["databricks"]
    jobs_dir = f"{remote_root}/jobs"
    outputs_dir = f"{remote_root}/outputs/{candidate_id}"
    output_json_remote = f"{outputs_dir}/registration.json"

    _copy_registration_assets(dbx, jobs_dir)
    local_root.mkdir(parents=True, exist_ok=True)
    local_candidate_dir = local_root / candidate_id
    local_candidate_dir.mkdir(parents=True, exist_ok=True)

    job_json = local_candidate_dir / "register.job.json"
    run_json = local_candidate_dir / "register_run.json"
    job_json.write_text(
        json.dumps(
            _registration_job(
                spec=spec,
                register_cfg=register_cfg,
                candidate_id=candidate_id,
                remote_root=remote_root,
                output_json_remote=output_json_remote,
                environment_version=eval_config.serverless_environment_version,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    subprocess.run(dbx + ["fs", "mkdirs", f"dbfs:{outputs_dir}"], check=True)
    proc = subprocess.run(
        dbx + ["api", "post", "/api/2.1/jobs/runs/submit", "--json", f"@{job_json}"],
        capture_output=True,
        text=True,
        check=True,
    )
    run_json.write_text(proc.stdout, encoding="utf-8")
    run_id = str(json.loads(proc.stdout)["run_id"])
    if wait:
        _wait_for_job_run(dbx, run_id, label=f"ml-asr register {candidate_id}")
        subprocess.run(
            dbx + ["fs", "cp", f"dbfs:{output_json_remote}", str(local_candidate_dir / "registration.json"), "--overwrite"],
            check=True,
        )
    return {
        "model_id": model_id,
        "candidate_id": candidate_id,
        "registered_model": spec.registered_model_fqdn,
        "run_id": run_id,
        "status": "completed" if wait else "started",
        "local_registration_json": str(local_candidate_dir / "registration.json"),
    }


def _register_config(model_id: str, *, config_path: str | None) -> dict[str, Any]:
    path = Path(config_path or DEFAULT_CONFIG_PATH)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    entry = ((raw.get("model_serving") or {}).get("models") or {}).get(model_id) or {}
    register = entry.get("register") if isinstance(entry.get("register"), dict) else {}
    return dict(register)


def _registration_job(
    *,
    spec: ServingSpec,
    register_cfg: dict[str, Any],
    candidate_id: str,
    remote_root: str,
    output_json_remote: str,
    environment_version: int,
) -> dict[str, Any]:
    family = str(register_cfg["family"])
    parameters = [
        "--candidate-id",
        candidate_id,
        "--family",
        family,
        "--base-model",
        str(register_cfg["base_model"]),
        "--language-code",
        str(register_cfg["language_code"]),
        "--language-name",
        str(register_cfg["language_name"]),
        "--adaptation-type",
        str(register_cfg.get("adaptation_type") or "oss_baseline"),
        "--fine-tuned-by-us",
        "true" if register_cfg.get("fine_tuned_by_us") else "false",
        "--registered-model",
        spec.registered_model_fqdn,
        "--package-dir",
        f"{remote_root}/packages",
        "--wrapper-path",
        f"{remote_root}/jobs/mlflow_oss_asr_pyfunc.py",
        "--requirements-dir",
        f"{remote_root}/jobs",
        "--output-json",
        output_json_remote,
    ]
    if family == "funasr":
        parameters.extend(["--funasr-hub", str(register_cfg.get("funasr_hub") or "ms")])
        if register_cfg.get("funasr_vad_model"):
            parameters.extend(["--funasr-vad-model", str(register_cfg["funasr_vad_model"])])
        if register_cfg.get("funasr_variant"):
            parameters.extend(["--funasr-variant", str(register_cfg["funasr_variant"])])
    if os.environ.get("ML_ASR_REGISTER_FORCE_DOWNLOAD", "").lower() == "true":
        parameters.append("--force-download")

    deps = [
        "mlflow",
        "huggingface_hub",
        "torch",
        "transformers",
        "accelerate",
        "qwen-asr",
        "librosa",
        "soundfile",
        "pandas",
    ]
    if family == "funasr":
        deps.extend(["funasr", "modelscope"])

    return {
        "run_name": f"ml-asr-register-{candidate_id}",
        "tasks": [
            {
                "task_key": "register_candidate",
                "environment_key": "ml_asr_register_env",
                "spark_python_task": {
                    "python_file": f"dbfs:{remote_root}/jobs/databricks_register_oss_candidate.py",
                    "parameters": parameters,
                },
            }
        ],
        "environments": [
            {
                "environment_key": "ml_asr_register_env",
                "spec": {
                    "environment_version": str(environment_version),
                    "dependencies": deps,
                },
            }
        ],
    }


def _copy_registration_assets(dbx: list[str], jobs_dir: str) -> None:
    repo_root = Path(__file__).resolve().parents[4]
    ml_asr_scripts = repo_root / "scripts" / "ml_asr"
    assets = [
        "databricks_register_oss_candidate.py",
        "mlflow_oss_asr_pyfunc.py",
        "funasr_serving_requirements.txt",
    ]
    subprocess.run(dbx + ["fs", "mkdirs", f"dbfs:{jobs_dir}"], check=True)
    for name in assets:
        subprocess.run(
            dbx + ["fs", "cp", str(ml_asr_scripts / name), f"dbfs:{jobs_dir}/{name}", "--overwrite"],
            check=True,
        )


def _wait_for_job_run(dbx: list[str], run_id: str, *, label: str) -> None:
    while True:
        proc = subprocess.run(
            dbx + ["jobs", "get-run", run_id, "--output", "json"],
            capture_output=True,
            text=True,
            check=True,
        )
        payload = json.loads(proc.stdout)
        state = payload.get("state") or {}
        lifecycle = str(state.get("life_cycle_state") or "")
        result = str(state.get("result_state") or "")
        url = str(payload.get("run_page_url") or "")
        print(f"  lifecycle={lifecycle} result={result} url={url}")
        if lifecycle in {"TERMINATED", "SKIPPED", "INTERNAL_ERROR"}:
            if result != "SUCCESS":
                raise RuntimeError(f"Databricks {label} failed: lifecycle={lifecycle} result={result} url={url}")
            return
        time.sleep(20)


def _databricks_profile() -> str:
    """Databricks CLI profile from config (databricks.profile in config.local.yaml)."""
    from genie_voice.ml_asr.runtime import databricks_profile

    return databricks_profile() or ""
