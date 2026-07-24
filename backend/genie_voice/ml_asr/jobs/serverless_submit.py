from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from genie_voice.ml_asr.config import EvalConfig, load_config


def submit_serverless_step(
    *,
    action: str,
    config_path: str | None = None,
    params: dict[str, Any] | None = None,
    wait: bool = True,
) -> dict[str, Any]:
    config = load_config(config_path=config_path)
    from genie_voice.ml_asr.runtime import databricks_profile

    profile = databricks_profile()
    dbx = ["databricks"]
    if profile:
        dbx = ["databricks", "--profile", profile]

    package_root = _sync_job_package(config, dbx)
    remote_config = f"{config.remote_results_dir}/ml_asr_eval.yaml"
    for remote_dir in {
        config.remote_results_dir,
        config.remote_jobs_dir,
        config.remote_datasets_root,
        config.remote_manifests_dir,
    }:
        subprocess.run([*dbx, "fs", "mkdirs", f"dbfs:{remote_dir}"], check=False)
    _copy_file(dbx, Path(config.config_path), remote_config)

    worker_remote = f"{config.remote_jobs_dir}/package/serverless_worker.py"
    job_json = Path(config.local_root) / "jobs" / f"{action}_job.json"
    run_json = Path(config.local_root) / "jobs" / f"{action}_run.json"
    job_json.parent.mkdir(parents=True, exist_ok=True)

    parameters = [
        "--config",
        remote_config,
        "--package-root",
        f"{config.remote_jobs_dir}/package",
        action,
    ]
    for key, value in (params or {}).items():
        if value is None:
            continue
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, list):
            for item in value:
                parameters.extend([flag, str(item)])
        elif isinstance(value, bool):
            if value:
                parameters.append(flag)
        else:
            parameters.extend([flag, str(value)])

    job_payload = {
        "run_name": f"ml-asr-{action}",
        "tasks": [
            {
                "task_key": "ml_asr_worker",
                "environment_key": "ml_asr_env",
                "spark_python_task": {
                    "python_file": f"dbfs:{worker_remote}",
                    "parameters": parameters,
                },
            }
        ],
        "environments": [
            {
                "environment_key": "ml_asr_env",
                "spec": {
                    "environment_version": str(config.serverless_environment_version),
                    "dependencies": _serverless_dependencies(action),
                    "environment_variables": _serverless_environment_variables(config),
                },
            }
        ],
    }
    job_json.write_text(json.dumps(job_payload, indent=2), encoding="utf-8")

    subprocess.run([*dbx, "fs", "mkdirs", f"dbfs:{config.remote_jobs_dir}"], check=False)
    proc = subprocess.run(
        [*dbx, "api", "post", "/api/2.1/jobs/runs/submit", "--json", f"@{job_json}"],
        check=True,
        capture_output=True,
        text=True,
    )
    run_json.write_text(proc.stdout, encoding="utf-8")
    payload = json.loads(proc.stdout)
    run_id = int(payload["run_id"])
    result = {"run_id": run_id, "run_json": str(run_json), "package_root": package_root, "action": action}
    if wait:
        result["state"] = _wait_for_run(dbx, run_id)
    return result


def _serverless_dependencies(action: str) -> list[str]:
    if action == "summarize":
        # Minimal deps — avoid numpy/pandas pins that break Databricks pyspark on worker boot.
        return ["pyyaml>=6", "python-dotenv>=1", "pydantic>=2"]

    base = [
        "pyyaml>=6",
        "databricks-sdk>=0.40",
        "python-dotenv>=1",
        "pydantic>=2",
        "httpx>=0.27",
    ]
    if action in {"prepare", "audit-dataset", "dataset-eval", "evaluate", "evaluate-all"}:
        base.extend(
            [
                "numpy>=1.26",
                "pandas>=2",
                "pyarrow>=15",
                "soundfile>=0.12",
                "huggingface_hub>=0.23",
            ]
        )
    return base


def _serverless_environment_variables(config: EvalConfig) -> dict[str, str]:
    env = {
        "ML_ASR_RUN_MODE": "serverless",
        "ML_ASR_PACKAGE_ROOT": f"{config.remote_jobs_dir}/package",
        "ML_ASR_JOBS_DIR": config.remote_jobs_dir,
        "ML_ASR_DEEPGRAM_SECRET_SCOPE": config.serverless_deepgram_secret_scope,
        "ML_ASR_DEEPGRAM_SECRET_KEY": config.serverless_deepgram_secret_key,
        "GENIE_CONFIG": f"{config.remote_jobs_dir}/package/config/config.yaml",
    }
    # HF token for model-weight downloads on the job. Source of truth is config
    # (secrets.hf_token in config.local.yaml); env is only a fallback.
    from genie_voice.config import get_settings

    hf = get_settings().secrets.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf:
        env["HF_TOKEN"] = hf
        env["HUGGING_FACE_HUB_TOKEN"] = hf
    return env


def _sync_job_package(config: EvalConfig, dbx: list[str]) -> str:
    repo_root = Path(config.config_path).resolve().parents[1]
    backend_root = repo_root / "backend"
    local_stage = Path(config.local_root) / "jobs" / "package"
    if local_stage.exists():
        import shutil

        shutil.rmtree(local_stage)
    local_stage.mkdir(parents=True, exist_ok=True)

    import shutil

    def _ignore_package(path: str, names: list[str]) -> set[str]:
        ignored: set[str] = set()
        for name in names:
            if name == "__pycache__" or name.endswith(".pyc"):
                ignored.add(name)
        return ignored

    shutil.copytree(
        backend_root / "genie_voice",
        local_stage / "genie_voice",
        ignore=_ignore_package,
    )
    config_dir = local_stage / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(repo_root / "config" / "config.yaml", config_dir / "config.yaml")
    local_config = repo_root / "config" / "config.local.yaml"
    if local_config.is_file():
        shutil.copy2(local_config, config_dir / "config.local.yaml")
    shutil.copy2(repo_root / "config" / "ml_asr_eval.yaml", local_stage / "ml_asr_eval.yaml")
    worker_src = backend_root / "genie_voice" / "ml_asr" / "jobs" / "serverless_worker.py"
    shutil.copy2(worker_src, local_stage / "serverless_worker.py")

    remote_root = config.remote_jobs_dir
    subprocess.run([*dbx, "fs", "rm", "-r", f"dbfs:{remote_root}/package"], check=False)
    subprocess.run([*dbx, "fs", "cp", "-r", str(local_stage), f"dbfs:{remote_root}/package", "--overwrite"], check=True)
    return f"dbfs:{remote_root}/package"


def _copy_file(dbx: list[str], local_path: Path, remote_path: str) -> None:
    subprocess.run([*dbx, "fs", "cp", str(local_path), f"dbfs:{remote_path}", "--overwrite"], check=True)


def _wait_for_run(dbx: list[str], run_id: int) -> dict[str, Any]:
    while True:
        proc = subprocess.run([*dbx, "jobs", "get-run", str(run_id), "--output", "json"], capture_output=True, text=True, check=True)
        payload = json.loads(proc.stdout)
        state = payload.get("state") or {}
        lifecycle = state.get("life_cycle_state")
        result = state.get("result_state")
        if lifecycle in {"TERMINATED", "SKIPPED", "INTERNAL_ERROR"}:
            if result != "SUCCESS":
                raise RuntimeError(f"Databricks run {run_id} failed: {json.dumps(state)}")
            return {
                "run_id": run_id,
                "lifecycle": lifecycle,
                "result": result,
                "run_page_url": payload.get("run_page_url"),
            }
        time.sleep(20)
