"""Submit serverless Databricks jobs to register + deploy realtime voice agents.

Runs LOCALLY (uses the databricks CLI + the merged config). For each requested
candidate it stages the agent code to a UC Volume and submits a serverless
one-time run with:

    register_<candidate>  (serverless)  ->  deploy_<candidate>  (serverless)

Registration packages the OSS checkpoint as an MLflow ResponsesAgent into UC and
sets the ``candidate`` alias; deployment promotes that alias to the agent Model
Serving endpoint. GPU is only used by the serving endpoint itself, so both job
tasks run on light serverless compute.

Examples:
    python scripts/ml_asr/submit_realtime_voice_jobs.py            # all candidates
    python scripts/ml_asr/submit_realtime_voice_jobs.py --stt qwen3_asr_1_7b_multilingual
    python scripts/ml_asr/submit_realtime_voice_jobs.py --no-deploy --wait
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from _realtime_config import databricks, load_config, realtime_voice

_HERE = Path(__file__).resolve().parent
_AGENT_FILES = [
    "register_realtime_voice_agent.py",
    "deploy_realtime_voice_models.py",
    "realtime_stt_agent.py",
    "realtime_tts_agent.py",
    "_realtime_config.py",
]


def _dbx(profile: str | None) -> list[str]:
    return ["databricks"] + (["--profile", profile] if profile else [])


def _run(cmd: list[str], *, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=check, capture_output=capture, text=True)


def _upload_agents(dbx: list[str], workspace_dir: str) -> dict[str, str]:
    """Stage agent files as Workspace Files (serverless-readable python_file)."""
    stage = Path(tempfile.mkdtemp(prefix="realtime_voice_agents_"))
    try:
        for name in _AGENT_FILES:
            shutil.copy2(_HERE / name, stage / name)
        _run([*dbx, "workspace", "import-dir", str(stage), workspace_dir, "--overwrite"])
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    return {name: f"{workspace_dir}/{name}" for name in _AGENT_FILES}


def _register_task(cid: str, candidate: dict, remote: dict[str, str], catalog: str, schema: str) -> dict:
    wrapper = remote["realtime_stt_agent.py" if candidate["modality"] == "stt" else "realtime_tts_agent.py"]
    registered = f"{catalog}.{schema}.{candidate['registered_model']}"
    return {
        "task_key": f"register_{cid}",
        "environment_key": "register_env",
        "spark_python_task": {
            "python_file": remote["register_realtime_voice_agent.py"],
            "parameters": [
                "--candidate-id", cid,
                "--modality", candidate["modality"],
                "--base-model", candidate["base_model"],
                "--registered-model", registered,
                "--package-dir", "/tmp/realtime_voice_packages",
                "--wrapper-path", wrapper,
                "--wrapper-class", candidate["wrapper_class"],
                "--output-json", f"/tmp/{cid}_registered.json",
            ],
        },
    }


def _deploy_task(cid: str, candidate: dict, remote: dict[str, str], serving: dict, catalog: str, schema: str) -> dict:
    registered = f"{catalog}.{schema}.{candidate['registered_model']}"
    params = [
        "--registered-model", registered,
        "--endpoint", candidate["endpoint"],
        "--workload-type", candidate.get("workload_type", serving.get("workload_type", "GPU_MEDIUM")),
        "--workload-size", candidate.get("workload_size", serving.get("workload_size", "Small")),
    ]
    if candidate.get("scale_to_zero", serving.get("scale_to_zero", False)):
        params.append("--scale-to-zero")
    return {
        "task_key": f"deploy_{cid}",
        "environment_key": "deploy_env",
        "depends_on": [{"task_key": f"register_{cid}"}],
        "spark_python_task": {"python_file": remote["deploy_realtime_voice_models.py"], "parameters": params},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stt", default=None, help="STT candidate id (default: all)")
    parser.add_argument("--tts", default=None, help="TTS candidate id (default: all)")
    parser.add_argument("--no-deploy", action="store_true", help="Register only")
    parser.add_argument("--wait", action="store_true", help="Block until the run finishes")
    args = parser.parse_args()

    config = load_config()
    rv = realtime_voice(config)
    db = databricks(config)
    profile = db.get("profile")
    catalog, schema = db["catalog"], db["schema"]
    serving = rv.get("serving") or {}
    env_version = str((config.get("pipeline") or {}).get("environment_version") or "3")
    hf_token = ((config.get("secrets") or {}).get("hf_token") or "").strip()

    stt = rv.get("stt_candidates") or {}
    tts = rv.get("tts_candidates") or {}
    selected: dict[str, dict] = {}
    if args.stt or args.tts:
        # Explicit selection: only the named candidate(s).
        if args.stt:
            selected[args.stt] = stt[args.stt]
        if args.tts:
            selected[args.tts] = tts[args.tts]
    else:
        selected.update(stt)
        selected.update(tts)
    if not selected:
        parser.error("No candidates selected/configured")

    run_as = db.get("run_as") or "me"
    code_dir = (serving.get("code_dir") or "").strip() or f"/Workspace/Users/{run_as}/realtime_voice_agents"

    dbx = _dbx(profile)
    remote = _upload_agents(dbx, code_dir)

    tasks: list[dict] = []
    for cid, candidate in selected.items():
        tasks.append(_register_task(cid, candidate, remote, catalog, schema))
        if not args.no_deploy:
            tasks.append(_deploy_task(cid, candidate, remote, serving, catalog, schema))

    register_env_vars = {"HF_TOKEN": hf_token, "HUGGING_FACE_HUB_TOKEN": hf_token} if hf_token else {}
    payload = {
        "run_name": "realtime-voice-register-deploy",
        "tasks": tasks,
        "environments": [
            {
                "environment_key": "register_env",
                "spec": {
                    "environment_version": env_version,
                    "dependencies": ["mlflow>=2.20", "huggingface_hub>=0.23", "pydantic>=2"],
                    "environment_variables": register_env_vars,
                },
            },
            {
                "environment_key": "deploy_env",
                "spec": {
                    "environment_version": env_version,
                    "dependencies": ["databricks-sdk>=0.40"],
                },
            },
        ],
    }

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(payload, handle, indent=2)
        job_json = handle.name

    proc = _run([*dbx, "api", "post", "/api/2.1/jobs/runs/submit", "--json", f"@{job_json}"], capture=True)
    submitted = json.loads(proc.stdout)
    run_id = int(submitted["run_id"])
    host = db.get("host", "").rstrip("/")
    print(json.dumps({"run_id": run_id, "run_page_url": f"{host}/jobs/runs/{run_id}", "tasks": [t["task_key"] for t in tasks]}, indent=2))

    if args.wait:
        print(json.dumps(_wait(dbx, run_id), indent=2))


def _wait(dbx: list[str], run_id: int) -> dict:
    while True:
        proc = _run([*dbx, "jobs", "get-run", str(run_id), "--output", "json"], capture=True)
        payload = json.loads(proc.stdout)
        state = payload.get("state") or {}
        lifecycle = state.get("life_cycle_state")
        if lifecycle in {"TERMINATED", "SKIPPED", "INTERNAL_ERROR"}:
            return {
                "run_id": run_id,
                "lifecycle": lifecycle,
                "result": state.get("result_state"),
                "run_page_url": payload.get("run_page_url"),
            }
        time.sleep(20)


if __name__ == "__main__":
    main()
