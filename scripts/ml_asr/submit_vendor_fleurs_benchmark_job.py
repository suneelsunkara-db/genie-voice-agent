"""Submit a separate Databricks job for measured vendor FLEURS STT baselines.

This does not change the benchmark table schema and does not collide with the
active Genie FLEURS job. Vendor results write a distinct dataset id:

  - fleurs_deepgram_stt

FLEURS is an STT benchmark, so only speech-to-text vendors are run here; TTS
quality is deliberately out of scope (see Benchmarks/MultilingualVoice/eval.sh).
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
_BENCHMARK_DIR = _REPO / "Benchmarks" / "MultilingualVoice"
sys.path.insert(0, str(_BENCHMARK_DIR))
sys.path.insert(0, str(_HERE))
os.environ.setdefault("MLV_BENCHMARK_DIR", str(_BENCHMARK_DIR))

from _realtime_config import databricks, load_config  # noqa: E402
from paths import (  # noqa: E402
    benchmark_results_dir,
    databricks_host,
    delta_catalog,
    delta_schema,
)
import languages as langmap  # noqa: E402


SECRET_SCOPE = "genie-voice"
DEEPGRAM_SECRET_KEY = "deepgram_api_key"
ELEVENLABS_SECRET_KEY = "elevenlabs_api_key"
HF_SECRET_KEY = "benchmark_hf_token"


def _upload_benchmark_code(client: WorkspaceClient, workspace_dir: str) -> tuple[str, str, str]:
    import subprocess

    dbx = ["databricks"]
    if client.config.profile:
        dbx += ["--profile", client.config.profile]

    benchmark_ws = f"{workspace_dir}/MultilingualVoice"
    config_ws = f"{workspace_dir}/config"
    excludes = [
        "results/**", "logs/**", "**/__pycache__/**", "**/*.pyc",
        "**/.venv/**", "**/.gitkeep", "**/benchmark_run.log",
    ]
    sync_cmd = [*dbx, "sync", str(_BENCHMARK_DIR), benchmark_ws, "--full"]
    for pattern in excludes:
        sync_cmd.extend(["--exclude", pattern])
    subprocess.run(sync_cmd, check=True)
    subprocess.run(
        [*dbx, "sync", str(_REPO / "config"), config_ws, "--full", "--exclude", "config.local.yaml"],
        check=True,
    )
    return f"{benchmark_ws}/vendor_fleurs_benchmark.py", f"{config_ws}/config.yaml", benchmark_ws


def _stage_vendor_secrets(client: WorkspaceClient, config: dict) -> None:
    import subprocess

    dbx = ["databricks"]
    if client.config.profile:
        dbx += ["--profile", client.config.profile]

    secrets = config.get("secrets") or {}
    deepgram_key = str(secrets.get("deepgram_api_key") or "").strip()
    elevenlabs_key = str(secrets.get("elevenlabs_api_key") or "").strip()
    hf_token = str(secrets.get("hf_token") or "").strip()
    subprocess.run([*dbx, "secrets", "create-scope", SECRET_SCOPE], check=False, capture_output=True)
    if deepgram_key:
        subprocess.run([*dbx, "secrets", "put-secret", SECRET_SCOPE, DEEPGRAM_SECRET_KEY,
                        "--string-value", deepgram_key], check=True)
    if elevenlabs_key:
        subprocess.run([*dbx, "secrets", "put-secret", SECRET_SCOPE, ELEVENLABS_SECRET_KEY,
                        "--string-value", elevenlabs_key], check=True)
    if hf_token:
        subprocess.run([*dbx, "secrets", "put-secret", SECRET_SCOPE, HF_SECRET_KEY,
                        "--string-value", hf_token], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit vendor FLEURS benchmark job")
    parser.add_argument("--vendors", default="deepgram")
    parser.add_argument("--languages", default="", help="comma-separated 2-letter codes")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument(
        "--max-parallel", type=int, default=6,
        help="max concurrent vendor/language tasks",
    )
    parser.add_argument("--wait", action="store_true", help="Block until the run finishes")
    args = parser.parse_args()

    config = load_config()
    db = databricks(config)
    profile = db.get("profile")
    run_as = db.get("run_as") or "me"
    env_version = str((config.get("pipeline") or {}).get("environment_version") or "3")
    host = databricks_host()
    results_dir = str(benchmark_results_dir())

    workspace_dir = f"/Workspace/Users/{run_as}/multilingual_voice_vendor_benchmark"
    client = WorkspaceClient(profile=profile) if profile else WorkspaceClient()
    remote_runner, remote_config, benchmark_ws = _upload_benchmark_code(client, workspace_dir)
    _stage_vendor_secrets(client, config)

    from databricks.sdk.service import compute

    run_id = "vendors_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base_params = [
        "--vendors", args.vendors,
        "--limit", str(args.limit),
        "--out-dir", results_dir,
        "--run-id", run_id,
        "--benchmark-dir", benchmark_ws,
        "--config", remote_config,
        "--databricks-host", host,
        "--secret-scope", SECRET_SCOPE,
        "--deepgram-secret-key", DEEPGRAM_SECRET_KEY,
        "--elevenlabs-secret-key", ELEVENLABS_SECRET_KEY,
        "--hf-secret-key", HF_SECRET_KEY,
    ]
    requested = [c.strip() for c in args.languages.split(",") if c.strip()] or None
    languages = langmap.resolve_languages("fleurs", requested)
    vendors = [v.strip() for v in args.vendors.split(",") if v.strip()]
    pairs = [(vendor, lang) for vendor in vendors for lang in languages]
    if not pairs:
        raise SystemExit("no vendor/language pairs resolved")

    lanes = max(1, min(args.max_parallel, len(pairs)))
    lane_tail: list[str | None] = [None] * lanes
    tasks: list[jobs.Task] = []
    for i, (vendor, lang) in enumerate(pairs):
        lane = i % lanes
        dep_key = lane_tail[lane]
        task_key = f"{vendor}_{lang}"
        task_params = base_params + ["--vendor", vendor, "--language", lang]
        depends = [jobs.TaskDependency(task_key=dep_key)] if dep_key else None
        tasks.append(jobs.Task(
            task_key=task_key,
            environment_key="mlv_vendor_env",
            depends_on=depends,
            run_if=jobs.RunIf.ALL_DONE if dep_key else None,
            spark_python_task=jobs.SparkPythonTask(
                python_file=remote_runner,
                parameters=task_params,
            ),
            max_retries=1,
            min_retry_interval_millis=30_000,
            retry_on_timeout=True,
            timeout_seconds=3600,
        ))
        lane_tail[lane] = task_key

    created = client.jobs.create(
        name=f"vendor-fleurs-benchmark-{run_id}",
        tasks=tasks,
        environments=[jobs.JobEnvironment(
            environment_key="mlv_vendor_env",
            spec=compute.Environment(
                environment_version=env_version,
                dependencies=[
                    "pyyaml>=6",
                    "huggingface_hub>=0.23",
                    "fsspec>=2023.1",
                    "pyarrow>=15",
                    "numpy>=1.26",
                    "soundfile>=0.12",
                    "databricks-sdk>=0.40",
                ],
            ),
        )],
        tags={"benchmark": "multilingual_voice_vendor", "run_id": run_id},
    )
    run = client.jobs.run_now(job_id=created.job_id)
    print(f"job_id: {created.job_id}")
    print(f"run_id: {run.run_id} (sweep run_id={run_id})")
    print(f"run_page_url: {client.config.host}/jobs/runs/{run.run_id}")
    print(f"results: Delta tables {delta_catalog()}.{delta_schema()}.benchmark_runs + benchmark_samples")
    print(f"tasks: {len(tasks)} ({len(vendors)} vendors x {len(languages)} languages, lanes={lanes})")
    if args.wait:
        result = run.result()
        print(f"final state: {result.state}")


if __name__ == "__main__":
    main()
