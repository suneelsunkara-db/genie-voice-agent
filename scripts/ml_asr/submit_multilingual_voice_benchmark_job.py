"""Submit a Databricks job with one task per (dataset, language) pair.

Each task runs ``run_benchmark.py`` for a single pair, writes a row to the
``benchmark_runs`` Delta table (plus per-sample rows to ``benchmark_samples``),
and exits. The platform provides parallelism, retry, timeout, and run history.

Resume = re-run the same job (same ``run_id``); each task checks Delta for its
pair in that ``run_id`` and exits early if already complete.

Usage:
    python scripts/ml_asr/submit_multilingual_voice_benchmark_job.py
    python scripts/ml_asr/submit_multilingual_voice_benchmark_job.py --dataset fleurs --languages en,ja --limit 5 --wait
    python scripts/ml_asr/submit_multilingual_voice_benchmark_job.py --resume  # re-run with prior run_id
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

# paths.py resolves local config from MLV_BENCHMARK_DIR. The submitter already
# knows that dir, so export it here rather than relying on a pre-set shell env
# var (the job tasks receive it via --benchmark-dir separately).
os.environ.setdefault("MLV_BENCHMARK_DIR", str(_BENCHMARK_DIR))

from _realtime_config import databricks, load_config  # noqa: E402
from paths import (  # noqa: E402
    benchmark_api_host,
    benchmark_api_prefix,
    benchmark_results_dir,
    benchmark_sp_credentials,
    databricks_host,
    delta_catalog,
    delta_schema,
)

SECRET_SCOPE = "genie-voice"
SP_SECRET_KEY = "benchmark_sp_client_secret"
HF_SECRET_KEY = "benchmark_hf_token"
APP_NAME = "genie-voice-agent"


def _upload_benchmark_code(client: WorkspaceClient, workspace_dir: str) -> tuple[str, str]:
    """Sync benchmark modules + committed config to Workspace Files.

    config.local.yaml is deliberately NOT synced — secrets travel via the
    secret scope, not the workspace tree.
    """
    import subprocess

    profile = client.config.auth_type
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
    return f"{benchmark_ws}/run_benchmark.py", f"{config_ws}/config.yaml", benchmark_ws


def _stage_secrets(client: WorkspaceClient, *, client_secret: str, hf_token: str) -> None:
    """Idempotently store job secrets in the secret scope (never in job params)."""
    import subprocess

    dbx = ["databricks"]
    if client.config.profile:
        dbx += ["--profile", client.config.profile]
    subprocess.run([*dbx, "secrets", "create-scope", SECRET_SCOPE], check=False, capture_output=True)
    subprocess.run([*dbx, "secrets", "put-secret", SECRET_SCOPE, SP_SECRET_KEY,
                    "--string-value", client_secret], check=True)
    if hf_token:
        subprocess.run([*dbx, "secrets", "put-secret", SECRET_SCOPE, HF_SECRET_KEY,
                        "--string-value", hf_token], check=True)


def _build_tasks(
    pairs: list[tuple[str, str]],
    *,
    remote_runner: str,
    remote_config: str,
    benchmark_dir: str,
    api_host: str,
    api_prefix: str,
    results_dir: str,
    host: str,
    client_id: str,
    run_id: str,
    hf_token: str,
    limit: int,
    tts_roundtrip: bool,
    max_audio_seconds: float,
    env_version: str,
    dataset_sel: str,
    languages_sel: str,
    max_parallel: int,
) -> list[jobs.Task]:
    base_params = [
        "--transport", "ws",
        "--api-host", api_host,
        "--api-prefix", api_prefix,
        "--limit", str(limit),
        "--out-dir", results_dir,
        "--run-id", run_id,
        "--benchmark-dir", benchmark_dir,
        "--config", remote_config,
        "--databricks-host", host,
        "--sp-client-id", client_id,
        "--secret-scope", SECRET_SCOPE,
        "--sp-secret-key", SP_SECRET_KEY,
    ]
    if hf_token:
        base_params += ["--hf-secret-key", HF_SECRET_KEY]
    if tts_roundtrip:
        base_params += ["--tts-roundtrip"]
    base_params += ["--max-audio-seconds", str(max_audio_seconds)]

    # Phase 1: one prepare task streams every pair from HuggingFace and stages
    # compact PCM artifacts to the Volume. It runs pairs sequentially, so the
    # heavyweight parquet decode is bounded to one pair's memory — never on the
    # parallel benchmark tasks (which previously OOM-killed the driver).
    prep_params = base_params + ["--mode", "prepare", "--dataset", dataset_sel]
    if languages_sel.strip():
        prep_params += ["--languages", languages_sel]
    prepare_task = jobs.Task(
        task_key="prepare",
        environment_key="mlv_env",
        spark_python_task=jobs.SparkPythonTask(
            python_file=remote_runner,
            parameters=prep_params,
        ),
        max_retries=1,  # idempotent: re-run skips already-staged pairs
        min_retry_interval_millis=30_000,
        retry_on_timeout=True,
        timeout_seconds=10_800,  # 3h to stage all pairs sequentially
    )

    # Phase 2: one benchmark task per pair (granular Delta results), but run them
    # in a bounded number of sequential "lanes" so only ~max_parallel WebSocket
    # clients hit the realtime app + serving endpoints at once. Running all pairs
    # at once overwhelms the shared endpoints and per-turn latency balloons into
    # minutes (queueing), blowing past the task timeout. Lanes cap concurrency
    # while keeping wall time reasonable.
    #
    # Each lane is a chain: pair -> pair -> pair. The lane head depends on the
    # prepare task (ALL_SUCCESS: no data => don't benchmark). Later pairs depend
    # on the previous pair in the lane with ALL_DONE, so one failed/timed-out pair
    # doesn't skip the rest of its lane.
    tasks: list[jobs.Task] = [prepare_task]
    lanes = max(1, min(max_parallel, len(pairs)))
    lane_tail = ["prepare"] * lanes
    for i, (dataset, lang) in enumerate(pairs):
        lane = i % lanes
        dep_key = lane_tail[lane]
        run_if = jobs.RunIf.ALL_SUCCESS if dep_key == "prepare" else jobs.RunIf.ALL_DONE
        key = f"{dataset}_{lang}"
        params = base_params + ["--mode", "benchmark", "--dataset", dataset, "--language", lang]
        tasks.append(jobs.Task(
            task_key=key,
            environment_key="mlv_env",
            depends_on=[jobs.TaskDependency(task_key=dep_key)],
            run_if=run_if,
            spark_python_task=jobs.SparkPythonTask(
                python_file=remote_runner,
                parameters=params,
            ),
            max_retries=2,
            min_retry_interval_millis=30_000,
            retry_on_timeout=True,
            timeout_seconds=3600,  # 1h per pair
        ))
        lane_tail[lane] = key
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit multilingual voice benchmark job")
    parser.add_argument("--dataset", default="all", choices=["all", "fleurs", "belebele", "ccfqa"])
    parser.add_argument("--languages", default="", help="comma-separated 2-letter codes")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument(
        "--max-parallel", type=int, default=6,
        help="max concurrent benchmark pairs (lanes) hitting the realtime API",
    )
    parser.add_argument("--wait", action="store_true", help="Block until the run finishes")
    parser.add_argument("--resume", help="Resume a prior sweep by run_id (skips completed pairs)")
    args = parser.parse_args()

    config = load_config()
    db = databricks(config)
    profile = db.get("profile")
    run_as = db.get("run_as") or "me"
    env_version = str((config.get("pipeline") or {}).get("environment_version") or "3")
    hf_token = str((config.get("secrets") or {}).get("hf_token") or "").strip()

    client_id, client_secret = benchmark_sp_credentials()
    if not (client_id and client_secret):
        raise SystemExit(
            "No service-principal credentials for app auth. Set "
            "realtime_voice.benchmark.auth.client_id/client_secret in config.local.yaml."
        )
    host = databricks_host()
    api_host = benchmark_api_host()
    api_prefix = benchmark_api_prefix()
    results_dir = str(benchmark_results_dir())

    # Resolve (dataset, language) pairs.
    import languages as langmap

    datasets = ["fleurs", "belebele", "ccfqa"] if args.dataset == "all" else [args.dataset]
    requested = [c.strip() for c in args.languages.split(",") if c.strip()] or None
    pairs: list[tuple[str, str]] = []
    for dataset in datasets:
        for lang in langmap.resolve_languages(dataset, requested):
            pairs.append((dataset, lang))
    if not pairs:
        raise SystemExit("no (dataset, language) pairs resolved")

    workspace_dir = f"/Workspace/Users/{run_as}/multilingual_voice_benchmark"
    client = WorkspaceClient(profile=profile) if profile else WorkspaceClient()
    remote_runner, remote_config, benchmark_ws = _upload_benchmark_code(client, workspace_dir)
    _stage_secrets(client, client_secret=client_secret, hf_token=hf_token)

    run_id = args.resume or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tasks = _build_tasks(
        pairs,
        remote_runner=remote_runner, remote_config=remote_config,
        benchmark_dir=benchmark_ws,
        api_host=api_host, api_prefix=api_prefix, results_dir=results_dir,
        host=host, client_id=client_id, run_id=run_id, hf_token=hf_token,
        limit=args.limit, tts_roundtrip=True, max_audio_seconds=120.0,
        env_version=env_version,
        dataset_sel=args.dataset, languages_sel=args.languages,
        max_parallel=args.max_parallel,
    )

    from databricks.sdk.service import compute

    created = client.jobs.create(
        name=f"multilingual-voice-benchmark-{run_id}",
        tasks=tasks,
        environments=[jobs.JobEnvironment(
            environment_key="mlv_env",
            spec=compute.Environment(
                environment_version=env_version,
                dependencies=[
                    "pyyaml>=6",
                    "websockets>=12,<13",
                    "huggingface_hub>=0.23",
                    "fsspec>=2023.1",  # HfFileSystem range-streaming reads
                    "pyarrow>=15",
                    "numpy>=1.26",
                    "soundfile>=0.12",
                    "requests>=2.31",
                    "databricks-sdk>=0.40",
                ],
            ),
        )],
        tags={"benchmark": "multilingual_voice", "run_id": run_id},
    )
    job_id = created.job_id
    run = client.jobs.run_now(job_id=job_id)
    run_id_num = run.run_id

    print(f"job_id: {job_id}")
    print(f"run_id: {run_id_num} (sweep run_id={run_id})")
    print(f"run_page_url: {client.config.host}/jobs/runs/{run_id_num}")
    print(f"api_host: {api_host}")
    print(f"results: Delta tables {delta_catalog()}.{delta_schema()}.benchmark_runs + benchmark_samples")
    print(f"tasks: {len(tasks)} (1 prepare + {len(pairs)} benchmark pairs)")

    if args.wait:
        # run_now returns a Wait[Run]; .result() blocks until the run terminates.
        result = run.result()
        print(f"final state: {result.state}")


if __name__ == "__main__":
    main()
