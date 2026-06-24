"""Deploy the Lakebase-first orchestration job.

Runs AS your U2M identity (no PAT). End to end it:
  1. builds the `genie_voice` wheel from `backend/`
  2. copies the wheel + `config/config.yaml` into a WORKSPACE FOLDER
     (/Workspace/Users/<you>/genie_voice_pipeline by default)
  3. creates-or-updates one job:
      - wheel task ingests reference files into UC
      - wheel task ingests streaming call files into Lakebase
       - wheel task verifies Lakebase CDF has published UC history tables
      - wheel task refreshes gold_call_insights as a UC Delta table
      - wheel task applies UC PK/FK metadata for Genie
       - wheel task reconciles the Genie space
  5. optionally runs the job and waits for completion.

The SDK Jobs surface varies by version, so calls are wrapped defensively.

Usage:
  python infra/jobs/deploy_pipeline.py                 # deploy + run orchestration
  python infra/jobs/deploy_pipeline.py --no-run        # deploy only
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import tempfile

sys.path.insert(0, "backend")

from genie_voice.config import get_settings  # noqa: E402
from genie_voice.databricks.client import current_user, get_workspace_client  # noqa: E402

WHEEL_DIST = "genie-voice-agent-backend"  # distribution name (pyproject project.name)
ENTRY_POINT = "genie-pipeline"
ENV_KEY = "genie_env"
LEGACY_JOB_NAMES = [
    "Genie Voice - Lakeflow Refresh",
    "Genie Voice - Voice Stream (bronze+silver)",
    "Genie Voice - Batch (reference+gold)",
]
LEGACY_PIPELINE_NAMES = ["Genie Voice - Lakeflow Pipeline"]


def _repo_root() -> str:
    return os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))


# --------------------------------------------------------------------------- #
# Workspace folder (where the job source + config live)
# --------------------------------------------------------------------------- #
def _ws_api_dir(s, user: str) -> str:
    """Workspace-API path (no /Workspace prefix) of the job source folder."""
    raw = s.pipeline.workspace_dir or f"/Users/{user}/genie_voice_pipeline"
    if raw.startswith("/Workspace/"):
        raw = raw[len("/Workspace"):]
    return raw.rstrip("/")


def _ws_fs(path: str) -> str:
    """Runtime FUSE path (/Workspace/...) used for library install + --config."""
    return path if path.startswith("/Workspace") else f"/Workspace{path}"


# --------------------------------------------------------------------------- #
# Build + copy source into the workspace
# --------------------------------------------------------------------------- #
def _build_wheel(out_dir: str) -> str:
    """Assemble the backend wheel using only the standard library.

    We can't use `pip wheel` here: this (locked-down) host has no PyPI access and
    no setuptools, so pip can't fetch a PEP 517 build backend. `genie_voice` is
    pure Python, so we build a valid PEP 427 wheel directly from the pyproject
    metadata + the package tree. The serverless job then installs this wheel and
    pulls its declared dependencies from the cluster's (reachable) pip index.
    """
    import base64
    import fnmatch
    import hashlib
    import re
    import tomllib
    import zipfile

    backend = os.path.join(_repo_root(), "backend")
    with open(os.path.join(backend, "pyproject.toml"), "rb") as fh:
        data = tomllib.load(fh)
    proj = data["project"]
    name, version = proj["name"], proj["version"]
    dist = re.sub(r"[-_.]+", "_", name)  # PEP 427 normalized distribution name
    distinfo = f"{dist}-{version}.dist-info"
    wheel_path = os.path.join(out_dir, f"{dist}-{version}-py3-none-any.whl")
    print(f"Building wheel (stdlib, no PyPI) -> {os.path.basename(wheel_path)}")

    include = (data.get("tool", {}).get("setuptools", {})
               .get("packages", {}).get("find", {}).get("include", ["*"]))
    pkgs = [d for d in sorted(os.listdir(backend))
            if os.path.isfile(os.path.join(backend, d, "__init__.py"))
            and any(fnmatch.fnmatch(d, pat) for pat in include)]
    if not pkgs:
        raise RuntimeError(f"no packages found under {backend} matching {include}")

    members: list[tuple[str, bytes]] = []
    for pkg in pkgs:
        for dirpath, dirnames, filenames in os.walk(os.path.join(backend, pkg)):
            dirnames[:] = [d for d in dirnames if d != "__pycache__"]
            for fn in sorted(filenames):
                if fn.endswith((".pyc", ".pyo")):
                    continue
                abspath = os.path.join(dirpath, fn)
                arc = os.path.relpath(abspath, backend).replace(os.sep, "/")
                with open(abspath, "rb") as fh:
                    members.append((arc, fh.read()))

    meta = [
        "Metadata-Version: 2.1",
        f"Name: {name}",
        f"Version: {version}",
    ]
    if proj.get("description"):
        meta.append(f"Summary: {proj['description']}")
    if proj.get("requires-python"):
        meta.append(f"Requires-Python: {proj['requires-python']}")
    for dep in proj.get("dependencies", []):
        meta.append(f"Requires-Dist: {dep}")
    for extra, deps in proj.get("optional-dependencies", {}).items():
        meta.append(f"Provides-Extra: {extra}")
        for dep in deps:
            meta.append(f'Requires-Dist: {dep}; extra == "{extra}"')

    wheel_meta = ("Wheel-Version: 1.0\n"
                  "Generator: genie-stdlib-wheel (1.0)\n"
                  "Root-Is-Purelib: true\n"
                  "Tag: py3-none-any\n")
    members.append((f"{distinfo}/METADATA", ("\n".join(meta) + "\n").encode()))
    members.append((f"{distinfo}/WHEEL", wheel_meta.encode()))
    members.append((f"{distinfo}/top_level.txt", ("".join(p + "\n" for p in pkgs)).encode()))
    scripts = proj.get("scripts", {})
    if scripts:
        ep = "[console_scripts]\n" + "".join(f"{k} = {v}\n" for k, v in scripts.items())
        members.append((f"{distinfo}/entry_points.txt", ep.encode()))

    def _record(arc: str, body: bytes) -> str:
        digest = base64.urlsafe_b64encode(hashlib.sha256(body).digest()).rstrip(b"=").decode()
        return f"{arc},sha256={digest},{len(body)}"

    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(wheel_path, "w", zipfile.ZIP_DEFLATED) as zf:
        records = []
        for arc, body in members:
            zf.writestr(arc, body)
            records.append(_record(arc, body))
        records.append(f"{distinfo}/RECORD,,")
        zf.writestr(f"{distinfo}/RECORD", ("\n".join(records) + "\n").encode())

    return wheel_path


def _ws_upload(client, local_path: str, api_path: str) -> None:
    from databricks.sdk.service.workspace import ImportFormat

    with open(local_path, "rb") as fh:
        client.workspace.upload(api_path, io.BytesIO(fh.read()),
                                format=ImportFormat.AUTO, overwrite=True)
    print(f"  copied -> /Workspace{api_path}")


def _volume_upload(client, local_path: str, volume_path: str) -> None:
    """Upload a runtime dependency to a stable UC Volume path."""
    parent = os.path.dirname(volume_path)
    try:
        client.files.create_directory(parent)
    except Exception:
        pass
    with open(local_path, "rb") as fh:
        client.files.upload(volume_path, io.BytesIO(fh.read()), overwrite=True)
    print(f"  copied -> {volume_path}")


def _volume_lib_path(s, filename: str) -> str:
    return (
        f"/Volumes/{s.databricks.catalog}/{s.databricks.schema_name}/"
        f"{s.volume.streaming_name}/libs/{filename}"
    )


# --------------------------------------------------------------------------- #
# Serverless job definition
# --------------------------------------------------------------------------- #
def _environment(s, wheel_ws_fs_path: str):
    """A serverless job environment whose only dependency is our wheel (installed
    from the stable UC Volume path). pip pulls the wheel's declared deps from PyPI;
    pyspark/pandas come from the serverless base image."""
    from databricks.sdk.service import compute, jobs

    return jobs.JobEnvironment(
        environment_key=ENV_KEY,
        spec=compute.Environment(
            environment_version=s.pipeline.environment_version,
            dependencies=[wheel_ws_fs_path],
        ),
    )


def _task(task_key: str, parameters: list[str], *, depends_on: list[str] | None = None):
    """A serverless python_wheel_task (no cluster - uses the job environment)."""
    from databricks.sdk.service import jobs

    return jobs.Task(
        task_key=task_key,
        depends_on=[jobs.TaskDependency(task_key=d) for d in (depends_on or [])] or None,
        python_wheel_task=jobs.PythonWheelTask(
            package_name=WHEEL_DIST,
            entry_point=ENTRY_POINT,
            parameters=parameters,
        ),
        environment_key=ENV_KEY,
    )


def _orchestration_job_settings(s, wheel: str, cfg: str, paused: bool) -> dict:
    """One job: ingest reference + calls in parallel, verify CDF, then publish Genie."""
    from databricks.sdk.service import jobs

    reference_ingest_task = _task(
        "batch_reference_ingest",
        ["--stage", "batch-reference-ingest", "--config", cfg],
    )
    call_ingest_task = _task(
        "call_lakebase_ingest",
        ["--stage", "call-lakebase-ingest", "--config", cfg],
    )
    cdf_check_task = _task(
        "lakebase_cdf_sync_check",
        ["--stage", "lakebase-cdf-sync-check", "--config", cfg],
        depends_on=["call_lakebase_ingest"],
    )
    gold_task = _task(
        "gold_insights_refresh",
        ["--stage", "gold-insights-refresh", "--config", cfg],
        depends_on=[
            "batch_reference_ingest",
            "lakebase_cdf_sync_check",
        ],
    )
    constraints_task = _task(
        "uc_constraints",
        ["--stage", "uc-constraints", "--config", cfg],
        depends_on=["gold_insights_refresh"],
    )
    data_quality_task = _task(
        "data_quality_check",
        ["--stage", "data-quality-check", "--config", cfg],
        depends_on=["uc_constraints"],
    )
    genie_task = _task(
        "genie_space",
        ["--stage", "genie-space", "--config", cfg],
        depends_on=["data_quality_check"],
    )
    settings: dict = {
        "name": s.pipeline.orchestration_job_name,
        "tasks": [
            reference_ingest_task,
            call_ingest_task,
            cdf_check_task,
            gold_task,
            constraints_task,
            data_quality_task,
            genie_task,
        ],
        "environments": [_environment(s, wheel)],
        "max_concurrent_runs": 1,
    }
    if paused:
        settings["trigger"] = jobs.TriggerSettings(pause_status=jobs.PauseStatus.PAUSED)
    return settings


def _find_job_id(client, name: str):
    for j in client.jobs.list():
        if j.settings and j.settings.name == name:
            return j.job_id
    return None


def _upsert(client, name: str, kw: dict):
    from databricks.sdk.service import jobs

    job_id = _find_job_id(client, name)
    if job_id:
        client.jobs.reset(job_id=job_id, new_settings=jobs.JobSettings(**kw))
        print(f"Updated job '{name}' (id {job_id})")
    else:
        job_id = client.jobs.create(**kw).job_id
        print(f"Created job '{name}' (id {job_id})")
    return job_id


def _delete_legacy_resources(client) -> None:
    for name in LEGACY_JOB_NAMES:
        for job in client.jobs.list(name=name):
            if job.job_id is not None:
                client.jobs.delete(job.job_id)
                print(f"Deleted legacy job '{name}' (id {job.job_id})")
    for pipeline in client.pipelines.list_pipelines():
        if getattr(pipeline, "name", None) not in LEGACY_PIPELINE_NAMES:
            continue
        pipeline_id = pipeline.pipeline_id
        if not pipeline_id:
            continue
        try:
            client.pipelines.stop_and_wait(pipeline_id)
        except Exception:
            pass
        client.pipelines.delete(pipeline_id, cascade=True, force=True)
        print(f"Deleted legacy pipeline '{pipeline.name}' (id {pipeline_id})")


def _run_and_wait(client, job_id: int, label: str) -> str:
    """Trigger the job and BLOCK until it reaches a terminal state. Returns the
    result_state (e.g. SUCCESS / FAILED), or 'UNKNOWN' if the SDK shape differs."""
    print(f"Running {label} (job {job_id}) and waiting for it to finish ...")
    try:
        waiter = client.jobs.run_now(job_id=job_id)
        run = waiter.result() if hasattr(waiter, "result") else waiter
        state = getattr(getattr(run, "state", None), "result_state", None)
        state = getattr(state, "value", state) or "UNKNOWN"
        print(f"  {label} finished: {state}")
        return str(state)
    except Exception as exc:  # noqa: BLE001
        print(f"  {label} run failed (the trigger still fires on new files): {exc}")
        return "ERROR"


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-run", action="store_true",
                    help="deploy, but skip the immediate pipeline refresh job")
    ap.add_argument("--full-refresh", action="store_true",
                    help="accepted for backward compatibility; orchestration tables are recreated by tasks")
    ap.add_argument("--paused", action="store_true", help="create the jobs paused")
    args = ap.parse_args()

    s = get_settings()

    client = get_workspace_client(s)
    me = current_user(client)
    user = s.databricks.run_as or me
    print(f"Authenticated as: {me} (run_as: {user})")
    _delete_legacy_resources(client)

    # 1-2. build wheel + copy source/config into stable locations. The runtime
    # wheel goes to a UC Volume path so serverless environment restore never
    # depends on ephemeral Workspace build folders.
    api_dir = _ws_api_dir(s, user)
    try:
        client.workspace.mkdirs(api_dir)
    except Exception as exc:  # noqa: BLE001
        print(f"Could not create workspace folder {api_dir}: {exc}")
        return

    with tempfile.TemporaryDirectory() as tmp:
        wheel_local = _build_wheel(tmp)
        wheel_name = os.path.basename(wheel_local)
        wheel_volume = _volume_lib_path(s, wheel_name)
        try:
            _volume_upload(client, wheel_local, wheel_volume)
            _ws_upload(client, os.path.join(_repo_root(), "config", "config.yaml"),
                       f"{api_dir}/config.yaml")
        except Exception as exc:  # noqa: BLE001
            print(f"Source copy to workspace failed: {exc}")
            return

    wheel_ws = wheel_volume
    cfg_ws = _ws_fs(f"{api_dir}/config.yaml")
    print(f"Workspace source folder: {_ws_fs(api_dir)}")
    print(f"Runtime wheel: {wheel_ws}")

    # 3. create/update the orchestration job.
    try:
        job_id = _upsert(
            client,
            s.pipeline.orchestration_job_name,
            _orchestration_job_settings(s, wheel_ws, cfg_ws, args.paused),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Job create/update failed: {exc}")
        return

    host = s.databricks_host.rstrip("/")
    print(f"Orchestration job: {host}/jobs/{job_id}")
    print(f"Sources: {s.raw_stt_path} / {s.reference_path}")

    # 5. Run refresh job now (reference ingest + call Lakebase ingest -> CDF check -> call UC ingest -> gold -> Genie).
    if args.no_run or args.paused:
        return
    state = _run_and_wait(
        client,
        job_id,
        "Batch reference ingest + Lakebase call ingest + CDF check + Gold refresh + UC constraints + DQ + Genie reconcile",
    )
    if state != "SUCCESS":
        raise RuntimeError(f"Orchestration job failed: {state}")


if __name__ == "__main__":
    main()
