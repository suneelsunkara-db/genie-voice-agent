"""Register one standalone realtime voice ResponsesAgent in Unity Catalog.

Works for both modalities (STT and TTS). The candidate wrapper is a
``mlflow.pyfunc.ResponsesAgent`` subclass, so MLflow auto-infers the
``agent/v1/responses`` signature and Databricks Model Serving deploys it as an
agent endpoint (the raw ``dataframe_records`` pyfunc path is intentionally not
used).

Run on Databricks serverless compute. Arguments are supplied from
``realtime_api/model_serving.yaml``.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import tempfile
from pathlib import Path

import mlflow
from huggingface_hub import snapshot_download
from mlflow.tracking import MlflowClient

# The Databricks GPU serving base ships torch 2.13.0+cu13, but only the cu118
# runtime wheels load on these nodes. The (working) STT env resolved to
# torch 2.7.1+cu118 via qwen-asr; pin the whole torch stack to that exact build so
# an unpinned resolution can't pull a cu12/cu13 variant whose libcudart is absent.
#
# torchvision must be pinned to the build that matches torch 2.7.1 (0.22.1).
# transformers imports torchvision at inference (image/feature processors), and an
# unpinned torchvision drifts to a wheel compiled against a different torch, which
# fails at predict time with "operator torchvision::nms does not exist" /
# "torchvision has no attribute 'extension'". Pinning makes the env reproducible.
_PIP_REQUIREMENTS = {
    "stt": ["mlflow>=2.20", "pydantic>=2", "torch==2.7.1", "torchaudio==2.7.1",
            "torchvision==0.22.1", "transformers", "accelerate", "qwen-asr",
            "numpy", "soundfile"],
    "tts": ["mlflow>=2.20", "pydantic>=2", "torch==2.7.1", "torchaudio==2.7.1",
            "torchvision==0.22.1", "transformers", "accelerate", "voxcpm",
            "numpy", "soundfile"],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--modality", required=True, choices=["stt", "tts"])
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--registered-model", required=True)
    parser.add_argument("--package-dir", required=True)
    parser.add_argument("--wrapper-path", required=True)
    parser.add_argument("--wrapper-class", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    # Use a guaranteed-writable, uniquely-owned scratch dir. The serverless node
    # rejected writes under a shared /tmp path, so mkdtemp (honours TMPDIR) is
    # used for both the HF hub cache and the materialised snapshot.
    base = Path(args.package_dir if _writable(args.package_dir) else tempfile.mkdtemp(prefix="realtime_voice_"))
    hf_cache = base / "hf_cache"
    model_dir = base / args.candidate_id / "model_snapshot"
    hf_cache.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_cache)
    os.environ["HF_HUB_CACHE"] = str(hf_cache)

    # HF token from config (secrets.hf_token); env is only a fallback.
    try:
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))
        from genie_voice.config import get_settings

        hf_token = get_settings().secrets.hf_token or None
    except Exception:  # noqa: BLE001
        hf_token = None
    hf_token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    snapshot_download(
        repo_id=args.base_model,
        local_dir=str(model_dir),
        local_dir_use_symlinks=False,
        cache_dir=str(hf_cache),
        token=hf_token,
    )
    metadata = {"candidate_id": args.candidate_id, "base_model": args.base_model, "modality": args.modality}
    agent_cls = _load_wrapper(Path(args.wrapper_path), args.wrapper_class)

    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    client = MlflowClient()
    experiment_name = "/Shared/realtime_voice_model_registration"
    experiment = client.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id if experiment else client.create_experiment(experiment_name)
    with mlflow.start_run(experiment_id=experiment_id, run_name=f"register-{args.candidate_id}") as run:
        # ResponsesAgent logging: MLflow sets the fixed agent signature and
        # appends {"task": "agent/v1/responses"} to metadata. No input_example
        # is passed so logging never executes predict (no torch/GPU at log time).
        info = mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=agent_cls(metadata),
            artifacts={"model_dir": str(model_dir)},
            registered_model_name=args.registered_model,
            pip_requirements=_PIP_REQUIREMENTS[args.modality],
            code_paths=[str(Path(args.wrapper_path))],
            metadata=metadata,
        )
        run_id = run.info.run_id

    version = _registered_version(client, args.registered_model, run_id)
    client.set_registered_model_alias(args.registered_model, "candidate", version.version)
    payload = {
        "registered_model": args.registered_model,
        "version": version.version,
        "model_uri": info.model_uri,
        "modality": args.modality,
    }
    Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


def _writable(path: str) -> bool:
    try:
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        probe = target / ".write_probe"
        probe.touch()
        probe.unlink()
        return True
    except OSError:
        return False


def _load_wrapper(path: Path, class_name: str):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load wrapper from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def _registered_version(client: MlflowClient, name: str, run_id: str):
    import time

    for _ in range(48):
        matches = [item for item in client.search_model_versions(f"name = '{name}'") if item.run_id == run_id]
        if matches:
            return max(matches, key=lambda item: int(item.version))
        time.sleep(5)
    raise TimeoutError(f"Model version did not appear for run {run_id}")


if __name__ == "__main__":
    main()
