"""Register an OSS ASR candidate in Unity Catalog (ml_asr pipeline worker)."""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import time
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from huggingface_hub import snapshot_download as hf_snapshot_download
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient


def load_pyfunc_class(wrapper_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("mlflow_oss_asr_pyfunc", wrapper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load pyfunc wrapper from {wrapper_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MultilingualASRModel


def pip_requirements_for_family(family: str, requirements_dir: Path) -> list[str]:
    base = [
        "mlflow",
        "torch",
        "transformers",
        "accelerate",
        "qwen-asr",
        "librosa",
        "soundfile",
        "pandas",
    ]
    if family == "funasr":
        req_path = requirements_dir / "funasr_serving_requirements.txt"
        return [line.strip() for line in req_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return base


def wait_for_registered_version(
    client: MlflowClient,
    registered_model: str,
    run_id: str,
    timeout_seconds: int = 240,
) -> Any:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        versions = [
            version
            for version in client.search_model_versions(f"name = '{registered_model}'")
            if version.run_id == run_id
        ]
        if versions:
            return max(versions, key=lambda item: int(item.version))
        time.sleep(5)
    raise TimeoutError(f"Timed out waiting for registered model version for run {run_id}")


def download_model_snapshot(
    *,
    base_model: str,
    hub: str,
    snapshot_dir: Path,
    force_download: bool,
) -> None:
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    if force_download and snapshot_dir.exists():
        for child in snapshot_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    has_files = any(snapshot_dir.iterdir()) if snapshot_dir.exists() else False
    if has_files and not force_download:
        return
    if hub == "hf":
        hf_snapshot_download(
            repo_id=base_model,
            local_dir=snapshot_dir,
            local_dir_use_symlinks=False,
        )
        return
    from modelscope.hub.snapshot_download import snapshot_download as ms_snapshot_download

    ms_snapshot_download(base_model, local_dir=str(snapshot_dir))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--family", choices=["qwen3", "whisper", "funasr"], required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--language-code", choices=["th", "id", "zh"], required=True)
    parser.add_argument("--language-name", required=True)
    parser.add_argument("--adaptation-type", choices=["oss_baseline", "finetuned_lora"], required=True)
    parser.add_argument("--fine-tuned-by-us", choices=["true", "false"], required=True)
    parser.add_argument("--registered-model", required=True)
    parser.add_argument("--package-dir", required=True)
    parser.add_argument("--wrapper-path", required=True)
    parser.add_argument("--requirements-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--funasr-hub", choices=["ms", "hf"], default="ms")
    parser.add_argument("--funasr-vad-model", default="")
    parser.add_argument("--funasr-variant", default="")
    parser.add_argument(
        "--experiment-name",
        default="/Users/suneel.sunkara@databricks.com/genie_ml_asr_model_registration",
    )
    args = parser.parse_args()

    package_dir = Path(args.package_dir)
    requirements_dir = Path(args.requirements_dir)
    snapshot_dir = package_dir / args.candidate_id / "model_snapshot"
    vad_snapshot_dir = package_dir / args.candidate_id / "vad_snapshot"
    download_model_snapshot(
        base_model=args.base_model,
        hub=args.funasr_hub if args.family == "funasr" else "hf",
        snapshot_dir=snapshot_dir,
        force_download=args.force_download,
    )
    vad_artifact: str | None = None
    if args.family == "funasr" and args.funasr_vad_model:
        download_model_snapshot(
            base_model=args.funasr_vad_model,
            hub=args.funasr_hub,
            snapshot_dir=vad_snapshot_dir,
            force_download=args.force_download,
        )
        vad_artifact = str(vad_snapshot_dir)

    metadata = {
        "status": "candidate",
        "candidate_id": args.candidate_id,
        "family": args.family,
        "base_model": args.base_model,
        "language_code": args.language_code,
        "language_name": args.language_name,
        "adaptation_type": args.adaptation_type,
        "fine_tuned_by_us": args.fine_tuned_by_us == "true",
        "registered_model": args.registered_model,
        "model_snapshot_path": str(snapshot_dir),
        "requires_invoice_postprocessing": True,
        "requires_real_recorded_holdout_before_production": True,
    }
    if args.family == "funasr":
        metadata.update(
            {
                "funasr_hub": args.funasr_hub,
                "funasr_vad_model": args.funasr_vad_model or None,
                "funasr_variant": args.funasr_variant or None,
            }
        )
    metadata_path = snapshot_dir.parent / "model_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    MultilingualASRModel = load_pyfunc_class(Path(args.wrapper_path))
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    client = MlflowClient()

    experiment = client.get_experiment_by_name(args.experiment_name)
    experiment_id = experiment.experiment_id if experiment else client.create_experiment(args.experiment_name)

    input_example = pd.DataFrame(
        [
            {
                "audio_b64": "UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA=",
                "mime_type": "audio/wav",
                "speaker": 1,
            }
        ]
    )
    output_example = pd.DataFrame(
        [
            {
                "raw_transcript": "example transcript",
                "transcript": "example transcript",
                "confidence": None,
                "model": args.candidate_id,
                "family": args.family,
                "base_model": args.base_model,
                "language": args.language_code,
                "adaptation_type": args.adaptation_type,
                "fine_tuned_by_us": args.fine_tuned_by_us == "true",
                "requires_invoice_postprocessing": True,
                "requires_real_recorded_holdout_before_production": True,
            }
        ]
    )
    signature = infer_signature(input_example, output_example)

    with mlflow.start_run(experiment_id=experiment_id, run_name=f"register-{args.candidate_id}") as run:
        mlflow.log_params(
            {
                "status": "candidate",
                "candidate_id": args.candidate_id,
                "family": args.family,
                "base_model": args.base_model,
                "language_code": args.language_code,
                "adaptation_type": args.adaptation_type,
                "fine_tuned_by_us": args.fine_tuned_by_us,
            }
        )
        mlflow.log_artifact(str(metadata_path), artifact_path="asr_candidate_package_raw")
        model_info = mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=MultilingualASRModel(metadata),
            artifacts={"model_dir": str(snapshot_dir), **({"vad_dir": vad_artifact} if vad_artifact else {})},
            registered_model_name=args.registered_model,
            signature=signature,
            input_example=input_example,
            pip_requirements=pip_requirements_for_family(args.family, requirements_dir),
            code_paths=[args.wrapper_path],
            metadata=metadata,
        )
        run_id = run.info.run_id

    version = wait_for_registered_version(client, args.registered_model, run_id)
    tags = {
        "status": "candidate",
        "candidate_id": args.candidate_id,
        "family": args.family,
        "base_model": args.base_model,
        "language_code": args.language_code,
        "adaptation_type": args.adaptation_type,
        "fine_tuned_by_us": args.fine_tuned_by_us,
        "requires_invoice_postprocessing": "true",
        "requires_real_recorded_holdout_before_production": "true",
    }
    for key, value in tags.items():
        client.set_model_version_tag(args.registered_model, version.version, key, value)
    client.set_registered_model_alias(args.registered_model, "candidate", version.version)

    result = {
        "registered_model": args.registered_model,
        "version": version.version,
        "alias": "candidate",
        "run_id": run_id,
        "model_uri": model_info.model_uri,
        "metadata": metadata,
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
