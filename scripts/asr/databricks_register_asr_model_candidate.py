"""Register the Genie Whisper LoRA ASR candidate model in Unity Catalog."""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient


def load_pyfunc_class(wrapper_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("mlflow_whisper_lora_pyfunc", wrapper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load pyfunc wrapper from {wrapper_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.WhisperLoraASRModel


def wait_for_registered_version(
    client: MlflowClient,
    registered_model: str,
    run_id: str,
    timeout_seconds: int = 180,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-artifacts", required=True)
    parser.add_argument("--evaluations", required=True)
    parser.add_argument("--lora-run-name", required=True)
    parser.add_argument("--registered-model", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--wrapper-path", required=True)
    parser.add_argument(
        "--experiment-name",
        default="/Users/suneel.sunkara@databricks.com/genie_asr_model_registration",
    )
    args = parser.parse_args()

    lora_dir = Path(args.model_artifacts) / "lora_runs" / args.lora_run_name
    adapter_dir = lora_dir / "adapter"
    processor_dir = lora_dir / "processor"
    run_config = lora_dir / "run_config.json"
    train_metrics = lora_dir / "train_metrics.json"
    comparison = (
        Path(args.evaluations)
        / f"asr_lora_invoice_postprocessed_comparison_{args.lora_run_name}.json"
    )
    error_report = (
        Path(args.evaluations)
        / f"asr_entity_error_analysis_invoice_postprocessed_{args.lora_run_name}.md"
    )

    required_paths = [
        adapter_dir,
        processor_dir,
        run_config,
        train_metrics,
        comparison,
        error_report,
        Path(args.wrapper_path),
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required registration artifacts: {missing}")

    metadata = {
        "status": "candidate",
        "base_model": args.base_model,
        "lora_run_name": args.lora_run_name,
        "registered_model": args.registered_model,
        "adapter_volume_path": str(adapter_dir),
        "processor_volume_path": str(processor_dir),
        "requires_invoice_postprocessing": True,
        "requires_real_recorded_holdout_before_production": True,
    }

    WhisperLoraASRModel = load_pyfunc_class(Path(args.wrapper_path))

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
                "confidence": 0.0,
                "model": "whisper_lora",
                "base_model": args.base_model,
                "lora_run_name": args.lora_run_name,
                "requires_invoice_postprocessing": True,
            }
        ]
    )
    signature = infer_signature(input_example, output_example)

    with mlflow.start_run(experiment_id=experiment_id, run_name=f"register-{args.lora_run_name}") as run:
        mlflow.log_param("status", "candidate")
        mlflow.log_param("base_model", args.base_model)
        mlflow.log_param("lora_run_name", args.lora_run_name)
        mlflow.log_param("requires_invoice_postprocessing", True)
        mlflow.log_param("requires_real_recorded_holdout_before_production", True)
        mlflow.log_artifact(str(run_config), artifact_path="asr_candidate_package_raw")
        mlflow.log_artifact(str(train_metrics), artifact_path="asr_candidate_package_raw")
        mlflow.log_artifact(str(comparison), artifact_path="asr_candidate_package_raw")
        mlflow.log_artifact(str(error_report), artifact_path="asr_candidate_package_raw")

        model_info = mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=WhisperLoraASRModel(metadata),
            artifacts={
                "adapter": str(adapter_dir),
                "processor": str(processor_dir),
            },
            registered_model_name=args.registered_model,
            signature=signature,
            input_example=input_example,
            pip_requirements=[
                "mlflow",
                "torch",
                "transformers",
                "peft",
                "librosa",
                "soundfile",
                "pandas",
            ],
            code_paths=[args.wrapper_path],
            metadata=metadata,
        )
        run_id = run.info.run_id

    version = wait_for_registered_version(client, args.registered_model, run_id)
    client.set_model_version_tag(args.registered_model, version.version, "status", "candidate")
    client.set_model_version_tag(args.registered_model, version.version, "base_model", args.base_model)
    client.set_model_version_tag(args.registered_model, version.version, "lora_run_name", args.lora_run_name)
    client.set_model_version_tag(
        args.registered_model,
        version.version,
        "requires_invoice_postprocessing",
        "true",
    )
    client.set_model_version_tag(
        args.registered_model,
        version.version,
        "requires_real_recorded_holdout_before_production",
        "true",
    )
    client.set_registered_model_alias(args.registered_model, "candidate", version.version)

    print(
        json.dumps(
            {
                "registered_model": args.registered_model,
                "version": version.version,
                "alias": "candidate",
                "run_id": run_id,
                "model_uri": model_info.model_uri,
                "metadata": metadata,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
