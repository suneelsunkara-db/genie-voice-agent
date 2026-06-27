"""Smoke-test the registered ASR UC model candidate using the app input contract."""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd


def first_manifest_row(manifest_path: Path) -> dict[str, Any]:
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            return json.loads(line)
    raise ValueError(f"No usable rows found in manifest: {manifest_path}")


def mime_type_for(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".wav":
        return "audio/wav"
    if suffix == ".mp3":
        return "audio/mpeg"
    if suffix in {".webm", ".weba"}:
        return "audio/webm"
    if suffix == ".m4a":
        return "audio/mp4"
    return "application/octet-stream"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registered-model", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")

    row = first_manifest_row(Path(args.manifest))
    audio_path = Path(row["audio_path"])
    audio_b64 = base64.b64encode(audio_path.read_bytes()).decode("ascii")
    model_uri = f"models:/{args.registered_model}@candidate"
    model = mlflow.pyfunc.load_model(model_uri)

    prediction = model.predict(
        pd.DataFrame(
            [
                {
                    "audio_b64": audio_b64,
                    "mime_type": mime_type_for(audio_path),
                    "speaker": int(row.get("speaker") or 1) if str(row.get("speaker") or "1").isdigit() else 1,
                }
            ]
        )
    )
    result = prediction[0] if isinstance(prediction, list) else prediction.to_dict(orient="records")[0]
    required = {
        "raw_transcript",
        "transcript",
        "confidence",
        "model",
        "base_model",
        "lora_run_name",
        "requires_invoice_postprocessing",
    }
    missing = sorted(required - set(result))
    if missing:
        raise AssertionError(f"Registered ASR candidate output missing keys: {missing}")
    if not str(result["transcript"]).strip():
        raise AssertionError("Registered ASR candidate returned an empty transcript")

    print(
        json.dumps(
            {
                "model_uri": model_uri,
                "clip_id": row.get("clip_id"),
                "audio_path": str(audio_path),
                "transcript": result["transcript"],
                "output_keys": sorted(result),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
