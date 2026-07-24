from __future__ import annotations

import base64
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from genie_voice.asr_eval.manifest import ASRGoldClip
from genie_voice.config import get_settings
from genie_voice.databricks.client import get_workspace_client
from genie_voice.i18n import LanguageCode

from genie_voice.ml_asr.audio import mime_type_for, read_audio_bytes, speaker_number
from genie_voice.ml_asr.config import ModelSpec
from genie_voice.ml_asr.types import TranscriptionResult


class DatabricksServingProvider:
    def __init__(self, spec: ModelSpec) -> None:
        if not spec.endpoint:
            raise ValueError(f"Model {spec.model_id} is missing endpoint")
        self.model_id = spec.model_id
        self.label = spec.label
        self.endpoint = spec.endpoint

    def transcribe(self, clip: ASRGoldClip, *, language: LanguageCode) -> TranscriptionResult:
        try:
            audio_b64 = base64.b64encode(read_audio_bytes(clip.audio_path)).decode("ascii")
            body = {
                "dataframe_records": [
                    {
                        "audio_b64": audio_b64,
                        "mime_type": mime_type_for(clip.audio_path, clip.audio_format),
                        "speaker": speaker_number(clip.speaker),
                        "language": language,
                    }
                ]
            }
            started = time.perf_counter()
            payload = _query_endpoint(self.endpoint, body)
            latency_ms = round((time.perf_counter() - started) * 1000)
            prediction = _first_prediction(payload)
            transcript = str(prediction.get("transcript") or prediction.get("raw_transcript") or "").strip()
            confidence = prediction.get("confidence")
            return TranscriptionResult(
                transcript=transcript,
                raw_transcript=str(prediction.get("raw_transcript") or transcript).strip(),
                latency_ms=latency_ms,
                confidence=float(confidence) if confidence is not None else None,
                raw=payload,
            )
        except Exception as exc:  # noqa: BLE001
            return TranscriptionResult(
                transcript="",
                raw_transcript="",
                latency_ms=0,
                confidence=None,
                error=str(exc),
            )


def _query_endpoint(endpoint: str, body: dict[str, Any]) -> dict[str, Any]:
    mode = os.environ.get("ML_ASR_DATABRICKS_QUERY_MODE", "").lower()
    if mode == "cli":
        return _query_endpoint_cli(endpoint, body)

    if _use_in_cluster_auth():
        client = _in_cluster_workspace_client()
        response = client.serving_endpoints.query(name=endpoint, **body)
        return _response_dict(response)

    if mode != "sdk":
        try:
            client = get_workspace_client(get_settings())
            response = client.serving_endpoints.query(name=endpoint, **body)
            return _response_dict(response)
        except Exception:
            return _query_endpoint_cli(endpoint, body)

    client = get_workspace_client(get_settings())
    response = client.serving_endpoints.query(name=endpoint, **body)
    return _response_dict(response)


def _use_in_cluster_auth() -> bool:
    return os.environ.get("ML_ASR_RUN_MODE") == "serverless"


def _in_cluster_workspace_client():
    from databricks.sdk import WorkspaceClient

    return WorkspaceClient()


def _query_endpoint_cli(endpoint: str, body: dict[str, Any]) -> dict[str, Any]:
    if os.environ.get("ML_ASR_RUN_MODE") == "serverless":
        raise RuntimeError(
            "Databricks CLI cannot query serving endpoints on serverless compute. "
            "Use the in-cluster WorkspaceClient (ML_ASR_RUN_MODE=serverless)."
        )
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as handle:
        json.dump(body, handle)
        request_path = handle.name
    try:
        cmd = ["databricks"]
        from genie_voice.ml_asr.runtime import databricks_profile

        profile = databricks_profile()
        if profile:
            cmd.extend(["--profile", profile])
        cmd.extend(["serving-endpoints", "query", endpoint, "--json", f"@{request_path}", "--output", "json"])
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=300)
        if proc.returncode != 0:
            raise RuntimeError((proc.stderr or proc.stdout or "Databricks query failed").strip())
        return json.loads(proc.stdout)
    finally:
        Path(request_path).unlink(missing_ok=True)


def _response_dict(response: Any) -> dict[str, Any]:
    if isinstance(response, dict):
        return response
    if hasattr(response, "as_dict"):
        return response.as_dict()
    predictions = getattr(response, "predictions", None)
    if predictions is not None:
        return {"predictions": predictions}
    return {}


def _first_prediction(payload: dict[str, Any]) -> dict[str, Any]:
    predictions = payload.get("predictions") or []
    if not predictions:
        return {}
    first = predictions[0]
    return first if isinstance(first, dict) else dict(first)
