"""MLflow pyfunc wrapper for multilingual ASR candidates.

The registered model loads model weights from MLflow artifacts, not directly
from Hugging Face during evaluation. That keeps quality tests aligned with the
candidate artifact that would later be served.
"""
from __future__ import annotations

import base64
import tempfile
from pathlib import Path
from typing import Any

import mlflow.pyfunc


MODEL_LANGUAGE_NAMES = {
    "th": "Thai",
    "id": "Indonesian",
    "zh": "Chinese",
}


class MultilingualASRModel(mlflow.pyfunc.PythonModel):
    """Lazy-loading multilingual ASR pyfunc model."""

    def __init__(self, metadata: dict[str, Any]) -> None:
        self.metadata = metadata
        self.model: Any = None
        self.family = str(metadata["family"])

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        model_dir = context.artifacts["model_dir"]
        if self.family == "qwen3":
            from qwen_asr import Qwen3ASRModel

            self.model = Qwen3ASRModel.from_pretrained(model_dir)
            return

        if self.family == "whisper":
            import torch
            from transformers import pipeline

            device = 0 if torch.cuda.is_available() else -1
            self.model = pipeline(
                "automatic-speech-recognition",
                model=model_dir,
                device=device,
            )
            return

        raise ValueError(f"Unsupported ASR family: {self.family}")

    def predict(self, context: mlflow.pyfunc.PythonModelContext, model_input: Any) -> list[dict[str, Any]]:
        if self.model is None:
            self.load_context(context)

        rows = model_input.to_dict(orient="records")
        outputs: list[dict[str, Any]] = []
        for row in rows:
            audio_path = self._materialize_audio(row)
            transcript = self._transcribe(audio_path, row)
            outputs.append(
                {
                    "raw_transcript": transcript,
                    "transcript": transcript,
                    "confidence": None,
                    "model": self.metadata["candidate_id"],
                    "family": self.family,
                    "base_model": self.metadata["base_model"],
                    "language": self.metadata["language_code"],
                    "adaptation_type": self.metadata.get("adaptation_type", "oss_baseline"),
                    "fine_tuned_by_us": bool(self.metadata.get("fine_tuned_by_us", False)),
                    "requires_invoice_postprocessing": True,
                    "requires_real_recorded_holdout_before_production": True,
                }
            )
        return outputs

    def _materialize_audio(self, row: dict[str, Any]) -> str:
        audio_b64 = row.get("audio_b64")
        if audio_b64:
            mime_type = str(row.get("mime_type") or "audio/webm")
            suffix = ".wav" if "wav" in mime_type else ".webm"
            audio_bytes = base64.b64decode(str(audio_b64))
            temp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            try:
                temp.write(audio_bytes)
                temp.flush()
                return temp.name
            finally:
                temp.close()

        audio_path = row.get("audio_path")
        if audio_path:
            return str(audio_path)

        raise ValueError("ASR model input requires either audio_b64 or audio_path")

    def _transcribe(self, audio_path: str, row: dict[str, Any]) -> str:
        if self.family == "qwen3":
            language = MODEL_LANGUAGE_NAMES.get(
                str(row.get("language") or self.metadata["language_code"]),
                str(self.metadata["language_name"]),
            )
            result = self.model.transcribe(audio_path, language=language)
            text = extract_transcript_text(result)
            return text if text else str(result).strip()

        if self.family == "whisper":
            import librosa

            kwargs: dict[str, Any] = {}
            language_code = str(row.get("language") or self.metadata.get("language_code") or "")
            language_name = MODEL_LANGUAGE_NAMES.get(language_code, str(self.metadata.get("language_name") or ""))
            language_name = language_name.lower()
            if language_name:
                kwargs["generate_kwargs"] = {"language": language_name, "task": "transcribe"}
            audio, sample_rate = librosa.load(audio_path, sr=16000, mono=True)
            result = self.model({"array": audio, "sampling_rate": sample_rate}, **kwargs)
            return extract_transcript_text(result)

        raise ValueError(f"Unsupported ASR family: {self.family}")


def extract_transcript_text(result: Any) -> str:
    if isinstance(result, dict):
        return str(result.get("text") or result.get("transcript") or "").strip()
    if isinstance(result, (list, tuple)):
        return " ".join(text for item in result if (text := extract_transcript_text(item))).strip()
    for attr in ("text", "transcript"):
        if hasattr(result, attr):
            value = getattr(result, attr)
            if value:
                return str(value).strip()
    return ""


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
