"""MLflow pyfunc wrapper for the Genie Whisper LoRA ASR candidate."""
from __future__ import annotations

import base64
import tempfile
from typing import Any

import mlflow.pyfunc


class WhisperLoraASRModel(mlflow.pyfunc.PythonModel):
    """Lazy-loading ASR pyfunc model.

    Registration does not require GPU. Inference loads Whisper + PEFT lazily and
    uses CUDA only when available in the serving environment.
    """

    def __init__(self, metadata: dict[str, Any]) -> None:
        self.metadata = metadata
        self.processor = None
        self.model = None
        self.device = None

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        import torch
        from peft import PeftModel
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        base_model = self.metadata["base_model"]
        processor_path = context.artifacts["processor"]
        adapter_path = context.artifacts["adapter"]

        self.processor = WhisperProcessor.from_pretrained(processor_path)
        model = WhisperForConditionalGeneration.from_pretrained(base_model)
        self.model = PeftModel.from_pretrained(model, adapter_path)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def predict(self, context: mlflow.pyfunc.PythonModelContext, model_input: Any) -> list[dict[str, Any]]:
        import librosa
        import torch

        if self.model is None or self.processor is None or self.device is None:
            self.load_context(context)

        rows = model_input.to_dict(orient="records")
        outputs: list[dict[str, Any]] = []
        for row in rows:
            audio, sample_rate = self._load_audio(row, librosa)
            inputs = self.processor.feature_extractor(
                audio,
                sampling_rate=sample_rate,
                return_tensors="pt",
            )
            with torch.no_grad():
                predicted_ids = self.model.generate(inputs.input_features.to(self.device))
            transcript = self.processor.tokenizer.batch_decode(
                predicted_ids,
                skip_special_tokens=True,
            )[0].strip()
            outputs.append(
                {
                    "raw_transcript": transcript,
                    "transcript": transcript,
                    "confidence": None,
                    "model": "whisper_lora",
                    "base_model": self.metadata["base_model"],
                    "lora_run_name": self.metadata["lora_run_name"],
                    "requires_invoice_postprocessing": True,
                }
            )
        return outputs

    def _load_audio(self, row: dict[str, Any], librosa: Any) -> tuple[Any, int]:
        audio_b64 = row.get("audio_b64")
        if audio_b64:
            mime_type = str(row.get("mime_type") or "audio/webm")
            suffix = ".wav" if "wav" in mime_type else ".webm"
            audio_bytes = base64.b64decode(str(audio_b64))
            with tempfile.NamedTemporaryFile(suffix=suffix) as temp:
                temp.write(audio_bytes)
                temp.flush()
                return librosa.load(temp.name, sr=16000, mono=True)

        audio_path = row.get("audio_path")
        if audio_path:
            return librosa.load(str(audio_path), sr=16000, mono=True)

        raise ValueError("ASR model input requires either audio_b64 or audio_path")
