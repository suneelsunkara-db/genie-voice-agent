"""Realtime STT packaged as an MLflow ``ResponsesAgent`` for Databricks.

Databricks Model Serving expects the Agent Framework contract
(``task = agent/v1/responses``), not a raw ``dataframe_records`` pyfunc. This
module wraps the OSS Qwen3-ASR checkpoint as a ``mlflow.pyfunc.ResponsesAgent``
so it deploys as a first-class agent endpoint.

Audio is a poor fit for the text-only ``input`` field, so it travels through the
Responses ``custom_inputs`` channel:

    {
      "input": [{"role": "user", "content": "transcribe"}],
      "custom_inputs": {
        "audio_b64": "<base64 pcm_s16le or wav>",
        "language": "en-US",       # optional; None => auto-detect
        "sample_rate_hz": 16000,
        "max_new_tokens": 1024     # optional per-request override (per audio chunk)
      }
    }

The transcript is returned both as a standard text output item (so the endpoint
is playground/Responses-API compatible) and in ``custom_outputs`` for
programmatic callers.

Inference API reference (qwen-asr package):
    model = Qwen3ASRModel.from_pretrained(dir, dtype=..., device_map="cuda:0")
    results = model.transcribe(audio=(np.ndarray, sr) | path, language=None)
    results[0].text, results[0].language

This file lives only under ``scripts/ml_asr`` and touches no existing pipeline.
"""
from __future__ import annotations

import base64
import io
import wave
from typing import Any

from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse

# Qwen3-ASR forces a language with its full English name; map common BCP 47
# primary subtags. Unknown tags fall back to None (auto-detection).
_LANGUAGE_NAMES = {
    "en": "English", "zh": "Chinese", "yue": "Cantonese", "ja": "Japanese",
    "ko": "Korean", "es": "Spanish", "fr": "French", "de": "German",
    "it": "Italian", "pt": "Portuguese", "ru": "Russian", "ar": "Arabic",
    "hi": "Hindi", "th": "Thai", "id": "Indonesian", "ms": "Malay",
    "vi": "Vietnamese", "nl": "Dutch", "pl": "Polish", "tr": "Turkish",
    "fa": "Persian", "fil": "Filipino", "cs": "Czech", "da": "Danish",
    "el": "Greek", "fi": "Finnish", "hu": "Hungarian", "ro": "Romanian",
    "sv": "Swedish",
}

# Qwen3-ASR splits long audio into ~30 s chunks and generates up to
# ``max_new_tokens`` PER CHUNK (see qwen_asr transcribe -> generate), stopping at
# EOS. 256 (our old value) is below the package's own default of 512 and
# truncated dense chunks, clipping long-audio transcripts. Because generation
# stops at EOS, a higher ceiling is nearly free for normal speech — it only
# removes the truncation cap. 1024 sits well above any real 30 s transcript
# (even fast/dense CJK) while still bounding runaway generation on silence/noise
# chunks. Callers can override per request via ``custom_inputs.max_new_tokens``.
_DEFAULT_MAX_NEW_TOKENS = 1024


class RealtimeSTTAgent(ResponsesAgent):
    """OSS Qwen3-ASR served through the Responses Agent contract."""

    def __init__(self, metadata: dict[str, Any] | None = None) -> None:
        self.metadata = metadata or {}
        self.model: Any = None
        self.inference_device = "uninitialized"

    def load_context(self, context) -> None:  # type: ignore[override]
        self._model_dir = context.artifacts["model_dir"]

    def _ensure_model(self) -> None:
        if self.model is not None:
            return
        import torch
        from qwen_asr import Qwen3ASRModel

        if torch.cuda.is_available():
            self.model = Qwen3ASRModel.from_pretrained(
                self._model_dir,
                dtype=torch.bfloat16,
                device_map="cuda:0",
                max_inference_batch_size=8,
                max_new_tokens=_DEFAULT_MAX_NEW_TOKENS,
            )
            self.inference_device = "cuda:0"
        else:
            self.model = Qwen3ASRModel.from_pretrained(
                self._model_dir,
                dtype=torch.float32,
                device_map="cpu",
                max_new_tokens=_DEFAULT_MAX_NEW_TOKENS,
            )
            self.inference_device = "cpu"

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        # MLflow runs predict on a default (audio-less) example at log time. Return
        # a valid empty response without importing torch / loading weights so
        # registration stays light; real calls always carry custom_inputs.
        if not request.custom_inputs:
            return ResponsesAgentResponse(
                output=[self.create_text_output_item(text="", id="stt-transcript")],
                custom_outputs={"transcript": "", "language": "", "inference_device": self.inference_device},
            )
        self._ensure_model()
        ci = dict(request.custom_inputs)
        audio_b64 = ci.get("audio_b64")
        if not audio_b64:
            raise ValueError("custom_inputs.audio_b64 is required for the STT agent")
        requested_language = ci.get("language")
        sample_rate_hz = int(ci.get("sample_rate_hz") or 16_000)

        # Optional per-request ceiling (per audio chunk). ``max_new_tokens`` lives
        # on the shared model object read by transcribe()'s generate() call, so an
        # override MUST be scoped to this request only — otherwise it leaks onto
        # the shared replica and silently changes the cap for every later caller
        # (e.g. a benchmark's long-passage override bleeding into realtime turns).
        # Set-and-restore keeps the endpoint default (_DEFAULT_MAX_NEW_TOKENS)
        # steady between requests.
        override = ci.get("max_new_tokens")
        previous_max_new_tokens = getattr(self.model, "max_new_tokens", _DEFAULT_MAX_NEW_TOKENS)
        if override is not None:
            self.model.max_new_tokens = int(override)

        samples, sample_rate = _decode_audio(base64.b64decode(audio_b64), sample_rate_hz)
        forced = _forced_language(requested_language)
        try:
            results = self.model.transcribe(audio=(samples, sample_rate), language=forced)
        finally:
            if override is not None:
                self.model.max_new_tokens = previous_max_new_tokens
        transcript, detected = _first_result(results)

        return ResponsesAgentResponse(
            output=[self.create_text_output_item(text=transcript, id="stt-transcript")],
            custom_outputs={
                "transcript": transcript,
                "language": requested_language or detected,
                "detected_language": detected,
                "inference_device": self.inference_device,
            },
        )


def _forced_language(language: str | None) -> str | None:
    if not language:
        return None
    return _LANGUAGE_NAMES.get(language.split("-", 1)[0].lower())


def _first_result(results: Any) -> tuple[str, str | None]:
    item = results[0] if isinstance(results, (list, tuple)) and results else results
    text = getattr(item, "text", None)
    language = getattr(item, "language", None)
    if text is None and isinstance(item, dict):
        text = item.get("text")
        language = item.get("language")
    return str(text or "").strip(), (str(language) if language else None)


def _decode_audio(audio: bytes, sample_rate_hz: int):
    """Return (float32 mono np.ndarray in [-1, 1], sample_rate)."""
    import numpy as np

    if audio[:4] == b"RIFF":
        with wave.open(io.BytesIO(audio), "rb") as handle:
            sample_rate = handle.getframerate()
            frames = handle.readframes(handle.getnframes())
            channels = handle.getnchannels()
        pcm = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
        if channels > 1:
            pcm = pcm.reshape(-1, channels).mean(axis=1)
        return pcm / 32768.0, sample_rate
    pcm = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
    return pcm, sample_rate_hz
