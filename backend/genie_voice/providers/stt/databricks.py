"""Databricks Model Serving STT adapter metadata.

The fine-tuned Whisper endpoint is utterance-level ASR. Runtime transcription is
called from the API upload path, while this adapter keeps the provider registry
and local mock/normalization pipeline valid when `providers.stt.active` is set
to `databricks`.
"""
from __future__ import annotations

from .deepgram import DeepgramSTT


class DatabricksSTT(DeepgramSTT):
    """Registry adapter for the Databricks-hosted fine-tuned Whisper model."""

    name = "databricks"
