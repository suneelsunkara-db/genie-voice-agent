from __future__ import annotations

from genie_voice.i18n import LanguageCode

from genie_voice.ml_asr.config import EvalConfig, ModelSpec
from genie_voice.ml_asr.providers.base import TranscriptionProvider
from genie_voice.ml_asr.providers.databricks_serving import DatabricksServingProvider
from genie_voice.ml_asr.providers.deepgram import DeepgramProvider


def build_provider(spec: ModelSpec) -> TranscriptionProvider:
    if spec.provider == "deepgram":
        return DeepgramProvider(spec)
    if spec.provider == "databricks_serving":
        return DatabricksServingProvider(spec)
    raise ValueError(f"Unsupported provider type: {spec.provider}")


def providers_for_language(config: EvalConfig, language: LanguageCode) -> list[TranscriptionProvider]:
    return [build_provider(spec) for spec in config.models_for_language(language)]
