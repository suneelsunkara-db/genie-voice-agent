"""Composable voice pipelines (one per public WebSocket capability)."""
from __future__ import annotations

from dataclasses import dataclass

from ..services import LanguageModel, SpeechToText, TextToSpeech


@dataclass(frozen=True)
class ServingBundle:
    stt: SpeechToText
    llm: LanguageModel
    tts: TextToSpeech
