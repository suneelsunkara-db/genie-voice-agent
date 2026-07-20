"""Named voice capabilities exposed as separate WebSocket routes."""
from __future__ import annotations

SPEECH_TO_TEXT = "speech-to-text"
SPEECH_LLM_TOOLASSIST_SPEECH = "speech-llm-toolassist-speech"
TEXT_TO_SPEECH = "text-to-speech"

# Deprecated alias for SPEECH_LLM_TOOLASSIST_SPEECH.
LEGACY_VOICE_PATH = "/v1/realtime/voice"

ALL = (SPEECH_TO_TEXT, SPEECH_LLM_TOOLASSIST_SPEECH, TEXT_TO_SPEECH)
