from __future__ import annotations

from typing import Protocol

from genie_voice.asr_eval.manifest import ASRGoldClip
from genie_voice.i18n import LanguageCode

from genie_voice.ml_asr.types import TranscriptionResult


class TranscriptionProvider(Protocol):
    model_id: str
    label: str

    def transcribe(self, clip: ASRGoldClip, *, language: LanguageCode) -> TranscriptionResult: ...
