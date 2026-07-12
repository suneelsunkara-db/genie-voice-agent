from __future__ import annotations

import json
import os
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from genie_voice.asr_eval.manifest import ASRGoldClip
from genie_voice.config import get_settings
from genie_voice.i18n import LANGUAGE_SPECS, LanguageCode

from genie_voice.ml_asr.audio import mime_type_for, read_audio_bytes
from genie_voice.ml_asr.config import ModelSpec
from genie_voice.ml_asr.runtime import is_volume_mode
from genie_voice.ml_asr.types import TranscriptionResult

DEEPGRAM_LISTEN_URL = "https://api.deepgram.com/v1/listen"


class DeepgramProvider:
    def __init__(self, spec: ModelSpec) -> None:
        self.model_id = spec.model_id
        self.label = spec.label
        self.model = spec.model or "nova-3"

    def transcribe(self, clip: ASRGoldClip, *, language: LanguageCode) -> TranscriptionResult:
        try:
            api_key = _resolve_deepgram_api_key()
            if not api_key:
                raise RuntimeError("DEEPGRAM_API_KEY is not configured")
            deepgram_language = LANGUAGE_SPECS[language].deepgram_language
            audio_bytes = read_audio_bytes(clip.audio_path)
            mime_type = mime_type_for(clip.audio_path, clip.audio_format)
            params = {
                "model": self.model,
                "language": deepgram_language,
                "smart_format": "true",
                "punctuate": "true",
            }
            request = Request(
                f"{DEEPGRAM_LISTEN_URL}?{urlencode(params)}",
                data=audio_bytes,
                method="POST",
            )
            request.add_header("Authorization", f"Token {api_key}")
            request.add_header("Content-Type", mime_type)
            request.add_header("Accept", "application/json")
            started = time.perf_counter()
            with urlopen(request, timeout=120) as response:
                raw = json.loads(response.read().decode("utf-8"))
            latency_ms = round((time.perf_counter() - started) * 1000)
            transcript, confidence = _extract_deepgram_result(raw)
            return TranscriptionResult(
                transcript=transcript,
                raw_transcript=transcript,
                latency_ms=latency_ms,
                confidence=confidence,
                raw=raw,
            )
        except Exception as exc:  # noqa: BLE001
            return TranscriptionResult(
                transcript="",
                raw_transcript="",
                latency_ms=0,
                confidence=None,
                error=str(exc),
            )


def _extract_deepgram_result(raw: dict) -> tuple[str, float | None]:
    channels = raw.get("results", {}).get("channels") or []
    alternatives = channels[0].get("alternatives") if channels else []
    alt = alternatives[0] if alternatives else {}
    transcript = str(alt.get("transcript") or "").strip()
    confidence = alt.get("confidence")
    return transcript, float(confidence) if confidence is not None else None


def _resolve_deepgram_api_key() -> str:
    if is_volume_mode():
        scope = os.environ.get("ML_ASR_DEEPGRAM_SECRET_SCOPE", "genie-voice")
        key_name = os.environ.get("ML_ASR_DEEPGRAM_SECRET_KEY", "deepgram_api_key")
        try:
            from pyspark.dbutils import DBUtils
            from pyspark.sql import SparkSession

            spark = SparkSession.builder.getOrCreate()
            secret = DBUtils(spark).secrets.get(scope, key_name).strip()
            if secret:
                return secret
        except Exception:
            pass
    api_key = get_settings().secrets.deepgram_api_key.strip()
    if not api_key:
        raise RuntimeError("DEEPGRAM_API_KEY is not configured")
    return api_key
