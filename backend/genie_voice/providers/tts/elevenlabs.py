"""ElevenLabs TTS adapter.

Owns the ElevenLabs "speaking" contract and maps it to canonical SpeechResult.
For deployment=local returns a deterministic fake (no API call); for
deployment=live calls the ElevenLabs API. All ElevenLabs specifics are confined
to this file.

Live response (with timestamps) shape:
  { audio_base64, alignment: { characters, character_start_times_seconds,
    character_end_times_seconds } }
"""
from __future__ import annotations

import base64
import os
from typing import Any

from genie_voice.models.contracts import CharTiming, SpeechResult

from ..base import TTSProvider


class ElevenLabsTTS(TTSProvider):
    name = "elevenlabs"

    def synthesize(self, text: str, *, voice: str | None = None) -> SpeechResult:
        voice_id = voice or self.options.get("voice_id", "Rachel")
        fmt = self.options.get("output_format", "mp3_44100_128")

        from genie_voice.config import get_settings

        if not get_settings().is_live or not os.environ.get("ELEVENLABS_API_KEY"):
            return self._mock(text, voice_id, fmt)
        return self._live(text, voice_id, fmt)

    def _mock(self, text: str, voice_id: str, fmt: str) -> SpeechResult:
        fake_audio = base64.b64encode(f"MOCK_AUDIO::{text}".encode()).decode()
        # Even spacing alignment as a stand-in.
        alignment = []
        t = 0.0
        for ch in text:
            alignment.append(CharTiming(char=ch, start=round(t, 3), end=round(t + 0.05, 3)))
            t += 0.05
        return SpeechResult(
            text=text,
            audio_b64=fake_audio,
            audio_format=fmt,
            provider=self.name,
            voice=voice_id,
            alignment=alignment,
            raw={"mock": True},
        )

    def _live(self, text: str, voice_id: str, fmt: str) -> SpeechResult:
        import httpx

        api_key = os.environ["ELEVENLABS_API_KEY"]
        model_id = self.options.get("model_id", "eleven_turbo_v2_5")
        url = (
            f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
            f"/with-timestamps?output_format={fmt}"
        )
        resp = httpx.post(
            url,
            headers={"xi-api-key": api_key, "content-type": "application/json"},
            json={"text": text, "model_id": model_id},
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()
        align = data.get("alignment", {}) or {}
        chars = align.get("characters", [])
        starts = align.get("character_start_times_seconds", [])
        ends = align.get("character_end_times_seconds", [])
        alignment = [
            CharTiming(char=c, start=s, end=e)
            for c, s, e in zip(chars, starts, ends)
        ]
        return SpeechResult(
            text=text,
            audio_b64=data.get("audio_base64", ""),
            audio_format=fmt,
            provider=self.name,
            voice=voice_id,
            alignment=alignment,
            raw=data,
        )
