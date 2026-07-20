"""Standalone FastAPI application for generic realtime browser voice sessions."""
from __future__ import annotations

import array
import json
import math
import os
import wave
from pathlib import Path
from typing import Callable

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .capabilities import LEGACY_VOICE_PATH, SPEECH_LLM_TOOLASSIST_SPEECH
from .config import RealtimeSettings, benchmark_summary_path, config_dir_from_env
from .pipelines import ServingBundle
from .services import DatabricksServing
from .ws.handler import ROUTES, capabilities_payload, make_ws_handler

# When VOICE_DEBUG_AUDIO=1, save each finalized turn's PCM to a WAV for inspection.
_DEBUG_AUDIO = os.getenv("VOICE_DEBUG_AUDIO") == "1"
_DEBUG_DIR = Path(os.getenv("VOICE_DEBUG_DIR", "/tmp/realtime_audio"))


def _benchmark_summary_file() -> Path:
    try:
        return benchmark_summary_path(config_dir_from_env())
    except Exception:
        override = os.getenv("MLV_RESULTS_DIR")
        if override:
            return Path(override) / "summary.json"
        raise


def _audio_stats(audio: bytes, sample_rate_hz: int) -> dict:
    """Peak/RMS level (dBFS) and duration of a PCM s16le buffer, for diagnostics."""
    samples = array.array("h")
    samples.frombytes(audio[: len(audio) - (len(audio) % 2)])
    n = len(samples)
    if not n:
        return {"ms": 0, "peak_dbfs": None, "rms_dbfs": None}
    peak = max(abs(s) for s in samples)
    rms = (sum(s * s for s in samples) / n) ** 0.5
    to_db = lambda v: round(20 * math.log10(max(v, 1) / 32768.0), 1)  # noqa: E731
    return {"ms": round(n / sample_rate_hz * 1000), "peak_dbfs": to_db(peak), "rms_dbfs": to_db(rms)}


def _save_debug_wav(audio: bytes, sample_rate_hz: int, session_id: str, turn_id: int) -> str:
    _DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    path = _DEBUG_DIR / f"{session_id[:8]}_turn{turn_id}.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate_hz)
        w.writeframes(audio)
    return str(path)


def create_app(
    *,
    settings: RealtimeSettings | None = None,
    bundle_factory: Callable[[RealtimeSettings], ServingBundle] | None = None,
) -> FastAPI:
    settings = settings or RealtimeSettings.resolve()
    bundle_factory = bundle_factory or _databricks_bundle
    app = FastAPI(title="Realtime Voice API", version="1.0.0")
    # The UI is a separate app (realtime_test_ui/) served from its own origin, so allow
    # cross-origin callers. WebSocket upgrades aren't CORS-gated, but this keeps
    # any HTTP endpoints (e.g. /healthz) reachable from the standalone client.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok", "service": "realtime-voice-api"}

    @app.get("/v1/languages")
    async def languages() -> dict:
        # End-to-end supported languages (STT ∩ TTS); lets the UI show them on
        # page load without opening a WebSocket session.
        langs = list(settings.supported_languages)
        return {"languages": langs, "count": len(langs)}

    @app.get("/v1/capabilities")
    async def capabilities() -> dict:
        return capabilities_payload(settings)

    @app.get("/v1/benchmarks")
    async def benchmarks() -> dict:
        """Latest multilingual voice scores from the UC Volume benchmark path.

        Populated by ``Benchmarks/MultilingualVoice/run_benchmark.py`` (Databricks
        job or ``eval.sh``): FLEURS STT + TTS round-trip, 2M-Belebele MCQ, and
        CCFQA spoken QA, plus per-stage latency, all measured on this API.
        """
        try:
            summary_path = _benchmark_summary_file()
        except Exception:
            return {
                "available": False,
                "message": (
                    "No benchmark results directory configured. Set "
                    "volume.multilingual_voice_benchmark_path or MLV_RESULTS_DIR."
                ),
            }
        if not summary_path.exists():
            return {
                "available": False,
                "message": (
                    "No benchmark summary found. Submit the multilingual voice "
                    "benchmark Databricks job (Benchmarks/MultilingualVoice/eval.sh)."
                ),
            }
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        payload["available"] = True
        payload["summary_path"] = str(summary_path)
        return payload

    def _on_turn_audio(audio: bytes, turn_id: int, sample_rate_hz: int, session_id: str) -> None:
        if not _DEBUG_AUDIO:
            return
        stats = _audio_stats(audio, sample_rate_hz)
        wav_path = _save_debug_wav(audio, sample_rate_hz, session_id, turn_id)
        from .observability import log_event

        log_event(
            "turn.audio",
            session_id=session_id,
            turn_id=turn_id,
            audio_bytes=len(audio),
            duration_ms=stats["ms"],
            peak_dbfs=stats["peak_dbfs"],
            rms_dbfs=stats["rms_dbfs"],
            wav=wav_path,
        )

    assist_spec = next(r for r in ROUTES if r.capability == SPEECH_LLM_TOOLASSIST_SPEECH)
    for spec in ROUTES:
        app.websocket(spec.path)(
            make_ws_handler(settings, bundle_factory, spec, on_turn_audio=_on_turn_audio)
        )

    # Deprecated alias for one release; routes to speech-llm-toolassist-speech.
    app.websocket(LEGACY_VOICE_PATH)(
        make_ws_handler(settings, bundle_factory, assist_spec, on_turn_audio=_on_turn_audio)
    )

    return app


def _databricks_bundle(settings: RealtimeSettings) -> ServingBundle:
    serving = DatabricksServing.from_workspace(
        stt_endpoint=settings.stt_endpoint,
        llm_endpoint=settings.llm_endpoint,
        tts_endpoint=settings.tts_endpoint,
        llm_temperature=settings.llm_temperature,
        llm_max_tokens=settings.llm_max_tokens,
        llm_tools_enabled=settings.llm_tools_enabled,
        llm_max_tool_iterations=settings.llm_max_tool_iterations,
        tts_inference_timesteps=settings.tts_inference_timesteps,
        tts_cfg_value=settings.tts_cfg_value,
    )
    return ServingBundle(stt=serving, llm=serving, tts=serving)
