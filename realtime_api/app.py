"""Standalone FastAPI application for generic realtime browser voice sessions."""
from __future__ import annotations

import array
import asyncio
import logging
import math
import threading
import time
import wave
from pathlib import Path
from typing import Awaitable, Callable

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from .benchmarks import load_benchmarks
from .capabilities import LEGACY_VOICE_PATH, SPEECH_LLM_TOOLASSIST_SPEECH
from .config import RealtimeSettings
from .languages import DEFAULT_TAG, language_options
from .pipelines import ServingBundle
from .services import DatabricksServing
from .ws.handler import ROUTES, capabilities_payload, make_ws_handler

logger = logging.getLogger("realtime_voice")


def warm_serving(
    settings: RealtimeSettings,
    bundle_factory: Callable[[RealtimeSettings], "ServingBundle | Awaitable[ServingBundle]"],
) -> None:
    """Prime the STT/LLM/TTS replicas off-thread so the first user turn is warm.

    Called from app startup. Builds one bundle and fires ``warmup()`` on a daemon
    thread so neither startup nor readiness is delayed. No-op when disabled, when
    the factory is async (the warm path only supports sync factories), or when the
    bundle's services don't expose ``warmup`` (e.g. test doubles).
    """
    if not settings.warmup_enabled:
        return

    def _work() -> None:
        try:
            bundle = bundle_factory(settings)
        except Exception as exc:  # noqa: BLE001
            logger.warning("realtime warmup: bundle build failed: %s", exc)
            return
        if asyncio.iscoroutine(bundle):
            bundle.close()  # can't drive an async factory from this thread
            return
        warmup = getattr(getattr(bundle, "stt", None), "warmup", None)
        if not callable(warmup):
            return
        started = time.perf_counter()
        try:
            results = warmup()
        except Exception as exc:  # noqa: BLE001
            logger.warning("realtime warmup failed: %s", exc)
            return
        logger.info(
            "realtime warmup complete in %.1fs: %s",
            time.perf_counter() - started,
            results,
        )

    threading.Thread(target=_work, daemon=True, name="realtime-warm").start()


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


def _save_debug_wav(
    audio: bytes, sample_rate_hz: int, session_id: str, turn_id: int, debug_dir: Path
) -> str:
    debug_dir.mkdir(parents=True, exist_ok=True)
    path = debug_dir / f"{session_id[:8]}_turn{turn_id}.wav"
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

    @app.on_event("startup")
    async def _warmup_serving() -> None:
        # Fires when this app runs standalone (``python -m realtime_api.server``).
        # When mounted under the main API (Databricks Apps / start_app.sh),
        # Starlette does NOT run a mounted sub-app's startup events, so the parent
        # triggers warm-up itself via ``warm_serving`` (see api/app/main.py).
        warm_serving(settings, bundle_factory)

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok", "service": "realtime-voice-api"}

    @app.get("/v1/languages")
    async def languages() -> dict:
        # End-to-end supported languages (STT ∩ TTS); lets the UI show them on
        # page load without opening a WebSocket session. Each option carries its
        # BCP-47 tag + English name; the client resolves native labels via
        # Intl.DisplayNames so there is no server-side per-language name map.
        options = language_options(settings.supported_languages)
        return {
            "languages": [item["code"] for item in options],
            "options": options,
            "default": DEFAULT_TAG,
            "count": len(options),
        }

    @app.get("/v1/capabilities")
    async def capabilities() -> dict:
        return capabilities_payload(settings)

    @app.get("/v1/benchmarks")
    async def benchmarks(run_id: str | None = None) -> dict:
        """Latest multilingual voice scores, read straight from the Delta tables.

        Source of truth is ``{catalog}.{schema}.benchmark_runs`` (written by the
        Databricks job / ``eval.sh``): FLEURS STT + TTS round-trip, 2M-Belebele MCQ,
        and CCFQA spoken QA, plus per-stage latency, all measured on this API.
        Returns the latest ``run_id`` unless one is passed explicitly.
        """
        return await run_in_threadpool(load_benchmarks, run_id)

    def _on_turn_audio(audio: bytes, turn_id: int, sample_rate_hz: int, session_id: str) -> None:
        if not settings.debug_audio:
            return
        stats = _audio_stats(audio, sample_rate_hz)
        wav_path = _save_debug_wav(
            audio, sample_rate_hz, session_id, turn_id, Path(settings.debug_audio_dir)
        )
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
        stt_warmup_passes=settings.stt_warmup_passes,
    )
    return ServingBundle(stt=serving, llm=serving, tts=serving)
