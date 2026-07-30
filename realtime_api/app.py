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
from .endpointing import EndpointModels
from .languages import language_payload
from .pipelines import ServingBundle
from .serving_factory import shared_serving
from .ws.handler import ROUTES, capabilities_payload, make_ws_handler

# Load built-in assistant profiles so they self-register (see profiles.py). This
# side-effecting import lives at the composition root ON PURPOSE — it's the single
# place that knows concrete profiles exist; the core voice loop stays generic and
# never imports a specific domain.
from . import tools as _billing_tools  # noqa: F401  — registers "billing" profile
from . import card_tools as _card_tools  # noqa: F401  — registers "card" profile
from . import concierge_tools as _concierge_tools  # noqa: F401  — registers "concierge"
from . import hls_tools as _hls_tools  # noqa: F401  — registers "hls" profile

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
    bundle_factory = bundle_factory or _default_bundle
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
        # page load without opening a WebSocket session. One canonical payload
        # (shared with the card page) — the client resolves native labels via
        # Intl.DisplayNames so there is no server-side per-language name map.
        return language_payload(settings.supported_languages)

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

    # Load the semantic end-of-turn models once (process-wide, thread-safe ONNX
    # sessions). None when disabled or unloadable -> sessions keep the energy VAD.
    endpoint_models = EndpointModels.load() if settings.endpointing_enabled else None

    assist_spec = next(r for r in ROUTES if r.capability == SPEECH_LLM_TOOLASSIST_SPEECH)
    for spec in ROUTES:
        app.websocket(spec.path)(
            make_ws_handler(
                settings,
                bundle_factory,
                spec,
                on_turn_audio=_on_turn_audio,
                endpoint_models=endpoint_models,
            )
        )

    # Deprecated alias for one release; routes to speech-llm-toolassist-speech.
    app.websocket(LEGACY_VOICE_PATH)(
        make_ws_handler(
            settings,
            bundle_factory,
            assist_spec,
            on_turn_audio=_on_turn_audio,
            endpoint_models=endpoint_models,
        )
    )

    return app


def _default_bundle(settings: RealtimeSettings) -> ServingBundle:
    """Default serving bundle: the ONE config-driven construction path.

    Returns a bundle backed by the process-wide ``shared_serving()`` singleton, so
    every WebSocket connection AND the startup warm-up share the SAME serving
    instance (one auth client; warm-up primes the very replicas that serve turns).
    There is no second (mlflow) construction path.
    """
    serving = shared_serving()
    return ServingBundle(stt=serving, llm=serving, tts=serving)
