"""Standalone FastAPI application for generic realtime browser voice sessions."""
from __future__ import annotations

import array
import asyncio
import json
import math
import os
import wave
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .config import RealtimeSettings
from .contracts import SessionStart
from .observability import log_event, new_session_id
from .services import DatabricksServing
from .session import VoicePipeline, VoiceSession

# When VOICE_DEBUG_AUDIO=1, save each finalized turn's PCM to a WAV for inspection.
_DEBUG_AUDIO = os.getenv("VOICE_DEBUG_AUDIO") == "1"
_DEBUG_DIR = Path(os.getenv("VOICE_DEBUG_DIR", "/tmp/realtime_audio"))


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
    pipeline_factory: Callable[[RealtimeSettings], VoicePipeline] | None = None,
) -> FastAPI:
    settings = settings or RealtimeSettings.resolve()
    pipeline_factory = pipeline_factory or _databricks_pipeline
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

    @app.websocket("/v1/realtime/voice")
    async def realtime_voice(websocket: WebSocket) -> None:
        await websocket.accept()
        session_id = new_session_id()
        log_event("ws.open", session_id=session_id)
        session: VoiceSession | None = None
        pipeline: VoicePipeline | None = None
        task: asyncio.Task | None = None
        try:
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    break
                if message.get("bytes") is not None:
                    if session is None:
                        await _send_error(websocket, "session_not_started", "Send session.start before audio.")
                        continue
                    began = session.add_audio(message["bytes"])
                    if session.busy:
                        # A reply is in flight. Ignore the mic (trailing speech /
                        # speaker->mic echo) so it can't finalize a new turn and
                        # cancel the reply. Only a *sustained* talk-over interrupts,
                        # and only when barge-in is enabled (needs headphones/AEC;
                        # otherwise the assistant's own echo would self-interrupt).
                        if (
                            settings.allow_barge_in
                            and task
                            and not task.done()
                            and session.voiced_ms >= settings.barge_in_ms
                        ):
                            task.cancel()
                            log_event("barge_in", session_id=session_id, turn_id=session.turn_id)
                            await websocket.send_json({"type": "playback.stop", "turn_id": session.barge_in()})
                        continue
                    if began:
                        log_event("speech.started", session_id=session_id, turn_id=session.turn_id + 1)
                        await websocket.send_json({"type": "speech.started", "turn_id": session.turn_id + 1})
                    if pipeline and session.should_finalize(
                        silence_ms=settings.vad_silence_ms,
                        max_turn_seconds=settings.max_turn_seconds,
                        min_speech_ms=settings.min_speech_ms,
                    ):
                        task = await _start_turn(websocket, pipeline, session, task, session_id)
                    continue
                if not message.get("text"):
                    continue
                payload = json.loads(message["text"])
                event_type = payload.get("type")
                if event_type == "session.start":
                    if session is not None:
                        await _send_error(websocket, "session_exists", "A session is already active.")
                        continue
                    session = VoiceSession(SessionStart.from_event(payload))
                    pipeline = pipeline_factory(settings)
                    log_event("session.start", session_id=session_id, language=session.config.language)
                    await websocket.send_json(
                        {
                            "type": "session.ready",
                            "language": session.config.language,
                            "sample_rate_hz": session.config.sample_rate_hz,
                            "supported_languages": list(settings.supported_languages),
                        }
                    )
                elif event_type == "audio.end":
                    if session is None or pipeline is None:
                        await _send_error(websocket, "session_not_started", "Send session.start before audio.end.")
                        continue
                    task = await _start_turn(websocket, pipeline, session, task, session_id)
                elif event_type == "barge_in":
                    if session is None:
                        await _send_error(websocket, "session_not_started", "Send session.start before barge_in.")
                        continue
                    if task and not task.done():
                        task.cancel()
                    log_event("barge_in", session_id=session_id, turn_id=session.turn_id)
                    await websocket.send_json({"type": "playback.stop", "turn_id": session.barge_in()})
                elif event_type == "session.stop":
                    break
                else:
                    await _send_error(websocket, "unsupported_event", f"Unsupported event type: {event_type!r}")
        except (ValueError, json.JSONDecodeError) as exc:
            try:
                await _send_error(websocket, "invalid_event", str(exc))
            except (RuntimeError, WebSocketDisconnect):
                pass
        except WebSocketDisconnect:
            pass
        finally:
            log_event("ws.close", session_id=session_id)
            if task and not task.done():
                task.cancel()
            # The client may have already disconnected; closing again raises.
            try:
                await websocket.close()
            except RuntimeError:
                pass

    return app


def _databricks_pipeline(settings: RealtimeSettings) -> VoicePipeline:
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
    return VoicePipeline(stt=serving, llm=serving, tts=serving, verify_mode=settings.verify_mode)


async def _emit_turn(
    websocket: WebSocket, pipeline: VoicePipeline, session: VoiceSession, turn_id: int, audio: bytes, session_id: str
) -> None:
    audio_chunks = 0
    try:
        async for event in pipeline.process_turn(session, turn_id, audio):
            etype = event.get("type")
            if etype == "transcript.final":
                log_event(
                    "transcript.final", session_id=session_id, turn_id=turn_id,
                    language=event.get("language"), text=event.get("text"),
                )
            elif etype == "response.text":
                log_event("response.text", session_id=session_id, turn_id=turn_id, text=event.get("text"))
            elif etype == "response.audio":
                audio_chunks += 1
                if event.get("final"):
                    log_event(
                        "response.audio.done", session_id=session_id, turn_id=turn_id,
                        chunks=audio_chunks, server_ttfb_ms=event.get("server_ttfb_ms"),
                    )
            await websocket.send_json(event)
    except asyncio.CancelledError:
        log_event("turn.cancelled", session_id=session_id, turn_id=turn_id)
        raise
    except Exception as exc:  # noqa: BLE001
        log_event("turn.error", session_id=session_id, turn_id=turn_id, error=str(exc))
        await _send_error(websocket, "inference_error", str(exc), turn_id=turn_id)
    finally:
        # Unlock the session and drop any echo/tail captured during the reply so
        # the next real utterance starts clean.
        session.busy = False
        session.discard_buffer()


async def _start_turn(
    websocket: WebSocket, pipeline: VoicePipeline, session: VoiceSession,
    previous_task: asyncio.Task | None, session_id: str,
) -> asyncio.Task | None:
    finished = session.finish_turn()
    if finished is None:
        await _send_error(websocket, "empty_audio", "No audio was received for this turn.")
        return previous_task
    if previous_task and not previous_task.done():
        previous_task.cancel()
    turn_id, audio = finished
    session.busy = True
    stats = _audio_stats(audio, session.config.sample_rate_hz)
    wav_path = _save_debug_wav(audio, session.config.sample_rate_hz, session_id, turn_id) if _DEBUG_AUDIO else None
    log_event(
        "turn.started", session_id=session_id, turn_id=turn_id, audio_bytes=len(audio),
        duration_ms=stats["ms"], peak_dbfs=stats["peak_dbfs"], rms_dbfs=stats["rms_dbfs"], wav=wav_path,
    )
    await websocket.send_json({"type": "turn.started", "turn_id": turn_id})
    return asyncio.create_task(_emit_turn(websocket, pipeline, session, turn_id, audio, session_id))


async def _send_error(websocket: WebSocket, code: str, message: str, *, turn_id: int | None = None) -> None:
    payload = {"type": "error", "code": code, "message": message}
    if turn_id is not None:
        payload["turn_id"] = turn_id
    await websocket.send_json(payload)
