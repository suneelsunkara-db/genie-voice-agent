"""Shared WebSocket loop parameterized by voice capability."""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import AsyncIterator, Awaitable, Callable, Literal

from fastapi import WebSocket, WebSocketDisconnect

from ..capabilities import (
    SPEECH_LLM_TOOLASSIST_SPEECH,
    SPEECH_TO_TEXT,
    TEXT_TO_SPEECH,
)
from ..config import RealtimeSettings
from ..contracts import SessionStart
from ..endpointing import EndpointModels, endpointer_for
from ..observability import log_event, new_session_id
from ..pipelines import ServingBundle
from ..pipelines import speech_llm_toolassist_speech as assist_pipeline
from ..pipelines import speech_to_text as stt_pipeline
from ..pipelines import text_to_speech as tts_pipeline
from ..session import VoiceSession
from ..voice_identity import load_voice_seed

logger = logging.getLogger(__name__)

CapabilityMode = Literal["speech-to-text", "speech-llm-toolassist-speech", "text-to-speech"]


@dataclass(frozen=True)
class RouteSpec:
    capability: CapabilityMode
    path: str
    accepts_audio: bool
    accepts_synthesize: bool
    supports_barge_in: bool


ROUTES: tuple[RouteSpec, ...] = (
    RouteSpec(
        capability=SPEECH_TO_TEXT,
        path="/v1/speech-to-text",
        accepts_audio=True,
        accepts_synthesize=False,
        supports_barge_in=False,
    ),
    RouteSpec(
        capability=SPEECH_LLM_TOOLASSIST_SPEECH,
        path="/v1/speech-llm-toolassist-speech",
        accepts_audio=True,
        # Also accept `synthesize`: agent-initiated flows (e.g. the card assistant's
        # opening greeting and its deep-dive spoken summary) speak text WITHOUT a
        # preceding STT turn. Routing them through THIS session — instead of a
        # separate /text-to-speech socket — means they share the session's locked
        # voice reference, so the caller hears ONE consistent voice for the whole
        # call rather than a different timbre per side-channel utterance.
        accepts_synthesize=True,
        supports_barge_in=True,
    ),
    RouteSpec(
        capability=TEXT_TO_SPEECH,
        path="/v1/text-to-speech",
        accepts_audio=False,
        accepts_synthesize=True,
        supports_barge_in=False,
    ),
)


def capabilities_payload(settings: RealtimeSettings) -> dict:
    stt_langs = list(settings.stt_languages or settings.supported_languages)
    tts_langs = list(settings.tts_languages or settings.supported_languages)
    conversation_langs = list(settings.supported_languages)
    return {
        SPEECH_TO_TEXT: {
            "path": "/v1/speech-to-text",
            "transport": "websocket",
            "input": ["audio"],
            "output": ["transcript.final"],
            "languages": stt_langs,
        },
        SPEECH_LLM_TOOLASSIST_SPEECH: {
            "path": "/v1/speech-llm-toolassist-speech",
            "transport": "websocket",
            "input": ["audio"],
            "output": ["transcript.final", "response.text", "response.audio"],
            "languages": conversation_langs,
        },
        TEXT_TO_SPEECH: {
            "path": "/v1/text-to-speech",
            "transport": "websocket",
            "input": ["synthesize"],
            "output": ["response.audio"],
            "languages": tts_langs,
        },
    }


async def handle_voice_ws(
    websocket: WebSocket,
    settings: RealtimeSettings,
    bundle: ServingBundle,
    spec: RouteSpec,
    *,
    on_turn_audio: Callable[[bytes, int, int, str], None] | None = None,
    endpoint_models: EndpointModels | None = None,
) -> None:
    await websocket.accept()
    session_id = new_session_id()
    log_event("ws.open", session_id=session_id, capability=spec.capability)
    session: VoiceSession | None = None
    task: asyncio.Task | None = None

    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break

            if message.get("bytes") is not None:
                if not spec.accepts_audio:
                    await _send_error(
                        websocket,
                        "unsupported_event",
                        f"{spec.capability} does not accept audio input",
                    )
                    continue
                if session is None:
                    await _send_error(websocket, "session_not_started", "Send session.start before audio.")
                    continue
                began = session.add_audio(message["bytes"])
                if spec.supports_barge_in and session.busy:
                    if (
                        settings.allow_barge_in
                        and task
                        and not task.done()
                        and session.voiced_ms >= settings.barge_in_ms
                    ):
                        parent_turn_id = session.turn_id
                        parent_goal = ""
                        active_meta = getattr(getattr(session, "active_turn", None), "meta", {})
                        if isinstance(active_meta, dict):
                            parent_goal = str(active_meta.get("utterance") or "")
                        task.cancel()
                        log_event("barge_in", session_id=session_id, turn_id=session.turn_id)
                        session.profile_state["_pending_barge"] = {
                            "parent_turn_id": parent_turn_id,
                            "parent_goal": parent_goal,
                        }
                        new_turn = session.barge_in(reserve_new_turn=False)
                        await websocket.send_json(
                            {
                                "type": "playback.stop",
                                "turn_id": new_turn,
                                "speech_epoch": session.speech_epoch,
                                "reason": "barge_in",
                            }
                        )
                    continue
                if began:
                    log_event("speech.started", session_id=session_id, turn_id=session.turn_id + 1)
                    await websocket.send_json({"type": "speech.started", "turn_id": session.turn_id + 1})
                # Manual turn mode: the client owns the boundary, so buffer the whole
                # utterance and finalize only on audio.end. The max_turn_seconds cap
                # still applies as a safety bound against unbounded buffering — it is
                # not a turn detector, just a ceiling.
                if session.manual_turns:
                    max_turn = session.config.max_turn_seconds or settings.max_turn_seconds
                    if session.turn_audio_ms >= max_turn * 1000:
                        task = await _start_audio_turn(
                            websocket, bundle, session, task, session_id, spec, settings, on_turn_audio
                        )
                    continue
                # Whole-utterance capture: no interim streaming. The turn is buffered
                # until end-of-speech and transcribed once. End-of-speech is decided
                # by the semantic endpointer (Silero + smart-turn) when available,
                # else the legacy energy VAD.
                if session.endpointer is not None:
                    if session.is_noise_timeout(settings.noise_discard_seconds):
                        # Ambient noise, never real speech: drop without transcribing.
                        log_event(
                            "turn.discarded",
                            session_id=session_id,
                            turn_id=session.turn_id + 1,
                            reason="no_speech",
                            audio_ms=round(session.turn_audio_ms),
                        )
                        session.discard_buffer()
                        continue
                    if await _smart_should_finalize(session, settings):
                        task = await _start_audio_turn(
                            websocket, bundle, session, task, session_id, spec, settings, on_turn_audio
                        )
                    continue
                if session.should_finalize(
                    silence_ms=session.config.vad_silence_ms or settings.vad_silence_ms,
                    max_turn_seconds=session.config.max_turn_seconds or settings.max_turn_seconds,
                    min_speech_ms=settings.min_speech_ms,
                ):
                    task = await _start_audio_turn(
                        websocket, bundle, session, task, session_id, spec, settings, on_turn_audio
                    )
                continue

            if not message.get("text"):
                continue
            payload = json.loads(message["text"])
            event_type = payload.get("type")

            if event_type == "session.start":
                if session is not None:
                    await _send_error(websocket, "session_exists", "A session is already active.")
                    continue
                try:
                    start = SessionStart.from_event(payload)
                except ValueError as exc:
                    await _send_error(websocket, "invalid_event", str(exc))
                    continue
                declared = payload.get("capability")
                if declared is not None and declared != spec.capability:
                    await _send_error(
                        websocket,
                        "capability_mismatch",
                        f"Route is {spec.capability}, session declared {declared}",
                    )
                    continue
                session = VoiceSession(start)
                session.session_id = session_id
                # Pin the selected app-wide voice from a server-owned allowlist.
                # Every page sends the same persisted variant key at session.start;
                # it cannot supply a path. The legacy clip is only a deployment
                # fallback until both committed variants are present.
                variant_paths = {
                    "female": settings.voice_reference_female_path,
                    "male": settings.voice_reference_male_path,
                }
                voice_seed = load_voice_seed(variant_paths[start.voice_variant])
                if voice_seed is None:
                    voice_seed = load_voice_seed(settings.voice_reference_path)
                if voice_seed is not None:
                    session.voice_reference_b64 = voice_seed.reference_b64
                    session.voice_id = voice_seed.voice_id
                # Bind OBO principal from Apps forwarded access token (or local
                # GENIE_OBO_LOCAL_TOKEN stand-in). Genie paths fail closed without it.
                from ..runtime.identity import resolve_session_principal

                session.principal = resolve_session_principal(websocket.headers)
                # Who supplies conversational continuity for governed workspace
                # turns. Once Genie One's own memory/recall does, this runtime stops
                # carrying a conversation handle (see runtime.workspace_conversation).
                session.workspace_conversation.upstream_memory = (
                    settings.genie_one_upstream_memory
                )
                # Manual turn mode: the client explicitly took ownership of the
                # end-of-turn boundary (``endpointing: false``), so the server does
                # no automatic finalization and waits for ``audio.end``. Anything
                # else inherits the server's endpointing.enabled default, keeping
                # the live voice loop's server-side turn detection unchanged.
                session.manual_turns = start.endpointing is False
                use_endpointing = (
                    start.endpointing if start.endpointing is not None else settings.endpointing_enabled
                )
                if use_endpointing and spec.accepts_audio:
                    session.endpointer = endpointer_for(
                        endpoint_models,
                        sample_rate_hz=start.sample_rate_hz,
                        stop_ms=settings.endpoint_stop_ms,
                        min_speech_ms=settings.min_speech_ms,
                        expected_language=start.expected_language,
                    )
                has_obo = bool(getattr(session.principal, "has_token", False))
                log_event(
                    "session.start",
                    session_id=session_id,
                    language=session.config.language,
                    expected_language=session.config.expected_language,
                    has_obo=has_obo,
                    # Same id on every session is the signal that the fixed voice is
                    # live; a null means this session will bootstrap its own.
                    voice_id=session.voice_id,
                    voice_variant=start.voice_variant,
                )
                await websocket.send_json(
                    {
                        "type": "session.ready",
                        "session_id": session_id,
                        "capability": spec.capability,
                        "language": session.config.language,
                        "sample_rate_hz": session.config.sample_rate_hz,
                        "supported_languages": list(settings.supported_languages),
                        "progressive_runtime": bool(settings.progressive_runtime),
                        "voice_variant": start.voice_variant,
                        "voice_id": session.voice_id,
                    }
                )
                # Pre-warm the spoken filler for the selected language while the
                # caller speaks their first (often cold, tool-heavy) turn, so a
                # slow turn plays "one moment" instead of dead air. Only when a
                # CONCRETE language was selected: with "auto" there's no language
                # yet, so warming here would generate a wrong-language clip; the
                # per-turn warm (after STT resolves the language) covers that case.
                if (
                    spec.capability == SPEECH_LLM_TOOLASSIST_SPEECH
                    and session.config.language
                    and session.config.language != "auto"
                ):
                    asyncio.ensure_future(
                        assist_pipeline.warm_filler(bundle, session.config.language)
                    )
            elif event_type == "audio.end":
                if not spec.accepts_audio:
                    continue
                if session is None:
                    await _send_error(websocket, "session_not_started", "Send session.start before audio.end.")
                    continue
                if not session.audio or session.busy:
                    continue
                task = await _start_audio_turn(
                    websocket, bundle, session, task, session_id, spec, settings, on_turn_audio
                )
            elif event_type == "synthesize":
                if not spec.accepts_synthesize:
                    await _send_error(
                        websocket,
                        "unsupported_event",
                        f"{spec.capability} does not accept synthesize events",
                    )
                    continue
                if session is None:
                    await _send_error(websocket, "session_not_started", "Send session.start before synthesize.")
                    continue
                text = str(payload.get("text") or "").strip()
                if not text:
                    await _send_error(websocket, "invalid_event", "synthesize.text is required")
                    continue
                language = payload.get("language")
                task = await _start_synthesize_turn(
                    websocket, bundle, session, task, session_id, text, language
                )
            elif event_type == "barge_in":
                if not spec.supports_barge_in:
                    await _send_error(websocket, "unsupported_event", "barge_in is not supported on this route")
                    continue
                if session is None:
                    await _send_error(websocket, "session_not_started", "Send session.start before barge_in.")
                    continue
                if task and not task.done():
                    task.cancel()
                if session.active_turn is not None:
                    try:
                        session.active_turn.cancel_turn()  # type: ignore[union-attr]
                    except Exception:  # noqa: BLE001
                        pass
                session.set_idle()
                log_event("barge_in", session_id=session_id, turn_id=session.turn_id)
                new_turn = session.barge_in(reserve_new_turn=False)
                await websocket.send_json(
                    {
                        "type": "playback.stop",
                        "turn_id": new_turn,
                        "speech_epoch": session.speech_epoch,
                        "reason": "barge_in",
                    }
                )
            elif event_type == "session.stop":
                if session is not None and session.audio and not session.busy:
                    task = await _start_audio_turn(
                        websocket, bundle, session, task, session_id, spec, settings, on_turn_audio
                    )
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
        log_event("ws.close", session_id=session_id, capability=spec.capability)
        if task and not task.done():
            # Let the in-flight final turn drain on graceful shutdown, but never
            # longer than one LLM turn budget (+5s buffer) — derived from the same
            # config value the pipeline enforces, so the two can't disagree.
            try:
                await asyncio.wait_for(
                    task,
                    timeout=max(
                        settings.llm_turn_timeout_s,
                        settings.deep_dive_read_timeout_s,
                    )
                    + 5,
                )
            except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                task.cancel()
        try:
            await websocket.close()
        except RuntimeError:
            pass


async def _smart_should_finalize(session: VoiceSession, settings: RealtimeSettings) -> bool:
    """End-of-turn decision for the semantic endpointer path.

    Hard cap first, then: for smart-turn languages, evaluate the completeness
    model (off the event loop) at each debounced Silero pause; for the fallback
    languages, finalize on a Silero pause of the configured end-of-utterance gap.
    """
    ep = session.endpointer
    max_turn = session.config.max_turn_seconds or settings.max_turn_seconds
    if session.turn_audio_ms >= max_turn * 1000:
        return True
    if not ep.has_speech:
        return False
    vad_silence = session.config.vad_silence_ms or settings.vad_silence_ms
    if ep.use_smart_turn:
        if ep.take_pause_candidate():
            loop = asyncio.get_event_loop()
            complete, _prob = await loop.run_in_executor(
                None, ep.smart_turn_complete, settings.smart_turn_threshold
            )
            if complete:
                return True
        # Safety net: a very long pause ends the turn even when smart-turn is
        # unsure (e.g. lower-accuracy languages like vi/zh). Bounds worst-case
        # latency to vad_silence_ms instead of the max_turn_seconds cap.
        return ep.silence_ms >= vad_silence
    # VAD-only path (languages smart-turn wasn't trained on): finalize on a
    # Silero pause of the configured end-of-utterance gap.
    return ep.silence_ms >= vad_silence


async def _start_audio_turn(
    websocket: WebSocket,
    bundle: ServingBundle,
    session: VoiceSession,
    previous_task: asyncio.Task | None,
    session_id: str,
    spec: RouteSpec,
    settings: RealtimeSettings,
    on_turn_audio: Callable[[bytes, int, int, str], None] | None,
) -> asyncio.Task | None:
    # Snapshot VAD counters BEFORE finish_turn resets them — this tells us how
    # much audio/voice the turn captured and why it ended (silence gap vs cap),
    # plus the adapted noise floor to sanity-check the gate against the room.
    audio_ms = round(session.turn_audio_ms)
    voiced_ms = round(session.voiced_ms)
    trailing_silence_ms = round(session.silence_ms)
    noise_floor = round(session.noise_floor)
    finished = session.finish_turn()
    if finished is None:
        await _send_error(websocket, "empty_audio", "No audio was received for this turn.")
        return previous_task
    if previous_task and not previous_task.done():
        previous_task.cancel()
    turn_id, audio = finished
    pretranscribed: tuple[str, str | None, int] | None = None
    pending_barge = session.profile_state.pop("_pending_barge", None)
    if isinstance(pending_barge, dict):
        from ..pipelines._shared import transcribe
        from ..runtime import classify_barge_intent

        pretranscribed = await transcribe(bundle, session, audio)
        transcript = pretranscribed[0]
        intent = classify_barge_intent(
            transcript,
            active_goal=str(pending_barge.get("parent_goal") or "") or None,
        )
        parent_turn_id = int(pending_barge.get("parent_turn_id") or max(0, turn_id - 1))
        log_event(
            f"barge.{intent}",
            session_id=session_id,
            turn_id=parent_turn_id,
            metric=f"barge_{intent}",
        )
        if intent == "stop":
            session.turn_id = parent_turn_id
            session.set_completed()
            await websocket.send_json(
                {
                    "type": "turn.event",
                    "turn_id": parent_turn_id,
                    "seq": 1,
                    "kind": "turn.cancelled",
                    "payload": {"reason": "user_stop"},
                }
            )
            return previous_task
        if intent == "amend":
            # Same logical turn id; work is restarted with the original utterance
            # already present in history plus this amendment.
            session.turn_id = parent_turn_id
            turn_id = parent_turn_id
    session.set_working()
    from ..runtime import SpeechScheduler, TurnState

    session.active_turn = TurnState(turn_id=turn_id)
    if pretranscribed is not None:
        session.active_turn.meta["pretranscribed"] = pretranscribed
    session.speech_scheduler = SpeechScheduler()
    if on_turn_audio is not None:
        on_turn_audio(audio, turn_id, session.config.sample_rate_hz, session_id)
    log_event(
        "turn.started",
        session_id=session_id,
        turn_id=turn_id,
        capability=spec.capability,
        audio_ms=audio_ms,
        voiced_ms=voiced_ms,
        trailing_silence_ms=trailing_silence_ms,
        vad_silence_ms=settings.vad_silence_ms,
        noise_floor=noise_floor,
    )
    await websocket.send_json({"type": "turn.started", "turn_id": turn_id})
    return asyncio.create_task(
        _emit_audio_turn(websocket, bundle, session, turn_id, audio, session_id, spec)
    )


async def _start_synthesize_turn(
    websocket: WebSocket,
    bundle: ServingBundle,
    session: VoiceSession,
    previous_task: asyncio.Task | None,
    session_id: str,
    text: str,
    language: object,
) -> asyncio.Task | None:
    """Speak ``text``.

    Progressive rule: if a turn is already ``working``, inject TTS on the same
    ``turn_id`` without cancelling the investigation or bumping the id. Idle
    sessions open a fresh synthesize turn as before.
    """
    lang = str(language) if language else None
    # Same-turn inject: active work must not be killed by spoken summaries.
    if previous_task and not previous_task.done() and session.busy:
        turn_id = session.turn_id
        scheduler = session.speech_scheduler
        if scheduler is not None:
            from ..runtime import SpeechKind, SpeechRequest

            if not scheduler.accept(SpeechRequest(kind=SpeechKind.INJECT, text=text)):
                log_event(
                    "speech.inject.skipped",
                    session_id=session_id,
                    turn_id=turn_id,
                    reason="budget",
                )
                return previous_task
        log_event("speech.inject", session_id=session_id, turn_id=turn_id)
        # Supersede any in-flight TTS for this turn (filler/progress/answer), tell
        # the client to flush, then speak the inject under the new speech_epoch.
        # Same turn_id is intentional (investigation continues); speech_epoch is
        # what prevents two voices overlapping.
        epoch = session.bump_speech_epoch()
        try:
            await websocket.send_json(
                {
                    "type": "playback.stop",
                    "turn_id": turn_id,
                    "speech_epoch": epoch,
                    "reason": "inject",
                }
            )
        except (WebSocketDisconnect, RuntimeError):
            return previous_task
        # Fire-and-forget inject; keep tracking the primary turn task.
        asyncio.create_task(
            _emit_synthesize_inject(websocket, bundle, session, turn_id, text, lang, session_id)
        )
        return previous_task

    if previous_task and not previous_task.done():
        previous_task.cancel()
    session.turn_id += 1
    turn_id = session.turn_id
    session.set_working()
    log_event("turn.started", session_id=session_id, turn_id=turn_id, capability=TEXT_TO_SPEECH)
    await websocket.send_json({"type": "turn.started", "turn_id": turn_id})
    return asyncio.create_task(
        _emit_synthesize_turn(websocket, bundle, session, turn_id, text, lang, session_id)
    )


async def _emit_synthesize_inject(
    websocket: WebSocket,
    bundle: ServingBundle,
    session: VoiceSession,
    turn_id: int,
    text: str,
    language: str | None,
    session_id: str,
) -> None:
    """Same-turn TTS: segment_final only — does not complete or cancel the turn."""
    try:
        async for event in tts_pipeline.process_turn(
            bundle, session, turn_id, text, language=language, mark_final=False
        ):
            if turn_id != session.turn_id:
                return
            await websocket.send_json(event)
    except asyncio.CancelledError:
        log_event("speech.inject.cancelled", session_id=session_id, turn_id=turn_id)
        raise
    except (WebSocketDisconnect, RuntimeError):
        log_event("speech.inject.aborted", session_id=session_id, turn_id=turn_id)
    except Exception as exc:  # noqa: BLE001
        logger.exception("synthesize inject turn %d failed", turn_id)
        log_event("speech.inject.error", session_id=session_id, turn_id=turn_id, error=repr(exc))



async def _emit_audio_turn(
    websocket: WebSocket,
    bundle: ServingBundle,
    session: VoiceSession,
    turn_id: int,
    audio: bytes,
    session_id: str,
    spec: RouteSpec,
) -> None:
    audio_chunks = 0
    try:
        pipeline = _audio_pipeline(spec, bundle, session, turn_id, audio)
        async for event in pipeline:
            etype = event.get("type")
            if etype == "transcript.final":
                log_event(
                    "transcript.final",
                    session_id=session_id,
                    turn_id=turn_id,
                    language=event.get("language"),
                    text=event.get("text"),
                )
            elif etype == "response.text":
                log_event("response.text", session_id=session_id, turn_id=turn_id, text=event.get("text"))
            elif etype == "tool.called":
                log_event("tool.called", session_id=session_id, turn_id=turn_id, name=event.get("name"))
            elif etype == "language.mismatch":
                log_event(
                    "language.mismatch",
                    session_id=session_id,
                    turn_id=turn_id,
                    expected=event.get("expected"),
                    detected=event.get("detected"),
                )
            elif etype == "response.audio":
                audio_chunks += 1
                if event.get("final"):
                    log_event(
                        "response.audio.done",
                        session_id=session_id,
                        turn_id=turn_id,
                        chunks=audio_chunks,
                        server_ttfb_ms=event.get("server_ttfb_ms"),
                    )
            await websocket.send_json(event)
    except asyncio.CancelledError:
        log_event("turn.cancelled", session_id=session_id, turn_id=turn_id)
        raise
    except (WebSocketDisconnect, RuntimeError) as exc:
        # Client went away mid-turn (disconnect, or send-after-close). Not an
        # inference failure — the turn's results simply have nowhere to go.
        if isinstance(exc, RuntimeError) and "send" not in str(exc).lower():
            logger.exception("turn %d failed", turn_id)
            log_event("turn.error", session_id=session_id, turn_id=turn_id, error=repr(exc))
        else:
            log_event("turn.aborted", session_id=session_id, turn_id=turn_id, reason="client_disconnected")
    except Exception as exc:  # noqa: BLE001
        logger.exception("turn %d failed", turn_id)
        log_event("turn.error", session_id=session_id, turn_id=turn_id, error=repr(exc))
        try:
            await _send_error(websocket, "inference_error", str(exc) or exc.__class__.__name__, turn_id=turn_id)
        except (RuntimeError, WebSocketDisconnect):
            pass
    finally:
        session.set_idle()
        session.discard_buffer()


async def _emit_synthesize_turn(
    websocket: WebSocket,
    bundle: ServingBundle,
    session: VoiceSession,
    turn_id: int,
    text: str,
    language: str | None,
    session_id: str,
) -> None:
    audio_chunks = 0
    try:
        async for event in tts_pipeline.process_turn(bundle, session, turn_id, text, language=language):
            if event.get("type") == "response.audio":
                audio_chunks += 1
                if event.get("final"):
                    log_event(
                        "response.audio.done",
                        session_id=session_id,
                        turn_id=turn_id,
                        chunks=audio_chunks,
                    )
            await websocket.send_json(event)
    except asyncio.CancelledError:
        log_event("turn.cancelled", session_id=session_id, turn_id=turn_id)
        raise
    except (WebSocketDisconnect, RuntimeError) as exc:
        if isinstance(exc, RuntimeError) and "send" not in str(exc).lower():
            logger.exception("synthesize turn %d failed", turn_id)
            log_event("turn.error", session_id=session_id, turn_id=turn_id, error=repr(exc))
        else:
            log_event("turn.aborted", session_id=session_id, turn_id=turn_id, reason="client_disconnected")
    except Exception as exc:  # noqa: BLE001
        logger.exception("synthesize turn %d failed", turn_id)
        log_event("turn.error", session_id=session_id, turn_id=turn_id, error=repr(exc))
        try:
            await _send_error(websocket, "inference_error", str(exc) or exc.__class__.__name__, turn_id=turn_id)
        except (RuntimeError, WebSocketDisconnect):
            pass
    finally:
        session.set_idle()


async def _audio_pipeline(
    spec: RouteSpec,
    bundle: ServingBundle,
    session: VoiceSession,
    turn_id: int,
    audio: bytes,
) -> AsyncIterator[dict]:
    if spec.capability == SPEECH_TO_TEXT:
        async for event in stt_pipeline.process_turn(bundle, session, turn_id, audio):
            yield event
    elif spec.capability == SPEECH_LLM_TOOLASSIST_SPEECH:
        async for event in assist_pipeline.process_turn(
            bundle, session, turn_id, audio, context=session.config.context
        ):
            yield event
    else:
        raise RuntimeError(f"audio pipeline not supported for {spec.capability}")


async def _send_error(
    websocket: WebSocket, code: str, message: str, *, turn_id: int | None = None
) -> None:
    payload = {"type": "error", "code": code, "message": message}
    if turn_id is not None:
        payload["turn_id"] = turn_id
    await websocket.send_json(payload)


def make_ws_handler(
    settings: RealtimeSettings,
    bundle_factory: Callable[[RealtimeSettings], ServingBundle | Awaitable[ServingBundle]],
    spec: RouteSpec,
    *,
    on_turn_audio: Callable[[bytes, int, int, str], None] | None = None,
    endpoint_models: EndpointModels | None = None,
):
    async def handler(websocket: WebSocket) -> None:
        bundle = bundle_factory(settings)
        if asyncio.iscoroutine(bundle):
            bundle = await bundle
        await handle_voice_ws(
            websocket,
            settings,
            bundle,
            spec,
            on_turn_audio=on_turn_audio,
            endpoint_models=endpoint_models,
        )

    return handler
