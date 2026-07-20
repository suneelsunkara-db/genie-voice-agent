"""Clients that drive voice turns against the realtime API capability routes.

Capability routing (one WebSocket per capability):
  - speech-to-text              : audio -> transcript.final
  - speech-llm-toolassist-speech: audio -> STT -> LLM(+tools) -> TTS
  - text-to-speech              : synthesize -> response.audio

Async model: one persistent background event loop (the IPython kernel owns the
main loop in serverless jobs). The runner's public API is synchronous and
bridges to that loop via run_coroutine_threadsafe.
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import random
import threading
import time
import wave
from dataclasses import dataclass, field
from typing import Any, Callable

ACCEPTED_RATES = {8_000, 16_000, 24_000, 48_000}

CAPABILITY_PATHS = {
    "speech-to-text": "/v1/speech-to-text",
    "speech-llm-toolassist-speech": "/v1/speech-llm-toolassist-speech",
    "text-to-speech": "/v1/text-to-speech",
}

DATASET_CAPABILITIES = {
    "fleurs": "speech-to-text",
    "belebele": "speech-llm-toolassist-speech",
    "ccfqa": "speech-llm-toolassist-speech",
}


def resample_pcm16(pcm: bytes, src_rate: int, dst_rate: int) -> bytes:
    import numpy as np

    if src_rate == dst_rate or not pcm:
        return pcm
    src = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
    n_dst = max(1, int(round(len(src) * dst_rate / src_rate)))
    x_src = np.linspace(0.0, 1.0, num=len(src), endpoint=False)
    x_dst = np.linspace(0.0, 1.0, num=n_dst, endpoint=False)
    out = np.interp(x_dst, x_src, src)
    return np.clip(out, -32768, 32767).astype("<i2").tobytes()


class _EventLoopThread:
    """One background thread + event loop for the whole process."""

    def __init__(self):
        self._loop = None
        self._thread = None
        self._lock = threading.Lock()

    def _ensure(self):
        with self._lock:
            if self._loop is not None:
                return self._loop
            loop = asyncio.new_event_loop()
            thread = threading.Thread(target=loop.run_forever, name="mlv-loop", daemon=True)
            thread.start()
            self._loop = loop
            self._thread = thread
            return loop

    def run(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self._ensure()).result()


_LOOP = _EventLoopThread()


def _run_async(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    return _LOOP.run(coro)


def _backoff_sleep(attempt, base=1.0, cap=30.0):
    delay = min(cap, base * (2 ** attempt))
    time.sleep(random.uniform(0, delay))


@dataclass
class TurnResult:
    transcript: str = ""
    detected_language: str | None = None
    response_text: str = ""
    tts_audio: bytes = b""
    tts_sample_rate: int = 0
    stt_ms: int | None = None
    llm_ms: int | None = None
    tts_first_ms: int | None = None
    client_ttfa_ms: int | None = None
    total_ms: int | None = None
    error: str | None = None
    roundtrip: dict[str, Any] = field(default_factory=dict)


def _decode_chunk(payload):
    import numpy as np

    raw = base64.b64decode(payload.get("audio_b64") or "")
    rate = int(payload.get("sample_rate_hz") or 24_000)
    mime = str(payload.get("mime_type") or "audio/pcm")
    if "wav" in mime:
        with wave.open(io.BytesIO(raw), "rb") as wf:
            rate = wf.getframerate()
            channels = wf.getnchannels()
            width = wf.getsampwidth()
            frames = wf.readframes(wf.getnframes())
        dtype = {1: np.int8, 2: "<i2", 4: "<i4"}.get(width, "<i2")
        samples = np.frombuffer(frames, dtype=dtype).astype(np.float32)
        if channels == 2:
            samples = samples.reshape(-1, 2).mean(axis=1)
        if width != 2:
            scale = 32767.0 / float(2 ** (8 * width - 1))
            samples = samples * scale
        return np.clip(samples, -32768, 32767).astype("<i2").tobytes(), rate
    return raw, rate


def _to_session_rate(pcm, rate):
    if rate in ACCEPTED_RATES:
        return pcm, rate
    return resample_pcm16(pcm, rate, 16_000), 16_000


def _ws_connect_kwargs(headers, timeout_s):
    """Build kwargs for websockets.connect, handling the extra_headers/additional_headers rename.

    websockets < 12 uses 'extra_headers'; >= 12 uses 'additional_headers'.
    The serverless environment may pin an older version despite our job spec.
    """
    import websockets

    kw = {
        "max_size": None,
        "ping_interval": None,
        "open_timeout": timeout_s,
        "close_timeout": 10,
    }
    # Detect which kwarg the installed version accepts.
    import inspect

    sig = inspect.signature(websockets.connect)
    if "additional_headers" in sig.parameters:
        kw["additional_headers"] = headers
    elif "extra_headers" in sig.parameters:
        kw["extra_headers"] = headers
    return kw


def detect_capabilities(host, *, prefix="", auth_token=None):
    """Probe which capability routes are reachable on the API host."""
    import requests

    host_clean = host.rstrip("/")
    for scheme in ("https://", "http://"):
        host_clean = host_clean.replace(scheme, "")
    secure = not (host_clean.startswith("localhost") or host_clean.startswith("127.0.0.1"))
    http_scheme = "https" if secure else "http"
    ws_scheme = "wss" if secure else "ws"
    path_prefix = ("/" + prefix.strip("/")) if prefix.strip("/") else ""

    headers = {}
    token = auth_token() if callable(auth_token) else auth_token
    if token:
        headers["Authorization"] = "Bearer " + str(token)

    available = {cap: False for cap in CAPABILITY_PATHS}
    cap_url = http_scheme + "://" + host_clean + path_prefix + "/v1/capabilities"
    try:
        resp = requests.get(cap_url, headers=headers, timeout=20)
        if resp.status_code == 200:
            body = resp.json()
            for cap in CAPABILITY_PATHS:
                if cap in body:
                    available[cap] = True
            return available
    except Exception:
        pass

    async def _probe(cap):
        import websockets

        route = CAPABILITY_PATHS[cap]
        url = ws_scheme + "://" + host_clean + path_prefix + route
        ws_headers = dict(headers)
        try:
            async with websockets.connect(url, **_ws_connect_kwargs(ws_headers, 15)) as ws:
                start_msg = json.dumps({
                    "type": "session.start",
                    "language": "en",
                    "sample_rate_hz": 16000,
                    "encoding": "pcm_s16le",
                })
                await ws.send(start_msg)
                msg = await asyncio.wait_for(ws.recv(), timeout=15)
                data = json.loads(msg)
                ok = data.get("type") == "session.ready"
                end_msg = json.dumps({"type": "audio.end"})
                await ws.send(end_msg)
                return cap, ok
        except Exception:
            return cap, False

    async def _probe_all():
        return list(await asyncio.gather(*(_probe(c) for c in CAPABILITY_PATHS)))

    for cap, ok in _run_async(_probe_all()):
        available[cap] = ok
    return available


def resolve_capability(requested, available):
    if available.get(requested):
        return requested
    raise RuntimeError(
        "Capability " + repr(requested) + " is not available on this API host"
    )


class WebSocketRealtimeClient:
    """Drives the realtime API over WebSocket (ws:// locally, wss:// remote)."""

    def __init__(self, host, *, prefix="", language="en", auth_token=None,
                 sample_rate_hz=16_000, chunk_ms=40, timeout_s=180.0,
                 default_capability="speech-llm-toolassist-speech",
                 available_capabilities=None):
        self.host = host.rstrip("/")
        self.prefix = ("/" + prefix.strip("/")) if prefix.strip("/") else ""
        self.language = language
        self._auth_token = auth_token
        self.sample_rate_hz = sample_rate_hz
        self.chunk_ms = chunk_ms
        self.timeout_s = timeout_s
        self.default_capability = default_capability
        self.available_capabilities = available_capabilities or {}

    def _bearer(self):
        if callable(self._auth_token):
            return self._auth_token()
        return self._auth_token

    def _url(self, capability):
        host = self.host
        for scheme in ("https://", "http://"):
            host = host.replace(scheme, "")
        secure = not (host.startswith("localhost") or host.startswith("127.0.0.1"))
        proto = "wss" if secure else "ws"
        route = CAPABILITY_PATHS[capability]
        return proto + "://" + host + self.prefix + route

    def _effective(self, capability):
        requested = capability or self.default_capability
        if self.available_capabilities:
            return resolve_capability(requested, self.available_capabilities)
        return requested

    def run_turn(self, pcm_s16le, *, sample_rate_hz, language=None, capability=None):
        cap = self._effective(capability)
        return _run_async(self._run_audio_turn(pcm_s16le, sample_rate_hz, language or self.language, cap))

    def synthesize(self, text, *, language=None, capability=None):
        cap = self._effective(capability or "text-to-speech")
        return _run_async(self._run_synthesize(text, language or self.language, cap))

    def roundtrip_tts(self, result, *, spoken_text, language=None):
        if not result.tts_audio:
            result.roundtrip = {"spoken_text": spoken_text, "reheard_text": None, "error": "no_tts_audio"}
            return result
        pcm, rate = _to_session_rate(result.tts_audio, result.tts_sample_rate or 24_000)
        heard = self.run_turn(pcm, sample_rate_hz=rate, language=language or self.language, capability="speech-to-text")
        result.roundtrip = {
            "spoken_text": spoken_text,
            "reheard_text": heard.transcript,
            "reheard_language": heard.detected_language,
            "error": heard.error,
        }
        return result

    async def _run_audio_turn(self, pcm_s16le, sample_rate_hz, language, capability):
        import websockets

        url = self._url(capability)
        headers = {}
        token = self._bearer()
        if token:
            headers["Authorization"] = "Bearer " + str(token)
        result = TurnResult()
        try:
            async with websockets.connect(url, **_ws_connect_kwargs(headers, self.timeout_s)) as ws:
                start_msg = json.dumps({
                    "type": "session.start",
                    "language": language,
                    "sample_rate_hz": sample_rate_hz,
                    "encoding": "pcm_s16le",
                })
                await ws.send(start_msg)
                await self._await_type(ws, "session.ready")
                await self._stream_audio(ws, pcm_s16le, sample_rate_hz)
                end_msg = json.dumps({"type": "audio.end"})
                await ws.send(end_msg)
                await self._collect(ws, result, stop_after_transcript=(capability == "speech-to-text"))
        except Exception as exc:
            result.error = type(exc).__name__ + ": " + str(exc)
        return result

    async def _run_synthesize(self, text, language, capability):
        import websockets

        url = self._url(capability)
        headers = {}
        token = self._bearer()
        if token:
            headers["Authorization"] = "Bearer " + str(token)
        result = TurnResult()
        try:
            async with websockets.connect(url, **_ws_connect_kwargs(headers, self.timeout_s)) as ws:
                start_msg = json.dumps({
                    "type": "session.start",
                    "language": language,
                    "sample_rate_hz": self.sample_rate_hz,
                    "encoding": "pcm_s16le",
                })
                await ws.send(start_msg)
                await self._await_type(ws, "session.ready")
                synth_msg = json.dumps({"type": "synthesize", "text": text, "language": language})
                await ws.send(synth_msg)
                await self._collect(ws, result, audio_only=True)
        except Exception as exc:
            result.error = type(exc).__name__ + ": " + str(exc)
        return result

    async def _stream_audio(self, ws, pcm, sample_rate_hz):
        bytes_per_chunk = max(2, int(sample_rate_hz * self.chunk_ms / 1000) * 2)
        chunk_interval = self.chunk_ms / 1000.0
        for offset in range(0, len(pcm), bytes_per_chunk):
            await ws.send(pcm[offset:offset + bytes_per_chunk])
            await asyncio.sleep(chunk_interval)

    async def _await_type(self, ws, want):
        while True:
            msg = await asyncio.wait_for(ws.recv(), timeout=self.timeout_s)
            if isinstance(msg, (bytes, bytearray)):
                continue
            data = json.loads(msg)
            if data.get("type") == want:
                return data
            if data.get("type") == "error":
                raise RuntimeError(data.get("message") or "api error")

    async def _collect(self, ws, result, *, stop_after_transcript=False, audio_only=False):
        start = time.perf_counter()
        pcm_parts = []
        first_audio_at = None
        while True:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=self.timeout_s)
            except asyncio.TimeoutError:
                if result.error is None:
                    result.error = "timeout waiting for response"
                break
            if isinstance(msg, (bytes, bytearray)):
                continue
            data = json.loads(msg)
            etype = data.get("type")
            if etype == "transcript.final":
                result.transcript = data.get("text") or ""
                result.detected_language = data.get("language")
                result.stt_ms = data.get("stt_ms")
                if stop_after_transcript:
                    break
            elif etype == "response.text":
                result.response_text = data.get("text") or ""
                result.llm_ms = data.get("llm_ms")
            elif etype == "response.audio":
                if first_audio_at is None:
                    first_audio_at = time.perf_counter()
                    result.client_ttfa_ms = round((first_audio_at - start) * 1000)
                    if data.get("tts_first_ms") is not None:
                        result.tts_first_ms = data.get("tts_first_ms")
                pcm, rate = _decode_chunk(data)
                pcm_parts.append(pcm)
                result.tts_sample_rate = rate
                if data.get("final"):
                    break
            elif etype == "error":
                result.error = data.get("message") or "api error"
                break
            elif etype == "turn.started" and audio_only:
                continue
            elif audio_only and etype not in {"response.audio", "error", "turn.started"}:
                continue
        result.tts_audio = b"".join(pcm_parts)
        result.total_ms = round((time.perf_counter() - start) * 1000)


class InProcessRealtimeClient:
    """Drives the API in-process via FastAPI TestClient (tests only, no network)."""

    def __init__(self, app, *, language="en", sample_rate_hz=16_000, chunk_ms=40):
        from fastapi.testclient import TestClient

        self._client = TestClient(app)
        self.language = language
        self.sample_rate_hz = sample_rate_hz
        self.chunk_ms = chunk_ms

    def _path(self, capability):
        return CAPABILITY_PATHS[capability]

    def run_turn(self, pcm_s16le, *, sample_rate_hz, language=None, capability=None):
        cap = capability or DATASET_CAPABILITIES.get("ccfqa", "speech-llm-toolassist-speech")
        result = TurnResult()
        lang = language or self.language
        try:
            with self._client.websocket_connect(self._path(cap)) as ws:
                ws.send_json({
                    "type": "session.start", "language": lang,
                    "sample_rate_hz": sample_rate_hz, "encoding": "pcm_s16le",
                })
                self._await_type(ws, "session.ready")
                bpc = max(2, int(sample_rate_hz * self.chunk_ms / 1000) * 2)
                for off in range(0, len(pcm_s16le), bpc):
                    ws.send_bytes(pcm_s16le[off:off + bpc])
                ws.send_json({"type": "audio.end"})
                self._collect(ws, result, stop_after_transcript=(cap == "speech-to-text"))
        except Exception as exc:
            result.error = type(exc).__name__ + ": " + str(exc)
        return result

    def synthesize(self, text, *, language=None, capability=None):
        result = TurnResult()
        lang = language or self.language
        try:
            with self._client.websocket_connect(self._path("text-to-speech")) as ws:
                ws.send_json({
                    "type": "session.start", "language": lang,
                    "sample_rate_hz": self.sample_rate_hz, "encoding": "pcm_s16le",
                })
                self._await_type(ws, "session.ready")
                ws.send_json({"type": "synthesize", "text": text, "language": lang})
                self._collect(ws, result, audio_only=True)
        except Exception as exc:
            result.error = type(exc).__name__ + ": " + str(exc)
        return result

    def roundtrip_tts(self, result, *, spoken_text, language=None):
        if not result.tts_audio:
            result.roundtrip = {"spoken_text": spoken_text, "reheard_text": None, "error": "no_tts_audio"}
            return result
        pcm, rate = _to_session_rate(result.tts_audio, result.tts_sample_rate or 24_000)
        heard = self.run_turn(pcm, sample_rate_hz=rate, language=language or self.language, capability="speech-to-text")
        result.roundtrip = {
            "spoken_text": spoken_text, "reheard_text": heard.transcript,
            "reheard_language": heard.detected_language, "error": heard.error,
        }
        return result

    def _await_type(self, ws, want):
        while True:
            data = ws.receive_json()
            if data.get("type") == want:
                return
            if data.get("type") == "error":
                raise RuntimeError(data.get("message") or "api error")

    def _collect(self, ws, result, *, stop_after_transcript=False, audio_only=False):
        start = time.perf_counter()
        pcm_parts = []
        first = None
        while True:
            data = ws.receive_json()
            etype = data.get("type")
            if etype == "transcript.final":
                result.transcript = data.get("text") or ""
                result.detected_language = data.get("language")
                result.stt_ms = data.get("stt_ms")
                if stop_after_transcript:
                    break
            elif etype == "response.text":
                result.response_text = data.get("text") or ""
                result.llm_ms = data.get("llm_ms")
            elif etype == "response.audio":
                if first is None:
                    first = time.perf_counter()
                    result.client_ttfa_ms = round((first - start) * 1000)
                    if data.get("tts_first_ms") is not None:
                        result.tts_first_ms = data.get("tts_first_ms")
                pcm, rate = _decode_chunk(data)
                pcm_parts.append(pcm)
                result.tts_sample_rate = rate
                if data.get("final"):
                    break
            elif etype == "error":
                result.error = data.get("message") or "api error"
                break
            elif etype == "turn.started" and audio_only:
                continue
        result.tts_audio = b"".join(pcm_parts)
        result.total_ms = round((time.perf_counter() - start) * 1000)
