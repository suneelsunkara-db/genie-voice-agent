"""Thin client for the Genie realtime voice API (HTTP metadata + WebSocket voice).

Self-contained on purpose: the MCP server ships/runs independently of the API
repo, so this re-implements the (small, stable) wire protocol rather than
importing server internals. Protocol, per ``realtime_api``:

  * HTTP GET  /                      -> service descriptor
             /v1/capabilities        -> capability routes + languages
             /v1/languages           -> end-to-end supported languages
             /v1/benchmarks[?run_id] -> FLEURS scores + latency
             /healthz                -> liveness
  * WS  /v1/speech-to-text                  audio  -> transcript.final
        /v1/text-to-speech                  synth  -> response.audio
        /v1/speech-llm-toolassist-speech    audio  -> transcript + response(.text/.audio) + tool.called

Every request carries ``Authorization: Bearer <token>`` when a token provider is
set (required for the deployed Databricks App; optional for a local server).
"""
from __future__ import annotations

import asyncio
import base64
import inspect
import io
import json
import time
import wave
from dataclasses import dataclass, field
from typing import Any, Callable
from urllib.parse import urlsplit

# Rates the API accepts as-is; anything else is resampled to 16 kHz before send.
ACCEPTED_RATES = {8_000, 16_000, 24_000, 48_000}

CAPABILITY_PATHS = {
    "speech-to-text": "/v1/speech-to-text",
    "text-to-speech": "/v1/text-to-speech",
    "speech-llm-toolassist-speech": "/v1/speech-llm-toolassist-speech",
}


# --------------------------------------------------------------------------- #
# Audio helpers (WAV <-> PCM s16le mono). numpy only when a conversion is
# actually needed, so plain 16-bit-mono WAVs work with the stdlib alone.
# --------------------------------------------------------------------------- #
def load_audio(path: str) -> tuple[bytes, int]:
    """Load a WAV file as mono PCM s16le. Returns ``(pcm_bytes, sample_rate)``."""
    with wave.open(path, "rb") as wf:
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
    if width == 2 and channels == 1:
        return frames, rate
    import numpy as np

    dtype = {1: np.uint8, 2: "<i2", 4: "<i4"}.get(width)
    if dtype is None:
        raise ValueError(f"unsupported WAV sample width: {width} bytes")
    samples = np.frombuffer(frames, dtype=dtype).astype(np.float32)
    if width == 1:  # 8-bit WAV is unsigned, centered at 128
        samples = (samples - 128.0) * 256.0
    elif width == 4:
        samples = samples / 65536.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return np.clip(samples, -32768, 32767).astype("<i2").tobytes(), rate


def _resample(pcm: bytes, src_rate: int, dst_rate: int) -> bytes:
    if src_rate == dst_rate or not pcm:
        return pcm
    import numpy as np

    src = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
    n_dst = max(1, int(round(len(src) * dst_rate / src_rate)))
    x_src = np.linspace(0.0, 1.0, num=len(src), endpoint=False)
    x_dst = np.linspace(0.0, 1.0, num=n_dst, endpoint=False)
    out = np.interp(x_dst, x_src, src)
    return np.clip(out, -32768, 32767).astype("<i2").tobytes()


def _coerce_rate(pcm: bytes, rate: int) -> tuple[bytes, int]:
    return (pcm, rate) if rate in ACCEPTED_RATES else (_resample(pcm, rate, 16_000), 16_000)


def _decode_chunk(payload: dict) -> tuple[bytes, int]:
    """Decode a ``response.audio`` event into ``(pcm_s16le, sample_rate)``."""
    raw = base64.b64decode(payload.get("audio_b64") or "")
    rate = int(payload.get("sample_rate_hz") or 24_000)
    if "wav" in str(payload.get("mime_type") or "audio/pcm"):
        with wave.open(io.BytesIO(raw), "rb") as wf:
            rate = wf.getframerate()
            raw = wf.readframes(wf.getnframes())
    return raw, rate


def write_wav(path: str, pcm: bytes, rate: int) -> None:
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)


def pcm_duration_ms(pcm: bytes, rate: int) -> int:
    return round(len(pcm) / 2 / max(rate, 1) * 1000)


@dataclass
class TurnResult:
    transcript: str = ""
    detected_language: str | None = None
    response_text: str = ""
    tools: list[dict[str, Any]] = field(default_factory=list)
    audio_pcm: bytes = b""
    sample_rate: int = 0
    stt_ms: int | None = None
    llm_ms: int | None = None
    tts_first_ms: int | None = None
    client_ttfa_ms: int | None = None
    total_ms: int | None = None
    error: str | None = None


# --------------------------------------------------------------------------- #
# API client
# --------------------------------------------------------------------------- #
class RealtimeVoiceAPIError(RuntimeError):
    pass


class RealtimeVoiceAPI:
    def __init__(
        self,
        base_url: str,
        *,
        token_provider: Callable[[], str] | None = None,
        timeout_s: float = 180.0,
        chunk_ms: int = 40,
    ) -> None:
        parts = urlsplit(base_url if "://" in base_url else f"https://{base_url}")
        self._host = parts.netloc
        self._prefix = "/" + parts.path.strip("/") if parts.path.strip("/") else ""
        secure = parts.scheme != "http"
        self._http = "https" if secure else "http"
        self._ws = "wss" if secure else "ws"
        self._token_provider = token_provider
        self.timeout_s = timeout_s
        self.chunk_ms = chunk_ms

    # ---- shared ---------------------------------------------------------- #
    @property
    def base_url(self) -> str:
        return f"{self._http}://{self._host}{self._prefix}"

    def _headers(self) -> dict[str, str]:
        if not self._token_provider:
            return {}
        token = self._token_provider()
        return {"Authorization": f"Bearer {token}"} if token else {}

    def _http_url(self, path: str) -> str:
        return f"{self._http}://{self._host}{self._prefix}{path}"

    def _ws_url(self, capability: str) -> str:
        return f"{self._ws}://{self._host}{self._prefix}{CAPABILITY_PATHS[capability]}"

    # ---- HTTP metadata --------------------------------------------------- #
    async def get(self, path: str) -> dict:
        return await asyncio.to_thread(self._get_sync, path)

    def _get_sync(self, path: str) -> dict:
        import requests

        resp = requests.get(self._http_url(path), headers=self._headers(), timeout=30)
        if resp.status_code == 401 or resp.status_code == 403:
            raise RealtimeVoiceAPIError(
                f"{resp.status_code} unauthorized calling {path}. Configure auth "
                "(GENIE_VOICE_TOKEN, or DATABRICKS_HOST + DATABRICKS_CLIENT_ID/SECRET, "
                "or DATABRICKS_CONFIG_PROFILE) — the Databricks App requires it."
            )
        if not resp.ok:
            raise RealtimeVoiceAPIError(f"{resp.status_code} {resp.reason} calling {path}")
        return resp.json()

    # ---- WebSocket voice ------------------------------------------------- #
    def _ws_kwargs(self) -> dict:
        """websockets<12 uses extra_headers; >=12 uses additional_headers."""
        import websockets

        kw: dict[str, Any] = {
            "max_size": None,
            "ping_interval": None,
            "open_timeout": self.timeout_s,
            "close_timeout": 10,
        }
        params = inspect.signature(websockets.connect).parameters
        headers = self._headers()
        if "additional_headers" in params:
            kw["additional_headers"] = headers
        elif "extra_headers" in params:
            kw["extra_headers"] = headers
        return kw

    async def _stream_audio(self, ws, pcm: bytes, rate: int) -> None:
        bytes_per_chunk = max(2, int(rate * self.chunk_ms / 1000) * 2)
        for off in range(0, len(pcm), bytes_per_chunk):
            await ws.send(pcm[off : off + bytes_per_chunk])

    async def _await_type(self, ws, want: str) -> dict:
        while True:
            msg = await asyncio.wait_for(ws.recv(), timeout=self.timeout_s)
            if isinstance(msg, (bytes, bytearray)):
                continue
            data = json.loads(msg)
            if data.get("type") == want:
                return data
            if data.get("type") == "error":
                raise RealtimeVoiceAPIError(data.get("message") or "api error")

    async def _collect(self, ws, result: TurnResult, *, mode: str) -> None:
        """Collect server events. ``mode`` in {"stt", "tts", "agent"}."""
        start = time.perf_counter()
        parts: list[bytes] = []
        first_audio = None
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
                if mode == "stt":
                    break
            elif etype == "response.text":
                result.response_text = data.get("text") or ""
                result.llm_ms = data.get("llm_ms")
            elif etype == "tool.called":
                result.tools.append(
                    {
                        "name": data.get("name"),
                        "arguments": data.get("arguments"),
                        "result": data.get("result"),
                    }
                )
            elif etype == "response.audio":
                if first_audio is None:
                    first_audio = time.perf_counter()
                    result.client_ttfa_ms = round((first_audio - start) * 1000)
                    if data.get("tts_first_ms") is not None:
                        result.tts_first_ms = data.get("tts_first_ms")
                pcm, rate = _decode_chunk(data)
                parts.append(pcm)
                result.sample_rate = rate
                if data.get("turn_final") is True:
                    break
            elif etype == "error":
                result.error = data.get("message") or "api error"
                break
        result.audio_pcm = b"".join(parts)
        result.total_ms = round((time.perf_counter() - start) * 1000)

    async def transcribe(self, pcm: bytes, rate: int, language: str = "auto") -> TurnResult:
        """Speech-to-text on a full utterance (batch, no truncation)."""
        pcm, rate = _coerce_rate(pcm, rate)
        result = TurnResult()
        try:
            import websockets

            async with websockets.connect(self._ws_url("speech-to-text"), **self._ws_kwargs()) as ws:
                # endpointing:false => audio.end is the ONLY turn boundary, so a
                # mid-clip pause can't split/truncate the supplied utterance.
                await ws.send(
                    json.dumps(
                        {
                            "type": "session.start",
                            "language": language,
                            "sample_rate_hz": rate,
                            "encoding": "pcm_s16le",
                            "endpointing": False,
                            "max_turn_seconds": 300,
                        }
                    )
                )
                await self._await_type(ws, "session.ready")
                await self._stream_audio(ws, pcm, rate)
                await ws.send(json.dumps({"type": "audio.end"}))
                await self._collect(ws, result, mode="stt")
        except Exception as exc:  # noqa: BLE001
            result.error = result.error or f"{type(exc).__name__}: {exc}"
        return result

    async def synthesize(self, text: str, language: str = "en", *, sample_rate_hz: int = 24_000) -> TurnResult:
        """Text-to-speech; returns PCM s16le audio in ``TurnResult.audio_pcm``."""
        result = TurnResult()
        try:
            import websockets

            async with websockets.connect(self._ws_url("text-to-speech"), **self._ws_kwargs()) as ws:
                await ws.send(
                    json.dumps(
                        {
                            "type": "session.start",
                            "language": language,
                            "sample_rate_hz": sample_rate_hz,
                            "encoding": "pcm_s16le",
                        }
                    )
                )
                await self._await_type(ws, "session.ready")
                await ws.send(json.dumps({"type": "synthesize", "text": text, "language": language}))
                await self._collect(ws, result, mode="tts")
        except Exception as exc:  # noqa: BLE001
            result.error = result.error or f"{type(exc).__name__}: {exc}"
        return result

    async def ask_agent(
        self,
        pcm: bytes,
        rate: int,
        *,
        language: str = "auto",
        profile: str | None = None,
        space_name: str | None = None,
        context: str | None = None,
    ) -> TurnResult:
        """Full voice turn: audio -> STT -> LLM(+tools) -> TTS."""
        pcm, rate = _coerce_rate(pcm, rate)
        result = TurnResult()
        try:
            import websockets

            url = self._ws_url("speech-llm-toolassist-speech")
            async with websockets.connect(url, **self._ws_kwargs()) as ws:
                start: dict[str, Any] = {
                    "type": "session.start",
                    "language": language,
                    "sample_rate_hz": rate,
                    "encoding": "pcm_s16le",
                    "endpointing": False,
                    "max_turn_seconds": 300,
                }
                if profile:
                    start["profile"] = profile
                if space_name:
                    start["space_name"] = space_name
                if context:
                    start["context"] = context
                await ws.send(json.dumps(start))
                await self._await_type(ws, "session.ready")
                await self._stream_audio(ws, pcm, rate)
                await ws.send(json.dumps({"type": "audio.end"}))
                await self._collect(ws, result, mode="agent")
        except Exception as exc:  # noqa: BLE001
            result.error = result.error or f"{type(exc).__name__}: {exc}"
        return result
