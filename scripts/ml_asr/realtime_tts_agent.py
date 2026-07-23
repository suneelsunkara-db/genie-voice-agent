"""Realtime TTS packaged as an MLflow ``ResponsesAgent`` for Databricks.

Mirrors ``realtime_stt_agent.py`` but in the synthesis direction. Text arrives in
the standard Responses ``input`` (or via ``custom_inputs.text``) and the rendered
audio is returned as base64 WAV in ``custom_outputs``:

    request:
      {
        "input": [{"role": "user", "content": "Hello there"}],
        "custom_inputs": {"language": "en-US",
                          "reference_audio_b64": "<optional base64 wav>"}
      }
    response.custom_outputs:
      {"audio_b64": "<base64 wav>", "mime_type": "audio/wav", "sample_rate_hz": 48000}

``reference_audio_b64`` (optional) is a base64 WAV whose voice VoxCPM2 clones, so
a caller can keep one consistent voice across many requests (e.g. a whole call).

Inference API reference (voxcpm package, VoxCPM2):
    model = VoxCPM.from_pretrained(dir, load_denoiser=False)
    wav = model.generate(text=..., cfg_value=2.0, inference_timesteps=10)  # np.ndarray
    sample_rate = model.tts_model.sample_rate  # 48000

VoxCPM2 is multilingual from the text itself; there is no language argument.

This file lives only under ``scripts/ml_asr`` and touches no existing pipeline.
"""
from __future__ import annotations

import base64
import io
import os
import tempfile
import time
import wave
from typing import Any


from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)


# The API synthesizes one sentence at a time, so warm the short/medium
# sentence-sized shape buckets torch.compile will actually see at serve time.
# Keeping these bounded also keeps container startup (where warmup runs) short.
_WARMUP_TEXTS = (
    "Hello.",
    "This is a normal length sentence used to warm the speech model.",
)


class RealtimeTTSAgent(ResponsesAgent):
    """OSS VoxCPM2 served through the Responses Agent contract."""

    def __init__(self, metadata: dict[str, Any] | None = None) -> None:
        self.metadata = metadata or {}
        self.model: Any = None
        self.inference_device = "uninitialized"
        # Latency knobs. inference_timesteps is the linear diffusion-step cost;
        # both are overridable per-request via custom_inputs for live profiling.
        self.cfg_value = float(self.metadata.get("cfg_value", 2.0))
        # 6 diffusion steps: profiled sweet spot (full multilingual quality incl.
        # Thai at the lowest safe latency). Overridable per-request via custom_inputs.
        self.inference_timesteps = int(self.metadata.get("inference_timesteps", 6))

    def load_context(self, context) -> None:  # type: ignore[override]
        self._model_dir = context.artifacts["model_dir"]
        # load_context runs in TWO different environments:
        #   1. Registration/log time: MLflow calls load_context + predict on a
        #      default example to validate the agent. That runs in the lightweight
        #      packaging job (CPU-only, no torch/voxcpm), so the model must NOT be
        #      loaded here -- the empty validation example is answered by the
        #      predict() short-circuit below without any weights.
        #   2. Serving startup: the full env (torch + GPU) is present, so we build
        #      and warm the model up-front. The container only becomes READY once
        #      inference is hot, moving the weight-load + torch.compile cost off the
        #      request path (no first-request cold start).
        # We therefore load only when the serving runtime is actually available.
        if _serving_runtime_available():
            self._ensure_model()
            self._warmup()

    def _ensure_model(self) -> None:
        if self.model is not None:
            return
        import torch
        from voxcpm import VoxCPM

        # VoxCPM selects CUDA automatically when available; record which device
        # actually backs inference for observability/promotion gates.
        self.model = VoxCPM.from_pretrained(self._model_dir, load_denoiser=False)
        self.inference_device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def _warmup(self) -> None:
        # Best-effort; a warmup failure must never block the endpoint from serving.
        ref_path: str | None = None
        try:
            for text in _WARMUP_TEXTS:
                try:
                    audio = self.model.generate(
                        text=text,
                        cfg_value=self.cfg_value,
                        inference_timesteps=self.inference_timesteps,
                        retry_badcase=False,
                    )
                except Exception:  # noqa: BLE001
                    continue
                # Capture one clip to warm the voice-cloning path below. A
                # reference prepends ref-audio tokens, changing the input shape
                # torch.compile specialises on; compiling that graph here (at
                # startup, off the request path) avoids a ~15 s one-off stall on
                # the first cloned turn of a call.
                if ref_path is None:
                    sr = int(getattr(getattr(self.model, "tts_model", None), "sample_rate", 48_000))
                    ref_path = _write_reference_wav(
                        base64.b64encode(_float_to_wav(audio, sr)).decode("ascii")
                    )
            if ref_path:
                try:
                    for _ in self.model.generate_streaming(
                        text=_WARMUP_TEXTS[-1],
                        cfg_value=self.cfg_value,
                        inference_timesteps=self.inference_timesteps,
                        retry_badcase=False,
                        reference_wav_path=ref_path,
                    ):
                        pass
                except Exception:  # noqa: BLE001
                    pass
        finally:
            _unlink(ref_path)

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        # MLflow runs predict on a default example at log time. Short-circuit with
        # a valid empty response when no custom_inputs are present so registration
        # never imports torch / loads weights; real calls always carry custom_inputs.
        if not request.custom_inputs:
            return ResponsesAgentResponse(
                output=[self.create_text_output_item(text="", id="tts-spoken-text")],
                custom_outputs={"audio_b64": "", "mime_type": "audio/wav", "sample_rate_hz": 0,
                                "language": "", "inference_device": self.inference_device},
            )
        self._ensure_model()
        ci = dict(request.custom_inputs)
        text = str(ci.get("text") or _last_user_text(request) or "").strip()
        if not text:
            raise ValueError("No text supplied for synthesis (custom_inputs.text or input message)")
        language = str(ci.get("language") or "")
        timesteps = int(ci.get("inference_timesteps") or self.inference_timesteps)
        cfg_value = float(ci.get("cfg_value") or self.cfg_value)

        started = time.perf_counter()
        gen_kwargs = dict(
            text=text,
            cfg_value=cfg_value,
            inference_timesteps=timesteps,
            retry_badcase=False,
        )
        ref_path = _write_reference_wav(ci.get("reference_audio_b64"))
        if ref_path:
            gen_kwargs["reference_wav_path"] = ref_path
        try:
            audio = self.model.generate(**gen_kwargs)
        finally:
            _unlink(ref_path)
        gen_ms = (time.perf_counter() - started) * 1000.0
        sample_rate_hz = int(getattr(getattr(self.model, "tts_model", None), "sample_rate", 48_000))
        wav_bytes = _float_to_wav(audio, sample_rate_hz)
        audio_ms = _audio_duration_ms(wav_bytes, sample_rate_hz)

        return ResponsesAgentResponse(
            output=[self.create_text_output_item(text=text, id="tts-spoken-text")],
            custom_outputs={
                "audio_b64": base64.b64encode(wav_bytes).decode("ascii"),
                "mime_type": "audio/wav",
                "sample_rate_hz": sample_rate_hz,
                "language": language,
                "inference_device": self.inference_device,
                "gen_ms": round(gen_ms, 1),
                "audio_ms": round(audio_ms, 1),
                # Real-time factor (<1.0 means faster than realtime).
                "rtf": round(gen_ms / audio_ms, 3) if audio_ms else None,
                "inference_timesteps": timesteps,
                "cfg_value": cfg_value,
            },
        )

    def predict_stream(self, request: ResponsesAgentRequest):  # type: ignore[override]
        """Stream audio as it is generated (time-to-first-audio ≪ full sentence).

        VoxCPM2's ``generate(streaming=True)`` yields ~80 ms PCM chunks per step.
        Each chunk is emitted as a ``ResponsesAgentStreamEvent`` whose
        ``custom_outputs`` carries raw base64 PCM16 (no per-chunk WAV header, so
        the client concatenates chunks directly). A final ``done`` event reports
        timing. This is the realtime win on the current GPU: the caller can begin
        playback after the first chunk instead of waiting ~3 s for the sentence.
        """
        item_id = "tts-spoken-text"
        if not request.custom_inputs:
            yield ResponsesAgentStreamEvent(
                type="response.output_item.done",
                item=self.create_text_output_item(text="", id=item_id),
                custom_outputs={"audio_pcm16_b64": "", "sample_rate_hz": 0, "final": True},
            )
            return
        self._ensure_model()
        ci = dict(request.custom_inputs)
        text = str(ci.get("text") or _last_user_text(request) or "").strip()
        if not text:
            raise ValueError("No text supplied for synthesis (custom_inputs.text or input message)")
        timesteps = int(ci.get("inference_timesteps") or self.inference_timesteps)
        cfg_value = float(ci.get("cfg_value") or self.cfg_value)
        sample_rate_hz = int(getattr(getattr(self.model, "tts_model", None), "sample_rate", 48_000))

        started = time.perf_counter()
        first_chunk_ms: float | None = None
        index = 0
        # generate_streaming() yields ~80 ms numpy chunks (it sets streaming=True
        # internally, so we must NOT pass streaming= ourselves).
        gen_kwargs = dict(
            text=text,
            cfg_value=cfg_value,
            inference_timesteps=timesteps,
            retry_badcase=False,
        )
        # A reference clip pins the voice: VoxCPM2 clones its timbre so every turn
        # in a session keeps the same voice. build_prompt_cache reads the file at
        # the start of generation, so it must survive until the generator drains.
        ref_path = _write_reference_wav(ci.get("reference_audio_b64"))
        if ref_path:
            gen_kwargs["reference_wav_path"] = ref_path
        try:
            for chunk in self.model.generate_streaming(**gen_kwargs):
                pcm = _float_to_pcm16(chunk)
                if not pcm:
                    continue
                if first_chunk_ms is None:
                    first_chunk_ms = (time.perf_counter() - started) * 1000.0
                yield ResponsesAgentStreamEvent(
                    **self.create_text_delta(delta=(text if index == 0 else ""), item_id=item_id),
                    custom_outputs={
                        "audio_pcm16_b64": base64.b64encode(pcm).decode("ascii"),
                        "chunk_index": index,
                        "sample_rate_hz": sample_rate_hz,
                        "final": False,
                    },
                )
                index += 1
        finally:
            _unlink(ref_path)

        total_ms = (time.perf_counter() - started) * 1000.0
        yield ResponsesAgentStreamEvent(
            type="response.output_item.done",
            item=self.create_text_output_item(text=text, id=item_id),
            custom_outputs={
                "final": True,
                "chunks": index,
                "sample_rate_hz": sample_rate_hz,
                "ttfb_ms": round(first_chunk_ms if first_chunk_ms is not None else total_ms, 1),
                "gen_ms": round(total_ms, 1),
                "inference_device": self.inference_device,
                "inference_timesteps": timesteps,
            },
        )


def _serving_runtime_available() -> bool:
    """True only when the heavy inference runtime is importable.

    Distinguishes the serving container (torch + voxcpm present) from the
    lightweight registration job (neither present), so eager model loading in
    ``load_context`` happens only where it is possible and desired.
    """
    import importlib.util

    try:
        return all(importlib.util.find_spec(name) is not None for name in ("torch", "voxcpm"))
    except Exception:  # noqa: BLE001
        return False


# Fixed reference-clip duration. Normalising every reference to one constant
# length gives the voice-cloning graph a single shape for torch.compile to
# specialise on, so it compiles exactly once (at startup warm-up) instead of
# recompiling (~15 s) on the first cloned turn of each new session.
_REFERENCE_SECONDS = 4.0


def _write_reference_wav(reference_audio_b64: Any) -> str | None:
    """Materialise a base64 WAV reference clip to a temp file for voice cloning.

    The clip is trimmed/padded to a fixed duration (see ``_REFERENCE_SECONDS``)
    so the compiled graph shape is constant across sessions. Returns the path
    (caller must ``_unlink`` it) or None when no reference was supplied or it
    could not be decoded (never fail synthesis over a bad reference).
    """
    if not reference_audio_b64:
        return None
    try:
        raw = base64.b64decode(str(reference_audio_b64))
    except Exception:  # noqa: BLE001
        return None
    if not raw:
        return None
    raw = _normalize_reference(raw)
    fd, path = tempfile.mkstemp(suffix=".wav")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
    except OSError:
        _unlink(path)
        return None
    return path


def _normalize_reference(wav_bytes: bytes) -> bytes:
    """Trim/pad a mono PCM16 WAV to exactly ``_REFERENCE_SECONDS``.

    Returns the input unchanged when it is not the mono/16-bit shape this service
    produces (be permissive; only normalise what we know how to)."""
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as reader:
            channels = reader.getnchannels()
            width = reader.getsampwidth()
            rate = reader.getframerate()
            frames = reader.readframes(reader.getnframes())
    except (wave.Error, EOFError, OSError):
        return wav_bytes
    if channels != 1 or width != 2 or rate <= 0:
        return wav_bytes
    target = int(rate * _REFERENCE_SECONDS) * 2  # bytes: 16-bit mono
    frames = frames[:target] if len(frames) >= target else frames + b"\x00" * (target - len(frames))
    out = io.BytesIO()
    with wave.open(out, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(rate)
        writer.writeframes(frames)
    return out.getvalue()


def _unlink(path: str | None) -> None:
    if not path:
        return
    try:
        os.unlink(path)
    except OSError:
        pass


def _last_user_text(request: ResponsesAgentRequest) -> str:
    for item in reversed([i.model_dump() for i in request.input]):
        if item.get("role") == "user":
            content = item.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") in ("input_text", "text"):
                        return str(part.get("text") or "")
    return ""


def _audio_duration_ms(wav_bytes: bytes, sample_rate_hz: int) -> float:
    # 44-byte WAV header + 16-bit mono PCM body.
    samples = max(len(wav_bytes) - 44, 0) / 2
    return samples / sample_rate_hz * 1000.0 if sample_rate_hz else 0.0


def _float_to_pcm16(audio: Any) -> bytes:
    """Raw little-endian PCM16 bytes (no WAV header) for a streamed chunk."""
    import numpy as np

    array = np.asarray(audio, dtype=np.float32).flatten()
    if array.size == 0:
        return b""
    array = np.clip(array, -1.0, 1.0)
    return (array * 32767.0).astype("<i2").tobytes()


def _float_to_wav(audio: Any, sample_rate_hz: int) -> bytes:
    import numpy as np

    array = np.asarray(audio, dtype=np.float32).flatten()
    array = np.clip(array, -1.0, 1.0)
    pcm = (array * 32767.0).astype(np.int16).tobytes()
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate_hz)
        handle.writeframes(pcm)
    return buffer.getvalue()
