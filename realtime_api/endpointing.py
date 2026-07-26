"""Semantic end-of-turn detection: Silero VAD (speech gate) + smart-turn v3.

Why this exists
---------------
The legacy energy-RMS VAD gates end-of-turn purely on a trailing-silence
threshold relative to an adaptive noise floor. In real rooms it fails two ways:
  * it holds a turn open on ambient noise (empty turns run to the 20 s cap), and
  * it mis-reads long between-clause pauses in real speech, so genuine turns also
    hit the cap — the caller waits many seconds for the agent to even start.

This module replaces that decision with a two-stage, audio-native pipeline that
mirrors modern voice stacks (Pipecat / LiveKit):

  1. **Silero VAD** (streaming, pure onnxruntime) is the *speech gate*. Only
     confirmed speech starts/keeps a turn; a buffer with no detected speech is
     discarded instead of transcribed (kills the empty 20 s turns).
  2. **smart-turn v3** (audio-native semantic end-of-turn model) decides, at each
     Silero-detected pause, whether the utterance is *complete*. If so the turn
     finalizes ~0.3 s after the caller stops; if not, we keep listening.

Both models run on CPU with numpy + onnxruntime (no torch / transformers). The
smart-turn feature extractor is the vendored numpy Whisper frontend.

Validated on real captured browser turns: real speech finalizes shortly after
speech ends (median ~9 s saved vs the 20 s cap), 0 premature cutoffs, and
no-speech turns are correctly discarded.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ._whisper_frontend import log_mel_features

logger = logging.getLogger(__name__)

_MODEL_DIR = Path(__file__).resolve().parent / "models"
_SILERO_PATH = _MODEL_DIR / "silero_vad.onnx"
_SMART_TURN_PATH = _MODEL_DIR / "smart_turn_v3.onnx"

# Silero v5 operates on 512-sample frames at 16 kHz (32 ms), prepending the last
# 64 samples of the prior frame as context.
_TARGET_SR = 16_000
_SILERO_FRAME = 512
_SILERO_CTX = 64
# smart-turn consumes up to 8 s of 16 kHz audio.
_SMART_TURN_MAX_SAMPLES = 8 * _TARGET_SR

# smart-turn v3 was trained on these 23 languages (primary subtags). Calls in a
# language outside this set fall back to VAD-only endpointing (Silero pause +
# the configured silence gap), since smart-turn's completeness signal is
# untrained there and could stall or cut turns.
SMART_TURN_LANGUAGES: frozenset[str] = frozenset(
    {
        "ar", "bn", "zh", "da", "nl", "de", "en", "fi", "fr", "hi", "id", "it",
        "ja", "ko", "mr", "no", "pl", "pt", "ru", "es", "tr", "uk", "vi",
    }
)


class EndpointModels:
    """Process-wide, thread-safe ONNX sessions shared across all voice sessions.

    onnxruntime ``InferenceSession.run`` is thread-safe; the per-turn *state*
    (Silero LSTM state, audio buffers) lives in :class:`TurnEndpointer`, not here.
    Construct once at app startup and inject into each session.
    """

    def __init__(self, silero_path: Path, smart_turn_path: Path) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 3
        providers = ["CPUExecutionProvider"]
        self.silero = ort.InferenceSession(str(silero_path), sess_options=opts, providers=providers)
        self.smart_turn = ort.InferenceSession(
            str(smart_turn_path), sess_options=opts, providers=providers
        )

    @classmethod
    def load(cls) -> "EndpointModels | None":
        """Load the bundled models, or return None if unavailable/unloadable.

        Returning None (rather than raising) lets the app degrade gracefully to
        the legacy energy VAD when the models or onnxruntime aren't present.
        """
        if not _SILERO_PATH.exists() or not _SMART_TURN_PATH.exists():
            logger.warning(
                "endpointing models not found under %s; falling back to energy VAD", _MODEL_DIR
            )
            return None
        try:
            models = cls(_SILERO_PATH, _SMART_TURN_PATH)
        except Exception as exc:  # noqa: BLE001 - onnxruntime import/load failure
            logger.warning("endpointing models failed to load (%s); falling back to energy VAD", exc)
            return None
        logger.info("endpointing models loaded (Silero VAD + smart-turn v3)")
        return models


class _LinearResampler:
    """Streaming linear resampler to 16 kHz with fractional-index carry-over.

    Adequate for VAD/smart-turn (both robust to mild resampling artefacts); the
    audio sent to STT keeps its original rate and is not touched here.
    """

    def __init__(self, input_rate: int) -> None:
        self._ratio = input_rate / _TARGET_SR
        self._pos = 0.0
        self._prev = np.zeros(1, dtype=np.float32)
        self._identity = input_rate == _TARGET_SR

    def push(self, samples: np.ndarray) -> np.ndarray:
        if self._identity:
            return samples.astype(np.float32, copy=False)
        # Prepend the last sample of the previous block so interpolation is
        # continuous across block boundaries.
        buf = np.concatenate([self._prev, samples.astype(np.float32)])
        # Output sample positions (in buf index space) starting from carry.
        out = []
        pos = self._pos
        limit = len(buf) - 1
        while pos < limit:
            i = int(pos)
            frac = pos - i
            out.append(buf[i] * (1.0 - frac) + buf[i + 1] * frac)
            pos += self._ratio
        self._pos = pos - (len(buf) - 1)  # carry fractional remainder past block end
        self._prev = buf[-1:]
        return np.asarray(out, dtype=np.float32)


class TurnEndpointer:
    """Per-session end-of-turn detector.

    Feed raw PCM (s16le) chunks at the session's capture rate via :meth:`feed`;
    it resamples to 16 kHz, runs streaming Silero VAD, and tracks speech/silence.
    :meth:`take_pause_candidate` flags a genuine post-speech pause (debounced),
    at which point the caller runs :meth:`smart_turn_complete` (off the event
    loop) to decide whether to finalize.
    """

    def __init__(
        self,
        models: EndpointModels,
        *,
        sample_rate_hz: int,
        stop_ms: int,
        min_speech_ms: int,
        use_smart_turn: bool,
    ) -> None:
        self._models = models
        self._sample_rate_hz = sample_rate_hz
        self._stop_ms = float(stop_ms)
        self._min_speech_ms = float(min_speech_ms)
        self.use_smart_turn = use_smart_turn
        self.reset()

    def reset(self) -> None:
        """Clear all per-turn state (Silero LSTM state, buffers, counters)."""
        self._resampler = _LinearResampler(self._sample_rate_hz)
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._ctx = np.zeros((1, _SILERO_CTX), dtype=np.float32)
        self._pending = np.zeros(0, dtype=np.float32)  # <512-sample tail awaiting a full frame
        self._audio16k = np.zeros(0, dtype=np.float32)  # rolling 16k buffer for smart-turn
        self.speech_ms = 0.0
        self.silence_ms = 0.0
        self.triggered = False
        self._last_eval_silence = 0.0

    def feed(self, pcm: bytes) -> bool:
        """Ingest a PCM s16le chunk. Returns True if this chunk *begins* speech."""
        if not pcm:
            return False
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        audio16k = self._resampler.push(samples)
        if audio16k.size:
            self._audio16k = np.concatenate([self._audio16k, audio16k])
            if self._audio16k.size > _SMART_TURN_MAX_SAMPLES:
                self._audio16k = self._audio16k[-_SMART_TURN_MAX_SAMPLES:]
        began = False
        buf = np.concatenate([self._pending, audio16k]) if self._pending.size else audio16k
        n_frames = buf.size // _SILERO_FRAME
        for i in range(n_frames):
            frame = buf[i * _SILERO_FRAME : (i + 1) * _SILERO_FRAME]
            was_triggered = self.triggered
            self._push_silero(frame)
            if self.triggered and not was_triggered:
                began = True
        self._pending = buf[n_frames * _SILERO_FRAME :].copy()
        return began

    def _push_silero(self, frame: np.ndarray) -> None:
        x = np.concatenate([self._ctx, frame[None, :]], axis=1)
        prob, self._state = self._models.silero.run(
            None,
            {"input": x, "state": self._state, "sr": np.array(_TARGET_SR, dtype=np.int64)},
        )
        self._ctx = x[:, -_SILERO_CTX:]
        frame_ms = _SILERO_FRAME / _TARGET_SR * 1000.0
        if float(prob.item()) >= 0.5:
            self.triggered = True
            self.speech_ms += frame_ms
            self.silence_ms = 0.0
            self._last_eval_silence = 0.0
        elif self.triggered:
            self.silence_ms += frame_ms

    @property
    def has_speech(self) -> bool:
        return self.speech_ms >= self._min_speech_ms

    def take_pause_candidate(self) -> bool:
        """True at most once per ``stop_ms`` of silence after enough speech.

        Debounced so smart-turn is evaluated when a pause first reaches
        ``stop_ms`` and then again only after another ``stop_ms`` of continued
        silence (avoids re-running smart-turn every frame during a long pause).
        """
        if self.speech_ms < self._min_speech_ms or self.silence_ms < self._stop_ms:
            return False
        if self.silence_ms - self._last_eval_silence < self._stop_ms:
            return False
        self._last_eval_silence = self.silence_ms
        return True

    def smart_turn_complete(self, threshold: float) -> tuple[bool, float]:
        """Run smart-turn on the buffered 16 kHz audio. Returns (complete, prob).

        CPU-bound (~9 ms); call via a thread executor from the async handler.
        """
        if self._audio16k.size == 0:
            return False, 0.0
        features = log_mel_features(self._audio16k)
        prob = float(self._models.smart_turn.run(None, {"input_features": features})[0].item())
        return prob >= threshold, prob


def endpointer_for(
    models: EndpointModels | None,
    *,
    sample_rate_hz: int,
    stop_ms: int,
    min_speech_ms: int,
    expected_language: str | None,
) -> TurnEndpointer | None:
    """Build a per-session endpointer, or None to keep the legacy energy VAD."""
    if models is None:
        return None
    primary = (expected_language or "").split("-")[0].lower()
    # "auto"/unknown -> assume a supported language (smart-turn is the default).
    use_smart_turn = primary == "" or primary == "auto" or primary in SMART_TURN_LANGUAGES
    return TurnEndpointer(
        models,
        sample_rate_hz=sample_rate_hz,
        stop_ms=stop_ms,
        min_speech_ms=min_speech_ms,
        use_smart_turn=use_smart_turn,
    )
