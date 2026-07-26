"""Self-contained numpy Whisper log-mel frontend for smart-turn endpointing.

Vendored from ``transformers.audio_utils`` / ``WhisperFeatureExtractor``
(Apache-2.0) so the realtime service does not depend on ``transformers``/``torch``.
Produces the exact ``input_features`` tensor smart-turn v3 expects: shape
(1, 80, 800). Validated bit-exact against the reference feature extractor.
"""
from __future__ import annotations

import numpy as np

_N_FFT = 400
_HOP = 160
_N_MELS = 80
_CHUNK_S = 8
_N_SAMPLES = _CHUNK_S * 16000  # 128000


def _hertz_to_mel(freq):
    min_log_hertz, min_log_mel = 1000.0, 15.0
    logstep = 27.0 / np.log(6.4)
    mels = 3.0 * freq / 200.0
    if isinstance(freq, np.ndarray):
        lr = freq >= min_log_hertz
        mels[lr] = min_log_mel + np.log(freq[lr] / min_log_hertz) * logstep
    elif freq >= min_log_hertz:
        mels = min_log_mel + np.log(freq / min_log_hertz) * logstep
    return mels


def _mel_to_hertz(mels):
    min_log_hertz, min_log_mel = 1000.0, 15.0
    logstep = np.log(6.4) / 27.0
    freq = 200.0 * mels / 3.0
    if isinstance(mels, np.ndarray):
        lr = mels >= min_log_mel
        freq[lr] = min_log_hertz * np.exp(logstep * (mels[lr] - min_log_mel))
    elif mels >= min_log_mel:
        freq = min_log_hertz * np.exp(logstep * (mels - min_log_mel))
    return freq


def _triangular_filter_bank(fft_freqs, filter_freqs):
    filter_diff = np.diff(filter_freqs)
    slopes = np.expand_dims(filter_freqs, 0) - np.expand_dims(fft_freqs, 1)
    down = -slopes[:, :-2] / filter_diff[:-1]
    up = slopes[:, 2:] / filter_diff[1:]
    return np.maximum(np.zeros(1), np.minimum(down, up))


def _mel_filter_bank():
    n_freq = 1 + _N_FFT // 2
    mel_min = _hertz_to_mel(0.0)
    mel_max = _hertz_to_mel(8000.0)
    mel_freqs = np.linspace(mel_min, mel_max, _N_MELS + 2)
    filter_freqs = _mel_to_hertz(mel_freqs)
    fft_freqs = np.linspace(0, 16000 // 2, n_freq)
    mel = _triangular_filter_bank(fft_freqs, filter_freqs)
    enorm = 2.0 / (filter_freqs[2 : _N_MELS + 2] - filter_freqs[:_N_MELS])
    mel *= np.expand_dims(enorm, 0)
    return mel  # (201, 80)


_MEL_FILTERS = _mel_filter_bank()
_WINDOW = np.hanning(_N_FFT + 1)[:-1]  # periodic hann, length 400


def _spectrogram(waveform: np.ndarray) -> np.ndarray:
    # center pad (reflect), matches audio_utils.spectrogram(center=True)
    pad = _N_FFT // 2
    waveform = np.pad(waveform, [(pad, pad)], mode="reflect").astype(np.float64)
    window = _WINDOW.astype(np.float64)
    num_frames = int(1 + np.floor((waveform.size - _N_FFT) / _HOP))
    n_freq = _N_FFT // 2 + 1
    spec = np.empty((num_frames, n_freq), dtype=np.complex64)
    buf = np.zeros(_N_FFT)
    t = 0
    for i in range(num_frames):
        buf[:_N_FFT] = waveform[t : t + _N_FFT] * window
        spec[i] = np.fft.rfft(buf)
        t += _HOP
    spec = (np.abs(spec, dtype=np.float64) ** 2.0).T
    spec = np.maximum(1e-10, np.dot(_MEL_FILTERS.T, spec))
    return np.log10(spec).astype(np.float32)


def log_mel_features(audio: np.ndarray) -> np.ndarray:
    """audio: float32 mono 16kHz, any length -> (1, 80, 800) input_features."""
    a = audio.astype(np.float32)
    if len(a) > _N_SAMPLES:
        a = a[-_N_SAMPLES:]
    elif len(a) < _N_SAMPLES:
        a = np.pad(a, (_N_SAMPLES - len(a), 0))
    # do_normalize: zero-mean unit-var over the full (padded) buffer
    a = (a - a.mean()) / np.sqrt(a.var() + 1e-7)
    log_spec = _spectrogram(a)  # (80, 801)
    log_spec = log_spec[:, :-1]  # drop last frame -> (80, 800)
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return np.expand_dims(log_spec.astype(np.float32), 0)
