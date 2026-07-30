"""Diagnostic: measure the ACTUAL VoxCPM TTS waveform the card/billing UI plays.

Runs the real production synthesis path (realtime_api serving, cfg_value from
config) and reports objective audio quality signals — peak, RMS, crest factor,
and clipping percentage — so we can tell whether the "too loud / not professional"
complaint is (a) clean-but-hot audio (a playback gain is the right fix) or
(b) clipped/over-driven at the source (a playback gain would NOT fix the harshness).

Writes /tmp/tts_noref.wav and /tmp/tts_withref.wav for listening.
"""
from __future__ import annotations

import base64
import io
import wave

import numpy as np

from realtime_api.config import RealtimeSettings
from realtime_api.serving_factory import build_serving

FULL = 32767.0


def analyze(pcm_bytes: bytes, sr: int, label: str) -> None:
    a = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    if a.size == 0:
        print(f"{label}: EMPTY audio")
        return
    peak = float(np.max(np.abs(a)))
    clip_pct = float(np.mean(np.abs(a) >= FULL * 0.999) * 100)
    near_pct = float(np.mean(np.abs(a) >= FULL * 0.95) * 100)
    rms = float(np.sqrt(np.mean((a / FULL) ** 2)))
    peak_db = 20 * np.log10(max(peak, 1) / FULL)
    rms_db = 20 * np.log10(max(rms, 1e-9))
    crest = peak_db - rms_db
    dur = a.size / sr if sr else 0
    print(
        f"{label}:\n"
        f"    sample_rate = {sr} Hz   duration = {dur:.2f}s   samples = {a.size}\n"
        f"    peak        = {peak:.0f} / {FULL:.0f}   ({peak_db:+.2f} dBFS)\n"
        f"    rms         = {rms:.4f}                 ({rms_db:+.2f} dBFS)\n"
        f"    crest factor= {crest:.1f} dB   (speech healthy ~12-20 dB; <8 dB = squashed/harsh)\n"
        f"    clipping    = {clip_pct:.3f}% at full-scale   {near_pct:.3f}% within 95% of full-scale\n"
    )


def save(pcm_bytes: bytes, sr: int, path: str) -> None:
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm_bytes)
    print(f"    wrote {path}")


def synth(serving, text: str, ref_b64: str | None):
    chunks = list(serving.synthesize_stream(text, language="en-US", reference_audio_b64=ref_b64))
    sr = chunks[0].sample_rate_hz if chunks else 0
    pcm = b"".join(c.pcm for c in chunks)
    return pcm, sr


def main() -> None:
    settings = RealtimeSettings.resolve()
    print(
        f"config: tts_endpoint={settings.tts_endpoint} "
        f"cfg_value={settings.tts_cfg_value} timesteps={settings.tts_inference_timesteps}\n"
    )
    serving = build_serving(settings)

    greeting = (
        "Hi Suneel, I'm Genie Agent, your EveryCard assistant. "
        "I can help you understand your latest statement or check your rewards."
    )
    pcm, sr = synth(serving, greeting, None)
    analyze(pcm, sr, "TURN 1  (greeting, NO reference — VoxCPM default voice)")
    save(pcm, sr, "/tmp/tts_noref.wav")

    # Mirror _lock_voice_reference: first ~4s wrapped as a WAV becomes the clone ref.
    ref_seconds = 4.0
    ref_pcm = pcm[: int(sr * 2 * ref_seconds)]
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(ref_pcm)
    ref_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    followup = "Your latest statement total is higher than last month, mostly from dining and travel."
    pcm2, sr2 = synth(serving, followup, ref_b64)
    analyze(pcm2, sr2, "TURN 2  (WITH self-clone reference — the rest of the call)")
    save(pcm2, sr2, "/tmp/tts_withref.wav")


if __name__ == "__main__":
    main()
