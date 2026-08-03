"""Realtime session state (VAD buffering and turn lifecycle)."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .contracts import SessionStart
from .endpointing import TurnEndpointer

# --- Adaptive energy VAD tuning ---------------------------------------------
# A single fixed RMS gate can't serve both a quiet mic (real speech falls under
# it → captured as silence → STT fragments) and a noisy room (ambient sits above
# it → the turn never ends). Instead we track the background noise floor and gate
# *relative* to it, with hysteresis so mid-word dips don't flap the state.
#
# on  = max(_ABS_MIN_RMS, floor * _ONSET_MULT): energy needed to START speech.
# off = on * _HANGOVER_RATIO: while already speaking, energy must fall below this
#       for a frame to count as trailing silence (prevents brief dips ending the
#       turn). The floor only adapts on non-speech frames, so speech never pulls
#       the threshold up after itself.
_ABS_MIN_RMS = 110.0  # ~ -49 dBFS; keeps digital/near silence from ever triggering
_ONSET_MULT = 2.2
_HANGOVER_RATIO = 0.55
_FLOOR_ALPHA = 0.08  # EMA weight for adapting the noise floor toward background
_INITIAL_NOISE_FLOOR = 80.0  # conservative seed before any calibration frames
# Hard ceiling on the noise floor: without it, quiet speech misclassified as
# background can ratchet the floor (and thus the onset gate) upward until real
# speech no longer registers. Capping the floor keeps the onset gate at/below
# ~_FLOOR_CEILING * _ONSET_MULT so normal speech always clears it.
_FLOOR_CEILING = 150.0


@dataclass
class VoiceSession:
    config: SessionStart
    # WS-connection id, set by the handler at accept time. Used to group per-turn
    # traces belonging to the same call in the observability view.
    session_id: str | None = None
    turn_id: int = 0
    audio: bytearray = field(default_factory=bytearray)
    speech_active: bool = False
    silence_ms: float = 0.0
    turn_audio_ms: float = 0.0
    voiced_ms: float = 0.0
    last_rms: float = 0.0
    # Running background-noise estimate, adapted across the whole call (NOT reset
    # per turn) so the gate stays calibrated to the caller's room/mic level.
    noise_floor: float = _INITIAL_NOISE_FLOOR
    busy: bool = False
    history: list = field(default_factory=list)
    # Session-scoped account-facts cache (customer_id -> facts), shared across
    # turns and passed into each turn's ToolContext. Lets a confirmation turn
    # that only calls apply_billing_action reuse facts read by an earlier
    # lookup_account turn. Invalidated by the billing tool on any write.
    account_store: dict = field(default_factory=dict)
    _cooldown_until: float = 0.0
    # Voice consistency: the agent's first synthesized turn is captured as a
    # reference clip (base64 WAV, set exactly once). Every later turn clones it
    # via VoxCPM2 reference-audio cloning, so one stable voice is used for the
    # whole call. Cloning from real audio is deterministic w.r.t. timbre, unlike
    # RNG seeding (which the deployed model does not even support).
    voice_reference_b64: str | None = None
    # Stable id for the clip above, derived from its bytes. The TTS endpoint caches
    # the clip under this id so later turns identify the voice without re-uploading
    # ~500KB of base64 audio on every turn (which dominated time-to-first-audio).
    voice_id: str | None = None
    # Semantic end-of-turn detector (Silero VAD + smart-turn). When set, it
    # replaces the energy VAD for speech gating and end-of-turn decisions; None
    # keeps the legacy energy VAD (also the graceful fallback when models can't
    # load). Injected by the handler at session.start.
    endpointer: TurnEndpointer | None = None
    # Client-managed turn mode (session started with ``endpointing: false``). The
    # handler does no automatic finalization in this mode and ends the turn only
    # on an explicit ``audio.end`` (plus the max_turn safety cap). Off by default,
    # so the live voice loop keeps server-side turn detection.
    manual_turns: bool = False
    # Small, opaque state a selected assistant profile persists across turns (via
    # its after_turn hook). Empty and unused by the default telco path; the engine
    # never inspects its contents, so it stays domain-agnostic.
    profile_state: dict[str, object] = field(default_factory=dict)

    def add_audio(self, frame: bytes) -> bool:
        """Append PCM audio and report whether this frame begins speech."""
        if len(frame) % 2:
            raise ValueError("PCM s16le audio frame must contain an even number of bytes")
        if self.endpointer is not None:
            return self._add_audio_smart(frame)
        return self._add_audio_energy(frame)

    def _add_audio_smart(self, frame: bytes) -> bool:
        """Silero-gated ingest: mirror speech/silence counters from the endpointer."""
        frame_ms = len(frame) / 2 / self.config.sample_rate_hz * 1000
        self.last_rms = _pcm_s16le_rms(frame)
        began = self.endpointer.feed(frame)
        self.speech_active = self.endpointer.triggered
        self.voiced_ms = self.endpointer.speech_ms
        self.silence_ms = self.endpointer.silence_ms
        self.turn_audio_ms += frame_ms
        self.audio.extend(frame)
        return began

    def _add_audio_energy(self, frame: bytes) -> bool:
        frame_ms = len(frame) / 2 / self.config.sample_rate_hz * 1000
        rms = _pcm_s16le_rms(frame)
        self.last_rms = rms

        on_threshold = max(_ABS_MIN_RMS, self.noise_floor * _ONSET_MULT)
        off_threshold = on_threshold * _HANGOVER_RATIO
        # Hysteresis: starting speech needs the (higher) onset threshold; once
        # speaking, we stay voiced until energy drops below the (lower) hangover
        # threshold, so a brief between-syllable dip isn't counted as silence.
        gate = off_threshold if self.speech_active else on_threshold
        is_voiced = bool(frame) and rms >= gate

        began_speech = is_voiced and not self.speech_active
        if is_voiced:
            self.speech_active = True
            self.silence_ms = 0.0
            self.voiced_ms += frame_ms
        else:
            if self.speech_active:
                self.silence_ms += frame_ms
            # Only non-speech frames update the floor, so speech can't inflate the
            # threshold. EMA tracks the room, clamped to a ceiling so misclassified
            # quiet speech can't ratchet the gate above real speech.
            self.noise_floor += _FLOOR_ALPHA * (rms - self.noise_floor)
            if self.noise_floor > _FLOOR_CEILING:
                self.noise_floor = _FLOOR_CEILING
        self.turn_audio_ms += frame_ms
        self.audio.extend(frame)
        return began_speech

    def should_finalize(self, *, silence_ms: int, max_turn_seconds: int, min_speech_ms: int = 0) -> bool:
        if time.monotonic() < self._cooldown_until:
            return False
        if self.turn_audio_ms >= max_turn_seconds * 1000:
            return True
        return (
            self.speech_active
            and self.voiced_ms >= min_speech_ms
            and self.silence_ms >= silence_ms
        )

    def set_cooldown(self, seconds: float) -> None:
        """Suppress turn finalization for `seconds` after the agent finishes speaking."""
        self._cooldown_until = time.monotonic() + seconds

    def finish_turn(self) -> tuple[int, bytes] | None:
        if not self.audio:
            return None
        self.turn_id += 1
        audio = bytes(self.audio)
        self._reset_turn()
        return self.turn_id, audio

    def barge_in(self) -> int:
        self.turn_id += 1
        self._reset_turn()
        return self.turn_id

    def discard_buffer(self) -> None:
        """Drop audio captured while busy (echo/tail) without bumping turn_id."""
        self._reset_turn()

    def is_noise_timeout(self, seconds: float) -> bool:
        """Smart-path only: buffer has run this long with no confirmed speech.

        Signals ambient noise that the energy VAD would have held open to the
        cap; the handler discards it instead of transcribing silence.
        """
        return (
            self.endpointer is not None
            and not self.endpointer.has_speech
            and self.turn_audio_ms >= seconds * 1000
        )

    def _reset_turn(self) -> None:
        self.audio.clear()
        self.speech_active = False
        self.silence_ms = 0.0
        self.turn_audio_ms = 0.0
        self.voiced_ms = 0.0
        if self.endpointer is not None:
            self.endpointer.reset()


def _pcm_s16le_rms(frame: bytes) -> float:
    samples = [
        int.from_bytes(frame[index : index + 2], byteorder="little", signed=True)
        for index in range(0, len(frame), 2)
    ]
    if not samples:
        return 0.0
    return (sum(sample * sample for sample in samples) / len(samples)) ** 0.5
