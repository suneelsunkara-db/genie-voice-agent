"""Tests for the app-wide fixed agent voice.

The user-visible bug these guard: with no reference clip, VoxCPM2 invents a
speaker on the first turn of every session, so the agent sounded like a different
person on each call and the Home greeting never matched the next page. Seeding
every session from one committed clip is what makes the voice identical
everywhere, so the invariant under test is "same voice_id, every session".

A bad or missing clip must degrade to the old per-session behaviour rather than
taking the agent mute, so the fallback paths are tested too.
"""
from __future__ import annotations

import asyncio
import wave
from pathlib import Path

from realtime_api.contracts import AudioChunk
from realtime_api.pipelines import ServingBundle
from realtime_api.pipelines._shared import stream_tts
from realtime_api.session import SessionStart, VoiceSession
from realtime_api.voice_identity import load_voice_seed, voice_id_for


class _RecordingTTS:
    """Records the reference/voice_id handed to every synthesis call."""

    def __init__(self) -> None:
        self.calls: list[tuple[str | None, str | None]] = []

    def synthesize_stream(
        self,
        text: str,
        *,
        language: str,
        reference_audio_b64: str | None = None,
        voice_id: str | None = None,
    ):
        self.calls.append((reference_audio_b64, voice_id))
        yield AudioChunk(pcm=b"\x00\x01" * 100, sample_rate_hz=48_000)


def _wav(path: Path, *, seconds: float = 4.0, channels: int = 1,
         width: int = 2, rate: int = 48_000, fill: bytes = b"\x01\x02") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(width)
        handle.setframerate(rate)
        handle.writeframes(fill * int(rate * seconds * channels * width // len(fill)))
    return path


def _session() -> VoiceSession:
    return VoiceSession(SessionStart.from_event({"language": "en-US", "sample_rate_hz": 16000}))


def _seed(session: VoiceSession, clip: Path) -> None:
    """Mirror what the WS handler does at session.start."""
    seed = load_voice_seed(str(clip))
    assert seed is not None
    session.voice_reference_b64 = seed.reference_b64
    session.voice_id = seed.voice_id


def _drain(session: VoiceSession, tts: _RecordingTTS) -> None:
    async def go() -> None:
        async for _ in stream_tts(
            ServingBundle(stt=None, llm=None, tts=tts), session, session.turn_id, "hello", "en-US"
        ):
            pass

    asyncio.run(go())


# --- the invariant the user is asking for ----------------------------------


def test_every_session_gets_the_same_voice(tmp_path) -> None:
    """Two independent sessions must clone one timbre — the whole point of the fix."""
    clip = _wav(tmp_path / "same" / "agent_voice.wav")
    first, second = _session(), _session()
    _seed(first, clip)
    _seed(second, clip)
    assert first.voice_id == second.voice_id
    assert first.voice_reference_b64 == second.voice_reference_b64


def test_first_turn_already_carries_the_reference(tmp_path) -> None:
    """The regression: turn 1 used to synthesize with reference_audio_b64=None.

    That unprompted turn is what produced an arbitrary speaker, and it then became
    the session's voice — so the very first call must already send the clip.
    """
    clip = _wav(tmp_path / "first" / "agent_voice.wav")
    session, tts = _session(), _RecordingTTS()
    _seed(session, clip)
    _drain(session, tts)
    reference, voice_id = tts.calls[0]
    assert reference is not None
    assert voice_id == session.voice_id


def test_seeded_voice_never_drifts(tmp_path) -> None:
    """Synthesized audio must not overwrite the pinned clip on any later turn."""
    clip = _wav(tmp_path / "drift" / "agent_voice.wav")
    session, tts = _session(), _RecordingTTS()
    _seed(session, clip)
    pinned = session.voice_id
    for _ in range(3):
        session.turn_id += 1
        _drain(session, tts)
    assert session.voice_id == pinned
    assert {call[1] for call in tts.calls} == {pinned}


def test_unseeded_session_bootstraps_its_own_voice() -> None:
    """Fallback behaviour: no clip configured still yields a within-call voice."""
    session, tts = _session(), _RecordingTTS()
    _drain(session, tts)
    assert tts.calls[0] == (None, None)
    assert session.voice_id is not None  # locked from the first turn's audio


# --- clip loading / validation ---------------------------------------------


def test_voice_id_is_content_addressed(tmp_path) -> None:
    assert voice_id_for(b"abc") == voice_id_for(b"abc")
    assert voice_id_for(b"abc") != voice_id_for(b"abd")


def test_valid_clip_reports_its_shape(tmp_path) -> None:
    clip = _wav(tmp_path / "shape" / "agent_voice.wav", seconds=4.0, rate=48_000)
    seed = load_voice_seed(str(clip))
    assert seed is not None
    assert seed.sample_rate_hz == 48_000
    assert round(seed.seconds) == 4
    assert seed.voice_id == voice_id_for(clip.read_bytes())


def test_unset_path_is_not_an_error() -> None:
    assert load_voice_seed(None) is None
    assert load_voice_seed("") is None


def test_missing_clip_falls_back(tmp_path) -> None:
    assert load_voice_seed(str(tmp_path / "nope" / "absent.wav")) is None


def test_stereo_clip_is_rejected(tmp_path) -> None:
    clip = _wav(tmp_path / "stereo" / "agent_voice.wav", channels=2)
    assert load_voice_seed(str(clip)) is None


def test_eight_bit_clip_is_rejected(tmp_path) -> None:
    clip = _wav(tmp_path / "eightbit" / "agent_voice.wav", width=1, fill=b"\x01")
    assert load_voice_seed(str(clip)) is None


def test_too_short_clip_is_rejected(tmp_path) -> None:
    """Under ~2s there is not enough signal to clone a stable timbre."""
    clip = _wav(tmp_path / "short" / "agent_voice.wav", seconds=0.5)
    assert load_voice_seed(str(clip)) is None


def test_too_long_clip_is_rejected(tmp_path) -> None:
    clip = _wav(tmp_path / "long" / "agent_voice.wav", seconds=25.0)
    assert load_voice_seed(str(clip)) is None


def test_non_wav_file_is_rejected(tmp_path) -> None:
    path = tmp_path / "junk" / "agent_voice.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not audio at all")
    assert load_voice_seed(str(path)) is None
