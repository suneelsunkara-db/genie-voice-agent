"""speech_epoch: same-turn inject must not overlap earlier TTS."""
from __future__ import annotations

import asyncio
import threading

from realtime_api.contracts import AudioChunk
from realtime_api.pipelines import ServingBundle
from realtime_api.pipelines._shared import stream_tts
from realtime_api.session import SessionStart, VoiceSession


class _GatedTTS:
    """Two immediate chunks, then wait on a threading.Event before the third."""

    def __init__(self) -> None:
        self.gate = threading.Event()

    def synthesize_stream(self, text, *, language, reference_audio_b64=None, voice_id=None):
        yield AudioChunk(pcm=b"\x00\x01" * 100, sample_rate_hz=48_000)
        yield AudioChunk(pcm=b"\x00\x01" * 100, sample_rate_hz=48_000)
        self.gate.wait(timeout=5)
        yield AudioChunk(pcm=b"\x00\x01" * 100, sample_rate_hz=48_000)


def _session() -> VoiceSession:
    return VoiceSession(SessionStart.from_event({"language": "en-US", "sample_rate_hz": 16000}))


def test_speech_epoch_bump_aborts_in_flight_stream_tts() -> None:
    session = _session()
    tts = _GatedTTS()
    events: list[dict] = []
    first_chunk = threading.Event()

    async def consume() -> None:
        async for event in stream_tts(
            ServingBundle(stt=None, llm=None, tts=tts),
            session,
            session.turn_id,
            "hello",
            "en-US",
        ):
            events.append(event)
            first_chunk.set()

    async def go() -> None:
        task = asyncio.create_task(consume())
        assert await asyncio.to_thread(first_chunk.wait, 5), "timed out waiting for first audio"
        assert events[0]["speech_epoch"] == 0
        session.bump_speech_epoch()
        tts.gate.set()
        await task

    asyncio.run(go())
    assert all(e["speech_epoch"] == 0 for e in events)
    # Lookahead yields chunk 1 when chunk 2 arrives; chunk 3 after the bump is dropped.
    assert len(events) == 1


def test_new_stream_after_bump_uses_new_epoch() -> None:
    session = _session()

    class _OneShot:
        def synthesize_stream(self, text, *, language, reference_audio_b64=None, voice_id=None):
            yield AudioChunk(pcm=b"\x00\x01" * 50, sample_rate_hz=48_000)

    async def drain() -> list[dict]:
        out: list[dict] = []
        async for event in stream_tts(
            ServingBundle(stt=None, llm=None, tts=_OneShot()),
            session,
            session.turn_id,
            "hi",
            "en-US",
        ):
            out.append(event)
        return out

    first = asyncio.run(drain())
    assert first and first[0]["speech_epoch"] == 0
    session.bump_speech_epoch()
    second = asyncio.run(drain())
    assert second and second[0]["speech_epoch"] == 1


def test_barge_in_bumps_speech_epoch() -> None:
    session = _session()
    assert session.speech_epoch == 0
    session.barge_in()
    assert session.speech_epoch == 1
