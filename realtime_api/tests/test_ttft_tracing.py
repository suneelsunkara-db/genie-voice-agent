"""Tests for time-to-first-audio (TTFT) capture on the voice turn trace.

TTFT is the latency a caller actually feels: how long the line stayed silent
before the agent's voice came back. It is captured inside ``stream_tts`` rather
than at each call site, so every path that speaks — the answer, the latency
filler, an intent-router confirmation, a language-switch prompt — reports it the
same way, and a new use case gets it for free.
"""
from __future__ import annotations

import asyncio

from realtime_api.contracts import AudioChunk, AudioResponse
from realtime_api.pipelines import ServingBundle
from realtime_api.pipelines._shared import stream_tts
from realtime_api.session import SessionStart, VoiceSession
from realtime_api.tracing import TurnTrace


class _StreamingTTS:
    """Streams three chunks; the last carries the endpoint's own timings."""

    def __init__(self, *, delay_s: float = 0.0):
        self._delay_s = delay_s
        self.calls = 0

    def synthesize_stream(
        self,
        text: str,
        *,
        language: str,
        reference_audio_b64: str | None = None,
        voice_id: str | None = None,
    ):
        self.calls += 1
        import time

        if self._delay_s:
            time.sleep(self._delay_s)
        yield AudioChunk(pcm=b"\x00\x01" * 100, sample_rate_hz=48_000)
        yield AudioChunk(pcm=b"\x00\x01" * 100, sample_rate_hz=48_000)
        yield AudioChunk(
            pcm=b"\x00\x01" * 100,
            sample_rate_hz=48_000,
            server_ttfb_ms=310.0,
            server_gen_ms=1400.0,
        )


class _BatchTTS:
    """Non-streaming endpoint: one full WAV per sentence."""

    def synthesize(self, text, *, language, reference_audio_b64=None, voice_id=None):
        return AudioResponse(audio=b"RIFF0000WAVE", mime_type="audio/wav", sample_rate_hz=48_000)


def _session() -> VoiceSession:
    return VoiceSession(SessionStart.from_event({"language": "en-US", "sample_rate_hz": 16000}))


def _trace() -> TurnTrace:
    return TurnTrace(session_id="s1", turn_id=1, capability="speech-llm-toolassist-speech")


def _drain(
    bundle, session, trace, text="Your bill went up because of roaming.", *, primary: bool = True
) -> list[dict]:
    # The turn id must match the session's current turn or stream_tts treats the
    # stream as superseded and stops.
    turn = session.turn_id

    async def run() -> list[dict]:
        return [
            e
            async for e in stream_tts(
                bundle, session, turn, text, "en-US", trace=trace, primary=primary
            )
        ]

    return asyncio.run(run())


def test_streaming_path_records_ttft_and_server_timings():
    tts = _StreamingTTS()
    trace = _trace()
    events = _drain(ServingBundle(stt=None, llm=None, tts=tts), _session(), trace)

    assert [e["type"] for e in events] == ["response.audio"] * 3
    assert trace.ttft_ms is not None and trace.ttft_ms > 0
    assert trace.tts_first_ms is not None
    assert trace.server_ttfb_ms == 310.0
    assert trace.server_gen_ms == 1400.0
    # ttft covers the whole turn, so it can never be under the TTS-only figure.
    assert trace.ttft_ms >= trace.tts_first_ms
    # With no filler, the silence ended exactly when the answer arrived.
    assert trace.answer_ttft_ms == trace.ttft_ms


def test_a_filler_ends_the_silence_but_does_not_own_the_answer_latency():
    """A filler speaks first, then the answer.

    The filler genuinely ends the dead air, so it owns ttft. But it is a canned
    phrase, not the reply, so answer_ttft_ms must move on to the real answer —
    otherwise a turn that talked over its own 3s delay would score as fast as one
    that answered immediately, which is exactly how slow turns hide.
    """
    session, trace = _session(), _trace()
    bundle = ServingBundle(stt=None, llm=None, tts=_StreamingTTS())

    _drain(bundle, session, trace, "One moment.", primary=False)
    filler_ttft = trace.ttft_ms
    assert filler_ttft is not None
    assert trace.answer_ttft_ms is None  # nothing has actually been answered yet
    assert trace.tts_first_ms is None  # the filler's synthesis is not the answer's

    _drain(bundle, session, trace, "Here is what I found.")
    assert trace.ttft_ms == filler_ttft
    assert trace.answer_ttft_ms is not None
    assert trace.answer_ttft_ms > filler_ttft
    assert trace.tts_first_ms is not None


def test_ttft_reflects_a_slow_endpoint():
    fast, slow = _trace(), _trace()
    _drain(ServingBundle(stt=None, llm=None, tts=_StreamingTTS()), _session(), fast)
    _drain(ServingBundle(stt=None, llm=None, tts=_StreamingTTS(delay_s=0.25)), _session(), slow)

    assert slow.ttft_ms > fast.ttft_ms
    assert slow.ttft_ms >= 250.0


def test_non_streaming_path_also_records_ttft():
    trace = _trace()
    _drain(ServingBundle(stt=None, llm=None, tts=_BatchTTS()), _session(), trace, "One sentence.")
    assert trace.ttft_ms is not None


def test_a_turn_that_never_speaks_has_no_ttft():
    """Dropped turns and Agent-Mode deep dives must stay NULL, not 0.

    A zero would drag every latency percentile down and hide real regressions.
    """
    trace = _trace()
    assert trace.ttft_ms is None
    assert trace.to_dict()["ttft_ms"] is None
    assert trace.to_dict()["answer_ttft_ms"] is None


def test_a_turn_that_only_played_a_filler_never_answered():
    """If the answer never arrived, answer_ttft stays NULL even though audio played."""
    trace = _trace()
    _drain(ServingBundle(stt=None, llm=None, tts=_StreamingTTS()), _session(), trace, primary=False)
    assert trace.ttft_ms is not None
    assert trace.to_dict()["answer_ttft_ms"] is None


def test_ttft_is_serialized_for_persistence():
    trace = _trace()
    _drain(ServingBundle(stt=None, llm=None, tts=_StreamingTTS()), _session(), trace)
    payload = trace.to_dict()

    assert payload["ttft_ms"] == round(trace.ttft_ms, 2)
    assert payload["answer_ttft_ms"] == round(trace.answer_ttft_ms, 2)
    assert payload["server_ttfb_ms"] == 310.0
    assert payload["server_gen_ms"] == 1400.0
    # total_ms is the whole turn and must not be confused for the latency metric.
    assert payload["total_ms"] >= payload["ttft_ms"]


def test_tracing_is_optional():
    """stream_tts must stay usable without a trace (e.g. the text-to-speech route)."""
    session = _session()

    async def run():
        return [
            e
            async for e in stream_tts(
                ServingBundle(stt=None, llm=None, tts=_StreamingTTS()),
                session, session.turn_id, "Hello.", "en-US",
            )
        ]

    assert len(asyncio.run(run())) == 3
