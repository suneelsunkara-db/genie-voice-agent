"""Unit tests for the standalone realtime voice API (no Databricks calls)."""
from __future__ import annotations

import struct

from fastapi.testclient import TestClient

from realtime_api.app import create_app
from realtime_api.config import RealtimeSettings
from realtime_api.contracts import AudioChunk, AudioResponse, SessionStart
from realtime_api.pipelines import ServingBundle
from realtime_api.session import VoiceSession


class FakeServices:
    """Deterministic STT/LLM/TTS stand-ins for the injected pipeline."""

    def transcribe(self, audio: bytes, *, language: str | None, sample_rate_hz: int) -> tuple[str, str | None]:
        return "hello world", "en-US"

    def respond(self, transcript: str, *, language: str, context: str | None = None) -> str:
        return f"you said {transcript}"

    def synthesize(self, text: str, *, language: str) -> AudioResponse:
        return AudioResponse(audio=b"\x00\x01", mime_type="audio/wav", sample_rate_hz=24_000)


def _settings() -> RealtimeSettings:
    return RealtimeSettings(
        stt_endpoint="stt",
        llm_endpoint="llm",
        tts_endpoint="tts",
        supported_languages=("en",),
        stt_languages=("en",),
        tts_languages=("en",),
    )


def _app() -> TestClient:
    fake = FakeServices()
    app = create_app(
        settings=_settings(),
        bundle_factory=lambda _s: ServingBundle(stt=fake, llm=fake, tts=fake),
    )
    return TestClient(app)


def _loud_frame(samples: int = 320, amplitude: int = 6000) -> bytes:
    return struct.pack("<" + "h" * samples, *([amplitude, -amplitude] * (samples // 2)))


def _drive_turn(ws) -> None:
    ws.send_bytes(_loud_frame())
    assert ws.receive_json()["type"] == "speech.started"
    ws.send_json({"type": "audio.end"})
    assert ws.receive_json()["type"] == "turn.started"


def test_healthz() -> None:
    with _app() as client:
        response = client.get("/healthz")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


def test_capabilities_endpoint() -> None:
    with _app() as client:
        response = client.get("/v1/capabilities")
        assert response.status_code == 200
        body = response.json()
        assert body["speech-to-text"]["path"] == "/v1/speech-to-text"
        assert body["speech-llm-toolassist-speech"]["path"] == "/v1/speech-llm-toolassist-speech"
        assert body["text-to-speech"]["path"] == "/v1/text-to-speech"


def test_benchmarks_endpoint() -> None:
    with _app() as client:
        response = client.get("/v1/benchmarks")
        assert response.status_code == 200
        body = response.json()
        assert "available" in body


def test_speech_llm_toolassist_speech_processes_a_finalized_turn() -> None:
    with _app() as client, client.websocket_connect("/v1/speech-llm-toolassist-speech") as ws:
        ws.send_json({"type": "session.start", "language": "en-US", "sample_rate_hz": 16000})
        ready = ws.receive_json()
        assert ready["type"] == "session.ready"
        assert ready["capability"] == "speech-llm-toolassist-speech"

        _drive_turn(ws)

        transcript = ws.receive_json()
        assert transcript["type"] == "transcript.final"
        assert transcript["text"] == "hello world"

        response_text = ws.receive_json()
        assert response_text["type"] == "response.text"
        assert response_text["text"] == "you said hello world"

        audio = ws.receive_json()
        assert audio["type"] == "response.audio"
        assert audio["audio_b64"]
        assert audio["chunk_index"] == 0
        assert audio["final"] is True


def test_legacy_voice_alias_routes_to_speech_llm_toolassist_speech() -> None:
    with _app() as client, client.websocket_connect("/v1/realtime/voice") as ws:
        ws.send_json({"type": "session.start", "language": "en-US", "sample_rate_hz": 16000})
        ready = ws.receive_json()
        assert ready["type"] == "session.ready"
        assert ready["capability"] == "speech-llm-toolassist-speech"
        _drive_turn(ws)
        assert ws.receive_json()["type"] == "transcript.final"
        assert ws.receive_json()["type"] == "response.text"
        assert ws.receive_json()["type"] == "response.audio"


def test_speech_to_text_emits_transcript_only() -> None:
    with _app() as client, client.websocket_connect("/v1/speech-to-text") as ws:
        ws.send_json({"type": "session.start", "language": "en-US", "sample_rate_hz": 16000})
        ready = ws.receive_json()
        assert ready["capability"] == "speech-to-text"
        _drive_turn(ws)
        transcript = ws.receive_json()
        assert transcript["type"] == "transcript.final"
        assert transcript["text"] == "hello world"


def test_text_to_speech_synthesize() -> None:
    with _app() as client, client.websocket_connect("/v1/text-to-speech") as ws:
        ws.send_json({"type": "session.start", "language": "en-US", "sample_rate_hz": 16000})
        ready = ws.receive_json()
        assert ready["capability"] == "text-to-speech"
        ws.send_json({"type": "synthesize", "text": "Hello there", "language": "en-US"})
        assert ws.receive_json()["type"] == "turn.started"
        audio = ws.receive_json()
        assert audio["type"] == "response.audio"
        assert audio["final"] is True


class MultiSentenceServices(FakeServices):
    def respond(self, transcript: str, *, language: str, context: str | None = None) -> str:
        return "First sentence. Second sentence! Third one?"


def test_websocket_streams_one_audio_chunk_per_sentence() -> None:
    fake = MultiSentenceServices()
    app = create_app(
        settings=_settings(),
        bundle_factory=lambda _s: ServingBundle(stt=fake, llm=fake, tts=fake),
    )
    with TestClient(app) as client, client.websocket_connect("/v1/speech-llm-toolassist-speech") as ws:
        ws.send_json({"type": "session.start", "language": "en-US", "sample_rate_hz": 16000})
        assert ws.receive_json()["type"] == "session.ready"
        ws.send_json({"type": "audio.end"})
        assert ws.receive_json()["code"] == "empty_audio"

        _drive_turn(ws)
        assert ws.receive_json()["type"] == "transcript.final"
        assert ws.receive_json()["type"] == "response.text"

        chunks = [ws.receive_json() for _ in range(3)]
        assert [c["chunk_index"] for c in chunks] == [0, 1, 2]
        assert [c["final"] for c in chunks] == [False, False, True]
        assert all(c["type"] == "response.audio" for c in chunks)


class StreamingServices(FakeServices):
    """TTS that streams PCM chunks (mirrors the deployed predict_stream path)."""

    def synthesize_stream(self, text: str, *, language: str):
        for _ in range(3):
            yield AudioChunk(pcm=b"\x00\x01\x02\x03", sample_rate_hz=48_000)


def test_websocket_streams_pcm_audio_chunks_when_supported() -> None:
    fake = StreamingServices()
    app = create_app(
        settings=_settings(),
        bundle_factory=lambda _s: ServingBundle(stt=fake, llm=fake, tts=fake),
    )
    with TestClient(app) as client, client.websocket_connect("/v1/speech-llm-toolassist-speech") as ws:
        ws.send_json({"type": "session.start", "language": "en-US", "sample_rate_hz": 16000})
        assert ws.receive_json()["type"] == "session.ready"
        _drive_turn(ws)
        assert ws.receive_json()["type"] == "transcript.final"
        assert ws.receive_json()["type"] == "response.text"

        chunks = [ws.receive_json() for _ in range(3)]
        assert all(c["type"] == "response.audio" for c in chunks)
        assert all(c["mime_type"] == "audio/pcm" and c["encoding"] == "pcm_s16le" for c in chunks)
        assert [c["chunk_index"] for c in chunks] == [0, 1, 2]
        assert [c["final"] for c in chunks] == [False, False, True]
        assert chunks[0].get("text")
        assert "text" not in chunks[1]


class FakeDeployClient:
    """Scripted mlflow-deployments stand-in for LLM tool-calling tests."""

    def __init__(self, responses: list[dict]) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    def predict(self, *, endpoint: str, inputs: dict) -> dict:
        self.calls.append(inputs)
        return self._responses.pop(0)


def test_respond_runs_tool_calling_loop_with_temperature() -> None:
    from realtime_api.services import DatabricksServing

    tool_call = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_current_time",
                                "arguments": '{"timezone": "Asia/Bangkok"}',
                            },
                        }
                    ],
                }
            }
        ]
    }
    final = {"choices": [{"message": {"role": "assistant", "content": "It's just past nine in Bangkok."}}]}
    client = FakeDeployClient([tool_call, final])
    serving = DatabricksServing(client=client, stt_endpoint="s", llm_endpoint="llm", tts_endpoint="t")

    text = serving.respond("what time is it in Bangkok?", language="en-US")

    assert text == "It's just past nine in Bangkok."
    assert client.calls[0]["temperature"] == 0.4
    assert client.calls[0]["tools"]
    assert any(m.get("role") == "tool" for m in client.calls[1]["messages"])


def test_respond_without_tools_returns_direct_text() -> None:
    from realtime_api.services import DatabricksServing

    client = FakeDeployClient([{"choices": [{"message": {"content": "Hello there."}}]}])
    serving = DatabricksServing(
        client=client, stt_endpoint="s", llm_endpoint="llm", tts_endpoint="t", llm_tools_enabled=False
    )

    assert serving.respond("hi", language="en-US") == "Hello there."
    assert "tools" not in client.calls[0]
    assert client.calls[0]["temperature"] == 0.4


def test_websocket_rejects_audio_before_session_start() -> None:
    with _app() as client, client.websocket_connect("/v1/speech-llm-toolassist-speech") as ws:
        ws.send_bytes(_loud_frame())
        error = ws.receive_json()
        assert error["type"] == "error"
        assert error["code"] == "session_not_started"


def test_session_accepts_non_validation_language_tags() -> None:
    session = VoiceSession(SessionStart.from_event({"language": "sw-KE", "sample_rate_hz": 16000}))
    assert session.config.language == "sw-KE"


def test_voice_session_finalizes_after_vad_silence() -> None:
    session = VoiceSession(SessionStart.from_event({"language": "en-US", "sample_rate_hz": 16000}))
    session.add_audio(_loud_frame())
    silence = struct.pack("<" + "h" * 320, *([0] * 320))
    for _ in range(60):
        session.add_audio(silence)
    assert session.should_finalize(silence_ms=700, max_turn_seconds=20)


def test_session_start_parses_turn_overrides_and_context() -> None:
    start = SessionStart.from_event(
        {
            "language": "en-US",
            "sample_rate_hz": 16000,
            "max_turn_seconds": 150,
            "vad_silence_ms": 600000,
            "context": "  Question: X  ",
        }
    )
    assert start.max_turn_seconds == 150
    assert start.vad_silence_ms == 600000
    assert start.context == "Question: X"


def test_session_start_defaults_have_no_overrides() -> None:
    start = SessionStart.from_event({"language": "en-US", "sample_rate_hz": 16000})
    assert start.max_turn_seconds is None
    assert start.vad_silence_ms is None
    assert start.context is None


def test_session_start_rejects_nonpositive_turn_override() -> None:
    import pytest

    with pytest.raises(ValueError):
        SessionStart.from_event(
            {"language": "en-US", "sample_rate_hz": 16000, "max_turn_seconds": 0}
        )


class ContextEchoServices(FakeServices):
    """Echoes the grounding context so the assist route can be asserted on."""

    def respond(self, transcript: str, *, language: str, context: str | None = None) -> str:
        return f"ctx:{context}"


def test_session_context_reaches_assist_llm() -> None:
    fake = ContextEchoServices()
    app = create_app(
        settings=_settings(),
        bundle_factory=lambda _s: ServingBundle(stt=fake, llm=fake, tts=fake),
    )
    with TestClient(app) as client, client.websocket_connect("/v1/speech-llm-toolassist-speech") as ws:
        ws.send_json(
            {
                "type": "session.start",
                "language": "en-US",
                "sample_rate_hz": 16000,
                "context": "Question: capital of France?\n1. Paris\n2. Rome",
            }
        )
        assert ws.receive_json()["type"] == "session.ready"
        _drive_turn(ws)
        assert ws.receive_json()["type"] == "transcript.final"
        response_text = ws.receive_json()
        assert response_text["type"] == "response.text"
        assert "Question: capital of France?" in response_text["text"]


def test_respond_appends_context_to_user_message() -> None:
    from realtime_api.services import DatabricksServing

    client = FakeDeployClient([{"choices": [{"message": {"content": "2"}}]}])
    serving = DatabricksServing(
        client=client, stt_endpoint="s", llm_endpoint="llm", tts_endpoint="t", llm_tools_enabled=False
    )

    out = serving.respond("the spoken passage", language="en-US", context="Question: Q?\n1. a\n2. b")

    assert out == "2"
    user_content = client.calls[0]["messages"][-1]["content"]
    assert "the spoken passage" in user_content
    assert "Question: Q?" in user_content
