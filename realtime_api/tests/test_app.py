"""Unit tests for the standalone realtime voice API (no Databricks calls)."""
from __future__ import annotations

import struct

from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from realtime_api.app import create_app
from realtime_api.config import RealtimeSettings
from realtime_api.contracts import AudioChunk, AudioResponse, SessionStart
from realtime_api.pipelines import ServingBundle
from realtime_api.session import VoiceSession


class FakeServices:
    """Deterministic STT/LLM/TTS stand-ins for the injected pipeline."""

    def transcribe(self, audio: bytes, *, language: str | None, sample_rate_hz: int) -> tuple[str, str | None]:
        return "hello world", "en-US"

    def respond(self, transcript: str, *, language: str, context: str | None = None, tool_ctx=None) -> str:
        return f"you said {transcript}"

    def phrase(self, intent: str, *, language: str) -> str:
        return f"[{language}] {intent}"

    def synthesize(
        self,
        text: str,
        *,
        language: str,
        reference_audio_b64: str | None = None,
        voice_id: str | None = None,
    ) -> AudioResponse:
        return AudioResponse(audio=b"\x00\x01", mime_type="audio/wav", sample_rate_hz=24_000)


def _settings() -> RealtimeSettings:
    return RealtimeSettings(
        stt_endpoint="stt",
        llm_endpoint="llm",
        tts_endpoint="tts",
        supported_languages=("en",),
        stt_languages=("en",),
        tts_languages=("en",),
        # These behavioral tests drive turns with synthetic square-wave frames,
        # which are (correctly) not real speech to the Silero gate. Exercise the
        # energy-VAD turn flow here; the semantic endpointer is covered by its
        # own tests below (test_endpointing.py logic + endpointer_for routing).
        endpointing_enabled=False,
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


def test_root_returns_api_descriptor_not_html() -> None:
    with _app() as client:
        response = client.get("/")
        assert response.status_code == 200
        body = response.json()
        assert body["service"] == "realtime-voice-api"
        # Advertised endpoints/WS routes let a caller discover the API from its base.
        assert body["endpoints"]["capabilities"].endswith("/v1/capabilities")
        assert "speech-to-text" in body["websockets"]


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


def test_whole_utterance_transcribed_once_no_partials() -> None:
    with _app() as client, client.websocket_connect("/v1/speech-llm-toolassist-speech") as ws:
        ws.send_json({"type": "session.start", "language": "en-US", "sample_rate_hz": 16000})
        assert ws.receive_json()["type"] == "session.ready"
        ws.send_bytes(_loud_frame())
        assert ws.receive_json()["type"] == "speech.started"
        for _ in range(30):
            ws.send_bytes(_loud_frame())

        ws.send_json({"type": "audio.end"})
        # No interim streaming: the whole utterance is transcribed once as final.
        types: list[str] = []
        for _ in range(8):
            types.append(ws.receive_json()["type"])
            if "transcript.final" in types:
                break
        assert "transcript.partial" not in types
        assert "transcript.final" in types


def test_language_gate_speaks_switch_prompt_and_drops_turn() -> None:
    with _app() as client, client.websocket_connect("/v1/speech-llm-toolassist-speech") as ws:
        # Agent selected Thai; fake STT detects en-US -> the whole turn is gated:
        # no partials, no transcript.final, no LLM reply. Instead the API warns
        # (language.mismatch) and *speaks* a switch-language prompt (response.audio).
        ws.send_json(
            {
                "type": "session.start",
                "language": "en-US",
                "expected_language": "th-TH",
                "sample_rate_hz": 16000,
            }
        )
        assert ws.receive_json()["type"] == "session.ready"
        ws.send_bytes(_loud_frame())
        assert ws.receive_json()["type"] == "speech.started"
        for _ in range(30):
            ws.send_bytes(_loud_frame())

        ws.send_json({"type": "audio.end"})
        types: list[str] = []
        for _ in range(12):
            msg = ws.receive_json()
            types.append(msg["type"])
            if msg["type"] == "response.audio" and msg.get("final"):
                break
        assert "transcript.partial" not in types
        assert "transcript.final" not in types  # off-language transcript is gated
        assert "response.text" not in types  # no agent reply
        assert "language.mismatch" in types  # visual warning for the UI
        assert "response.audio" in types  # spoken switch-language prompt


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


def test_language_mismatch_gates_transcript_on_stt_route() -> None:
    with _app() as client, client.websocket_connect("/v1/speech-to-text") as ws:
        # Agent picked Thai but the fake STT always detects en-US. This route has
        # no TTS, so the gate suppresses the off-language transcript.final and
        # only emits the language.mismatch warning.
        ws.send_json(
            {
                "type": "session.start",
                "language": "en-US",
                "expected_language": "th-TH",
                "sample_rate_hz": 16000,
            }
        )
        assert ws.receive_json()["capability"] == "speech-to-text"
        _drive_turn(ws)
        mismatch = ws.receive_json()
        assert mismatch["type"] == "language.mismatch"
        assert mismatch["expected"] == "th-TH"
        assert mismatch["detected"] == "en-US"


def test_no_language_mismatch_when_selection_matches() -> None:
    with _app() as client, client.websocket_connect("/v1/speech-to-text") as ws:
        ws.send_json(
            {
                "type": "session.start",
                "language": "en-US",
                "expected_language": "en-US",
                "sample_rate_hz": 16000,
            }
        )
        assert ws.receive_json()["capability"] == "speech-to-text"
        _drive_turn(ws)
        assert ws.receive_json()["type"] == "transcript.final"
        ws.send_json({"type": "session.stop"})
        # Drain remaining events until the socket closes; none may be a mismatch.
        try:
            for _ in range(4):
                msg = ws.receive_json()
                assert msg["type"] != "language.mismatch"
                if msg["type"] == "session.closed":
                    break
        except WebSocketDisconnect:
            pass


def test_language_mismatch_canonicalizes_name_vs_tag() -> None:
    """Regression: STT reports a name ('chinese') while selection is a tag
    ('zh-CN'). These are the same language and must NOT trigger a mismatch."""
    from realtime_api.pipelines._shared import language_mismatch

    def _sess(expected: str):
        return VoiceSession(
            SessionStart.from_event(
                {"type": "session.start", "language": "auto", "expected_language": expected}
            )
        )

    # Same language reported in different forms -> no gate.
    assert language_mismatch(_sess("zh-CN"), "chinese") is None
    assert language_mismatch(_sess("zh-CN"), "zh") is None
    assert language_mismatch(_sess("en-US"), "english") is None
    # Genuinely different language -> gate fires with a canonical detected tag.
    assert language_mismatch(_sess("zh-CN"), "english") == {"expected": "zh-CN", "detected": "en-US"}
    # Unmappable detection -> never gate (a false mismatch blocks the whole turn).
    assert language_mismatch(_sess("zh-CN"), "xyz") is None


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
    def respond(self, transcript: str, *, language: str, context: str | None = None, tool_ctx=None) -> str:
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
        # A stray audio.end with nothing buffered is a no-op (persistent-call
        # flow): it must not error or cancel, and a real turn still proceeds.
        ws.send_json({"type": "audio.end"})

        _drive_turn(ws)
        assert ws.receive_json()["type"] == "transcript.final"
        assert ws.receive_json()["type"] == "response.text"

        chunks = [ws.receive_json() for _ in range(3)]
        assert [c["chunk_index"] for c in chunks] == [0, 1, 2]
        assert [c["final"] for c in chunks] == [False, False, True]
        assert all(c["type"] == "response.audio" for c in chunks)


class StreamingServices(FakeServices):
    """TTS that streams PCM chunks (mirrors the deployed predict_stream path)."""

    def synthesize_stream(
        self,
        text: str,
        *,
        language: str,
        reference_audio_b64: str | None = None,
        voice_id: str | None = None,
    ):
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


def test_bare_tool_call_json_is_routed_not_spoken() -> None:
    """Regression: on a non-English turn the model sometimes prints a tool call as
    bare JSON content ({"use_case": ...}) instead of a structured tool_call. It must
    be EXECUTED (state recorded) and must never leak into the spoken/on-screen reply.
    """
    from realtime_api import card_tools
    from realtime_api.services import DatabricksServing
    from realtime_api.tool_registry import ToolContext, run_tool

    thai_reply = "กำลังดึงข้อมูลเชิงลึกจากใบแจ้งยอดให้นะคะ"
    client = FakeDeployClient(
        [
            # The model picked the topic but rendered the tool call as content.
            {"choices": [{"message": {"role": "assistant", "content": '{"use_case": "statement_insights"}'}}]},
            # After the tool runs it speaks a normal Thai confirmation.
            {"choices": [{"message": {"role": "assistant", "content": thai_reply}}]},
        ]
    )
    serving = DatabricksServing(client=client, stt_endpoint="s", llm_endpoint="llm", tts_endpoint="t")
    ctx = ToolContext(customer_id="CH-1", call_id="c1", profile_state={})

    text, invocations = serving.respond_with_tools(
        "ข้อมูลเชิงลึกจากใบแจ้งยอด",
        language="th-TH",
        tool_ctx=ctx,
        system_prompt="{language}",
        tools_override=card_tools._card_tools_spec(),
        tool_runner=lambda n, a, c: run_tool(n, a, c, profile="card"),
    )

    assert text == thai_reply
    assert "{" not in text  # no raw JSON leaked into the spoken reply
    assert [i["name"] for i in invocations] == ["select_use_case"]
    assert ctx.profile_state["use_case"] == "statement_insights"


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

    def respond(self, transcript: str, *, language: str, context: str | None = None, tool_ctx=None) -> str:
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
