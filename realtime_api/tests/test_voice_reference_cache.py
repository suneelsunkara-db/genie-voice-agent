"""Voice-reference caching: upload the clip once, then address it by id.

Re-sending the ~500KB base64 reference clip on every turn was ~1.7s of the ~3.4s
time-to-first-audio (measured: the endpoint materialises the clip in ~25ms, so the
cost was upload, not work). These tests pin both halves of the fix:

* client (``DatabricksServing``): sends the clip on the first cloned turn, then
  only ``voice_id``, and recovers from a replica that lost the cached clip by
  resending it and retrying the SAME turn -- so a caller never hears a turn
  rendered in a different voice, and never hears duplicated audio.
* endpoint (``RealtimeTTSAgent._resolve_reference``): caches per ``voice_id`` with
  LRU eviction, keeps cached files across requests, and reports a miss instead of
  silently synthesizing in the wrong voice.
"""
from __future__ import annotations

import base64
import importlib.util
import io
import os
import pickle
import sys
import types
import wave
from pathlib import Path
from typing import Any

import pytest

from realtime_api.services import DatabricksServing


def _wav_b64(seconds: float = 0.1, sample_rate: int = 48_000) -> str:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x01" * int(sample_rate * seconds))
    return base64.b64encode(buffer.getvalue()).decode("ascii")


# --------------------------------------------------------------------------
# client side
# --------------------------------------------------------------------------


class _ScriptedClient:
    """Serving client that replays scripted SSE responses and records requests."""

    def __init__(self, responses: list[list[dict[str, Any]]]) -> None:
        self._responses = responses
        self.calls: list[dict[str, Any]] = []

    def predict_stream(self, *, endpoint: str, inputs: dict) -> Any:
        self.calls.append(dict(inputs["custom_inputs"]))
        events = self._responses[min(len(self.calls) - 1, len(self._responses) - 1)]
        return iter(events)

    def predict(self, *, endpoint: str, inputs: dict) -> dict:
        self.calls.append(dict(inputs["custom_inputs"]))
        events = self._responses[min(len(self.calls) - 1, len(self._responses) - 1)]
        return events[0]


def _audio_stream(chunks: int = 2) -> list[dict[str, Any]]:
    pcm = base64.b64encode(b"\x00\x01\x02\x03").decode("ascii")
    events: list[dict[str, Any]] = [
        {"custom_outputs": {"audio_pcm16_b64": pcm, "sample_rate_hz": 48_000, "final": False}}
        for _ in range(chunks)
    ]
    events.append({"custom_outputs": {"final": True, "ttfb_ms": 70.0, "gen_ms": 900.0}})
    return events


def _cache_miss_stream() -> list[dict[str, Any]]:
    return [{"custom_outputs": {"final": True, "chunks": 0, "voice_cache_miss": True}}]


def _serving(client: Any) -> DatabricksServing:
    return DatabricksServing(
        client=client, stt_endpoint="stt", llm_endpoint="llm", tts_endpoint="tts"
    )


def test_clip_uploaded_once_then_addressed_by_id() -> None:
    client = _ScriptedClient([_audio_stream()])
    serving = _serving(client)
    clip = _wav_b64()

    for _ in range(3):
        chunks = list(
            serving.synthesize_stream("hello", language="en-US", reference_audio_b64=clip, voice_id="v1")
        )
        assert chunks, "each turn must still produce audio"

    assert len(client.calls) == 3
    assert client.calls[0].get("reference_audio_b64") == clip
    assert client.calls[0].get("voice_id") == "v1"
    # The whole point: later turns carry the id but not the payload.
    for call in client.calls[1:]:
        assert call.get("voice_id") == "v1"
        assert "reference_audio_b64" not in call


def test_cache_miss_resends_clip_and_retries_same_turn() -> None:
    # turn 1 succeeds (uploads clip); turn 2 misses, then succeeds on the retry.
    client = _ScriptedClient([_audio_stream(), _cache_miss_stream(), _audio_stream(3)])
    serving = _serving(client)
    clip = _wav_b64()

    list(serving.synthesize_stream("one", language="en-US", reference_audio_b64=clip, voice_id="v1"))
    chunks = list(
        serving.synthesize_stream("two", language="en-US", reference_audio_b64=clip, voice_id="v1")
    )

    assert len(client.calls) == 3
    assert "reference_audio_b64" not in client.calls[1], "id-only attempt comes first"
    assert client.calls[2].get("reference_audio_b64") == clip, "retry resends the clip"
    # Retry happens before any chunk is emitted, so audio is neither lost nor doubled.
    assert len(chunks) == 3


def test_cache_miss_retries_at_most_once() -> None:
    client = _ScriptedClient([_audio_stream(), _cache_miss_stream(), _cache_miss_stream()])
    serving = _serving(client)
    clip = _wav_b64()

    list(serving.synthesize_stream("one", language="en-US", reference_audio_b64=clip, voice_id="v1"))
    chunks = list(
        serving.synthesize_stream("two", language="en-US", reference_audio_b64=clip, voice_id="v1")
    )

    assert chunks == []
    assert len(client.calls) == 3, "one retry only; never an unbounded resend loop"


def test_cache_miss_without_a_clip_does_not_retry() -> None:
    client = _ScriptedClient([_cache_miss_stream()])
    serving = _serving(client)

    chunks = list(serving.synthesize_stream("hello", language="en-US", voice_id="v1"))

    assert chunks == []
    assert len(client.calls) == 1, "nothing to resend, so no retry"


def test_no_voice_id_keeps_sending_the_clip() -> None:
    """Callers without a voice_id keep the previous behaviour."""
    client = _ScriptedClient([_audio_stream()])
    serving = _serving(client)
    clip = _wav_b64()

    for _ in range(2):
        list(serving.synthesize_stream("hello", language="en-US", reference_audio_b64=clip))

    assert all(call.get("reference_audio_b64") == clip for call in client.calls)
    assert all("voice_id" not in call for call in client.calls)


def test_non_streaming_synthesize_also_caches_by_id() -> None:
    audio_b64 = _wav_b64()
    ok = [{"custom_outputs": {"audio_b64": audio_b64, "mime_type": "audio/wav", "sample_rate_hz": 48_000}}]
    client = _ScriptedClient([ok])
    serving = _serving(client)

    for _ in range(2):
        serving.synthesize("hello", language="en-US", reference_audio_b64=audio_b64, voice_id="v1")

    assert client.calls[0].get("reference_audio_b64") == audio_b64
    assert "reference_audio_b64" not in client.calls[1]


# --------------------------------------------------------------------------
# endpoint side
# --------------------------------------------------------------------------


def _load_tts_agent_module():
    """Load the serving-side TTS agent, stubbing mlflow if it is absent.

    The wrapper subclasses ``mlflow.pyfunc.ResponsesAgent``, which only exists in
    the registration/serving environments, not the app venv. The reference-cache
    logic under test never touches mlflow, so a minimal stub suffices to import it.
    """
    if "mlflow" not in sys.modules:
        mlflow = types.ModuleType("mlflow")
        pyfunc = types.ModuleType("mlflow.pyfunc")
        pyfunc.ResponsesAgent = type("ResponsesAgent", (), {})
        responses = types.ModuleType("mlflow.types.responses")
        for name in ("ResponsesAgentRequest", "ResponsesAgentResponse", "ResponsesAgentStreamEvent"):
            setattr(responses, name, type(name, (), {}))
        mlflow_types = types.ModuleType("mlflow.types")
        mlflow_types.responses = responses
        mlflow.pyfunc = pyfunc
        mlflow.types = mlflow_types
        sys.modules.update(
            {
                "mlflow": mlflow,
                "mlflow.pyfunc": pyfunc,
                "mlflow.types": mlflow_types,
                "mlflow.types.responses": responses,
            }
        )
    path = Path(__file__).resolve().parents[2] / "scripts" / "ml_asr" / "realtime_tts_agent.py"
    spec = importlib.util.spec_from_file_location("realtime_tts_agent_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tts_agent_module():
    return _load_tts_agent_module()


@pytest.fixture
def agent(tts_agent_module):
    instance = tts_agent_module.RealtimeTTSAgent({})
    yield instance
    for path in list(instance._voice_cache.values()):
        tts_agent_module._unlink(path)


def test_keyed_reference_is_cached_and_reused(agent) -> None:
    clip = _wav_b64()

    path, owned, miss = agent._resolve_reference({"voice_id": "v1", "reference_audio_b64": clip})
    assert not miss and not owned, "cached clips are owned by the cache, not the request"
    assert path and os.path.exists(path)

    again, owned_again, miss_again = agent._resolve_reference({"voice_id": "v1"})
    assert (again, owned_again, miss_again) == (path, False, False)
    assert os.path.exists(path), "the cached file must survive across requests"


def test_unknown_voice_id_reports_miss(agent) -> None:
    path, owned, miss = agent._resolve_reference({"voice_id": "never-seen"})
    assert miss is True
    assert path is None and owned is False


def test_lost_cache_file_reports_miss(agent, tts_agent_module) -> None:
    clip = _wav_b64()
    path, _, _ = agent._resolve_reference({"voice_id": "v1", "reference_audio_b64": clip})
    tts_agent_module._unlink(path)

    resolved, _, miss = agent._resolve_reference({"voice_id": "v1"})
    assert miss is True and resolved is None
    assert "v1" not in agent._voice_cache, "stale entry is dropped so the resend repopulates"


def test_unkeyed_reference_stays_per_request(agent) -> None:
    path, owned, miss = agent._resolve_reference({"reference_audio_b64": _wav_b64()})
    assert owned is True and miss is False
    assert path and os.path.exists(path)
    assert not agent._voice_cache, "an unkeyed clip must not enter the cache"
    os.unlink(path)


def test_lru_eviction_removes_oldest_clip(agent, tts_agent_module, monkeypatch) -> None:
    monkeypatch.setattr(tts_agent_module, "_VOICE_CACHE_MAX", 2)
    clip = _wav_b64()

    first, _, _ = agent._resolve_reference({"voice_id": "v1", "reference_audio_b64": clip})
    agent._resolve_reference({"voice_id": "v2", "reference_audio_b64": clip})
    agent._resolve_reference({"voice_id": "v3", "reference_audio_b64": clip})

    assert list(agent._voice_cache) == ["v2", "v3"]
    assert not os.path.exists(first), "evicted clip is deleted, not leaked on disk"
    assert agent._resolve_reference({"voice_id": "v1"})[2] is True


def test_agent_state_stays_picklable(agent, tts_agent_module) -> None:
    """Registration cloudpickles the agent, and a lock in ``__dict__`` breaks it.

    This is not hypothetical: holding the cache lock as a plain attribute failed
    ``mlflow.pyfunc.log_model`` with MODEL_SERIALIZATION_FAILED and blocked deploy.
    """
    path, _, _ = agent._resolve_reference({"voice_id": "v1", "reference_audio_b64": _wav_b64()})
    try:
        state = agent.__getstate__()
        assert "_voice_cache_lock" not in state
        assert "_voice_cache" not in state
        agent.__setstate__(pickle.loads(pickle.dumps(state)))
        # Rebuilt empty on the serving side, and the lock is usable again.
        assert agent._voice_cache == {}
        assert agent._resolve_reference({"voice_id": "v1"})[2] is True
    finally:
        tts_agent_module._unlink(path)


def test_recaching_same_voice_replaces_old_file(agent) -> None:
    clip = _wav_b64()
    first, _, _ = agent._resolve_reference({"voice_id": "v1", "reference_audio_b64": clip})
    second, _, _ = agent._resolve_reference({"voice_id": "v1", "reference_audio_b64": clip})

    assert second != first
    assert not os.path.exists(first), "superseded clip is unlinked"
    assert os.path.exists(second)
