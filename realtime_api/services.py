"""Stateless inference adapters for Databricks Model Serving.

The STT/TTS/LLM endpoints are Databricks Agent Framework endpoints
(``ResponsesAgent``, ``task = agent/v1/responses``). We therefore query them with
the OpenAI Responses shape (``input`` + ``custom_inputs``) via the MLflow
deployments client and read structured payloads back from ``custom_outputs``.
"""
from __future__ import annotations

import base64
import datetime
import json
from dataclasses import dataclass
from typing import Any, Iterator, Protocol

from .contracts import AudioChunk, AudioResponse

# Generic, domain-agnostic tool set for the voice assistant. Kept business-free
# on purpose (new API/UI): a single get_current_time tool is enough to exercise
# the full tool-calling loop end-to-end without inventing business context.
_TOOLS_SPEC: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current date and time, optionally for a specific IANA timezone.",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "IANA timezone name, e.g. 'Asia/Bangkok'. Defaults to UTC.",
                    }
                },
            },
        },
    }
]


def _run_tool(name: str, arguments: dict[str, Any]) -> str:
    if name == "get_current_time":
        tz_name = str(arguments.get("timezone") or "UTC")
        try:
            from zoneinfo import ZoneInfo

            now = datetime.datetime.now(ZoneInfo(tz_name))
        except Exception:  # noqa: BLE001
            tz_name = "UTC"
            now = datetime.datetime.now(datetime.timezone.utc)
        return json.dumps(
            {
                "timezone": tz_name,
                "iso": now.isoformat(timespec="seconds"),
                "spoken": now.strftime("%A, %B %d, %Y at %I:%M %p"),
            }
        )
    return json.dumps({"error": f"unknown tool: {name}"})


class SpeechToText(Protocol):
    def transcribe(
        self, audio: bytes, *, language: str | None, sample_rate_hz: int
    ) -> tuple[str, str | None]: ...


class LanguageModel(Protocol):
    def respond(self, transcript: str, *, language: str, context: str | None = None) -> str: ...


class TextToSpeech(Protocol):
    def synthesize(self, text: str, *, language: str) -> AudioResponse: ...


class StreamingTextToSpeech(TextToSpeech, Protocol):
    def synthesize_stream(self, text: str, *, language: str) -> Iterator[AudioChunk]: ...


class _SdkDeployClient:
    """mlflow-free serving client backed by the Databricks SDK.

    Exposes ``predict`` (WorkspaceClient) and ``predict_stream`` (SSE via
    requests), matching the tiny surface ``DatabricksServing`` needs. Lets the
    API run against live endpoints without installing mlflow, using the same
    Databricks auth profile as the CLI.
    """

    def __init__(self, profile: str | None = None) -> None:
        from databricks.sdk import WorkspaceClient

        self._w = WorkspaceClient(profile=profile or None)
        self._host = self._w.config.host.rstrip("/")
        self._headers = {**dict(self._w.config.authenticate() or {}), "Content-Type": "application/json"}

    def predict(self, *, endpoint: str, inputs: dict) -> dict:
        return self._w.api_client.do("POST", f"/serving-endpoints/{endpoint}/invocations", body=inputs)

    def predict_stream(self, *, endpoint: str, inputs: dict):
        import requests

        body = {**inputs, "stream": True}
        url = f"{self._host}/serving-endpoints/{endpoint}/invocations"
        with requests.post(url, headers=self._headers, json=body, stream=True, timeout=180) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=True):
                if line and line.startswith("data:"):
                    payload = line[len("data:"):].strip()
                    if payload and payload != "[DONE]":
                        try:
                            yield json.loads(payload)
                        except json.JSONDecodeError:
                            continue


@dataclass(frozen=True)
class DatabricksServing:
    """Thin, injectable Model Serving adapter; no legacy API dependencies."""

    client: Any
    stt_endpoint: str
    llm_endpoint: str
    tts_endpoint: str
    llm_temperature: float = 0.4
    llm_max_tokens: int = 512
    llm_tools_enabled: bool = True
    llm_max_tool_iterations: int = 3
    tts_inference_timesteps: int = 8
    tts_cfg_value: float = 2.0

    @classmethod
    def from_workspace(
        cls,
        *,
        stt_endpoint: str,
        llm_endpoint: str,
        tts_endpoint: str,
        llm_temperature: float = 0.4,
        llm_max_tokens: int = 512,
        llm_tools_enabled: bool = True,
        llm_max_tool_iterations: int = 3,
        tts_inference_timesteps: int = 8,
        tts_cfg_value: float = 2.0,
    ) -> "DatabricksServing":
        from mlflow.deployments import get_deploy_client

        return cls(
            client=get_deploy_client("databricks"),
            stt_endpoint=stt_endpoint,
            llm_endpoint=llm_endpoint,
            tts_endpoint=tts_endpoint,
            llm_temperature=llm_temperature,
            llm_max_tokens=llm_max_tokens,
            llm_tools_enabled=llm_tools_enabled,
            llm_max_tool_iterations=llm_max_tool_iterations,
            tts_inference_timesteps=tts_inference_timesteps,
            tts_cfg_value=tts_cfg_value,
        )

    @classmethod
    def from_sdk(
        cls,
        *,
        stt_endpoint: str,
        llm_endpoint: str,
        tts_endpoint: str,
        profile: str | None = None,
        llm_temperature: float = 0.4,
        llm_max_tokens: int = 512,
        llm_tools_enabled: bool = True,
        llm_max_tool_iterations: int = 3,
        tts_inference_timesteps: int = 6,
        tts_cfg_value: float = 2.0,
    ) -> "DatabricksServing":
        """Build against live endpoints using the Databricks SDK (no mlflow)."""
        return cls(
            client=_SdkDeployClient(profile),
            stt_endpoint=stt_endpoint,
            llm_endpoint=llm_endpoint,
            tts_endpoint=tts_endpoint,
            llm_temperature=llm_temperature,
            llm_max_tokens=llm_max_tokens,
            llm_tools_enabled=llm_tools_enabled,
            llm_max_tool_iterations=llm_max_tool_iterations,
            tts_inference_timesteps=tts_inference_timesteps,
            tts_cfg_value=tts_cfg_value,
        )

    def _predict(self, endpoint: str, *, text: str, custom_inputs: dict[str, Any]) -> dict[str, Any]:
        response = self.client.predict(
            endpoint=endpoint,
            inputs={"input": [{"role": "user", "content": text}], "custom_inputs": custom_inputs},
        )
        return response if isinstance(response, dict) else dict(response)

    def transcribe(
        self, audio: bytes, *, language: str | None = None, sample_rate_hz: int
    ) -> tuple[str, str | None]:
        # language=None => Qwen3-ASR auto-detects; the detected tag is returned so
        # the LLM and TTS stages can follow the caller's spoken language.
        response = self._predict(
            self.stt_endpoint,
            text="transcribe",
            custom_inputs={
                "audio_b64": base64.b64encode(audio).decode("ascii"),
                "language": language,
                "sample_rate_hz": sample_rate_hz,
            },
        )
        custom = _custom_outputs(response)
        transcript = str(custom.get("transcript") or _output_text(response) or "").strip()
        # Empty transcript = silence/noise picked up by VAD. Return it as an empty
        # string (the caller drops the turn) rather than raising, so brief noise
        # doesn't surface as an error in the UI.
        detected = custom.get("detected_language") or custom.get("language")
        return transcript, (str(detected) if detected else None)

    def respond(self, transcript: str, *, language: str, context: str | None = None) -> str:
        # The middle stage is a Databricks foundation-model chat endpoint
        # (databricks-qwen3-next-80b-a3b-instruct), queried with ChatCompletions
        # shape, not the Responses Agent contract used by STT/TTS. Qwen3-Next
        # accepts ``temperature`` and ``tools`` (unlike claude-sonnet-5).
        system = (
            "You are a warm, helpful voice assistant on a live phone-style call. Engage directly "
            "and try to help with whatever the caller asks: answer their question, walk them through "
            "next steps, or ask one brief clarifying question if you truly need a detail. Never refuse "
            "or deflect with generic lines like 'contact customer support' or 'I can't help with that' "
            "\u2014 stay in the conversation and be genuinely useful. Reply in one to three short spoken "
            "sentences with no markdown, no lists, and no emoji. Use the available tools when they help "
            f"answer accurately. Always respond in the user's language ({language})."
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": _compose_user_content(transcript, context)},
        ]
        tools = _TOOLS_SPEC if self.llm_tools_enabled else None

        for _ in range(max(1, self.llm_max_tool_iterations)):
            message = self._chat(messages, tools=tools)
            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                text = _message_text(message).strip()
                if not text:
                    raise RuntimeError("LLM endpoint returned no response text")
                return text
            # Record the assistant tool request, then run each tool and feed the
            # results back so the model can compose a final spoken answer.
            messages.append(
                {"role": "assistant", "content": message.get("content") or "", "tool_calls": tool_calls}
            )
            for call in tool_calls:
                fn = call.get("function") or {}
                try:
                    arguments = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    arguments = {}
                result = _run_tool(str(fn.get("name") or ""), arguments)
                messages.append({"role": "tool", "tool_call_id": call.get("id"), "content": result})

        # Tool budget exhausted: force a plain text answer (no further tools).
        message = self._chat(messages, tools=None)
        text = _message_text(message).strip()
        if not text:
            raise RuntimeError("LLM endpoint returned no response text after tool calls")
        return text

    def _chat(self, messages: list[dict[str, Any]], *, tools: list[dict[str, Any]] | None) -> dict[str, Any]:
        inputs: dict[str, Any] = {
            "messages": messages,
            "max_tokens": self.llm_max_tokens,
            "temperature": self.llm_temperature,
        }
        if tools:
            inputs["tools"] = tools
            inputs["tool_choice"] = "auto"
        response = self.client.predict(endpoint=self.llm_endpoint, inputs=inputs)
        payload = response if isinstance(response, dict) else dict(response)
        choices = payload.get("choices") or []
        if choices and isinstance(choices[0], dict):
            return choices[0].get("message") or {}
        return {}

    def synthesize(self, text: str, *, language: str) -> AudioResponse:
        response = self._predict(
            self.tts_endpoint,
            text=text,
            custom_inputs={
                "text": text,
                "language": language,
                "inference_timesteps": self.tts_inference_timesteps,
                "cfg_value": self.tts_cfg_value,
            },
        )
        custom = _custom_outputs(response)
        encoded = str(custom.get("audio_b64") or "")
        if not encoded:
            raise RuntimeError("TTS endpoint returned no audio")
        return AudioResponse(
            audio=base64.b64decode(encoded),
            mime_type=str(custom.get("mime_type") or "audio/wav"),
            sample_rate_hz=int(custom.get("sample_rate_hz") or 24_000),
        )

    def synthesize_stream(self, text: str, *, language: str) -> Iterator[AudioChunk]:
        """Yield PCM audio chunks as the TTS agent generates them.

        Invokes the endpoint's ``predict_stream`` (VoxCPM2 ``generate(streaming=
        True)``): each SSE event carries a ~80 ms base64 PCM16 slice in
        ``custom_outputs.audio_pcm16_b64``. Emitting these as they arrive lets the
        client start playback long before the sentence finishes generating.
        """
        stream = self.client.predict_stream(
            endpoint=self.tts_endpoint,
            inputs={
                "input": [{"role": "user", "content": text}],
                "custom_inputs": {
                    "text": text,
                    "language": language,
                    "inference_timesteps": self.tts_inference_timesteps,
                    "cfg_value": self.tts_cfg_value,
                },
            },
        )
        # One-chunk lookahead so the server's final timing (gen_ms/ttfb_ms), which
        # arrives in the last SSE event AFTER all audio, can be attached to the
        # last audio chunk we emit.
        pending: AudioChunk | None = None
        server_ttfb_ms: float | None = None
        server_gen_ms: float | None = None
        for event in stream:
            payload = event if isinstance(event, dict) else dict(event)
            custom = _custom_outputs(payload)
            pcm_b64 = custom.get("audio_pcm16_b64")
            if pcm_b64:
                if pending is not None:
                    yield pending
                pending = AudioChunk(
                    pcm=base64.b64decode(pcm_b64),
                    sample_rate_hz=int(custom.get("sample_rate_hz") or 48_000),
                )
            elif custom.get("final"):
                server_ttfb_ms = _as_float(custom.get("ttfb_ms"))
                server_gen_ms = _as_float(custom.get("gen_ms"))
        if pending is not None:
            yield AudioChunk(
                pcm=pending.pcm,
                sample_rate_hz=pending.sample_rate_hz,
                server_ttfb_ms=server_ttfb_ms,
                server_gen_ms=server_gen_ms,
            )


def _compose_user_content(transcript: str, context: str | None) -> str:
    """Combine the spoken transcript with optional textual grounding context.

    The transcript is what the caller said (for Belebele, the spoken passage);
    the context carries any non-spoken grounding the caller supplied (for
    Belebele, the question + answer options + how to respond). Kept as a labelled
    single user message so the model treats the context as part of the request.
    """
    if not context:
        return transcript
    if not transcript.strip():
        return context
    return f"{transcript}\n\n{context}"


def _message_text(message: dict[str, Any]) -> str:
    """Extract assistant text from a ChatCompletions message object."""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(part.get("text") or "")
            for part in content
            if isinstance(part, dict) and part.get("type") in ("text", "output_text")
        )
    return ""


def _as_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _custom_outputs(response: dict[str, Any]) -> dict[str, Any]:
    custom = response.get("custom_outputs")
    return custom if isinstance(custom, dict) else {}


def _output_text(response: dict[str, Any]) -> str:
    """Extract concatenated text from a Responses ``output`` array."""
    parts: list[str] = []
    for item in response.get("output") or []:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for chunk in content:
                if isinstance(chunk, dict) and chunk.get("type") in ("output_text", "text"):
                    parts.append(str(chunk.get("text") or ""))
    return "".join(parts).strip()
