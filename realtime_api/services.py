"""Stateless inference adapters for Databricks Model Serving.

The STT/TTS/LLM endpoints are Databricks Agent Framework endpoints
(``ResponsesAgent``, ``task = agent/v1/responses``). We therefore query them with
the OpenAI Responses shape (``input`` + ``custom_inputs``) via the MLflow
deployments client and read structured payloads back from ``custom_outputs``.
"""
from __future__ import annotations

import base64
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterator, Protocol

from .contracts import AudioChunk, AudioResponse
from .tool_registry import ToolContext, run_tool, tools_spec

if TYPE_CHECKING:
    from .tracing import TurnTrace

logger = logging.getLogger("realtime_voice")


class SpeechToText(Protocol):
    def transcribe(
        self, audio: bytes, *, language: str | None, sample_rate_hz: int
    ) -> tuple[str, str | None]: ...


class LanguageModel(Protocol):
    def respond(self, transcript: str, *, language: str, context: str | None = None, tool_ctx: ToolContext | None = None) -> str: ...

    def phrase(self, intent: str, *, language: str) -> str:
        """Generate one short spoken sentence for `intent`, in `language`.

        Tool-free, context-free single-shot generation used for fixed spoken
        moments (e.g. a "one moment" acknowledgment, or a "switch language"
        prompt). The multilingual model renders the correct language for any
        supported BCP-47 tag, so there are no hardcoded per-language phrase
        tables and no English fallback.
        """
        ...


class TextToSpeech(Protocol):
    def synthesize(
        self,
        text: str,
        *,
        language: str,
        reference_audio_b64: str | None = None,
        voice_id: str | None = None,
    ) -> AudioResponse: ...


class StreamingTextToSpeech(TextToSpeech, Protocol):
    def synthesize_stream(
        self,
        text: str,
        *,
        language: str,
        reference_audio_b64: str | None = None,
        voice_id: str | None = None,
    ) -> Iterator[AudioChunk]: ...


class _SdkDeployClient:
    """mlflow-free serving client backed by the Databricks SDK.

    Exposes ``predict`` (WorkspaceClient) and ``predict_stream`` (SSE via
    requests), matching the tiny surface ``DatabricksServing`` needs. Lets the
    API run against live endpoints without installing mlflow, using the same
    Databricks auth profile as the CLI.
    """

    def __init__(
        self,
        profile: str | None = None,
        *,
        predict_timeout_s: float = 45.0,
        stream_timeout_s: float = 180.0,
    ) -> None:
        from databricks.sdk import WorkspaceClient

        # Timeouts are config-sourced (realtime_voice.timeouts) — no hardcoded
        # literals. predict = synchronous STT/LLM; stream = long TTS synth SSE.
        self._predict_timeout_s = predict_timeout_s
        self._stream_timeout_s = stream_timeout_s
        self._w = WorkspaceClient(profile=profile or None)
        self._host = self._w.config.host.rstrip("/")

    def _auth_headers(self) -> dict[str, str]:
        """Fresh auth headers for one request.

        Must be re-read per call: the SP OAuth token the app runs on lives 60
        minutes, and the SDK only refreshes it when asked. Caching this dict at
        construction pins one token for the life of the process, so every
        serving call starts returning 403 an hour after startup.
        """
        return {**dict(self._w.config.authenticate() or {}), "Content-Type": "application/json"}

    def predict(self, *, endpoint: str, inputs: dict) -> dict:
        import requests as _requests

        url = f"{self._host}/serving-endpoints/{endpoint}/invocations"
        resp = _requests.post(
            url, headers=self._auth_headers(), json=inputs, timeout=self._predict_timeout_s
        )
        resp.raise_for_status()
        return resp.json()

    def predict_stream(self, *, endpoint: str, inputs: dict):
        import requests

        body = {**inputs, "stream": True}
        url = f"{self._host}/serving-endpoints/{endpoint}/invocations"
        with requests.post(
            url,
            headers=self._auth_headers(),
            json=body,
            stream=True,
            timeout=self._stream_timeout_s,
        ) as resp:
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
    stt_warmup_passes: int = 3
    # Voice ids whose reference clip this process has already uploaded. A session
    # reference is ~500KB base64, and re-sending it every turn cost ~1.7s of
    # time-to-first-audio (pure upload; the endpoint materialises it in ~25ms), so
    # the clip goes up once and later turns identify it by id alone. Runtime cache,
    # hence excluded from equality/repr.
    _voice_ids_sent: set[str] = field(default_factory=set, compare=False, repr=False)

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
        predict_timeout_s: float = 45.0,
        tts_stream_timeout_s: float = 180.0,
    ) -> "DatabricksServing":
        """Build against live endpoints using the Databricks SDK (no mlflow)."""
        return cls(
            client=_SdkDeployClient(
                profile,
                predict_timeout_s=predict_timeout_s,
                stream_timeout_s=tts_stream_timeout_s,
            ),
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

    def warmup(self) -> dict[str, Any]:
        """Best-effort priming of the STT/LLM/TTS serving replicas.

        The GPU replicas pay a one-time warm-up (CUDA/kernel init, first forward
        pass) on their first inference after (re)deploy or scale-up. Firing one
        tiny request per endpoint at app startup moves that cost off the first
        *real* user turn (which otherwise shows up as an inflated STT/LLM/TTS time
        in the UI). Each ping is independently guarded — a slow or unreachable
        endpoint degrades warm-up, it never breaks startup.
        """
        import time

        results: dict[str, Any] = {}

        def _timed(name: str, fn) -> None:
            started = time.perf_counter()
            try:
                fn()
                results[name] = {"ok": True, "ms": round((time.perf_counter() - started) * 1000)}
            except Exception as exc:  # noqa: BLE001
                results[name] = {
                    "ok": False,
                    "ms": round((time.perf_counter() - started) * 1000),
                    "error": str(exc),
                }

        # 0.3 s of silence @16 kHz s16le: runs the ASR encoder + a decode step
        # (empty transcript) to warm the replica without needing real speech.
        # A freshly (re)deployed GPU replica's warm-up curve spans several
        # inferences (kernel autotune / graph capture), not just one — observed
        # ~24s then ~16s then ~1.5s. So drive STT a few passes to push through it,
        # leaving the first real user turn on the fast path. Off-thread, so the
        # extra passes never delay startup/readiness.
        silence = b"\x00\x00" * 4800
        for i in range(self.stt_warmup_passes):
            _timed(f"stt{i + 1}", lambda: self.transcribe(silence, language=None, sample_rate_hz=16_000))
        _timed("llm", lambda: self.respond("hi", language="en"))
        _timed("tts", lambda: self.synthesize("Hello.", language="en"))
        return results

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

    def respond(self, transcript: str, *, language: str, context: str | None = None, tool_ctx: ToolContext | None = None) -> str:
        text, _ = self.respond_with_tools(transcript, language=language, context=context, tool_ctx=tool_ctx)
        return text

    def phrase(self, intent: str, *, language: str) -> str:
        system = (
            "You are a voice assistant's phrasing helper. Produce a single short, "
            "natural spoken sentence that fulfills the request, written in the "
            f"language identified by the BCP-47 code '{language}'. Output ONLY that "
            "sentence — no quotes, no explanation, no translation, no alternatives."
        )
        message = self._chat(
            [{"role": "system", "content": system}, {"role": "user", "content": intent}],
            tools=None,
        )
        return _message_text(message).strip()

    def respond_with_tools(
        self, transcript: str, *, language: str, context: str | None = None,
        tool_ctx: Any | None = None, history: list[dict[str, str]] | None = None,
        trace: "TurnTrace | None" = None,
        system_prompt: str | None = None,
        tools_override: list[dict[str, Any]] | None = None,
        tool_runner: Any | None = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Like respond(), but also returns a list of tool invocations for UI emission.

        When ``trace`` is provided, every LLM iteration (with the FULL messages
        array sent, including text-only history — the key thing to inspect for
        tool-calling bugs) and every tool call (arguments + result) is recorded as
        a span. Recording is in-memory only; persistence happens off the hot path.

        The pipeline always passes ``system_prompt``, ``tools_override``, and
        ``tool_runner`` from the resolved profile — the fallback to
        ``tools_spec(profile="billing")`` is only a safety net for tests.
        """
        from .tools import BILLING_SYSTEM_PROMPT as _DEFAULT_PROMPT
        system = (system_prompt or _DEFAULT_PROMPT).format(language=language)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
        ]
        for msg in (history or []):
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": _compose_user_content(transcript, context)})
        if tools_override is not None:
            tools = tools_override if self.llm_tools_enabled else None
        else:
            tools = tools_spec(profile="billing") if self.llm_tools_enabled else None
        tool_names = [t["function"]["name"] for t in tools] if tools else []
        ctx = tool_ctx or ToolContext()
        runner = tool_runner or (lambda n, a, c: run_tool(n, a, c, profile="billing"))
        tool_invocations: list[dict[str, Any]] = []

        for iteration in range(max(1, self.llm_max_tool_iterations)):
            _t = time.perf_counter()
            llm_span = None
            if trace is not None:
                llm_span = trace.span(
                    f"llm.iteration.{iteration}", "LLM",
                    input={
                        "messages": [dict(m) for m in messages],
                        "tools_available": tool_names,
                        "tool_choice": "auto" if tools else None,
                        "temperature": self.llm_temperature,
                        "max_tokens": self.llm_max_tokens,
                        "endpoint": self.llm_endpoint,
                    },
                )
            message = self._chat(messages, tools=tools)
            tool_calls = message.get("tool_calls") or []
            assistant_content = _message_text(message)
            # Some serving endpoints don't return the structured `tool_calls` field and
            # instead print the call as inline text in the content. Parse those so the
            # tool actually EXECUTES, and strip the markup so it never leaks into the
            # spoken/on-screen transcript.
            if not tool_calls:
                inline, cleaned = _extract_inline_tool_calls(assistant_content)
                if inline:
                    tool_calls = [
                        {
                            "id": f"inline_{i}",
                            "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])},
                        }
                        for i, c in enumerate(inline)
                    ]
                    assistant_content = cleaned
            logger.info(
                "llm _chat iter %d: %dms tool_calls=%d",
                iteration, round((time.perf_counter() - _t) * 1000), len(tool_calls),
            )
            if llm_span is not None:
                llm_span.set_output({
                    "content": assistant_content,
                    "tool_calls": tool_calls,
                }).set_attribute("tool_call_count", len(tool_calls)).set_attribute(
                    "tool_calls_emitted", [((c.get("function") or {}).get("name")) for c in tool_calls]
                ).end()
            if not tool_calls:
                text = _strip_tool_markup(assistant_content).strip()
                if not text:
                    raise RuntimeError("LLM endpoint returned no response text")
                return text, tool_invocations
            messages.append(
                {"role": "assistant", "content": assistant_content, "tool_calls": tool_calls}
            )
            for call in tool_calls:
                fn = call.get("function") or {}
                try:
                    arguments = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    arguments = {}
                name = str(fn.get("name") or "")
                tool_span = None
                if trace is not None:
                    tool_span = trace.span(f"tool.{name}", "TOOL", input=arguments)
                _tt = time.perf_counter()
                result = runner(name, arguments, ctx)
                logger.info("tool %s: %dms", name, round((time.perf_counter() - _tt) * 1000))
                messages.append({"role": "tool", "tool_call_id": call.get("id"), "content": result})
                try:
                    parsed_result = json.loads(result)
                except json.JSONDecodeError:
                    parsed_result = result
                if tool_span is not None:
                    tool_span.set_output(parsed_result).set_attribute(
                        "tool_call_id", call.get("id")
                    ).end()
                tool_invocations.append({"name": name, "arguments": arguments, "result": parsed_result})

        _t = time.perf_counter()
        final_span = None
        if trace is not None:
            final_span = trace.span(
                "llm.final", "LLM",
                input={"messages": [dict(m) for m in messages], "tools_available": []},
            )
        message = self._chat(messages, tools=None)
        logger.info("llm _chat final: %dms", round((time.perf_counter() - _t) * 1000))
        if final_span is not None:
            final_span.set_output({"content": message.get("content")}).end()
        text = _strip_tool_markup(_message_text(message)).strip()
        if not text:
            raise RuntimeError("LLM endpoint returned no response text after tool calls")
        return text, tool_invocations

    def summarize(
        self,
        *,
        system: str,
        user: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        endpoint: str | None = None,
    ) -> str:
        """One-shot, tool-free system+user completion returning plain text.

        Public entrypoint for callers that need a bounded LLM completion (e.g. the
        deep-dive spoken 'why' summary and report translation) WITHOUT reaching
        into the private ``_chat`` or spinning up a second serving client.
        ``temperature``/``max_tokens`` are per-call overrides of the instance
        defaults; ``endpoint`` routes this one call to a different model (the
        text-to-text conversion model) while the voice turns keep ``llm_endpoint``.
        """
        message = self._chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tools=None,
            temperature=temperature,
            max_tokens=max_tokens,
            endpoint=endpoint,
        )
        return _message_text(message).strip()

    def _chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        endpoint: str | None = None,
    ) -> dict[str, Any]:
        inputs: dict[str, Any] = {
            "messages": messages,
            "max_tokens": self.llm_max_tokens if max_tokens is None else max_tokens,
            "temperature": self.llm_temperature if temperature is None else temperature,
        }
        if tools:
            inputs["tools"] = tools
            inputs["tool_choice"] = "auto"
        response = self.client.predict(endpoint=endpoint or self.llm_endpoint, inputs=inputs)
        payload = response if isinstance(response, dict) else dict(response)
        choices = payload.get("choices") or []
        if choices and isinstance(choices[0], dict):
            return choices[0].get("message") or {}
        return {}

    def _tts_inputs(
        self,
        text: str,
        language: str,
        reference_audio_b64: str | None,
        voice_id: str | None,
        *,
        send_reference: bool,
    ) -> dict[str, Any]:
        custom_inputs: dict[str, Any] = {
            "text": text,
            "language": language,
            "inference_timesteps": self.tts_inference_timesteps,
            "cfg_value": self.tts_cfg_value,
        }
        if voice_id:
            custom_inputs["voice_id"] = voice_id
        if reference_audio_b64 and send_reference:
            custom_inputs["reference_audio_b64"] = reference_audio_b64
        return custom_inputs

    def _send_reference(self, reference_audio_b64: str | None, voice_id: str | None) -> bool:
        """Whether this request must carry the reference clip itself.

        Only the first cloned turn of a voice uploads the clip; later turns name
        the endpoint's cached copy via ``voice_id``. Without a ``voice_id`` there
        is nothing to cache against, so the clip goes every time (old behaviour).
        """
        if not reference_audio_b64:
            return False
        if not voice_id:
            return True
        return voice_id not in self._voice_ids_sent

    def synthesize(
        self,
        text: str,
        *,
        language: str,
        reference_audio_b64: str | None = None,
        voice_id: str | None = None,
    ) -> AudioResponse:
        send_reference = self._send_reference(reference_audio_b64, voice_id)
        while True:
            response = self._predict(
                self.tts_endpoint,
                text=text,
                custom_inputs=self._tts_inputs(
                    text, language, reference_audio_b64, voice_id, send_reference=send_reference
                ),
            )
            custom = _custom_outputs(response)
            if send_reference and voice_id:
                self._voice_ids_sent.add(voice_id)
            if custom.get("voice_cache_miss") and not send_reference and reference_audio_b64:
                # This replica no longer holds the session voice (fresh container or
                # a different replica). Resend the clip and retry the same turn so
                # the caller never hears a turn rendered in another voice.
                if voice_id:
                    self._voice_ids_sent.discard(voice_id)
                send_reference = True
                continue
            encoded = str(custom.get("audio_b64") or "")
            if not encoded:
                raise RuntimeError("TTS endpoint returned no audio")
            return AudioResponse(
                audio=base64.b64decode(encoded),
                mime_type=str(custom.get("mime_type") or "audio/wav"),
                sample_rate_hz=int(custom.get("sample_rate_hz") or 24_000),
            )

    def synthesize_stream(
        self,
        text: str,
        *,
        language: str,
        reference_audio_b64: str | None = None,
        voice_id: str | None = None,
    ) -> Iterator[AudioChunk]:
        """Yield PCM audio chunks as the TTS agent generates them.

        Invokes the endpoint's ``predict_stream`` (VoxCPM2 ``generate(streaming=
        True)``): each SSE event carries a ~80 ms base64 PCM16 slice in
        ``custom_outputs.audio_pcm16_b64``. Emitting these as they arrive lets the
        client start playback long before the sentence finishes generating.

        ``reference_audio_b64`` (a base64 WAV) pins the voice: the endpoint clones
        its timbre so every turn in a session keeps the same voice. Passing a
        ``voice_id`` uploads that clip only once -- later turns send the id alone,
        which is what keeps the clip off the critical path.

        A ``voice_cache_miss`` means the replica cannot resolve the id and produced
        no audio, so the clip is resent and the turn retried once. The retry happens
        before any chunk is emitted, so it can never duplicate or split audio.
        """
        send_reference = self._send_reference(reference_audio_b64, voice_id)
        while True:
            status: dict[str, Any] = {}
            emitted = 0
            for chunk in self._synthesize_stream_once(
                text,
                language=language,
                reference_audio_b64=reference_audio_b64,
                voice_id=voice_id,
                send_reference=send_reference,
                status=status,
            ):
                emitted += 1
                yield chunk
            if send_reference and voice_id:
                self._voice_ids_sent.add(voice_id)
            if not (status.get("voice_cache_miss") and emitted == 0 and not send_reference):
                return
            if voice_id:
                self._voice_ids_sent.discard(voice_id)
            if not reference_audio_b64:
                return
            send_reference = True

    def _synthesize_stream_once(
        self,
        text: str,
        *,
        language: str,
        reference_audio_b64: str | None,
        voice_id: str | None,
        send_reference: bool,
        status: dict[str, Any],
    ) -> Iterator[AudioChunk]:
        """One streaming synthesis attempt; reports stream-level flags via ``status``."""
        stream = self.client.predict_stream(
            endpoint=self.tts_endpoint,
            inputs={
                "input": [{"role": "user", "content": text}],
                "custom_inputs": self._tts_inputs(
                    text, language, reference_audio_b64, voice_id, send_reference=send_reference
                ),
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
                if custom.get("voice_cache_miss"):
                    status["voice_cache_miss"] = True
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


_TOOL_CALL_TAG_RE = re.compile(r"</?\s*tool_call\s*>", re.IGNORECASE)


def _iter_json_objects(text: str) -> Iterator[tuple[str, Any]]:
    """Yield (raw_substring, parsed) for each top-level ``{...}`` JSON object.

    A brace-matching scanner (string/escape aware) so it tolerates the malformed
    markup some endpoints emit (e.g. unpaired ``<tool_call>`` tags) that a strict
    regex would miss.
    """
    i, n = 0, len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        depth, j, in_str, esc = 0, i, False, False
        while j < n:
            ch = text[j]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    raw = text[i : j + 1]
                    try:
                        yield raw, json.loads(raw)
                    except json.JSONDecodeError:
                        pass
                    break
            j += 1
        i = j + 1


def _extract_inline_tool_calls(content: str) -> tuple[list[dict[str, Any]], str]:
    """Parse tool calls a model emitted as inline TEXT and strip them from the text.

    Some serving endpoints don't return the structured ``tool_calls`` field and
    instead print the call in the message content, e.g.::

        <tool_call> {"name": "start_deep_dive", "arguments": {...}} </tool_call>

    Left unhandled, that raw markup both (a) leaks into the spoken + on-screen
    transcript and (b) means the tool never actually runs. We only trigger when a
    ``<tool_call>`` marker is present (so ordinary prose containing JSON is never
    misread as a tool call), then extract every ``{"name": ...}`` object.

    Returns ``(calls, cleaned_text)`` where ``calls`` is ``[{"name", "arguments"}]``.
    """
    if not content or "tool_call" not in content.lower():
        return [], content
    calls: list[dict[str, Any]] = []
    cleaned = content
    for raw, obj in _iter_json_objects(content):
        if isinstance(obj, dict) and isinstance(obj.get("name"), str):
            args = obj.get("arguments")
            calls.append({"name": obj["name"], "arguments": args if isinstance(args, dict) else {}})
            cleaned = cleaned.replace(raw, "")
    cleaned = _TOOL_CALL_TAG_RE.sub("", cleaned).strip()
    return calls, cleaned


def _strip_tool_markup(text: str) -> str:
    """Defense-in-depth: remove any stray inline tool-call markup from spoken text."""
    _, cleaned = _extract_inline_tool_calls(text)
    return cleaned if cleaned else text if "tool_call" not in (text or "").lower() else ""


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
