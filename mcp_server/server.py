"""MCP server for the Genie realtime voice API.

Run it (stdio transport, for Cursor / Claude Desktop):

    GENIE_VOICE_API_URL=https://<app>.databricksapps.com/realtime \\
    DATABRICKS_HOST=https://<workspace>.cloud.databricks.com \\
    DATABRICKS_CLIENT_ID=<sp-client-id> DATABRICKS_CLIENT_SECRET=<sp-secret> \\
    python -m mcp_server.server

Environment:
  GENIE_VOICE_API_URL  realtime API base incl. mount prefix (default: the
                       deployed app's /realtime). Point at http://localhost:8000
                       for a local `python -m realtime_api.server`.
  GENIE_VOICE_OUT_DIR  where synthesized / response audio WAVs are written
                       (default: the OS temp dir).
  auth                 see mcp_server.auth (GENIE_VOICE_TOKEN | DATABRICKS_* ).

Exposed tools: describe_api, get_capabilities, list_languages, get_benchmarks,
health, synthesize_speech, transcribe_audio, ask_voice_agent.
"""
from __future__ import annotations

import logging
import os
import sys
import tempfile
import time
from typing import Any

# FastMCP lived in the SDK as ``mcp.server.fastmcp`` through the 1.x line; the
# ``mcp`` 2.0 release moved it out into the standalone ``fastmcp`` package. Try
# the bundled path first (matches our pinned ``mcp<2``), fall back to standalone.
try:
    from mcp.server.fastmcp import FastMCP
except ModuleNotFoundError:  # mcp >= 2: FastMCP is the separate `fastmcp` package
    from fastmcp import FastMCP

from .auth import token_provider_from_env
from .client import RealtimeVoiceAPI, load_audio, pcm_duration_ms, write_wav

DEFAULT_API_URL = "https://genie-voice-agent-3644297589119053.aws.databricksapps.com/realtime"

logger = logging.getLogger("genie_voice_mcp")
logging.basicConfig(level=logging.INFO, stream=sys.stderr)

mcp = FastMCP("genie-voice")

_api: RealtimeVoiceAPI | None = None


def _resolve_target() -> tuple[str, "Any", str]:
    """Resolve the realtime API base URL + auth for this runtime.

    Three cases, in order:
      1. GENIE_VOICE_API_URL explicitly set -> use it, with env-configured auth.
      2. Running INSIDE the Databricks App (DATABRICKS_APP_PORT set) -> call the
         co-hosted /realtime mount over loopback. Platform auth is enforced at
         ingress, so loopback needs NO bearer (and skipping it avoids a per-call
         OAuth mint against the app's own SP).
      3. Otherwise (local stdio client) -> the deployed app URL + env auth.
    """
    explicit = os.environ.get("GENIE_VOICE_API_URL")
    if explicit:
        provider, mode = token_provider_from_env()
        return explicit, provider, mode
    port = os.environ.get("DATABRICKS_APP_PORT")
    if port:
        return f"http://localhost:{port}/realtime", None, "loopback (in-app, no auth)"
    provider, mode = token_provider_from_env()
    return DEFAULT_API_URL, provider, mode


def _client() -> RealtimeVoiceAPI:
    """Lazily build the API client (reads env at first use)."""
    global _api
    if _api is None:
        url, provider, mode = _resolve_target()
        logger.info("genie-voice MCP -> %s (auth: %s)", url, mode)
        _api = RealtimeVoiceAPI(url, token_provider=provider)
    return _api


def _out_dir() -> str:
    d = os.environ.get("GENIE_VOICE_OUT_DIR") or tempfile.gettempdir()
    os.makedirs(d, exist_ok=True)
    return d


# --------------------------------------------------------------------------- #
# Metadata tools (HTTP)
# --------------------------------------------------------------------------- #
@mcp.tool()
async def describe_api() -> dict:
    """Return the realtime voice API's service descriptor: name, version, and the
    HTTP + WebSocket endpoints it exposes. Use this first to confirm connectivity
    and auth."""
    return await _client().get("/")


@mcp.tool()
async def get_capabilities() -> dict:
    """List the voice capabilities the API offers (speech-to-text, text-to-speech,
    speech-llm-toolassist-speech), each with its WebSocket path, input/output event
    types, and supported languages."""
    return await _client().get("/v1/capabilities")


@mcp.tool()
async def list_languages() -> dict:
    """List the languages supported end-to-end (speech-to-text ∩ text-to-speech),
    as BCP-47 tags with display metadata."""
    return await _client().get("/v1/languages")


@mcp.tool()
async def get_benchmarks(run_id: str | None = None) -> dict:
    """Return the latest multilingual voice benchmark results (FLEURS transcription
    accuracy — WER/CER — plus per-stage latency) measured on this API. Pass a
    specific ``run_id`` to fetch that run instead of the latest."""
    path = f"/v1/benchmarks?run_id={run_id}" if run_id else "/v1/benchmarks"
    return await _client().get(path)


@mcp.tool()
async def health() -> dict:
    """Liveness probe for the realtime voice API."""
    return await _client().get("/healthz")


# --------------------------------------------------------------------------- #
# Voice tools (WebSocket)
# --------------------------------------------------------------------------- #
@mcp.tool()
async def synthesize_speech(text: str, language: str = "en", out_path: str | None = None) -> dict:
    """Synthesize speech from text using the API's text-to-speech voice models.

    Writes a mono 16-bit WAV file and returns its path. ``language`` is a BCP-47
    tag (e.g. "en", "es", "zh"); it must be a text-to-speech-supported language
    (see get_capabilities). ``out_path`` overrides the default output location.
    """
    if not text.strip():
        return {"error": "text is empty"}
    result = await _client().synthesize(text, language)
    if result.error:
        return {"error": result.error}
    if not result.audio_pcm:
        return {"error": "no audio returned"}
    path = out_path or os.path.join(_out_dir(), f"tts_{language}_{int(time.time() * 1000)}.wav")
    write_wav(path, result.audio_pcm, result.sample_rate)
    return {
        "audio_path": path,
        "language": language,
        "sample_rate_hz": result.sample_rate,
        "duration_ms": pcm_duration_ms(result.audio_pcm, result.sample_rate),
        "tts_first_ms": result.tts_first_ms,
    }


@mcp.tool()
async def transcribe_audio(audio_path: str, language: str = "auto") -> dict:
    """Transcribe a WAV audio file to text using the API's speech-to-text model.

    ``audio_path`` must be a readable WAV file (any sample rate; mono/stereo, 8/16/32-bit
    are converted automatically). ``language`` is "auto" (detect) or a BCP-47 tag to
    force. Returns the transcript and the detected language.
    """
    if not os.path.isfile(audio_path):
        return {"error": f"file not found: {audio_path}"}
    try:
        pcm, rate = load_audio(audio_path)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"could not read WAV: {exc}"}
    result = await _client().transcribe(pcm, rate, language=language)
    if result.error:
        return {"error": result.error}
    return {
        "transcript": result.transcript,
        "detected_language": result.detected_language,
        "input_duration_ms": pcm_duration_ms(pcm, rate),
        "stt_ms": result.stt_ms,
    }


@mcp.tool()
async def ask_voice_agent(
    audio_path: str,
    language: str = "auto",
    profile: str | None = None,
    space_name: str | None = None,
    context: str | None = None,
    save_response_audio: bool = True,
) -> dict:
    """Run a full voice-agent turn on a spoken WAV: speech-to-text → LLM (with
    tools) → text-to-speech.

    Returns what the caller said (``transcript``), the agent's reply
    (``response_text``), any tools it invoked (``tools``), and — when
    ``save_response_audio`` is true — the path to the spoken reply WAV.
    ``profile`` selects the Genie product (``billing`` = Space, ``card`` = Space
    + Agent Mode, ``knowledge`` = Genie One). ``space_name`` retargets Space /
    Agent Mode to a space the caller can run; Genie One ignores it.
    ``context`` injects extra textual grounding for the turn.
    """
    if not os.path.isfile(audio_path):
        return {"error": f"file not found: {audio_path}"}
    try:
        pcm, rate = load_audio(audio_path)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"could not read WAV: {exc}"}
    result = await _client().ask_agent(
        pcm, rate, language=language, profile=profile, space_name=space_name, context=context
    )
    if result.error and not (result.transcript or result.response_text):
        return {"error": result.error}
    out: dict[str, Any] = {
        "transcript": result.transcript,
        "detected_language": result.detected_language,
        "response_text": result.response_text,
        "tools": result.tools,
        "stt_ms": result.stt_ms,
        "llm_ms": result.llm_ms,
        "client_ttfa_ms": result.client_ttfa_ms,
    }
    if result.error:
        out["warning"] = result.error
    if save_response_audio and result.audio_pcm:
        path = os.path.join(_out_dir(), f"agent_reply_{int(time.time() * 1000)}.wav")
        write_wav(path, result.audio_pcm, result.sample_rate)
        out["response_audio_path"] = path
        out["response_audio_ms"] = pcm_duration_ms(result.audio_pcm, result.sample_rate)
    return out


def main() -> None:
    """Console entry point: run the MCP server over stdio."""
    mcp.run()


if __name__ == "__main__":
    main()
