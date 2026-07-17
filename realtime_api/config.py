"""Configuration for the standalone realtime voice API.

Endpoint settings are resolved (in priority order):
1. ``VOICE_API_*`` environment variables (explicit override), else
2. the shared ``realtime_voice:`` block in ``config/config.yaml`` (+ local),
   using the promoted STT/TTS candidate endpoints and ``llm_endpoint``.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RealtimeSettings:
    stt_endpoint: str
    llm_endpoint: str
    tts_endpoint: str
    sample_rate_hz: int = 16_000
    # Trailing silence required to treat the turn as finished (end-of-speech
    # endpointing). Set generously so ordinary pauses *between clauses* don't end
    # the turn and make the assistant reply while you're mid-sentence. ~1.4s is a
    # forgiving end-of-utterance gap; lower it for snappier (but choppier) turns.
    vad_silence_ms: int = 1400
    max_turn_seconds: int = 20
    # Minimum voiced audio before a turn can finalize; filters brief blips and
    # speaker->mic echo (e.g. the assistant's own voice) from firing turns.
    min_speech_ms: int = 400
    # While the assistant is replying, only a *sustained* talk-over of this much
    # voiced audio counts as a barge-in (interrupt). Long enough to reject echo
    # and backchannels ("mm-hmm"), short enough that a real interruption lands.
    barge_in_ms: int = 700
    # Voice barge-in (interrupting the reply by talking) is only reliable with
    # headphones or AEC. With echo-cancellation off on speakers, the assistant's
    # own voice re-enters the mic and falsely interrupts, so this is OFF by
    # default. Enable with VOICE_ALLOW_BARGE_IN=1 when using headphones.
    allow_barge_in: bool = False
    # LLM sampling + tool-calling knobs. Qwen3-Next accepts `temperature` and
    # `tools` (unlike claude-sonnet-5), so both are enabled by default.
    llm_temperature: float = 0.4
    llm_max_tokens: int = 512
    llm_tools_enabled: bool = True
    llm_max_tool_iterations: int = 3
    # Verification mode: instead of answering, the assistant reports the detected
    # language + verbatim transcript and speaks it back (to confirm STT/language
    # ID). Toggle with VOICE_VERIFY_MODE=0/1.
    verify_mode: bool = True
    # TTS latency/quality knobs forwarded to the VoxCPM agent per request.
    # 6 diffusion steps is the profiled sweet spot: full multilingual quality
    # (incl. Thai) at the lowest safe latency on GPU_MEDIUM.
    tts_inference_timesteps: int = 6
    tts_cfg_value: float = 2.0
    # End-to-end supported languages (STT ∩ TTS) as BCP 47 primary subtags,
    # resolved from config. The UI reports these to the user.
    supported_languages: tuple[str, ...] = ()

    @classmethod
    def from_env(cls) -> "RealtimeSettings":
        required = {
            "VOICE_API_STT_ENDPOINT": os.getenv("VOICE_API_STT_ENDPOINT", ""),
            "VOICE_API_LLM_ENDPOINT": os.getenv("VOICE_API_LLM_ENDPOINT", ""),
            "VOICE_API_TTS_ENDPOINT": os.getenv("VOICE_API_TTS_ENDPOINT", ""),
        }
        missing = [key for key, value in required.items() if not value]
        if missing:
            raise RuntimeError(f"Missing realtime API settings: {', '.join(missing)}")
        return cls(
            stt_endpoint=required["VOICE_API_STT_ENDPOINT"],
            llm_endpoint=required["VOICE_API_LLM_ENDPOINT"],
            tts_endpoint=required["VOICE_API_TTS_ENDPOINT"],
        )

    @classmethod
    def from_config(cls, config_dir: str | Path | None = None) -> "RealtimeSettings":
        rv = _load_realtime_voice(config_dir)
        stt = _first_endpoint(rv.get("stt_candidates"))
        tts = _first_endpoint(rv.get("tts_candidates"))
        llm = str(rv.get("llm_endpoint") or "")
        missing = [n for n, v in (("stt", stt), ("llm", llm), ("tts", tts)) if not v]
        if missing:
            raise RuntimeError(f"realtime_voice config is missing endpoints: {', '.join(missing)}")
        tts_defaults = rv.get("tts_defaults") or {}
        llm_defaults = rv.get("llm_defaults") or {}
        return cls(
            stt_endpoint=stt,
            llm_endpoint=llm,
            tts_endpoint=tts,
            llm_temperature=float(llm_defaults.get("temperature", 0.4)),
            llm_max_tokens=int(llm_defaults.get("max_tokens", 512)),
            llm_tools_enabled=bool(llm_defaults.get("tools_enabled", True)),
            llm_max_tool_iterations=int(llm_defaults.get("max_tool_iterations", 3)),
            tts_inference_timesteps=int(tts_defaults.get("inference_timesteps", 6)),
            tts_cfg_value=float(tts_defaults.get("cfg_value", 2.0)),
            supported_languages=_supported_languages(rv),
        )

    @classmethod
    def resolve(cls, config_dir: str | Path | None = None) -> "RealtimeSettings":
        """Env override first, then the shared config block."""
        settings = cls.from_env() if os.getenv("VOICE_API_STT_ENDPOINT") else cls.from_config(config_dir)
        import dataclasses

        verify = os.getenv("VOICE_VERIFY_MODE")
        if verify is not None:
            settings = dataclasses.replace(settings, verify_mode=verify == "1")
        barge = os.getenv("VOICE_ALLOW_BARGE_IN")
        if barge is not None:
            settings = dataclasses.replace(settings, allow_barge_in=barge == "1")
        return settings


def _repo_config_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "config"


def _load_merged(config_dir: str | Path | None = None) -> dict[str, Any]:
    import yaml

    directory = Path(config_dir) if config_dir else _repo_config_dir()
    merged: dict[str, Any] = {}
    for name in ("config.yaml", "config.local.yaml"):
        path = directory / name
        if path.exists():
            merged = _deep_merge(merged, yaml.safe_load(path.read_text(encoding="utf-8")) or {})
    return merged


def _load_realtime_voice(config_dir: str | Path | None) -> dict[str, Any]:
    block = _load_merged(config_dir).get("realtime_voice")
    if not block:
        raise RuntimeError("No 'realtime_voice' block found in config")
    return block


def databricks_profile(config_dir: str | Path | None = None) -> str | None:
    """Databricks CLI profile for SDK auth (config block, then env, else default)."""
    profile = (_load_merged(config_dir).get("databricks") or {}).get("profile")
    return profile or os.getenv("DATABRICKS_CONFIG_PROFILE") or None


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _first_endpoint(candidates: dict[str, Any] | None) -> str:
    for candidate in (candidates or {}).values():
        endpoint = candidate.get("endpoint")
        if endpoint:
            return str(endpoint)
    return ""


def _first_supported(candidates: dict[str, Any] | None) -> list[str]:
    for candidate in (candidates or {}).values():
        langs = candidate.get("supported_languages")
        if langs:
            return [str(x).lower() for x in langs]
    return []


def _supported_languages(rv: dict[str, Any]) -> tuple[str, ...]:
    """End-to-end languages = STT ∩ TTS (a language needs both to round-trip).

    Ordered by the TTS list (the narrower set), so the UI shows a stable order.
    """
    stt = set(_first_supported(rv.get("stt_candidates")))
    tts = _first_supported(rv.get("tts_candidates"))
    if not stt or not tts:
        return tuple(tts or sorted(stt))
    return tuple(code for code in tts if code in stt)
