"""Configuration for the standalone realtime voice API.

Everything is read from the shared ``realtime_voice:`` block in
``config/config.yaml`` (deep-merged with ``config/config.local.yaml`` locally):
the promoted STT/TTS candidate endpoints, ``llm_endpoint``, and the runtime knobs
(barge-in, warmup, debug). There are no environment overrides; the only env this
module reads is ``GENIE_CONFIG``, which merely points at which config file to load
on Databricks jobs/apps.
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
    # endpointing). Set generously so the caller can speak their FULL sentence —
    # ordinary mid-sentence / between-clause pauses must not end the turn and make
    # the assistant reply to half a sentence. 2.5s is a deliberately forgiving
    # end-of-utterance gap (favors "let them finish" over snappy turnaround).
    vad_silence_ms: int = 2500
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
    # default. Enable via realtime_voice.allow_barge_in in config when on headphones.
    allow_barge_in: bool = False
    # LLM sampling + tool-calling knobs. Qwen3-Next accepts `temperature` and
    # `tools` (unlike claude-sonnet-5), so both are enabled by default.
    llm_temperature: float = 0.4
    llm_max_tokens: int = 512
    llm_tools_enabled: bool = True
    llm_max_tool_iterations: int = 3
    # TTS latency/quality knobs forwarded to the VoxCPM agent per request.
    # 6 diffusion steps is the profiled sweet spot: full multilingual quality
    # (incl. Thai) at the lowest safe latency on GPU_MEDIUM.
    tts_inference_timesteps: int = 6
    tts_cfg_value: float = 2.0
    # End-to-end supported languages (STT ∩ TTS) as BCP 47 primary subtags,
    # resolved from config. The UI reports these to the user.
    supported_languages: tuple[str, ...] = ()
    stt_languages: tuple[str, ...] = ()
    tts_languages: tuple[str, ...] = ()
    # Startup priming of the serving replicas (off-thread). Disable via
    # realtime_voice.warmup: false in config (e.g. for fast local iteration).
    warmup_enabled: bool = True
    # STT warm-up passes fired at startup. A cold GPU replica's warm-up spans
    # several inferences, so >1 pass reliably lands the first real turn warm.
    stt_warmup_passes: int = 3
    # Diagnostics: dump each finalized turn's PCM to a WAV (realtime_voice.debug_audio).
    debug_audio: bool = False
    debug_audio_dir: str = "/tmp/realtime_audio"

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
            allow_barge_in=bool(rv.get("allow_barge_in", False)),
            llm_temperature=float(llm_defaults.get("temperature", 0.4)),
            llm_max_tokens=int(llm_defaults.get("max_tokens", 512)),
            llm_tools_enabled=bool(llm_defaults.get("tools_enabled", True)),
            llm_max_tool_iterations=int(llm_defaults.get("max_tool_iterations", 3)),
            tts_inference_timesteps=int(tts_defaults.get("inference_timesteps", 6)),
            tts_cfg_value=float(tts_defaults.get("cfg_value", 2.0)),
            stt_languages=tuple(_first_supported(rv.get("stt_candidates"))),
            tts_languages=tuple(_first_supported(rv.get("tts_candidates"))),
            supported_languages=_supported_languages(rv),
            warmup_enabled=bool(rv.get("warmup", True)),
            stt_warmup_passes=max(1, int(rv.get("stt_warmup_passes", 3))),
            debug_audio=bool(rv.get("debug_audio", False)),
            debug_audio_dir=str(rv.get("debug_audio_dir", "/tmp/realtime_audio")),
        )

    @classmethod
    def resolve(cls, config_dir: str | Path | None = None) -> "RealtimeSettings":
        """Resolve from the shared config block (the single source of truth)."""
        return cls.from_config(config_dir)


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
    """Databricks CLI profile for SDK auth, from the config file only.

    Real value lives in ``config.local.yaml`` for local dev; on Databricks Apps
    ``databricks.profile`` is empty and the injected service-principal OAuth
    creds are used (``profile=None``). An unfilled template placeholder like
    ``"<your-databricks-profile>"`` is treated as unset.
    """
    profile = (_load_merged(config_dir).get("databricks") or {}).get("profile") or None
    if profile and str(profile).startswith("<"):
        return None
    return profile


def _databricks_block(config_dir: str | Path | None = None) -> dict[str, Any]:
    return _load_merged(config_dir or config_dir_from_env()).get("databricks") or {}


def delta_catalog(config_dir: str | Path | None = None) -> str:
    """UC catalog holding the benchmark result tables (databricks.catalog)."""
    return str(_databricks_block(config_dir).get("catalog") or "").strip()


def delta_schema(config_dir: str | Path | None = None) -> str:
    """UC schema holding the benchmark result tables (databricks.schema)."""
    return str(_databricks_block(config_dir).get("schema") or "").strip()


def sql_warehouse_id(config_dir: str | Path | None = None) -> str:
    """SQL warehouse used to query the benchmark Delta tables.

    From ``databricks.sql_warehouse_id`` in the config file (source of truth).
    """
    return str(_databricks_block(config_dir).get("sql_warehouse_id") or "").strip()


def config_dir_from_env() -> Path | None:
    """Config directory when ``GENIE_CONFIG`` points at a yaml file (Databricks jobs/apps)."""
    raw = os.getenv("GENIE_CONFIG")
    if not raw:
        return None
    path = Path(raw)
    return path.parent if path.is_file() else path


def resolve_volume_path(template: str, config_dir: str | Path | None = None) -> str:
    cfg = _load_merged(config_dir or config_dir_from_env())
    db = cfg.get("databricks") or {}
    vol = cfg.get("volume") or {}
    return template.format(
        catalog=db.get("catalog", ""),
        schema=db.get("schema", ""),
        batch_volume=vol.get("batch_name", "raw_batch_data"),
        streaming_volume=vol.get("streaming_name", "raw_streaming_data"),
    )


def benchmark_results_dir(config_dir: str | Path | None = None) -> Path:
    """UC Volume directory for multilingual voice benchmark artifacts (from config)."""
    cfg = _load_merged(config_dir or config_dir_from_env())
    template = str((cfg.get("volume") or {}).get("multilingual_voice_benchmark_path") or "").strip()
    if not template:
        raise RuntimeError("volume.multilingual_voice_benchmark_path is not configured")
    return Path(resolve_volume_path(template, config_dir))


def benchmark_summary_path(config_dir: str | Path | None = None) -> Path:
    return benchmark_results_dir(config_dir) / "summary.json"


def benchmark_api_host(config_dir: str | Path | None = None) -> str:
    block = (_load_merged(config_dir or config_dir_from_env()).get("realtime_voice") or {}).get("benchmark") or {}
    host = str(block.get("api_host") or "").strip()
    if host:
        return host.rstrip("/")
    raise RuntimeError("Set realtime_voice.benchmark.api_host in config")


def benchmark_api_prefix(config_dir: str | Path | None = None) -> str:
    block = (_load_merged(config_dir or config_dir_from_env()).get("realtime_voice") or {}).get("benchmark") or {}
    return str(block.get("api_prefix") or "realtime").strip("/")


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
